//! Dalivinci: an AI service that speaks the OpenAI chat-completions API behind the AI Assistant
//! panel: a local model through Ollama (`http://localhost:11434/v1`), Gemini's OpenAI-compatible
//! endpoint, or any other service that offers one (desktop only).
//!
//! Unlike Claude Code, these services don't run tools themselves, so this module is the agent
//! loop: it starts PhotoCraft's MCP server in bridge mode ([`McpLink`]) as an MCP client, offers the
//! model the bridge tools it may use ([`crate::ai_tools::allowed`]) as functions, runs the calls the
//! model asks for, and hands back their results (a screenshot or preview goes back as an image, for
//! vision models), until the model answers without asking for more. Each turn runs on a background
//! thread; nothing is sent anywhere until the user sends a message.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};

use crate::ai_panel::{Assistant, Event, McpLink};
use crate::ai_tools::{allowed, describe, instructions};

/// Most tool-call rounds in one turn, so a confused model can't loop forever.
const MAX_ROUNDS: usize = 24;
/// Longest a single request may take: local models on a CPU can be slow.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Most characters of one tool result sent back to the model.
const MAX_RESULT_CHARS: usize = 20_000;
/// Most characters of a tool's description sent to the model.
const MAX_DESCRIPTION_CHARS: usize = 1_200;
/// Most lines read from the MCP server while waiting for one reply.
const MAX_MCP_LINES: usize = 10_000;
/// Waits before retrying a request the service turned away as busy (500, 502, 503, 504).
const BUSY_RETRIES: [Duration; 3] = [Duration::from_secs(3), Duration::from_secs(8), Duration::from_secs(20)];

/// Added to the instructions for these services: they see images only when a tool returns one.
const HOW_TO_LOOK: &str = " To see the image, call doc_render_preview; the picture comes back in the next message.";

/// Which service, where, and how to sign in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceConfig {
    /// For messages, e.g. "Ollama" or "Gemini".
    pub label: &'static str,
    /// The API's base address, ending before `/chat/completions`.
    pub base_url: String,
    pub model: String,
    /// Sent as a bearer token; `None` for local services.
    pub api_key: Option<String>,
}

pub struct OpenAiCompatible {
    config: ServiceConfig,
    can_edit: bool,
    link: Option<McpLink>,
    /// The conversation so far, as sent to the service (images dropped after their turn).
    history: Arc<Mutex<Vec<Value>>>,
    rx: Option<Receiver<Event>>,
    stop: Arc<AtomicBool>,
    said: bool,
}

impl OpenAiCompatible {
    pub fn new(config: ServiceConfig, can_edit: bool) -> Self {
        OpenAiCompatible { config, can_edit, link: None, history: Arc::default(), rx: None, stop: Arc::default(), said: false }
    }
}

impl Assistant for OpenAiCompatible {
    fn name(&self) -> &'static str {
        self.config.label
    }

    fn needs_bridge(&self) -> bool {
        true
    }

    fn connect(&mut self, link: McpLink) {
        self.link = Some(link);
    }

    fn start(&mut self, prompt: &str, _now: f64) {
        self.stop();
        self.said = false;
        self.stop = Arc::default();
        let (tx, rx) = channel();
        self.rx = Some(rx);
        let Some(link) = self.link.clone() else {
            let _ = tx.send(Event::Text(format!("{} isn't connected to this window.", self.config.label)));
            let _ = tx.send(Event::Done);
            return;
        };
        let turn = Turn { config: self.config.clone(), can_edit: self.can_edit, link, history: Arc::clone(&self.history), stop: Arc::clone(&self.stop), tx };
        let prompt = prompt.to_owned();
        std::thread::spawn(move || turn.run(&prompt));
    }

    fn poll(&mut self, _now: f64) -> Vec<Event> {
        let Some(rx) = self.rx.as_ref() else { return Vec::new() };
        let mut events = Vec::new();
        for event in rx.try_iter() {
            match event {
                Event::Text(text) => {
                    let text = text.trim();
                    if !text.is_empty() {
                        events.push(Event::Text(if self.said { format!("\n\n{text}") } else { text.to_owned() }));
                        self.said = true;
                    }
                }
                e => events.push(e),
            }
        }
        if events.contains(&Event::Done) {
            self.rx = None;
        }
        events
    }

    fn stop(&mut self) {
        // The worker notices between steps; whatever it was waiting for is then ignored.
        self.stop.store(true, Ordering::Relaxed);
        self.rx = None;
    }
}

impl Drop for OpenAiCompatible {
    fn drop(&mut self) {
        self.stop();
    }
}

/// One turn, on its own thread.
struct Turn {
    config: ServiceConfig,
    can_edit: bool,
    link: McpLink,
    history: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    tx: Sender<Event>,
}

impl Turn {
    fn run(self, prompt: &str) {
        if let Err(e) = self.run_inner(prompt) {
            let _ = self.tx.send(Event::Text(e));
        }
        let _ = self.tx.send(Event::Done);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// A chat request, retried a few times while the service says it is busy. None if Stop was
    /// pressed while waiting.
    fn chat_retrying(&self, agent: &ureq::Agent, messages: &[Value], tools: &[Value]) -> Option<Result<Value, String>> {
        let mut waits = BUSY_RETRIES.iter();
        loop {
            match chat(agent, &self.config, messages, tools) {
                Err(ChatError { status: Some(500 | 502..=504), message }) => {
                    let Some(&wait) = waits.next() else { return Some(Err(message)) };
                    let _ = self.tx.send(Event::Action(format!("{} is busy; trying again in {} s", self.config.label, wait.as_secs())));
                    let until = std::time::Instant::now() + wait;
                    while std::time::Instant::now() < until {
                        if self.stopped() {
                            return None;
                        }
                        std::thread::sleep(Duration::from_millis(200));
                    }
                }
                other => return Some(other.map_err(|e| e.message)),
            }
        }
    }

    fn run_inner(&self, prompt: &str) -> Result<(), String> {
        let mut mcp = Mcp::start(&self.link)?;
        let tools = mcp.tools(self.can_edit)?;
        let mut messages = self.history.lock().unwrap_or_else(PoisonError::into_inner).clone();
        if messages.is_empty() {
            messages.push(json!({ "role": "system", "content": format!("{}{HOW_TO_LOOK}", instructions(self.can_edit)) }));
        }
        messages.push(json!({ "role": "user", "content": prompt }));
        let agent = agent();
        let mut finished = false;
        for _ in 0..MAX_ROUNDS {
            if self.stopped() {
                return Ok(());
            }
            let reply = match self.chat_retrying(&agent, &messages, &tools) {
                Some(reply) => reply?,
                None => return Ok(()),
            };
            if self.stopped() {
                return Ok(());
            }
            let text = reply.get("content").and_then(Value::as_str).unwrap_or_default().to_owned();
            if !text.trim().is_empty() {
                let _ = self.tx.send(Event::Text(text.clone()));
            }
            let calls = tool_calls(&reply);
            messages.push(assistant_message(&text, &reply));
            if calls.is_empty() {
                finished = true;
                break;
            }
            let mut images = Vec::new();
            for call in calls {
                if self.stopped() {
                    return Ok(());
                }
                let (content, mut pictures) = if allowed(&call.name, self.can_edit) {
                    let _ = self.tx.send(Event::Action(describe(&call.name, Some(&call.arguments))));
                    mcp.call(&call.name, &call.arguments)
                } else {
                    (format!("Error: the tool {} isn't available to you.", call.name), Vec::new())
                };
                images.append(&mut pictures);
                messages.push(json!({ "role": "tool", "tool_call_id": call.id, "content": content }));
            }
            if !images.is_empty() {
                let mut parts = vec![json!({ "type": "text", "text": "The image returned by the tool:" })];
                parts.extend(images.into_iter().map(|url| json!({ "type": "image_url", "image_url": { "url": url } })));
                messages.push(json!({ "role": "user", "content": parts }));
            }
        }
        if !finished {
            let _ = self.tx.send(Event::Text(format!("Stopped after {MAX_ROUNDS} rounds of tool calls without a final answer.")));
        }
        *self.history.lock().unwrap_or_else(PoisonError::into_inner) = without_images(messages);
        Ok(())
    }
}

/// A tool call the model asked for.
#[derive(Debug, PartialEq)]
struct Call {
    id: String,
    name: String,
    arguments: Value,
}

/// The tool calls in an assistant message. Arguments come as a JSON string (OpenAI, Gemini) or an
/// object (some local servers); anything unreadable becomes `{}`.
fn tool_calls(message: &Value) -> Vec<Call> {
    let Some(calls) = message.get("tool_calls").and_then(Value::as_array) else { return Vec::new() };
    calls
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let f = c.get("function")?;
            let name = f.get("name")?.as_str()?.to_owned();
            let arguments = match f.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                Some(v @ Value::Object(_)) => v.clone(),
                _ => json!({}),
            };
            let id = c.get("id").and_then(Value::as_str).map_or_else(|| format!("call_{i}"), str::to_owned);
            Some(Call { id, name, arguments })
        })
        .collect()
}

/// The assistant message as it goes back into the conversation.
fn assistant_message(text: &str, reply: &Value) -> Value {
    let mut m = json!({ "role": "assistant", "content": text });
    if let Some(calls) = reply.get("tool_calls").filter(|c| c.as_array().is_some_and(|a| !a.is_empty())) {
        m["tool_calls"] = calls.clone();
    }
    m
}

/// The conversation with each turn's images replaced by a note, so later turns stay small.
fn without_images(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|mut m| {
            if m.get("role").and_then(Value::as_str) == Some("user") && m.get("content").is_some_and(Value::is_array) {
                m["content"] = json!("(An image the tool returned was shown here.)");
            }
            m
        })
        .collect()
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(REQUEST_TIMEOUT)).http_status_as_error(false).build().into()
}

/// A failed chat request: the HTTP status, if the service answered, and what to tell the user.
#[derive(Debug)]
struct ChatError {
    status: Option<u16>,
    message: String,
}

impl From<String> for ChatError {
    fn from(message: String) -> Self {
        ChatError { status: None, message }
    }
}

/// One chat-completions request; returns the reply message.
fn chat(agent: &ureq::Agent, config: &ServiceConfig, messages: &[Value], tools: &[Value]) -> Result<Value, ChatError> {
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let mut body = json!({ "model": config.model, "messages": messages });
    if !tools.is_empty() {
        body["tools"] = json!(tools);
        body["tool_choice"] = json!("auto");
    }
    let mut request = agent.post(&url);
    if let Some(key) = config.api_key.as_deref().filter(|k| !k.is_empty()) {
        request = request.header("Authorization", &format!("Bearer {key}"));
    }
    let mut response = request.send_json(&body).map_err(|e| unreachable_message(config, &e.to_string()))?;
    let status = response.status().as_u16();
    let reply = response.body_mut().read_json::<Value>();
    // An error reply may not be JSON (a gateway's busy page); its status still counts.
    let reply = match reply {
        Ok(reply) => reply,
        Err(_) if !(200..300).contains(&status) => Value::Null,
        Err(e) => return Err(format!("{}: the reply wasn't readable ({e}).", config.label).into()),
    };
    if !(200..300).contains(&status) {
        return Err(ChatError { status: Some(status), message: http_error(config, status, &reply) });
    }
    Ok(reply.pointer("/choices/0/message").cloned().ok_or_else(|| format!("{}: the reply had no message.", config.label))?)
}

fn unreachable_message(config: &ServiceConfig, error: &str) -> String {
    if config.api_key.is_none() {
        format!("Couldn't reach {} at {} ({error}). Is it running? For Ollama: `ollama serve`.", config.label, config.base_url)
    } else {
        format!("Couldn't reach {} ({error}). Check the internet connection.", config.label)
    }
}

fn http_error(config: &ServiceConfig, status: u16, reply: &Value) -> String {
    let detail = reply
        .pointer("/error/message")
        .or_else(|| reply.pointer("/0/error/message"))
        .or_else(|| reply.get("error"))
        .and_then(Value::as_str)
        .unwrap_or("no details");
    let hint = match status {
        401 | 403 => " Check the API key.",
        404 => " Check the model name and address.",
        429 => {
            " The service's rate limit or quota was reached: wait a minute, or pick another model. Free tiers allow only a few requests a minute, and some models (image generators especially) have no free quota."
        }
        500 | 502..=504 => " The model is busy right now: try again shortly, or pick another model.",
        _ => "",
    };
    format!("{} answered with an error ({status}): {detail}.{hint}", config.label)
}

/// The models a service offers (`GET /models`), for the panel's model list.
pub fn list_models(config: &ServiceConfig) -> Result<Vec<String>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).http_status_as_error(false).build().into();
    let mut request = agent.get(&format!("{}/models", config.base_url.trim_end_matches('/')));
    if let Some(key) = config.api_key.as_deref().filter(|k| !k.is_empty()) {
        request = request.header("Authorization", &format!("Bearer {key}"));
    }
    let mut response = request.call().map_err(|e| unreachable_message(config, &e.to_string()))?;
    let status = response.status().as_u16();
    let reply: Value = response.body_mut().read_json().map_err(|e| format!("{}: the model list wasn't readable ({e}).", config.label))?;
    if !(200..300).contains(&status) {
        return Err(http_error(config, status, &reply));
    }
    let mut models: Vec<String> = reply
        .get("data")
        .and_then(Value::as_array)
        .map(|d| d.iter().filter_map(|m| m.get("id").and_then(Value::as_str)).map(|id| id.trim_start_matches("models/").to_owned()).collect())
        .unwrap_or_default();
    models.sort();
    Ok(models)
}

/// A minimal MCP client for PhotoCraft's server in bridge mode, over its stdin and stdout.
struct Mcp {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next_id: u64,
}

impl Mcp {
    fn start(link: &McpLink) -> Result<Self, String> {
        let _ = std::fs::create_dir_all(&link.work_dir);
        let mut child = Command::new(&link.command)
            .args(&link.args)
            .current_dir(&link.work_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Couldn't start Dalivinci's helper ({}): {e}", link.command.display()))?;
        let (Some(input), Some(output)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            return Err("Couldn't talk to Dalivinci's helper.".into());
        };
        let mut mcp = Mcp { child, input, output: BufReader::new(output), next_id: 1 };
        mcp.request("initialize", json!({ "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "dalivinci", "version": "1" } }))?;
        mcp.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))?;
        Ok(mcp)
    }

    fn send(&mut self, message: &Value) -> Result<(), String> {
        writeln!(self.input, "{message}").and_then(|()| self.input.flush()).map_err(|e| format!("Dalivinci's helper stopped: {e}"))
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        let mut line = String::new();
        for _ in 0..MAX_MCP_LINES {
            line.clear();
            match self.output.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let Ok(reply) = serde_json::from_str::<Value>(&line) else { continue };
            if reply.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = reply.get("error") {
                return Err(format!("Dalivinci's helper: {}", error.get("message").and_then(Value::as_str).unwrap_or("error")));
            }
            return Ok(reply.get("result").cloned().unwrap_or(Value::Null));
        }
        Err("Dalivinci's helper stopped answering.".into())
    }

    /// The bridge tools the model may use, as chat-completions functions.
    fn tools(&mut self, can_edit: bool) -> Result<Vec<Value>, String> {
        let listed = self.request("tools/list", json!({}))?;
        Ok(functions(&listed, can_edit))
    }

    /// Run a tool: its text, and its images as data URLs.
    fn call(&mut self, name: &str, arguments: &Value) -> (String, Vec<String>) {
        match self.request("tools/call", json!({ "name": name, "arguments": arguments })) {
            Ok(result) => tool_result(&result),
            Err(e) => (format!("Error: {e}"), Vec::new()),
        }
    }
}

impl Drop for Mcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `tools/list`'s tools the model may use, as chat-completions functions.
fn functions(listed: &Value, can_edit: bool) -> Vec<Value> {
    listed
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    let name = t.get("name")?.as_str()?;
                    allowed(name, can_edit).then(|| {
                        let description: String =
                            t.get("description").and_then(Value::as_str).unwrap_or_default().chars().take(MAX_DESCRIPTION_CHARS).collect();
                        let parameters = t.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
                        json!({ "type": "function", "function": { "name": name, "description": description, "parameters": parameters } })
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A `tools/call` result as text for the model, plus its images as data URLs.
fn tool_result(result: &Value) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut images = Vec::new();
    for part in result.get("content").and_then(Value::as_array).into_iter().flatten() {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
            }
            Some("image") => {
                if let Some(data) = part.get("data").and_then(Value::as_str) {
                    let mime = part.get("mimeType").and_then(Value::as_str).unwrap_or("image/png");
                    images.push(format!("data:{mime};base64,{data}"));
                }
            }
            _ => {}
        }
    }
    if text.chars().count() > MAX_RESULT_CHARS {
        text = text.chars().take(MAX_RESULT_CHARS).collect::<String>() + "\n… (cut short)";
    }
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        text = format!("Error: {text}");
    }
    if text.is_empty() {
        text = if images.is_empty() { "(no output)".into() } else { "(an image; it follows in the next message)".into() };
    }
    (text, images)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_calls_are_read_from_strings_or_objects() {
        let reply = json!({ "role": "assistant", "content": null, "tool_calls": [
            { "id": "a1", "type": "function", "function": { "name": "doc_inspect", "arguments": "{}" } },
            { "id": "a2", "type": "function", "function": { "name": "command_run", "arguments": { "id": "layer.new.layer" } } },
            { "type": "function", "function": { "name": "command_run", "arguments": "not json" } },
            { "type": "function" }
        ]});
        assert_eq!(
            tool_calls(&reply),
            [
                Call { id: "a1".into(), name: "doc_inspect".into(), arguments: json!({}) },
                Call { id: "a2".into(), name: "command_run".into(), arguments: json!({ "id": "layer.new.layer" }) },
                Call { id: "call_2".into(), name: "command_run".into(), arguments: json!({}) },
            ]
        );
        assert!(tool_calls(&json!({ "content": "hi" })).is_empty());
        assert_eq!(assistant_message("ok", &json!({ "tool_calls": [] })), json!({ "role": "assistant", "content": "ok" }));
    }

    #[test]
    fn only_allowed_tools_are_offered() {
        let listed = json!({ "tools": [
            { "name": "doc_inspect", "description": "Document state", "inputSchema": { "type": "object" } },
            { "name": "command_run", "description": "Run a command" },
            { "name": "doc_save", "description": "Save" },
            { "name": "control_call" }
        ]});
        let names = |fs: Vec<Value>| fs.iter().map(|f| f["function"]["name"].as_str().unwrap().to_owned()).collect::<Vec<_>>();
        assert_eq!(names(functions(&listed, true)), ["doc_inspect", "command_run"]);
        assert_eq!(names(functions(&listed, false)), ["doc_inspect"]);
        assert_eq!(functions(&listed, true)[1]["function"]["parameters"], json!({ "type": "object", "properties": {} }));
        assert!(functions(&json!({}), true).is_empty());
    }

    #[test]
    fn tool_results_become_text_and_images() {
        let r = json!({ "content": [{ "type": "image", "data": "AAAA", "mimeType": "image/png" }, { "type": "text", "text": "640x400" }] });
        assert_eq!(tool_result(&r), ("640x400".to_owned(), vec!["data:image/png;base64,AAAA".to_owned()]));
        let failed = json!({ "isError": true, "content": [{ "type": "text", "text": "no document" }] });
        assert_eq!(tool_result(&failed).0, "Error: no document");
        let long = json!({ "content": [{ "type": "text", "text": "x".repeat(MAX_RESULT_CHARS + 50) }] });
        assert!(tool_result(&long).0.ends_with("(cut short)"));
        assert_eq!(tool_result(&json!({})).0, "(no output)");
    }

    #[test]
    fn images_leave_the_conversation_after_their_turn() {
        let messages = vec![
            json!({ "role": "user", "content": "hi" }),
            json!({ "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } }] }),
            json!({ "role": "tool", "tool_call_id": "a", "content": "ok" }),
        ];
        let kept = without_images(messages);
        assert_eq!(kept[0]["content"], "hi");
        assert!(kept[1]["content"].as_str().unwrap().contains("image"));
        assert!(!kept.iter().any(|m| m.to_string().contains("base64")));
    }

    #[test]
    fn service_errors_say_what_to_check() {
        let gemini = ServiceConfig { label: "Gemini", base_url: "https://x/v1".into(), model: "m".into(), api_key: Some("k".into()) };
        assert!(http_error(&gemini, 401, &json!([{ "error": { "message": "API key not valid" } }])).contains("API key not valid. Check the API key."));
        assert!(http_error(&gemini, 404, &json!({ "error": { "message": "model not found" } })).contains("Check the model name"));
        assert!(http_error(&gemini, 429, &json!({ "error": { "message": "quota exceeded" } })).contains("no free quota"));
        let ollama = ServiceConfig { label: "Ollama", base_url: "http://localhost:11434/v1".into(), model: "qwen3-vl:30b".into(), api_key: None };
        assert!(unreachable_message(&ollama, "connection refused").contains("ollama serve"));
    }

    #[test]
    fn a_busy_reply_keeps_its_status_even_without_json() {
        use std::io::Read;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = "<html>Service Unavailable</html>";
            let _ = write!(
                stream,
                "HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        });
        let config = ServiceConfig { label: "Gemini", base_url: format!("http://127.0.0.1:{port}/v1"), model: "m".into(), api_key: Some("k".into()) };
        let err = chat(&agent(), &config, &[json!({ "role": "user", "content": "hi" })], &[]).unwrap_err();
        assert_eq!(err.status, Some(503));
        assert!(err.message.contains("busy"), "{}", err.message);
    }

    #[test]
    fn an_unconnected_service_answers_with_why() {
        let mut ai = OpenAiCompatible::new(ServiceConfig { label: "Ollama", base_url: String::new(), model: String::new(), api_key: None }, true);
        ai.start("hi", 0.0);
        let events = ai.poll(0.0);
        assert!(matches!(events.first(), Some(Event::Text(t)) if t.contains("isn't connected")), "{events:?}");
        assert_eq!(events.last(), Some(&Event::Done));
    }
}

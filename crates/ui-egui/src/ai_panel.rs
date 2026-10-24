//! Window › AI Assistant (Dalivinci): a side panel, between the canvas and the dock, where you chat
//! with an AI that edits the open document. The panel shows the conversation, an action log of the
//! commands the AI ran, and a Stop button that ends the AI's turn at once.
//!
//! The AI behind the panel is an [`Assistant`], picked at the top of the panel ([`Provider`]):
//! [`Echo`], a stand-in that repeats what you typed and runs no commands, or Claude Code
//! (`ai_claude`), which reaches this window through PhotoCraft's MCP bridge and, for now, can only
//! look. Assistants report what they do as [`Event`]s, polled each frame, so a slow one never
//! blocks the UI.
//!
//! Automation drives the panel like the menu does: `window.panel.aiAssistant` shows or hides it,
//! `ai.provider` (`{"provider": "echo" | "claudeCode"}`) picks the AI, `ai.send` (`{"text": …}`),
//! `ai.stop` and `ai.clear` act on the conversation, and its state is in
//! `UiState::ai` for the control channel to read.
//!
//! Its text is English only for now: the fork's strings aren't in PhotoCraft's translation catalogs.

use egui::{Align, Layout, RichText, Sense};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::menus::MenuItem;
use crate::theme::Tokens;
use crate::{ControlRequest, PhotocraftApp};

/// Window › AI Assistant.
pub const TOGGLE: &str = "window.panel.aiAssistant";
const SEND: &str = "ai.send";
const STOP: &str = "ai.stop";
const CLEAR: &str = "ai.clear";
const PROVIDER: &str = "ai.provider";
const SETTINGS: &str = "ai.settings";

/// The longest message the panel accepts, in characters. A chat message, not a document.
const MAX_INPUT_CHARS: usize = 8_000;
/// Conversation entries kept; older ones are dropped first.
const MAX_MESSAGES: usize = 200;
/// Action log entries kept.
const MAX_ACTIONS: usize = 500;

/// Who wrote a conversation entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    /// A note from the panel itself (an error, a stopped turn).
    Notice,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

/// Which AI the panel talks to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Provider {
    #[default]
    Echo,
    /// Claude Code, allowed to edit the open document.
    ClaudeCode,
    /// Claude Code, only allowed to look.
    ClaudeCodeLookOnly,
    /// A local model through Ollama's OpenAI-compatible API.
    Ollama,
    /// Google's Gemini through its OpenAI-compatible API (needs an API key).
    Gemini,
}

/// Gemini's OpenAI-compatible API.
pub const GEMINI_URL: &str = "https://generativelanguage.googleapis.com/v1beta/openai";

impl Provider {
    pub const ALL: [Provider; 5] = [Provider::Echo, Provider::ClaudeCode, Provider::ClaudeCodeLookOnly, Provider::Ollama, Provider::Gemini];

    pub fn label(self) -> &'static str {
        match self {
            Provider::Echo => "Echo (test AI)",
            Provider::ClaudeCode => "Claude Code",
            Provider::ClaudeCodeLookOnly => "Claude Code (look only)",
            Provider::Ollama => "Ollama (local)",
            Provider::Gemini => "Gemini",
        }
    }

    /// Sends the conversation, and pictures of the document, off this computer.
    pub fn online(self) -> bool {
        matches!(self, Provider::ClaudeCode | Provider::ClaudeCodeLookOnly | Provider::Gemini)
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    fn assistant(self, settings: &AiSettings, gemini_key: Option<&str>) -> Box<dyn Assistant> {
        #[cfg(not(target_arch = "wasm32"))]
        use crate::ai_openai::{OpenAiCompatible, ServiceConfig};
        match self {
            Provider::Echo => Box::new(Echo::default()),
            #[cfg(not(target_arch = "wasm32"))]
            Provider::ClaudeCode => Box::new(crate::ai_claude::ClaudeCode::new(true)),
            #[cfg(not(target_arch = "wasm32"))]
            Provider::ClaudeCodeLookOnly => Box::new(crate::ai_claude::ClaudeCode::new(false)),
            #[cfg(not(target_arch = "wasm32"))]
            Provider::Ollama => Box::new(OpenAiCompatible::new(
                ServiceConfig { label: "Ollama", base_url: settings.ollama_url.clone(), model: settings.ollama_model.clone(), api_key: None },
                true,
            )),
            #[cfg(not(target_arch = "wasm32"))]
            Provider::Gemini => match gemini_key.filter(|k| !k.trim().is_empty()) {
                Some(key) => Box::new(OpenAiCompatible::new(
                    ServiceConfig {
                        label: "Gemini",
                        base_url: settings.gemini_url.clone(),
                        model: settings.gemini_model.clone(),
                        api_key: Some(key.trim().to_owned()),
                    },
                    true,
                )),
                None => Box::new(Unavailable("Gemini needs an API key: paste it under Connection above (get one free at aistudio.google.com).")),
            },
            #[cfg(target_arch = "wasm32")]
            _ => Box::new(Unavailable("This AI needs the desktop app.")),
        }
    }
}

/// Where the AI services are and which models to use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AiSettings {
    pub ollama_url: String,
    pub ollama_model: String,
    pub gemini_model: String,
    /// Gemini's API address (its OpenAI-compatible endpoint), changeable for a proxy.
    pub gemini_url: String,
}

impl Default for AiSettings {
    fn default() -> Self {
        AiSettings {
            ollama_url: "http://localhost:11434/v1".into(),
            ollama_model: "qwen3-vl:30b".into(),
            gemini_model: "gemini-2.5-flash".into(),
            gemini_url: GEMINI_URL.into(),
        }
    }
}

/// How an AI's helper reaches this window: the command that runs PhotoCraft's MCP server in bridge
/// mode, and the private control connection it talks to (see `Services::ai_bridge`).
pub struct Bridge {
    pub mcp_command: std::path::PathBuf,
    pub mcp_args: Vec<String>,
    /// An empty folder the AI program runs in, so it doesn't pick up a project from wherever the
    /// app was started.
    pub work_dir: std::path::PathBuf,
    /// Requests from the bridge, handled like the control server's.
    pub control: std::sync::mpsc::Receiver<ControlRequest>,
    /// The desktop automation policy (no ambient file access, …), installed for the AI's requests
    /// when the app wasn't started with `--control`, which installs it already.
    pub authorize_command: crate::AutomationCommandFn,
    pub authorize_step: fn(&str, &serde_json::Value) -> photocraft_engine::Result<()>,
}

/// The part of a [`Bridge`] an assistant keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpLink {
    pub command: std::path::PathBuf,
    pub args: Vec<String>,
    pub work_dir: std::path::PathBuf,
}

/// The current assistant, made on first use. Not part of the UI state's data: a clone starts with
/// none and makes its own.
#[derive(Default)]
struct Slot(Option<Box<dyn Assistant>>);

impl Clone for Slot {
    fn clone(&self) -> Self {
        Slot(None)
    }
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Slot(assistant)" } else { "Slot(empty)" })
    }
}

/// The Connection section's state: the key being typed, and a service's model list.
#[derive(Default)]
struct ConnectionForm {
    key_input: String,
    models: Vec<String>,
    models_rx: Option<std::sync::mpsc::Receiver<Result<Vec<String>, String>>>,
    note: Option<String>,
}

impl Clone for ConnectionForm {
    fn clone(&self) -> Self {
        ConnectionForm::default()
    }
}

impl std::fmt::Debug for ConnectionForm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionForm").field("models", &self.models.len()).finish_non_exhaustive()
    }
}

/// Gemini's key file in the AI folder, readable only by this user.
const GEMINI_KEY_FILE: &str = "gemini.key";

/// Read Gemini's key once: its private file, else `GEMINI_API_KEY`.
fn load_gemini_key(app: &mut PhotocraftApp) {
    if app.ui.ai.key_loaded {
        return;
    }
    app.ui.ai.key_loaded = true;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let from_file = app.services.ai_dir.as_ref().and_then(|d| std::fs::read_to_string(d.join(GEMINI_KEY_FILE)).ok());
        let from_env = || std::env::var("GEMINI_API_KEY").ok();
        app.ui.ai.gemini_key = from_file.or_else(from_env).map(|k| k.trim().to_owned()).filter(|k| !k.is_empty());
    }
}

/// Save (or, with an empty key, forget) Gemini's key in its private file.
#[cfg(not(target_arch = "wasm32"))]
fn store_gemini_key(app: &mut PhotocraftApp, key: &str) -> Result<(), String> {
    let dir = app.services.ai_dir.clone().ok_or("there's no settings folder to keep the key in")?;
    let path = dir.join(GEMINI_KEY_FILE);
    let key = key.trim();
    if key.is_empty() {
        let _ = std::fs::remove_file(&path);
        app.ui.ai.gemini_key = None;
    } else {
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        std::io::Write::write_all(&mut file, key.as_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        app.ui.ai.gemini_key = Some(key.to_owned());
    }
    app.ui.ai.reconfigure();
    Ok(())
}

/// AI Assistant panel view state. The conversation is exposed to automation but never restored
/// from settings: only whether the panel is open survives a restart.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AiPanelUi {
    #[serde(default)]
    pub open: bool,
    /// The AI picked at the top of the panel.
    #[serde(default)]
    pub provider: Provider,
    /// Only AIs that run on this computer may be used: nothing is sent off it.
    #[serde(default)]
    pub local_only: bool,
    #[serde(default)]
    pub settings: AiSettings,
    /// Gemini's API key, read from its private file (or `GEMINI_API_KEY`) when first needed. Never
    /// in the settings file.
    #[serde(skip)]
    gemini_key: Option<String>,
    #[serde(skip)]
    key_loaded: bool,
    /// The connection section's edits, and the models a service listed.
    #[serde(skip)]
    form: ConnectionForm,
    #[serde(default, skip_deserializing)]
    pub messages: Vec<Message>,
    /// What the AI did, oldest first: one line per command it ran.
    #[serde(default, skip_deserializing)]
    pub actions: Vec<String>,
    /// The AI is answering (its reply is still arriving).
    #[serde(default, skip_deserializing)]
    pub busy: bool,
    /// The message being typed.
    #[serde(skip)]
    pub input: String,
    #[serde(skip)]
    assistant: Slot,
    /// The bridge's link once it started (it starts on the first message to an AI that needs it).
    #[serde(skip)]
    link: Option<McpLink>,
    #[serde(skip)]
    turn_start: Option<TurnStart>,
}

impl PartialEq for AiPanelUi {
    fn eq(&self, other: &Self) -> bool {
        self.open == other.open
            && self.provider == other.provider
            && self.local_only == other.local_only
            && self.settings == other.settings
            && self.messages == other.messages
            && self.actions == other.actions
            && self.busy == other.busy
    }
}

/// What an [`Assistant`] reports while it works on a turn.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// More of the reply's text.
    Text(String),
    /// A command it ran, as one line for the action log.
    Action(String),
    /// The turn is over.
    Done,
}

/// An AI the panel can talk to. `start` begins a turn; `poll` returns what happened since the last
/// poll (`now` is the UI clock in seconds) and must not block; `stop` ends the turn early.
pub trait Assistant {
    /// Shown at the top of the panel, e.g. "Echo (test AI)".
    fn name(&self) -> &'static str;
    /// Whether it needs the MCP bridge; [`AiPanelUi::send`] hands it over with [`Self::connect`].
    fn needs_bridge(&self) -> bool {
        false
    }
    fn connect(&mut self, _link: McpLink) {}
    fn start(&mut self, prompt: &str, now: f64);
    fn poll(&mut self, now: f64) -> Vec<Event>;
    fn stop(&mut self);
}

/// The stand-in AI: replies by repeating your message, streamed a few characters at a time so the
/// Stop button has something to stop, and runs no commands.
#[derive(Clone, Debug, Default)]
pub struct Echo {
    /// The whole reply, its characters, and how many have been sent; `None` between turns.
    turn: Option<(Vec<char>, usize, f64)>,
}

impl Echo {
    /// Characters per second.
    const SPEED: f64 = 80.0;
}

impl Assistant for Echo {
    fn name(&self) -> &'static str {
        "Echo (test AI)"
    }

    fn start(&mut self, prompt: &str, now: f64) {
        let reply = format!(
            "You said: \u{201c}{}\u{201d}\n\nI'm the stand-in echo AI, so I only repeat what you type and can't change your image yet. A real AI connects in the next milestone.",
            prompt.trim()
        );
        self.turn = Some((reply.chars().collect(), 0, now));
    }

    fn poll(&mut self, now: f64) -> Vec<Event> {
        let Some((chars, sent, started)) = self.turn.as_mut() else { return Vec::new() };
        let due = (((now - *started).max(0.0) * Self::SPEED) as usize).min(chars.len());
        let mut events = Vec::new();
        if due > *sent {
            events.push(Event::Text(chars.get(*sent..due).unwrap_or_default().iter().collect()));
            *sent = due;
        }
        if *sent >= chars.len() {
            events.push(Event::Action("Echo AI: no commands run".into()));
            events.push(Event::Done);
            self.turn = None;
        }
        events
    }

    fn stop(&mut self) {
        self.turn = None;
    }
}

/// An AI that can't run (not on this build, or not set up): every turn answers with why.
struct Unavailable(&'static str);

impl Assistant for Unavailable {
    fn name(&self) -> &'static str {
        "Unavailable"
    }
    fn start(&mut self, _prompt: &str, _now: f64) {}
    fn poll(&mut self, _now: f64) -> Vec<Event> {
        vec![Event::Text(self.0.into()), Event::Done]
    }
    fn stop(&mut self) {}
}

impl AiPanelUi {
    fn assistant(&mut self) -> &mut Box<dyn Assistant> {
        let (provider, settings, key) = (self.provider, self.settings.clone(), self.gemini_key.clone());
        self.assistant.0.get_or_insert_with(|| provider.assistant(&settings, key.as_deref()))
    }

    /// Talk to another AI: ends the current turn and starts a new conversation. With Local only
    /// on, an online AI is refused.
    pub fn set_provider(&mut self, provider: Provider) -> Result<(), String> {
        if self.local_only && provider.online() {
            return Err(format!("{} sends your conversation and pictures off this computer: turn off Local only to use it", provider.label()));
        }
        if provider != self.provider {
            self.clear();
            self.provider = provider;
            self.assistant.0 = None;
            self.form.models.clear();
            self.form.models_rx = None;
        }
        Ok(())
    }

    /// Turn Local only on or off. Turning it on while an online AI is picked switches to Ollama.
    pub fn set_local_only(&mut self, on: bool) {
        self.local_only = on;
        if on && self.provider.online() {
            let _ = self.set_provider(Provider::Ollama);
        }
    }

    /// New service settings or key: the next message uses them (and the AI starts a new memory).
    fn reconfigure(&mut self) {
        if !self.busy {
            self.assistant.0 = None;
        }
    }

    /// Whether the current AI needs the bridge and doesn't have it yet.
    pub fn wants_bridge(&mut self) -> bool {
        self.link.is_none() && self.assistant().needs_bridge()
    }

    /// Send `text` as your message and start the AI's turn. Empty text, or a send while the AI is
    /// still answering, is refused.
    pub fn send(&mut self, text: &str, now: f64) -> Result<(), String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("type a message first".into());
        }
        if text.chars().count() > MAX_INPUT_CHARS {
            return Err(format!("messages are limited to {MAX_INPUT_CHARS} characters"));
        }
        if self.busy {
            return Err("the AI is still answering: stop it first".into());
        }
        if self.local_only && self.provider.online() {
            return Err("Local only is on: pick an AI that runs on this computer".into());
        }
        if let Some(link) = self.link.clone() {
            self.assistant().connect(link);
        }
        self.push(Role::User, text.into());
        self.push(Role::Assistant, String::new());
        self.busy = true;
        self.assistant().start(text, now);
        Ok(())
    }

    /// End the AI's turn now, keeping what it already said.
    pub fn stop(&mut self) {
        if self.busy {
            self.assistant().stop();
            self.busy = false;
            self.push(Role::Notice, "Stopped.".into());
        }
    }

    /// Start a new conversation.
    pub fn clear(&mut self) {
        if let Some(a) = self.assistant.0.as_mut() {
            a.stop();
        }
        // A new conversation, also for the AI: its memory of this one goes too.
        self.assistant.0 = None;
        self.busy = false;
        self.messages.clear();
        self.actions.clear();
    }

    /// Take in what the AI did since the last frame.
    pub fn poll(&mut self, now: f64) {
        if !self.busy {
            return;
        }
        for event in self.assistant().poll(now) {
            match event {
                Event::Text(more) => match self.messages.last_mut() {
                    Some(m) if m.role == Role::Assistant => m.text.push_str(&more),
                    _ => self.push(Role::Assistant, more),
                },
                Event::Action(line) => {
                    self.actions.push(line);
                    let over = self.actions.len().saturating_sub(MAX_ACTIONS);
                    self.actions.drain(..over);
                }
                Event::Done => self.busy = false,
            }
        }
    }

    pub fn assistant_name(&self) -> &'static str {
        self.provider.label()
    }

    fn push(&mut self, role: Role, text: String) {
        self.messages.push(Message { role, text });
        let over = self.messages.len().saturating_sub(MAX_MESSAGES);
        self.messages.drain(..over);
    }
}

/// Send `text` from the panel or automation: starts the bridge first when the AI needs it.
pub fn send(app: &mut PhotocraftApp, text: &str) -> Result<(), String> {
    load_gemini_key(app);
    if !app.ui.ai.busy && !text.trim().is_empty() && app.ui.ai.wants_bridge() {
        let link = connect_bridge(app)?;
        app.ui.ai.link = Some(link);
    }
    let now = app.ui.ai_clock;
    let before = app.session.active().map(|d| d.doc.clone());
    app.ui.ai.send(text, now)?;
    app.ui.ai.turn_start = before.map(|doc| TurnStart { doc, label: step_label(text) });
    Ok(())
}

/// The document as the AI's turn found it, so its edits can become one undo step.
#[derive(Clone)]
pub(crate) struct TurnStart {
    doc: std::sync::Arc<photocraft_doc::Document>,
    label: String,
}

impl std::fmt::Debug for TurnStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnStart").field("label", &self.label).finish_non_exhaustive()
    }
}

/// History label for an AI turn: "AI: " and the start of what was asked.
fn step_label(prompt: &str) -> String {
    const MAX: usize = 40;
    let words = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if words.chars().count() <= MAX {
        return format!("AI: {words}");
    }
    let cut: String = words.chars().take(MAX).collect();
    // Back to the last whole word.
    let short = cut.rfind(' ').and_then(|at| cut.get(..at)).unwrap_or(&cut).trim_end_matches([',', '.', ';', ':']);
    format!("AI: {short}\u{2026}")
}

/// Once a turn is over, fold the steps the AI added to the active document into one, so a single
/// Ctrl+Z (or one History row) takes the whole turn back. Nothing is folded when the turn changed
/// nothing, or when the document the turn started from is no longer in the history (the user
/// undid past it, or the history limit dropped it).
fn fold_turn(app: &mut PhotocraftApp) {
    if app.ui.ai.busy {
        return;
    }
    let Some(start) = app.ui.ai.turn_start.take() else { return };
    let Some(st) = app.session.active_mut() else { return };
    if std::sync::Arc::ptr_eq(&st.doc, &start.doc) {
        return;
    }
    let past = st.history.past_len();
    let Some(at) = (0..past).rev().find(|&i| st.history.state(i).is_some_and(|d| std::sync::Arc::ptr_eq(&d, &start.doc))) else { return };
    for _ in at.saturating_add(1)..past {
        st.history.purge_last();
    }
    st.history.set_current_label(start.label);
    // The next edit is a step of its own, not a continuation of the AI's last one.
    st.coalesce = None;
}

/// Start the private connection an AI program uses to reach this window.
fn connect_bridge(app: &mut PhotocraftApp) -> Result<McpLink, String> {
    let Some(start) = app.services.ai_bridge.as_mut() else {
        return Err("this build of Dalivinci can't connect an AI program".into());
    };
    let bridge = start()?;
    if app.services.automation_command.is_none() {
        app.services.automation_command = Some(bridge.authorize_command);
    }
    if app.services.automation_authorize.is_none() {
        app.services.automation_authorize = Some(bridge.authorize_step);
    }
    // Started with `--control` too: both connections feed one queue.
    app.control_rx = Some(match app.control_rx.take() {
        None => bridge.control,
        Some(existing) => merge(existing, bridge.control),
    });
    Ok(McpLink { command: bridge.mcp_command, args: bridge.mcp_args, work_dir: bridge.work_dir })
}

/// One receiver for the requests of two.
pub(crate) fn merge(a: std::sync::mpsc::Receiver<ControlRequest>, b: std::sync::mpsc::Receiver<ControlRequest>) -> std::sync::mpsc::Receiver<ControlRequest> {
    let (tx, rx) = std::sync::mpsc::channel();
    for source in [a, b] {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for request in source {
                if tx.send(request).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

/// Ids this module owns.
pub fn handles(id: &str) -> bool {
    matches!(id, TOGGLE | SEND | STOP | CLEAR | PROVIDER | SETTINGS)
}

/// Menu checkmark for Window › AI Assistant.
pub fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    (id == TOGGLE).then_some(app.ui.ai.open)
}

/// Run one of this module's ids. `window.panel.aiAssistant` toggles the panel, or sets it with
/// `{"open": bool}`.
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if id == SEND {
        return Some(match params.get("text").and_then(Value::as_str) {
            Some(text) => send(app, text).map(|()| {
                app.ui.ai.open = true;
                json!({ "busy": app.ui.ai.busy })
            }),
            None => Err("ai.send needs {\"text\": \"…\"}".into()),
        });
    }
    let ai = &mut app.ui.ai;
    Some(match id {
        TOGGLE => {
            ai.open = params.get("open").and_then(Value::as_bool).unwrap_or(!ai.open);
            Ok(json!({ "open": ai.open }))
        }
        STOP => {
            ai.stop();
            Ok(json!({ "busy": false }))
        }
        CLEAR => {
            ai.clear();
            Ok(json!({}))
        }
        PROVIDER => match params.get("provider").cloned().map(serde_json::from_value::<Provider>) {
            Some(Ok(p)) => ai.set_provider(p).map(|()| json!({ "provider": p })),
            _ => Err("ai.provider needs {\"provider\": \"echo\" | \"claudeCode\" | \"claudeCodeLookOnly\" | \"ollama\" | \"gemini\"}".into()),
        },
        SETTINGS => {
            let mut merged = serde_json::to_value(&ai.settings).unwrap_or_default();
            if let (Some(m), Some(changes)) = (merged.as_object_mut(), params.as_object()) {
                for (k, v) in changes {
                    if m.contains_key(k) {
                        m.insert(k.clone(), v.clone());
                    }
                }
            }
            match serde_json::from_value::<AiSettings>(merged) {
                Ok(settings) => {
                    ai.settings = settings;
                    ai.reconfigure();
                    serde_json::to_value(&ai.settings).map_err(|e| e.to_string())
                }
                Err(e) => Err(format!("ai.settings: {e}")),
            }
        }
        _ => return None,
    })
}

/// Add Window › AI Assistant after Window › Timeline (or at the end of the Window menu).
pub fn insert_menu_items(app: &PhotocraftApp, items: &mut Vec<MenuItem>) {
    let anchor = items.iter().position(|i| i.id == "window.panel.timeline");
    let Some(path) = anchor.and_then(|a| items.get(a)).map(|i| i.path.clone()).or_else(|| Some(vec!["Window".to_string()])) else { return };
    let item = MenuItem { id: TOGGLE.into(), label: "AI Assistant".into(), path, shortcut: None, enabled: true, checked: Some(app.ui.ai.open), color: None };
    let at = anchor.map_or_else(|| items.iter().rposition(|i| i.path.first().is_some_and(|p| p == "Window")).map_or(items.len(), |p| p + 1), |a| a + 1);
    items.insert(at.min(items.len()), item);
}

/// The panel, docked on the canvas side of the right-hand dock.
pub fn side_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let now = ui.ctx().input(|i| i.time);
    app.ui.ai_clock = now;
    app.ui.ai.poll(now);
    fold_turn(app);
    if app.ui.ai.busy {
        ui.ctx().request_repaint();
    }
    if !app.ui.ai.open {
        return;
    }
    let t = Tokens::get(ui.ctx());
    crate::direction::end_panel("ai_assistant")
        .resizable(true)
        .default_size(320.0)
        .size_range(260.0..=560.0)
        .frame(egui::Frame::NONE.fill(t.dock).inner_margin(egui::Margin::same(8)))
        .show(ui, |ui| panel_body(app, ui, &t));
}

fn panel_body(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let mut to_send: Option<String> = None;
    let mut pick: Option<Provider> = None;
    let mut local_only: Option<bool> = None;
    load_gemini_key(app);
    let ai = &mut app.ui.ai;
    // Header: title, the connected AI, and close.
    ui.horizontal(|ui| {
        ui.label(RichText::new("AI Assistant").strong().color(t.text));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if crate::icons::button(ui, "x", 20.0, false, "Close").clicked() {
                ai.open = false;
            }
            if crate::icons::button(ui, "trash", 20.0, false, "New conversation").clicked() {
                ai.clear();
            }
        });
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("AI:").small().color(t.text_dim));
        egui::ComboBox::from_id_salt("ai_provider").selected_text(ai.assistant_name()).show_ui(ui, |ui| {
            for p in Provider::ALL.into_iter().filter(|p| !(ai.local_only && p.online())) {
                if ui.selectable_label(p == ai.provider, p.label()).clicked() {
                    pick = Some(p);
                }
            }
        });
    });
    let mut local = ai.local_only;
    if ui
        .checkbox(&mut local, RichText::new("Local only").small().color(t.text_dim))
        .on_hover_text("Only AIs that run on this computer: your conversation and pictures stay here.")
        .changed()
    {
        local_only = Some(local);
    }
    #[cfg(not(target_arch = "wasm32"))]
    if matches!(ai.provider, Provider::Ollama | Provider::Gemini) {
        connection(app, ui, t);
    }
    let ai = &mut app.ui.ai;
    ui.separator();

    // Input area pinned to the bottom; the conversation fills what's left.
    egui::Panel::bottom("ai_input").frame(egui::Frame::NONE.fill(t.dock)).show(ui, |ui| {
        ui.add_space(6.0);
        egui::CollapsingHeader::new(RichText::new(format!("Actions ({})", ai.actions.len())).color(t.text_dim)).id_salt("ai_actions").show(ui, |ui| {
            egui::ScrollArea::vertical().max_height(120.0).stick_to_bottom(true).show(ui, |ui| {
                if ai.actions.is_empty() {
                    ui.weak("Commands the AI runs appear here.");
                }
                for line in &ai.actions {
                    ui.label(RichText::new(line).small().monospace().color(t.text_dim));
                }
            });
        });
        ui.add_space(4.0);
        let edit = egui::TextEdit::multiline(&mut ai.input)
            .hint_text("Ask the AI to edit your image…")
            .desired_rows(3)
            .desired_width(f32::INFINITY)
            .char_limit(MAX_INPUT_CHARS);
        let response = ui.add(edit);
        // Enter sends; Shift+Enter starts a new line.
        let enter = response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
        ui.add_space(4.0);
        let mut send = enter;
        ui.horizontal(|ui| {
            if ai.busy {
                if crate::widgets::secondary_button(ui, "Stop", 80.0).clicked() {
                    ai.stop();
                }
                ui.weak("Answering…");
            } else if crate::widgets::primary_button(ui, "Send", 80.0).clicked() {
                send = true;
            }
        });
        if send && !ai.busy {
            // Enter's newline lands in the text before we see the key; it's trimmed by `send`.
            to_send = Some(std::mem::take(&mut ai.input));
            response.request_focus();
        }
        ui.add_space(2.0);
    });

    if let Some(p) = pick
        && let Err(e) = app.ui.ai.set_provider(p)
    {
        app.ui.ai.push(Role::Notice, e);
    }
    if let Some(on) = local_only {
        app.ui.ai.set_local_only(on);
    }
    if let Some(text) = to_send
        && let Err(e) = send(app, &text)
        && !text.trim().is_empty()
    {
        app.ui.ai.input = text;
        // Shown in the conversation, where the user is looking.
        app.ui.ai.push(Role::Notice, e);
    }

    let ai = &app.ui.ai;
    egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        if ai.messages.is_empty() {
            ui.add_space(12.0);
            ui.weak("Ask for an edit, such as \u{201c}add a soft vignette\u{201d}. Every change the AI makes can be undone with Ctrl+Z.");
        }
        for m in &ai.messages {
            message(ui, t, m);
        }
    });
}

/// The Connection section for Ollama and Gemini: address, model (with the service's list), and
/// Gemini's API key.
#[cfg(not(target_arch = "wasm32"))]
fn connection(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let ai = &mut app.ui.ai;
    if let Some(rx) = ai.form.models_rx.as_ref()
        && let Ok(result) = rx.try_recv()
    {
        ai.form.models_rx = None;
        match result {
            Ok(models) if models.is_empty() => ai.form.note = Some("The service listed no models.".into()),
            Ok(models) => {
                ai.form.models = models;
                ai.form.note = None;
            }
            Err(e) => ai.form.note = Some(e),
        }
    }
    let gemini = ai.provider == Provider::Gemini;
    let needs_key = gemini && ai.gemini_key.is_none();
    let mut save_key: Option<String> = None;
    let mut changed = false;
    egui::CollapsingHeader::new(RichText::new("Connection").small().color(t.text_dim)).id_salt("ai_connection").default_open(needs_key).show(ui, |ui| {
        let settings = &mut ai.settings;
        if !gemini {
            ui.label(RichText::new("Address").small().color(t.text_dim));
            changed |= ui.add(egui::TextEdit::singleline(&mut settings.ollama_url).desired_width(f32::INFINITY)).lost_focus();
        }
        ui.label(RichText::new("Model").small().color(t.text_dim));
        let model = if gemini { &mut settings.gemini_model } else { &mut settings.ollama_model };
        ui.horizontal(|ui| {
            changed |= ui.add(egui::TextEdit::singleline(model).desired_width(ui.available_width() - 70.0)).lost_focus();
            let listing = ai.form.models_rx.is_some();
            if ui
                .add_enabled(!listing && (!gemini || ai.gemini_key.is_some()), egui::Button::new(if listing { "…" } else { "List" }))
                .on_hover_text("Ask the service which models it has")
                .clicked()
            {
                let config = crate::ai_openai::ServiceConfig {
                    label: if gemini { "Gemini" } else { "Ollama" },
                    base_url: if gemini { settings.gemini_url.clone() } else { settings.ollama_url.clone() },
                    model: String::new(),
                    api_key: if gemini { ai.gemini_key.clone() } else { None },
                };
                let (tx, rx) = std::sync::mpsc::channel();
                let ctx = ui.ctx().clone();
                std::thread::spawn(move || {
                    let _ = tx.send(crate::ai_openai::list_models(&config));
                    ctx.request_repaint();
                });
                ai.form.models_rx = Some(rx);
            }
        });
        if !ai.form.models.is_empty() {
            let mut picked = None;
            egui::ScrollArea::vertical().id_salt("ai_models").max_height(110.0).show(ui, |ui| {
                for m in &ai.form.models {
                    let current = if gemini { &settings.gemini_model } else { &settings.ollama_model };
                    if ui.selectable_label(m == current, RichText::new(m).small()).clicked() {
                        picked = Some(m.clone());
                    }
                }
            });
            // Picking a model closes the list; List opens it again.
            if let Some(m) = picked {
                if gemini {
                    settings.gemini_model = m;
                } else {
                    settings.ollama_model = m;
                }
                ai.form.models.clear();
                changed = true;
            }
        }
        if gemini {
            ui.label(RichText::new("API key").small().color(t.text_dim));
            if ai.gemini_key.is_some() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Saved on this computer").small().color(t.text_dim));
                    if ui.small_button("Forget").clicked() {
                        save_key = Some(String::new());
                    }
                });
            } else {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut ai.form.key_input)
                            .password(true)
                            .hint_text("Paste your key")
                            .desired_width(ui.available_width() - 70.0),
                    );
                    if ui.add_enabled(!ai.form.key_input.trim().is_empty(), egui::Button::new("Save")).clicked() {
                        save_key = Some(std::mem::take(&mut ai.form.key_input));
                    }
                });
                ui.label(
                    RichText::new("Free keys: aistudio.google.com. On the free tier Google may use what you send to improve its products.")
                        .small()
                        .color(t.text_faint),
                );
            }
        }
        if !gemini {
            ui.label(
                RichText::new("Tip: start Ollama with OLLAMA_CONTEXT_LENGTH=32768 (or more). Its default of 4096 is too small for the tools and pictures, and the model loses track.")
                    .small()
                    .color(t.text_faint),
            );
        }
        if let Some(note) = &ai.form.note {
            ui.label(RichText::new(note).small().color(t.warning));
        }
    });
    if changed {
        ai.reconfigure();
    }
    if let Some(key) = save_key
        && let Err(e) = store_gemini_key(app, &key)
    {
        app.ui.ai.form.note = Some(e);
    }
}

fn message(ui: &mut egui::Ui, t: &Tokens, m: &Message) {
    ui.add_space(6.0);
    let (who, fill, color) = match m.role {
        Role::User => ("You", t.accent_soft, t.text),
        Role::Assistant => ("AI", t.card, t.text),
        Role::Notice => ("", egui::Color32::TRANSPARENT, t.text_faint),
    };
    egui::Frame::NONE.fill(fill).corner_radius(t.radius).inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        if !who.is_empty() {
            ui.label(RichText::new(who).small().strong().color(t.text_dim));
        }
        let text = if m.text.is_empty() { "…" } else { m.text.as_str() };
        ui.add(egui::Label::new(RichText::new(text).color(color)).wrap().sense(Sense::hover()));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_until_done(ai: &mut AiPanelUi, mut now: f64) -> f64 {
        for _ in 0..10_000 {
            now += 0.05;
            ai.poll(now);
            if !ai.busy {
                break;
            }
        }
        now
    }

    #[test]
    fn echo_streams_a_reply_and_logs_that_it_ran_nothing() {
        let mut ai = AiPanelUi::default();
        ai.send("  add a vignette \n", 0.0).unwrap();
        assert!(ai.busy);
        assert_eq!(ai.messages.first(), Some(&Message { role: Role::User, text: "add a vignette".into() }));
        // Part-way through, only part of the reply has arrived.
        ai.poll(0.2);
        let partial = ai.messages.last().unwrap().text.clone();
        assert!(!partial.is_empty() && ai.busy);
        run_until_done(&mut ai, 0.2);
        let reply = &ai.messages.last().unwrap().text;
        assert!(reply.starts_with("You said: \u{201c}add a vignette\u{201d}") && reply.len() > partial.len(), "{reply}");
        assert_eq!(ai.actions, ["Echo AI: no commands run"]);
        assert!(!ai.busy);
    }

    #[test]
    fn stop_ends_the_turn_and_keeps_what_was_said() {
        let mut ai = AiPanelUi::default();
        ai.send("hello", 0.0).unwrap();
        ai.poll(0.1);
        let said = ai.messages.get(1).unwrap().text.clone();
        ai.stop();
        assert!(!ai.busy);
        assert_eq!(ai.messages.get(1).unwrap().text, said, "the partial reply stays");
        assert_eq!(ai.messages.last().unwrap(), &Message { role: Role::Notice, text: "Stopped.".into() });
        ai.poll(10.0);
        assert_eq!(ai.messages.len(), 3, "nothing arrives after Stop");
        assert!(ai.actions.is_empty());
    }

    #[test]
    fn bad_sends_are_refused_without_changing_the_conversation() {
        let mut ai = AiPanelUi::default();
        assert!(ai.send("   \n ", 0.0).is_err());
        assert!(ai.send(&"x".repeat(MAX_INPUT_CHARS + 1), 0.0).is_err());
        assert!(ai.messages.is_empty() && !ai.busy);
        ai.send("one", 0.0).unwrap();
        assert!(ai.send("two", 0.0).is_err(), "one turn at a time");
        assert_eq!(ai.messages.len(), 2);
    }

    #[test]
    fn the_conversation_and_log_are_capped() {
        let mut ai = AiPanelUi::default();
        let mut now = 0.0;
        for i in 0..(MAX_MESSAGES / 2 + 5) {
            ai.send(&format!("m{i}"), now).unwrap();
            now = run_until_done(&mut ai, now);
        }
        assert_eq!(ai.messages.len(), MAX_MESSAGES);
        ai.clear();
        assert!(ai.messages.is_empty() && ai.actions.is_empty());
    }

    #[test]
    fn automation_toggles_sends_and_rejects_bad_params() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        assert_eq!(menu(&mut app, TOGGLE, &json!({})), Some(Ok(json!({"open": true}))));
        assert_eq!(menu(&mut app, TOGGLE, &json!({"open": false})), Some(Ok(json!({"open": false}))));
        assert!(matches!(menu(&mut app, SEND, &json!({})), Some(Err(_))));
        assert!(matches!(menu(&mut app, SEND, &json!({"text": 5})), Some(Err(_))));
        assert_eq!(menu(&mut app, SEND, &json!({"text": "hi"})), Some(Ok(json!({"busy": true}))));
        assert!(app.ui.ai.open, "sending opens the panel");
        assert_eq!(menu(&mut app, STOP, &json!({})), Some(Ok(json!({"busy": false}))));
        assert_eq!(menu(&mut app, CLEAR, &json!(null)), Some(Ok(json!({}))));
        assert!(matches!(menu(&mut app, PROVIDER, &json!({"provider": "nope"})), Some(Err(_))));
        assert_eq!(menu(&mut app, PROVIDER, &json!({"provider": "claudeCode"})), Some(Ok(json!({"provider": "claudeCode"}))));
        assert_eq!(app.ui.ai.provider, Provider::ClaudeCode);
        // Without the desktop's bridge service, a message to Claude says why instead of failing silently.
        assert!(matches!(menu(&mut app, SEND, &json!({"text": "hi"})), Some(Err(e)) if e.contains("can't connect")));
        assert!(app.ui.ai.messages.is_empty());
        assert_eq!(menu(&mut app, "window.panel.timeline", &json!({})), None);
        assert_eq!(checked(&app, TOGGLE), Some(true));
    }

    #[test]
    fn an_ai_turn_becomes_one_undo_step() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        app.run("layer.new.layer", json!({"name": "Mine"})).unwrap();
        let mine = app.session.active().unwrap().history.past_len();
        // The AI's turn: three edits, then it's done.
        app.ui.ai.turn_start =
            app.session.active().map(|d| TurnStart { doc: d.doc.clone(), label: step_label("add  three\nlayers please, and make them nice and tidy for me") });
        for name in ["A", "B", "C"] {
            app.run("layer.new.layer", json!({"name": name})).unwrap();
        }
        assert_eq!(app.session.active().unwrap().history.past_len(), mine + 3);
        fold_turn(&mut app);
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), mine + 1, "one step for the whole turn");
        assert_eq!(st.history.undo_label(), Some("AI: add three layers please, and make them\u{2026}"));
        assert!(app.ui.ai.turn_start.is_none());
        // One undo takes the whole turn back, keeping the user's own step.
        app.run("edit.undo", json!({})).unwrap();
        let names: Vec<String> = app.session.active().unwrap().doc.walk().iter().map(|(_, _, l)| l.name.clone()).collect();
        assert!(names.iter().any(|n| n == "Mine") && !names.iter().any(|n| ["A", "B", "C"].contains(&n.as_str())), "{names:?}");
    }

    #[test]
    fn a_turn_that_changed_nothing_or_was_undone_folds_nothing() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        let start = app.session.active().map(|d| TurnStart { doc: d.doc.clone(), label: "AI: x".into() });
        app.ui.ai.turn_start = start.clone();
        let before = app.session.active().unwrap().history.entries();
        fold_turn(&mut app);
        assert_eq!(app.session.active().unwrap().history.entries(), before, "nothing changed, nothing folded");
        // The user undid past where the turn started: its document isn't in the history anymore.
        app.ui.ai.turn_start = start;
        app.run("edit.undo", json!({})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        let before = app.session.active().unwrap().history.entries();
        fold_turn(&mut app);
        assert_eq!(app.session.active().unwrap().history.entries(), before);
        // Still answering: wait.
        app.ui.ai.turn_start = app.session.active().map(|d| TurnStart { doc: d.doc.clone(), label: "AI: y".into() });
        app.ui.ai.busy = true;
        fold_turn(&mut app);
        assert!(app.ui.ai.turn_start.is_some());
    }

    #[test]
    fn connecting_the_ai_turns_the_automation_policy_on() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        assert!(app.services.automation_command.is_none(), "a plain launch has no policy");
        fn deny_paths(id: &str, _: &Value) -> photocraft_engine::Result<()> {
            if id == "layer.exportAs" { Err(photocraft_engine::EngineError::Other("no paths".into())) } else { Ok(()) }
        }
        app.services.ai_bridge = Some(Box::new(|| {
            Ok(Bridge {
                mcp_command: "photocraft-cli".into(),
                mcp_args: vec!["mcp".into()],
                work_dir: std::env::temp_dir(),
                control: std::sync::mpsc::channel().1,
                authorize_command: Box::new(|id, _| if id == "layer.exportAs" { Err("no paths".into()) } else { Ok(()) }),
                authorize_step: deny_paths,
            })
        }));
        let link = connect_bridge(&mut app).unwrap();
        assert_eq!(link.args, ["mcp"]);
        let authorize = app.services.automation_command.as_ref().unwrap();
        assert!(authorize("layer.exportAs", &json!({"path": "/tmp/x.png"})).is_err());
        assert!(authorize("layer.new.layer", &json!({})).is_ok());
        assert!(app.services.automation_authorize.is_some());
        assert!(app.control_rx.is_some(), "the bridge's requests are read");
    }

    #[test]
    fn a_policy_from_control_launches_is_kept() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.services.automation_command = Some(Box::new(|_, _| Err("the --control policy".into())));
        app.services.ai_bridge = Some(Box::new(|| {
            Ok(Bridge {
                mcp_command: "photocraft-cli".into(),
                mcp_args: Vec::new(),
                work_dir: std::env::temp_dir(),
                control: std::sync::mpsc::channel().1,
                authorize_command: Box::new(|_, _| Ok(())),
                authorize_step: |_, _| Ok(()),
            })
        }));
        connect_bridge(&mut app).unwrap();
        let authorize = app.services.automation_command.as_ref().unwrap();
        assert_eq!(authorize("anything", &json!({})), Err("the --control policy".to_string()));
    }

    #[test]
    fn another_ai_drops_the_model_list() {
        let mut ai = AiPanelUi::default();
        ai.set_provider(Provider::Ollama).unwrap();
        ai.form.models = vec!["qwen3-vl:30b".into()];
        ai.set_provider(Provider::Gemini).unwrap();
        assert!(ai.form.models.is_empty(), "Ollama's models are not Gemini's");
    }

    #[test]
    fn local_only_keeps_online_ais_out() {
        let mut ai = AiPanelUi::default();
        ai.set_provider(Provider::Gemini).unwrap();
        ai.set_local_only(true);
        assert_eq!(ai.provider, Provider::Ollama, "turning it on leaves the online AI");
        for online in [Provider::Gemini, Provider::ClaudeCode, Provider::ClaudeCodeLookOnly] {
            assert!(ai.set_provider(online).is_err(), "{online:?}");
        }
        assert!(ai.set_provider(Provider::Echo).is_ok() && ai.set_provider(Provider::Ollama).is_ok());
        // Even if a stale setting says otherwise, nothing online is sent.
        ai.provider = Provider::Gemini;
        assert!(ai.send("hi", 0.0).is_err());
        ai.set_local_only(false);
        assert!(ai.set_provider(Provider::Gemini).is_ok());
        assert!(!Provider::Ollama.online() && !Provider::Echo.online());
    }

    #[test]
    fn gemini_without_a_key_says_where_to_get_one() {
        let mut ai = AiPanelUi::default();
        ai.set_provider(Provider::Gemini).unwrap();
        ai.link = Some(McpLink { command: "x".into(), args: Vec::new(), work_dir: std::env::temp_dir() });
        ai.send("hi", 0.0).unwrap();
        ai.poll(0.1);
        let reply = &ai.messages.last().unwrap().text;
        assert!(reply.contains("API key") && reply.contains("aistudio.google.com"), "{reply}");
        assert!(!ai.busy);
    }

    #[cfg(unix)]
    #[test]
    fn the_gemini_key_is_kept_in_a_private_file_not_the_settings() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("dalivinci-key-{}", std::process::id()));
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.services.ai_dir = Some(dir.clone());
        store_gemini_key(&mut app, "  secret-123 \n").unwrap();
        let file = dir.join(GEMINI_KEY_FILE);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "secret-123");
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(!serde_json::to_string(&app.ui.ai).unwrap().contains("secret"), "never in the UI state");
        // A fresh start reads it back.
        let mut again = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        again.services.ai_dir = Some(dir.clone());
        load_gemini_key(&mut again);
        assert_eq!(again.ui.ai.gemini_key.as_deref(), Some("secret-123"));
        // Forget.
        store_gemini_key(&mut app, "").unwrap();
        assert!(!file.exists() && app.ui.ai.gemini_key.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_window_menu_lists_the_panel_once() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let items = crate::menus::menu_items(&app);
        let ours: Vec<_> = items.iter().filter(|i| i.id == TOGGLE).collect();
        assert_eq!(ours.len(), 1);
        assert_eq!(ours.first().and_then(|i| i.path.first()).map(String::as_str), Some("Window"));
    }
}

//! Dalivinci: Claude Code behind the AI Assistant panel (desktop only).
//!
//! Each turn runs the `claude` program the user installed and signed in to, non-interactively
//! (`claude -p … --output-format stream-json`), and reads its events line by line on a background
//! thread. The conversation continues across turns with `--resume <session id>`.
//!
//! Claude reaches this window only through PhotoCraft's MCP server in bridge mode
//! ([`McpLink`]), and for now only through its look-only tools: Claude Code's own tools are
//! switched off (`--tools ""`, so no shell or file access), other MCP servers are ignored
//! (`--strict-mcp-config`), the look-only tools are pre-approved and every other tool is refused
//! without asking (`--permission-mode dontAsk`, plus an explicit deny list).

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::Value;

use crate::ai_panel::{Assistant, Event, McpLink};

/// The MCP server's name in Claude's config; its tools are `mcp__dalivinci__<tool>`.
const SERVER: &str = "dalivinci";

/// Tools that only look: pre-approved.
const LOOK_TOOLS: [&str; 8] =
    ["session_list", "doc_inspect", "doc_render_preview", "render_preview", "command_list", "jobs_list", "ui_inspect", "ui_screenshot"];

/// Tools that change the document, files or the window: refused for now.
const CHANGE_TOOLS: [&str; 16] = [
    "doc_open",
    "doc_place",
    "doc_export_layers",
    "doc_export_data_sets",
    "doc_import_data_sets",
    "doc_new",
    "doc_save",
    "doc_export",
    "doc_select",
    "doc_close",
    "command_run",
    "command_batch",
    "jobs_cancel",
    "ui_pointer",
    "ui_menu_invoke",
    "ui_set",
];
/// Raw control-protocol calls: refused.
const RAW_TOOLS: [&str; 1] = ["control_call"];

const SYSTEM_PROMPT: &str = "You are the AI assistant inside Dalivinci, a fork of the PhotoCraft image editor, talking to \
the user in a small side panel. You reach the open document only through the dalivinci tools. Use doc_inspect for the \
layers and document state, ui_screenshot or doc_render_preview to see the image, ui_inspect for the tools and panels, \
and command_list to find out which editing commands exist. In this version you can look but not change anything: \
describe what you see, and when asked for an edit, explain which menu commands and settings the user can use to do it \
themselves. Keep replies short and plain; the panel is narrow.";

/// Shown instead of a reply when Claude Code couldn't start Dalivinci's helper.
const CONNECTION_FAILED: &str = "Claude Code started, but its connection to Dalivinci didn't, so it can't see this window. \
Its log of this turn is claude-last-turn.log in DalivinciData/ai beside the program.";

/// Most bytes of the program's error output kept for an error message.
const STDERR_TAIL: usize = 2_000;

/// What the reader thread passes on.
#[derive(Debug, PartialEq)]
enum Msg {
    Event(Event),
    /// The conversation's id, for the next turn's `--resume`.
    Session(String),
}

#[derive(Default)]
pub struct ClaudeCode {
    link: Option<McpLink>,
    session: Option<String>,
    child: Option<Child>,
    rx: Option<Receiver<Msg>>,
    /// Text already came in this turn: the next message starts a new paragraph.
    said: bool,
}

impl Assistant for ClaudeCode {
    fn name(&self) -> &'static str {
        "Claude Code (read-only)"
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
        let (tx, rx) = channel();
        self.rx = Some(rx);
        let Some(link) = self.link.clone() else {
            fail(&tx, "Claude Code isn't connected to this window.".into());
            return;
        };
        match spawn(&link, prompt, self.session.as_deref()) {
            Ok(mut child) => {
                read_output(&mut child, tx);
                self.child = Some(child);
            }
            Err(e) => fail(&tx, e),
        }
    }

    fn poll(&mut self, _now: f64) -> Vec<Event> {
        let Some(rx) = self.rx.as_ref() else { return Vec::new() };
        let mut events = Vec::new();
        for msg in rx.try_iter() {
            match msg {
                Msg::Session(id) => self.session = Some(id),
                Msg::Event(Event::Text(text)) => {
                    let text = text.trim();
                    if !text.is_empty() {
                        events.push(Event::Text(if self.said { format!("\n\n{text}") } else { text.to_owned() }));
                        self.said = true;
                    }
                }
                Msg::Event(e) => events.push(e),
            }
        }
        if events.contains(&Event::Done) {
            // Usually it has exited already; after an early stop (a failed connection) it hasn't.
            self.stop();
        }
        events
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            // Collect the exit status off the UI thread so no zombie is left behind.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
        self.rx = None;
    }
}

impl Drop for ClaudeCode {
    fn drop(&mut self) {
        self.stop();
    }
}

fn fail(tx: &Sender<Msg>, message: String) {
    let _ = tx.send(Msg::Event(Event::Text(message)));
    let _ = tx.send(Msg::Event(Event::Done));
}

/// The `claude` program: `$DALIVINCI_CLAUDE` when set, else the native installer's
/// `~/.local/bin/claude`, else the first `claude` on the PATH (a desktop launcher may not have
/// `~/.local/bin` on it).
///
/// Under WSL the PATH also lists Windows' folders, and the Windows Claude Code found there runs its
/// MCP servers through cmd.exe, which can't start this Linux build's helper: the connection closes
/// at once. So on Linux, copies on Windows drives (`/mnt/<drive>/…`) and `.exe` files are skipped,
/// and if that's all there is, the panel says so.
fn program() -> Result<std::path::PathBuf, String> {
    if cfg!(windows) {
        return Ok("claude".into());
    }
    find_program(std::env::var_os("DALIVINCI_CLAUDE"), std::env::var_os("HOME").map(std::path::PathBuf::from), std::env::var_os("PATH"), |p| p.is_file())
}

fn find_program(
    chosen: Option<std::ffi::OsString>,
    home: Option<std::path::PathBuf>,
    path: Option<std::ffi::OsString>,
    exists: impl Fn(&std::path::Path) -> bool,
) -> Result<std::path::PathBuf, String> {
    if let Some(p) = chosen.filter(|p| !p.is_empty()) {
        return Ok(p.into());
    }
    let local = home.map(|h| h.join(".local/bin/claude"));
    let on_path = path.iter().flat_map(std::env::split_paths).flat_map(|d| [d.join("claude"), d.join("claude.exe")]);
    let mut windows = None;
    for candidate in local.into_iter().chain(on_path) {
        if !exists(&candidate) {
            continue;
        }
        if on_windows_side(&candidate) {
            windows.get_or_insert(candidate);
        } else {
            return Ok(candidate);
        }
    }
    Err(match windows {
        Some(w) => format!(
            "Only the Windows version of Claude Code was found ({}), and it can't start Dalivinci's Linux helper. Install Claude Code inside WSL with: curl -fsSL https://claude.ai/install.sh | bash \u{2014} then run ~/.local/bin/claude once to sign in, and try again.",
            w.display()
        ),
        None => NOT_INSTALLED.into(),
    })
}

const NOT_INSTALLED: &str = "Claude Code isn't installed where Dalivinci can find it. Install it (curl -fsSL https://claude.ai/install.sh | bash), run `claude` once to sign in, then try again.";

/// A Windows program seen from WSL: on a Windows drive (`/mnt/c/…`) or an `.exe`.
fn on_windows_side(p: &std::path::Path) -> bool {
    let mut parts = p.components().map(|c| c.as_os_str().to_string_lossy().into_owned());
    let drive = matches!((parts.next().as_deref(), parts.next().as_deref(), parts.next()), (Some("/"), Some("mnt"), Some(d)) if d.len() == 1);
    drive || p.extension().is_some_and(|e| e.eq_ignore_ascii_case("exe"))
}

/// Claude Code's debug log of the latest turn, beside the work folder: what to look at, or send,
/// when the connection fails.
fn debug_log(link: &McpLink) -> std::path::PathBuf {
    link.work_dir.parent().unwrap_or(&link.work_dir).join("claude-last-turn.log")
}

/// The MCP config handed to Claude: just this window's bridge.
fn mcp_config(link: &McpLink) -> String {
    serde_json::json!({ "mcpServers": { SERVER: { "command": link.command, "args": link.args } } }).to_string()
}

fn tool_list(tools: &[&str]) -> String {
    tools.iter().map(|t| format!("mcp__{SERVER}__{t}")).collect::<Vec<_>>().join(",")
}

/// The arguments for one turn.
fn args(link: &McpLink, prompt: &str, session: Option<&str>) -> Vec<String> {
    let mut deny: Vec<&str> = CHANGE_TOOLS.to_vec();
    deny.extend(RAW_TOOLS);
    let mut a = vec![
        "-p".to_owned(),
        prompt.to_owned(),
        "--output-format".into(),
        "stream-json".into(),
        "--verbose".into(),
        "--tools".into(),
        String::new(),
        "--strict-mcp-config".into(),
        "--mcp-config".into(),
        mcp_config(link),
        "--allowedTools".into(),
        tool_list(&LOOK_TOOLS),
        "--disallowedTools".into(),
        tool_list(&deny),
        "--permission-mode".into(),
        "dontAsk".into(),
        "--append-system-prompt".into(),
        SYSTEM_PROMPT.into(),
    ];
    if let Some(id) = session {
        a.extend(["--resume".into(), id.into()]);
    }
    a
}

fn spawn(link: &McpLink, prompt: &str, session: Option<&str>) -> Result<Child, String> {
    let _ = std::fs::create_dir_all(&link.work_dir);
    let log = debug_log(link);
    let _ = std::fs::remove_file(&log);
    Command::new(program()?)
        .args(args(link, prompt, session))
        .arg("--debug-file")
        .arg(&log)
        .current_dir(&link.work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => NOT_INSTALLED.to_owned(),
            _ => format!("Couldn't start Claude Code: {e}"),
        })
}

/// Read the program's events on background threads. If it ends without a final result, the panel
/// gets the end of its error output instead.
fn read_output(child: &mut Child, tx: Sender<Msg>) {
    let tail = Arc::new(Mutex::new(String::new()));
    if let Some(err) = child.stderr.take() {
        let tail = Arc::clone(&tail);
        std::thread::spawn(move || {
            let mut buf = [0u8; 1024];
            let mut err = err;
            while let Ok(n) = err.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let mut t = tail.lock().unwrap_or_else(PoisonError::into_inner);
                t.push_str(&String::from_utf8_lossy(buf.get(..n).unwrap_or_default()));
                let over = t.len().saturating_sub(STDERR_TAIL);
                if over > 0 {
                    let cut = (over..=t.len()).find(|&i| t.is_char_boundary(i)).unwrap_or(t.len());
                    t.drain(..cut);
                }
            }
        });
    }
    let Some(out) = child.stdout.take() else {
        fail(&tx, "Couldn't read Claude Code's output.".into());
        return;
    };
    std::thread::spawn(move || {
        let mut finished = false;
        for line in BufReader::new(out).lines() {
            let Ok(line) = line else { break };
            for msg in parse_line(&line) {
                finished |= msg == Msg::Event(Event::Done);
                if tx.send(msg).is_err() {
                    return;
                }
            }
        }
        if !finished {
            // Give the error reader a moment to catch the last of it.
            std::thread::sleep(std::time::Duration::from_millis(200));
            let t = tail.lock().unwrap_or_else(PoisonError::into_inner).trim().to_owned();
            let why = if t.is_empty() { "it ended without an answer".to_owned() } else { t };
            fail(&tx, format!("Claude Code stopped: {why}"));
        }
    });
}

/// What one line of `--output-format stream-json` means for the panel.
fn parse_line(line: &str) -> Vec<Msg> {
    let Ok(v) = serde_json::from_str::<Value>(line) else { return Vec::new() };
    let mut out = Vec::new();
    let session = || v.get("session_id").and_then(Value::as_str).map(|s| Msg::Session(s.to_owned()));
    match v.get("type").and_then(Value::as_str) {
        Some("system") if v.get("subtype").and_then(Value::as_str) == Some("init") => {
            out.extend(session());
            let ours =
                v.get("mcp_servers").and_then(Value::as_array).and_then(|all| all.iter().find(|m| m.get("name").and_then(Value::as_str) == Some(SERVER)));
            if ours.and_then(|m| m.get("status")).and_then(Value::as_str) == Some("failed") {
                out.push(Msg::Event(Event::Text(CONNECTION_FAILED.into())));
                out.push(Msg::Event(Event::Done));
            }
        }
        Some("assistant") => {
            let parts = v.pointer("/message/content").and_then(Value::as_array).cloned().unwrap_or_default();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            out.push(Msg::Event(Event::Text(t.to_owned())));
                        }
                    }
                    Some("tool_use") => {
                        let name = part.get("name").and_then(Value::as_str).unwrap_or("a tool");
                        out.push(Msg::Event(Event::Action(describe(name))));
                    }
                    _ => {}
                }
            }
        }
        Some("result") => {
            out.extend(session());
            let failed =
                v.get("is_error").and_then(Value::as_bool).unwrap_or(false) || v.get("subtype").and_then(Value::as_str).is_some_and(|s| s != "success");
            if failed {
                let why = v.get("result").and_then(Value::as_str).filter(|s| !s.trim().is_empty()).unwrap_or("Claude Code reported an error.");
                out.push(Msg::Event(Event::Text(format!("Claude Code: {why}"))));
            }
            out.push(Msg::Event(Event::Done));
        }
        _ => {}
    }
    out
}

/// One action-log line for a tool Claude used.
fn describe(tool: &str) -> String {
    let short = tool.strip_prefix(&format!("mcp__{SERVER}__")).unwrap_or(tool);
    let what = match short {
        "session_list" => "Listed the open documents",
        "doc_inspect" => "Read the layers and document state",
        "doc_render_preview" | "render_preview" => "Looked at the image",
        "ui_screenshot" => "Took a screenshot of the window",
        "ui_inspect" => "Read the tool and panel state",
        "command_list" => "Looked up the available commands",
        "jobs_list" => "Checked background jobs",
        _ => return format!("Tried {short}"),
    };
    format!("{what} ({short})")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> McpLink {
        McpLink {
            command: "/opt/dalivinci/photocraft-cli".into(),
            args: vec!["mcp".into(), "--bridge".into(), "127.0.0.1:4321".into(), "--control-token-file".into(), "/t/bridge.token".into()],
            work_dir: "/t/ai".into(),
        }
    }

    fn value_after<'a>(a: &'a [String], flag: &str) -> &'a str {
        let i = a.iter().position(|x| x == flag).unwrap();
        a.get(i + 1).unwrap()
    }

    #[test]
    fn a_turn_is_locked_down_to_looking() {
        let a = args(&link(), "what's on the canvas?", None);
        assert_eq!(a.get(..2).unwrap(), ["-p", "what's on the canvas?"]);
        assert_eq!(value_after(&a, "--tools"), "", "no shell or file tools");
        assert!(a.iter().any(|x| x == "--strict-mcp-config"));
        assert_eq!(value_after(&a, "--permission-mode"), "dontAsk");
        let allowed = value_after(&a, "--allowedTools");
        let denied = value_after(&a, "--disallowedTools");
        for t in LOOK_TOOLS {
            assert!(allowed.split(',').any(|x| x == format!("mcp__dalivinci__{t}")), "{t}");
        }
        for t in CHANGE_TOOLS.iter().chain(RAW_TOOLS.iter()) {
            assert!(denied.split(',').any(|x| x == format!("mcp__dalivinci__{t}")), "{t}");
            assert!(!allowed.contains(&format!("__{t},")) && !allowed.ends_with(&format!("__{t}")), "{t} must not be allowed");
        }
        let config: Value = serde_json::from_str(value_after(&a, "--mcp-config")).unwrap();
        assert_eq!(config["mcpServers"]["dalivinci"]["args"][2], "127.0.0.1:4321");
        assert!(!a.iter().any(|x| x == "--resume"));
        let next = args(&link(), "and now?", Some("abc-123"));
        assert_eq!(value_after(&next, "--resume"), "abc-123");
    }

    #[test]
    fn every_bridge_tool_is_either_allowed_or_denied() {
        let names: Vec<&str> = LOOK_TOOLS.iter().chain(CHANGE_TOOLS.iter()).chain(RAW_TOOLS.iter()).copied().collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "no tool is in two lists");
    }

    #[test]
    fn stream_json_lines_become_panel_events() {
        assert_eq!(parse_line(r#"{"type":"system","subtype":"init","session_id":"s1","tools":[]}"#), [Msg::Session("s1".into())]);
        assert_eq!(
            parse_line(
                r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Two layers."},{"type":"tool_use","name":"mcp__dalivinci__doc_inspect","input":{}}]}}"#
            ),
            [Msg::Event(Event::Text("Two layers.".into())), Msg::Event(Event::Action("Read the layers and document state (doc_inspect)".into()))]
        );
        assert_eq!(
            parse_line(r#"{"type":"result","subtype":"success","is_error":false,"result":"Two layers.","session_id":"s1"}"#),
            [Msg::Session("s1".into()), Msg::Event(Event::Done)]
        );
        assert_eq!(
            parse_line(r#"{"type":"result","subtype":"error_during_execution","is_error":true,"session_id":"s1"}"#),
            [Msg::Session("s1".into()), Msg::Event(Event::Text("Claude Code: Claude Code reported an error.".into())), Msg::Event(Event::Done)]
        );
        // Tool results, partial junk and other lines are ignored, never a panic.
        assert!(parse_line(r#"{"type":"user","message":{"content":[{"type":"tool_result"}]}}"#).is_empty());
        assert!(parse_line("not json").is_empty());
        assert!(parse_line(r#"{"type":"assistant","message":{"content":"oops"}}"#).is_empty());
        assert_eq!(describe("mcp__dalivinci__command_run"), "Tried command_run");
    }

    #[test]
    fn replies_from_several_messages_become_paragraphs() {
        let (tx, rx) = channel();
        let mut c = ClaudeCode::default();
        c.rx = Some(rx);
        tx.send(Msg::Session("s9".into())).unwrap();
        tx.send(Msg::Event(Event::Text("Let me look.".into()))).unwrap();
        tx.send(Msg::Event(Event::Action("Looked at the image (ui_screenshot)".into()))).unwrap();
        tx.send(Msg::Event(Event::Text("  It's a beach at sunset.\n".into()))).unwrap();
        tx.send(Msg::Event(Event::Done)).unwrap();
        let events = c.poll(0.0);
        assert_eq!(
            events,
            [
                Event::Text("Let me look.".into()),
                Event::Action("Looked at the image (ui_screenshot)".into()),
                Event::Text("\n\nIt's a beach at sunset.".into()),
                Event::Done
            ]
        );
        assert_eq!(c.session.as_deref(), Some("s9"), "the next turn resumes this conversation");
        assert!(c.rx.is_none());
    }

    #[test]
    fn a_failed_connection_ends_the_turn_with_why() {
        let line = r#"{"type":"system","subtype":"init","session_id":"s2","mcp_servers":[{"name":"dalivinci","status":"failed"}]}"#;
        let msgs = parse_line(line);
        assert_eq!(msgs.first(), Some(&Msg::Session("s2".into())));
        assert!(matches!(msgs.get(1), Some(Msg::Event(Event::Text(t))) if t.contains("connection to Dalivinci didn't")));
        assert_eq!(msgs.last(), Some(&Msg::Event(Event::Done)));
        // Connected, still starting, or someone else's server failing: carry on.
        for status in [r#""connected""#, r#""pending""#] {
            let ok = format!(
                r#"{{"type":"system","subtype":"init","session_id":"s2","mcp_servers":[{{"name":"dalivinci","status":{status}}},{{"name":"other","status":"failed"}}]}}"#
            );
            assert_eq!(parse_line(&ok), [Msg::Session("s2".into())], "{status}");
        }
    }

    #[test]
    fn the_linux_claude_wins_over_windows_copies_seen_from_wsl() {
        use std::path::{Path, PathBuf};
        let have = |files: &'static [&'static str]| move |p: &Path| files.iter().any(|f| Path::new(f) == p);
        let path = Some(std::ffi::OsString::from("/usr/bin:/mnt/c/Users/me/AppData/Roaming/npm:/home/me/bin"));
        let home = Some(PathBuf::from("/home/me"));
        // The Windows copy comes first on the PATH, but the Linux one is used.
        let found = find_program(None, home.clone(), path.clone(), have(&["/mnt/c/Users/me/AppData/Roaming/npm/claude", "/home/me/bin/claude"]));
        assert_eq!(found, Ok(PathBuf::from("/home/me/bin/claude")));
        // The native installer's copy first.
        let found = find_program(None, home.clone(), path.clone(), have(&["/home/me/.local/bin/claude", "/usr/bin/claude"]));
        assert_eq!(found, Ok(PathBuf::from("/home/me/.local/bin/claude")));
        // Only Windows' copy: say so, naming it.
        let found = find_program(None, home.clone(), path.clone(), have(&["/mnt/c/Users/me/AppData/Roaming/npm/claude"]));
        assert!(matches!(&found, Err(e) if e.contains("Only the Windows version") && e.contains("/mnt/c/Users/me/AppData/Roaming/npm/claude")), "{found:?}");
        assert!(matches!(find_program(None, home.clone(), path.clone(), have(&["/home/me/bin/claude.exe"])), Err(e) if e.contains("Windows")));
        // Nothing at all.
        assert_eq!(find_program(None, home.clone(), path.clone(), have(&[])), Err(NOT_INSTALLED.into()));
        // An explicit choice wins.
        assert_eq!(find_program(Some("/opt/claude".into()), home, path, have(&[])), Ok(PathBuf::from("/opt/claude")));
        assert!(!on_windows_side(Path::new("/mnt/data/claude")), "a mounted Linux disk isn't Windows");
    }

    #[test]
    fn a_missing_program_or_link_answers_with_why() {
        let mut c = ClaudeCode::default();
        c.start("hi", 0.0);
        let events = c.poll(0.0);
        assert!(matches!(events.first(), Some(Event::Text(t)) if t.contains("isn't connected")), "{events:?}");
        assert_eq!(events.last(), Some(&Event::Done));
    }
}

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
    ClaudeCode,
}

impl Provider {
    pub const ALL: [Provider; 2] = [Provider::Echo, Provider::ClaudeCode];

    pub fn label(self) -> &'static str {
        match self {
            Provider::Echo => "Echo (test AI)",
            Provider::ClaudeCode => "Claude Code (read-only)",
        }
    }

    fn assistant(self) -> Box<dyn Assistant> {
        match self {
            Provider::Echo => Box::new(Echo::default()),
            #[cfg(not(target_arch = "wasm32"))]
            Provider::ClaudeCode => Box::new(crate::ai_claude::ClaudeCode::default()),
            #[cfg(target_arch = "wasm32")]
            Provider::ClaudeCode => Box::new(Unavailable("Claude Code needs the desktop app.")),
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

/// AI Assistant panel view state. The conversation is exposed to automation but never restored
/// from settings: only whether the panel is open survives a restart.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AiPanelUi {
    #[serde(default)]
    pub open: bool,
    /// The AI picked at the top of the panel.
    #[serde(default)]
    pub provider: Provider,
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
}

impl PartialEq for AiPanelUi {
    fn eq(&self, other: &Self) -> bool {
        self.open == other.open
            && self.provider == other.provider
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

/// An AI this build can't run: every turn answers with why.
#[cfg(target_arch = "wasm32")]
struct Unavailable(&'static str);

#[cfg(target_arch = "wasm32")]
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
        let provider = self.provider;
        self.assistant.0.get_or_insert_with(|| provider.assistant())
    }

    /// Talk to another AI: ends the current turn and starts a new conversation.
    pub fn set_provider(&mut self, provider: Provider) {
        if provider != self.provider {
            self.clear();
            self.provider = provider;
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
    if !app.ui.ai.busy && !text.trim().is_empty() && app.ui.ai.wants_bridge() {
        let link = connect_bridge(app)?;
        app.ui.ai.link = Some(link);
    }
    let now = app.ui.ai_clock;
    app.ui.ai.send(text, now)
}

/// Start the private connection an AI program uses to reach this window.
fn connect_bridge(app: &mut PhotocraftApp) -> Result<McpLink, String> {
    let Some(start) = app.services.ai_bridge.as_mut() else {
        return Err("this build of Dalivinci can't connect an AI program".into());
    };
    let bridge = start()?;
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
    matches!(id, TOGGLE | SEND | STOP | CLEAR | PROVIDER)
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
            Some(Ok(p)) => {
                ai.set_provider(p);
                Ok(json!({ "provider": p }))
            }
            _ => Err("ai.provider needs {\"provider\": \"echo\" | \"claudeCode\"}".into()),
        },
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
            for p in Provider::ALL {
                if ui.selectable_label(p == ai.provider, p.label()).clicked() {
                    pick = Some(p);
                }
            }
        });
    });
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

    if let Some(p) = pick {
        app.ui.ai.set_provider(p);
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
    fn the_window_menu_lists_the_panel_once() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let items = crate::menus::menu_items(&app);
        let ours: Vec<_> = items.iter().filter(|i| i.id == TOGGLE).collect();
        assert_eq!(ours.len(), 1);
        assert_eq!(ours.first().and_then(|i| i.path.first()).map(String::as_str), Some("Window"));
    }
}

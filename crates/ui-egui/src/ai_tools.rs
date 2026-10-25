//! Dalivinci: what every AI behind the AI Assistant panel may do, and what it's told.
//!
//! The AI reaches this window through PhotoCraft's MCP server in bridge mode. It may always use the
//! look-only tools; with editing on, the tools that run engine commands on the open document. The
//! rest (files, switching documents, pointer and menu driving, raw control calls) stay refused, and
//! the app's automation policy still screens every command it runs.

use serde_json::Value;

/// The MCP server's name in Claude's config; its tools are `mcp__dalivinci__<tool>`.
pub(crate) const SERVER: &str = "dalivinci";

/// Tools that only look: pre-approved.
pub(crate) const LOOK_TOOLS: [&str; 8] =
    ["session_list", "doc_inspect", "doc_render_preview", "render_preview", "command_list", "jobs_list", "ui_inspect", "ui_screenshot"];

/// Tools that edit the open document: allowed when editing is on.
pub(crate) const EDIT_TOOLS: [&str; 3] = ["command_run", "command_batch", "jobs_cancel"];

/// Tools that open, save or export files, switch documents, or drive the pointer and menus:
/// always refused.
pub(crate) const CHANGE_TOOLS: [&str; 13] = [
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
    "ui_pointer",
    "ui_menu_invoke",
    "ui_set",
];
/// Raw control-protocol calls: refused.
pub(crate) const RAW_TOOLS: [&str; 1] = ["control_call"];

pub(crate) const ABOUT: &str = "You are the AI assistant inside Dalivinci, a fork of the PhotoCraft image editor, talking to the \
user in a small side panel. You reach the open document only through the dalivinci tools. Use doc_inspect for the layers \
and document state, ui_screenshot or doc_render_preview to see the image, ui_inspect for the tools and panels, and \
command_list (with a filter) to find out which editing commands exist and what parameters they take. Keep replies short \
and plain; the panel is narrow.";

pub(crate) const LOOK_ONLY: &str = " In this mode you can look but not change anything: describe what you see, and when asked \
for an edit, explain which menu commands and settings the user can use to do it themselves.";

pub(crate) const CAN_EDIT: &str = " You can edit the open document with command_run and command_batch. Do what was asked and \
nothing more. Prefer edits that keep the original pixels: adjustment layers, new layers, masks and layer styles over \
changing the image layer directly. Look up a command's id and parameters with command_list before running it, and look \
at the result afterwards. Everything you do in one reply becomes a single undo step, so the user can take it all back \
with one Ctrl+Z; say so if they seem unhappy with the result. You cannot open, save or export files: when the work is \
done, remind the user to save. For new pixels from the picture model set up in the panel's Pictures section, \
run command_run with id ai.picture and params {\"prompt\": \"…\", \"mode\": \"repaint\" | \"change\" | \"create\", \
\"strength\": 0.1-1}; the result arrives as a new layer. Pick the mode carefully. repaint is the usual one: it changes \
only the selected area of the open picture, so to add, remove or replace something, first look at the image, then \
select the area where it goes with select.rect {\"x\", \"y\", \"width\", \"height\"} in document pixels, then run \
ai.picture with mode repaint and strength 0.8-1. change restyles the whole picture (strength 0.3-0.6 keeps it \
recognisable). create makes a new, unrelated picture: use it only when the user asks for a brand-new picture, never \
to edit the open one. Write the prompt as a caption of what the selected area should show afterwards, matching the \
photo (for example \"a smiling man in a white shirt sitting on a cream leather boat seat, bright daylight\"), not as \
an instruction. The reply waits for the picture; if it says it is still busy, run ai.picture.status to wait again. \
For a plain colour or tone change, use an adjustment layer instead of a picture.";

/// One action-log line for a tool the AI used (`input` is its arguments).
pub(crate) fn describe(tool: &str, input: Option<&Value>) -> String {
    let short = tool.strip_prefix(&format!("mcp__{SERVER}__")).unwrap_or(tool);
    let id = |v: &Value| v.get("id").and_then(Value::as_str).map(str::to_owned);
    let what = match short {
        "session_list" => "Listed the open documents",
        "doc_inspect" => "Read the layers and document state",
        "doc_render_preview" | "render_preview" => "Looked at the image",
        "ui_screenshot" => "Took a screenshot of the window",
        "ui_inspect" => "Read the tool and panel state",
        "command_list" => "Looked up the available commands",
        "jobs_list" => "Checked background jobs",
        "jobs_cancel" => "Cancelled a background job",
        "command_run" => return format!("Ran {}", input.and_then(id).unwrap_or_else(|| "a command".into())),
        "command_batch" => {
            let ids: Vec<String> = input.and_then(|i| i.get("steps")).and_then(Value::as_array).map(|s| s.iter().filter_map(id).collect()).unwrap_or_default();
            return match ids.as_slice() {
                [] => "Ran several commands".into(),
                [one] => format!("Ran {one}"),
                many => format!("Ran {} commands: {}", many.len(), many.join(", ")),
            };
        }
        _ => return format!("Tried {short}"),
    };
    format!("{what} ({short})")
}

/// Whether the AI may call `tool` (a bridge tool name, without the `mcp__dalivinci__` prefix).
pub(crate) fn allowed(tool: &str, can_edit: bool) -> bool {
    LOOK_TOOLS.contains(&tool) || (can_edit && EDIT_TOOLS.contains(&tool))
}

/// The instructions an AI gets.
pub(crate) fn instructions(can_edit: bool) -> String {
    format!("{ABOUT}{}", if can_edit { CAN_EDIT } else { LOOK_ONLY })
}

//! Dalivinci: the AI Assistant panel's private connection to this window.
//!
//! Started on the panel's first message to an AI program (Claude Code), never at launch: the
//! control server listens on a free loopback port with a fresh token, written to a private file
//! for `photocraft-cli mcp --bridge`, which the AI program runs as its MCP server.

use photocraft_ui_egui::AiBridgeFn;
use photocraft_ui_egui::ai_panel::Bridge;

pub fn service(ctx: egui::Context) -> AiBridgeFn {
    Box::new(move || {
        let dir = crate::app_dirs::config_dir().ok_or("there's no settings folder to keep the AI's connection token in")?.join("ai");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let cli = std::env::current_exe()
            .map_err(|e| format!("couldn't find this program: {e}"))?
            .with_file_name(format!("photocraft-cli{}", std::env::consts::EXE_SUFFIX));
        if !cli.is_file() {
            return Err(format!("{} is missing: it ships next to the app", cli.display()));
        }
        remove_stale_tokens(&dir);
        // A fresh token each launch: an old file is replaced, not reused.
        let token_file = dir.join(format!("bridge-{}.token", std::process::id()));
        let _ = std::fs::remove_file(&token_file);
        let token = photocraft_automation::security::server_token(None, Some(&token_file)).map_err(|e| e.to_string())?;
        let (port, control) = crate::control_server::start_on_free_port(token, ctx.clone()).map_err(|e| format!("couldn't start the AI's connection: {e}"))?;
        Ok(Bridge {
            mcp_command: cli,
            mcp_args: vec!["mcp".into(), "--bridge".into(), format!("127.0.0.1:{port}"), "--control-token-file".into(), token_file.display().to_string()],
            work_dir: dir.join("workdir"),
            control,
        })
    })
}

/// Token files of earlier launches that have ended (Linux, where `/proc` says which are running).
fn remove_stale_tokens(dir: &std::path::Path) {
    if !cfg!(target_os = "linux") {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(pid) = name.strip_prefix("bridge-").and_then(|n| n.strip_suffix(".token")) else { continue };
        if pid.parse::<u32>().is_ok() && !std::path::Path::new("/proc").join(pid).exists() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    #[test]
    fn tokens_of_ended_launches_are_removed_and_running_ones_kept() {
        let dir = std::env::temp_dir().join(format!("dalivinci-ai-tokens-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mine = dir.join(format!("bridge-{}.token", std::process::id()));
        let ended = dir.join("bridge-4000000000.token");
        let other = dir.join("notes.txt");
        for f in [&mine, &ended, &other] {
            std::fs::write(f, "x").unwrap();
        }
        super::remove_stale_tokens(&dir);
        assert!(mine.exists() && other.exists());
        assert!(!ended.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

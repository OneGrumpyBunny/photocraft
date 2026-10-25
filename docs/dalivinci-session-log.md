# Dalivinci session log

Notes for picking the work back up. The plan and checklist live in
[dalivinci-roadmap.md](dalivinci-roadmap.md); this file records where each session stopped.
Private details (network addresses, machine names, keys) are kept out of this public repo.

## Session ending 2026-10-25

### Where things stand

- Branch `ai-panel`, last commit `97f74f5`; the latest Linux build (`dalivinci-97f74f5-linux-x86_64`)
  is installed in WSL and verified by hash.
- Builds: GitHub Actions workflow `Dalivinci builds` runs on every push to `ai-panel` (Windows
  portable and Linux tarball, about 15 minutes). GitHub had switched Actions off on the fork
  ("Workflows on this fork have been disabled"); the user re-enabled it in the Actions tab. If
  builds stop starting again, check that page first.
- Backup: the user mirrors the GitHub repo to their own Gitea after major milestones.

### Tested by the user and working

- Claude Code: describe and edit, one "AI: …" undo step per reply.
- Ollama `qwen3-vl:30b` (local, with `OLLAMA_CONTEXT_LENGTH=32768`): describe and edit.
- Gemini `gemini-3.8-flash`: describe works; edits hit free-tier quota/busy errors (now retried
  and explained). Edit test still to do.
- Pictures (ComfyUI 0.41 on the user's GPU server, two RTX 3060 12 GB):
  - List finds the models; Create works well with `z_image_turbo_bf16` (results look like the
    user's own ComfyUI Z-Image template, as expected).
  - Change image works (re-imagines the whole picture, as designed).
  - A chat AI (Qwen) can call `ai.picture` itself.
  - **Repaint selection is the weak spot**: the subject is plausible and the rest of the photo is
    untouched, but it doesn't fit its surroundings. The soft mask + Differential Diffusion
    (`97f74f5`) removed the hard seam; the content still ignores the scene. Cause: Z-Image and
    SDXL base aren't inpainting models.

### Changes this session (oldest first)

- Model list closes after a pick; Gemini default `gemini-3.8-flash`; busy (5xx) retries, clearer
  quota (429) message.
- M6 Pictures from ComfyUI: client, workflows (Z-Image-style diffusion models and SDXL-style
  checkpoints), create / change / repaint, result as a new layer in one undo step, Local only
  allows only local/LAN addresses, `ai.picture` / `ai.picture.status` / `ai.picture.settings`
  for chat AIs and scripts (settings refused while an AI is answering).
- Reasoning text (`</think>`) from Qwen3 and similar models hidden from the chat.
- Chat-AI instructions: repaint (select first) for edits, change to restyle, create only for a
  new picture; prompts are captions.
- Claude Code Model box (Connection › Model) passes `--model`, so its model can be pinned.
- Repaints blend: feathered mask (about 5% of the selection, 4–48 px) for ComfyUI and the layer
  mask, sampled through `DifferentialDiffusion`.

### Next session: decide and build

1. **Repaint quality** (proposed, waiting for the user's go-ahead):
   - Send the whole picture as context for a repaint when it's small enough (up to about
     1.5 MP), instead of only the selection plus a margin. Small change in `ai_pictures.rs`
     (`region`).
   - Support an inpainting model: when a checkpoint with "inpaint" in its name is installed
     (proposed: SDXL Inpainting 0.1, `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, about
     7 GB), use it for Repaint with `InpaintModelConditioning`, keep Z-Image for Create and
     Change. The user downloads the model onto the server into ComfyUI's
     `models/checkpoints/`.
2. Gemini edit test when the quota allows (or billing is turned on).
3. From the roadmap: approval prompts for file actions; more AI programs; saved conversations;
   icon.

### Useful to know

- Update steps in WSL: copy the zip from Windows Downloads, `sha256sum` it and compare with the
  artifact digest on the run page, `unzip -o`, `tar -xzf … -C ~/dalivinci`, run `./photocraft`
  from the new folder.
- Ollama model names are tags: `ollama cp qwen3-vl:30b qwen3-vl-30b-pinned` freezes the current
  copy. Dalivinci never pulls or updates models.
- Diagnostics: the panel's Actions list shows every tool an AI called; Claude Code writes
  `ai/claude-last-turn.log` in the settings folder.
- The local build machine (cloud session) can't reach the user's network; testing against
  ComfyUI and Ollama happens on the user's side.

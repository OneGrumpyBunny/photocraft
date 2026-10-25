# Dalivinci roadmap

Status of the AI side panel on the `ai-panel` branch of this fork.

## Done

- [x] M1: renamed fork, Windows portable and Linux (WSL) builds
- [x] M2: AI Assistant panel with a stand-in echo AI
- [x] M3: Claude Code, read-only
- [x] M4: Claude Code edits, one undo step per reply
- [x] M5: Ollama and Gemini, Local only mode
- [x] Model list closes after a pick; Gemini default `gemini-3.8-flash`; busy retry
- [x] Claude Code Model box (pinned `--model`); model reasoning hidden from the chat
- [x] Tested by the user: Claude Code edit, Ollama (qwen3-vl:30b) describe and edit,
      Gemini describe

## M6: Pictures from ComfyUI

The user runs ComfyUI 0.41 on another machine on their network (address kept out of this
public repo). Models there: Z-Image Turbo (int8) with Qwen3 4B text encoder and `ae` VAE,
and SDXL base. No inpainting model yet.

- [x] Reachable from WSL; firewall allows the LAN on ComfyUI's port
- [x] Client: list models, upload image and mask, queue a workflow, wait, fetch the result
- [x] Workflows for Z-Image-style diffusion models and SDXL-style checkpoints:
      create, change the whole image, repaint the selection
- [x] Result as a new layer in one undo step, masked to the selection for a repaint
- [x] Local only allows ComfyUI on this computer or the local network only
- [x] Chat AIs: `command_run` `ai.picture` (waits for the result), `ai.picture.status`
- [x] Tests against a stand-in ComfyUI, and in the running app
- [x] User test against the real ComfyUI: List, Create and Change work (Z-Image bf16)
- [x] Repaint blends at the edges (feathered mask, Differential Diffusion)
- [ ] Repaint fits its surroundings: whole picture as context when small; support an
      inpainting checkpoint (e.g. SDXL Inpainting 0.1) with InpaintModelConditioning
- [ ] Maybe: steps/size options

## Later

- [ ] Gemini edit test once the free quota resets (or billing is on)
- [ ] Approval prompts for file actions (open, save, export), currently refused
- [ ] More AI programs (Codex, Gemini CLI), saved conversations per document, quick prompts
- [ ] Dalivinci icon; code signing if builds are shared

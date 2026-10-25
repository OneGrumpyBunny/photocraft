# Dalivinci roadmap

Status of the AI side panel on the `ai-panel` branch of this fork.

## Done

- [x] M1: renamed fork, Windows portable and Linux (WSL) builds
- [x] M2: AI Assistant panel with a stand-in echo AI
- [x] M3: Claude Code, read-only
- [x] M4: Claude Code edits, one undo step per reply
- [x] M5: Ollama and Gemini, Local only mode
- [x] Model list closes after a pick; Gemini default `gemini-3.8-flash`; busy retry
- [x] Tested by the user: Claude Code edit, Ollama (qwen3-vl:30b) describe and edit,
      Gemini describe

## Next: Stable Diffusion (ComfyUI)

The user runs ComfyUI on another machine on their network (address kept out of this
public repo; ComfyUI's default port is 8188).

- [ ] Confirm the setup: ComfyUI version, which checkpoint(s) and an inpainting model,
      and that WSL can reach the machine (`curl http://<address>:8188/system_stats`)
- [ ] Confirm ComfyUI listens on the network (`--listen`) and decide whether the
      network is trusted (ComfyUI has no authentication)
- [ ] Decide the use: a direct Generate box, a tool the chat AIs can call, or both
- [ ] Connection settings: ComfyUI address, checkpoint, steps/strength defaults
- [ ] Client: upload image and mask (`/upload/image`), queue a workflow (`/prompt`),
      wait for it (`/history` or the websocket), fetch the result (`/view`)
- [ ] Workflows: text-to-image, image-to-image, inpaint the selection (selection as the mask)
- [ ] Result goes in as a new layer above the source, aligned to the selection, one undo step
- [ ] Counts as local for Local only only when the address is on the user's own network;
      say so in the panel
- [ ] Tool for chat AIs (for example `image_generate`) with prompt, strength, selection only
- [ ] Tests against a mock ComfyUI server; end-to-end test by the user

## Later

- [ ] Gemini edit test once the free quota resets (or billing is on)
- [ ] Approval prompts for file actions (open, save, export), currently refused
- [ ] More AI programs (Codex, Gemini CLI), saved conversations per document, quick prompts
- [ ] Dalivinci icon; code signing if builds are shared

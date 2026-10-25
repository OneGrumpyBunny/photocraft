//! Pictures from a ComfyUI server: create one from a prompt, change the whole image, or repaint
//! the selection. The result arrives as a new layer above the active one, in one undo step; a
//! repaint gets the selection as its layer mask, so nothing outside the selection changes.
//!
//! The image and mask are sent at about one megapixel (what the models are trained on) and the
//! result is scaled back to the document's size. A repaint sends only the selection and a margin
//! around it, so small areas keep their detail.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use photocraft_color::PixelFormat;
use photocraft_doc::{Layer, LayerMask};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;

/// Make a picture: `{"prompt", "mode": "create" | "change" | "repaint", "strength"?}`. Over the
/// control channel the reply waits for the picture (up to [`REPLY_WAIT`]).
pub const PICTURE: &str = "ai.picture";
/// Whether a picture is still being made; over the control channel, waits for it like [`PICTURE`].
pub const STATUS: &str = "ai.picture.status";
/// Change the Pictures settings (`comfyUrl`, `model`, `family`, `textEncoder`, `vae`,
/// `strength`); `{"list": true}` asks ComfyUI for its models and fills in missing choices.
pub const PICTURE_SETTINGS: &str = "ai.picture.settings";
/// How long a control request waits for a picture before answering that it's still coming
/// (below the bridge's 55-second call timeout).
const REPLY_WAIT: Duration = Duration::from_secs(45);

/// What a picture request does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Mode {
    /// A new picture from the prompt, the size of the canvas.
    #[default]
    Create,
    /// The whole image, changed by the prompt.
    Change,
    /// Only the selection, repainted by the prompt.
    Repaint,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Create, Mode::Change, Mode::Repaint];

    pub fn label(self) -> &'static str {
        match self {
            Mode::Create => "Create",
            Mode::Change => "Change image",
            Mode::Repaint => "Repaint selection",
        }
    }
}

/// The model family decides the workflow: SDXL-style checkpoints hold everything in one file;
/// Z-Image-style diffusion models need a text encoder and a VAE beside them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Family {
    #[default]
    Checkpoint,
    Diffusion,
}

/// Where ComfyUI is and which models to use.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PictureSettings {
    pub comfy_url: String,
    pub model: String,
    pub family: Family,
    /// For a diffusion model: its text encoder and VAE files.
    pub text_encoder: String,
    pub vae: String,
    /// How much a change or repaint may differ from the original, 0–1.
    pub strength: f32,
}

impl Default for PictureSettings {
    fn default() -> Self {
        PictureSettings {
            comfy_url: "http://localhost:8188".into(),
            model: String::new(),
            family: Family::Checkpoint,
            text_encoder: String::new(),
            vae: String::new(),
            strength: 0.7,
        }
    }
}

/// A model ComfyUI listed, and its family.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelChoice {
    pub file: String,
    pub family: Family,
}

/// What ComfyUI has: its models, and the files a diffusion model needs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Catalog {
    pub models: Vec<ModelChoice>,
    pub text_encoders: Vec<String>,
    pub vaes: Vec<String>,
}

impl Catalog {
    /// The settings with this catalog's choices filled in where they're missing or gone.
    pub fn fill(&self, settings: &mut PictureSettings) {
        if !self.models.iter().any(|m| m.file == settings.model) {
            // Prefer a diffusion model (faster, newer) over a checkpoint.
            if let Some(m) = self.models.iter().find(|m| m.family == Family::Diffusion).or_else(|| self.models.first()) {
                settings.model = m.file.clone();
                settings.family = m.family;
            }
        }
        if !self.text_encoders.contains(&settings.text_encoder) {
            settings.text_encoder =
                self.text_encoders.iter().find(|f| f.to_lowercase().contains("qwen")).or_else(|| self.text_encoders.first()).cloned().unwrap_or_default();
        }
        if !self.vaes.contains(&settings.vae) {
            settings.vae = self.vaes.iter().find(|f| f.to_lowercase().starts_with("ae")).or_else(|| self.vaes.first()).cloned().unwrap_or_default();
        }
    }
}

/// The picture a request made, at the size it was sent.
struct Made {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

/// A request in flight.
struct Job {
    mode: Mode,
    label: String,
    /// The document it was made for (its index in the session), and the area the result covers.
    doc_index: usize,
    region: Rect,
    /// The selection when the request started: a repaint's layer mask.
    selection: Option<Surface>,
    stop: Arc<AtomicBool>,
    rx: Receiver<Result<Made, String>>,
}

/// The Pictures section's state.
#[derive(Default)]
pub struct PicturesUi {
    pub settings: PictureSettings,
    pub mode: Mode,
    pub prompt: String,
    /// The last error or result, shown under the button.
    pub note: Option<String>,
    catalog: Catalog,
    catalog_rx: Option<Receiver<Result<Catalog, String>>>,
    job: Option<Job>,
    /// Control requests waiting for the current picture, and when each must be answered.
    waiters: Vec<(Sender<Value>, Instant)>,
}

impl Clone for PicturesUi {
    fn clone(&self) -> Self {
        PicturesUi { settings: self.settings.clone(), mode: self.mode, ..Default::default() }
    }
}

impl std::fmt::Debug for PicturesUi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PicturesUi").field("settings", &self.settings).field("busy", &self.job.is_some()).finish_non_exhaustive()
    }
}

impl PartialEq for PicturesUi {
    fn eq(&self, other: &Self) -> bool {
        self.settings == other.settings && self.mode == other.mode
    }
}

impl Serialize for PicturesUi {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        json!({ "settings": self.settings, "mode": self.mode, "busy": self.job.is_some(), "note": self.note, "models": self.catalog.models.iter().map(|m| &m.file).collect::<Vec<_>>() }).serialize(s)
    }
}

impl<'de> Deserialize<'de> for PicturesUi {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize, Default)]
        #[serde(default)]
        struct Saved {
            settings: PictureSettings,
            mode: Mode,
        }
        let saved = Saved::deserialize(d)?;
        Ok(PicturesUi { settings: saved.settings, mode: saved.mode, ..Default::default() })
    }
}

impl PicturesUi {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    /// Asking ComfyUI for its models.
    pub fn listing(&self) -> bool {
        self.catalog_rx.is_some()
    }

    pub fn stop(&mut self) {
        if let Some(job) = self.job.take() {
            job.stop.store(true, Ordering::Relaxed);
            self.finish(Err("Stopped.".into()));
        }
    }

    /// Answer a control request once the current picture is done (see [`PICTURE`]).
    pub fn wait(&mut self, reply: Sender<Value>) {
        if self.job.is_some() {
            self.waiters.push((reply, Instant::now() + REPLY_WAIT));
        } else {
            let _ = reply.send(json!({ "ok": true, "result": { "busy": false, "note": self.note } }));
        }
    }

    /// A picture request ended: show the outcome and answer whoever waits for it.
    fn finish(&mut self, outcome: Result<String, String>) {
        for (reply, _) in self.waiters.drain(..) {
            let _ = reply.send(match &outcome {
                Ok(note) => json!({ "ok": true, "result": { "busy": false, "note": note } }),
                Err(e) => json!({ "ok": false, "error": e }),
            });
        }
        self.note = Some(match outcome {
            Ok(note) | Err(note) => note,
        });
    }

    /// Answer waiters whose time is up that the picture is still coming.
    fn answer_overdue(&mut self) {
        let now = Instant::now();
        self.waiters.retain(|(reply, due)| {
            if now < *due {
                return true;
            }
            let _ = reply.send(json!({ "ok": true, "result": { "busy": true, "note": format!("Still making the picture; call {STATUS} to wait for it.") } }));
            false
        });
    }
}

/// Run [`PICTURE`] or [`STATUS`]. `None` for other ids.
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    match id {
        STATUS => Some(Ok(json!({ "busy": app.ui.ai.pictures.busy(), "note": app.ui.ai.pictures.note }))),
        PICTURE_SETTINGS => Some((|| {
            // An AI could point ComfyUI somewhere else and send the user's pictures there.
            if app.ui.ai.busy {
                return Err("Not while the AI is answering.".to_owned());
            }
            let pictures = &mut app.ui.ai.pictures;
            let mut merged = serde_json::to_value(&pictures.settings).unwrap_or_default();
            if let (Some(m), Some(changes)) = (merged.as_object_mut(), params.as_object()) {
                for (k, v) in changes {
                    if m.contains_key(k) {
                        m.insert(k.clone(), v.clone());
                    }
                }
            }
            pictures.settings = serde_json::from_value(merged).map_err(|e| format!("{PICTURE_SETTINGS}: {e}"))?;
            #[cfg(not(target_arch = "wasm32"))]
            if params.get("list").and_then(Value::as_bool) == Some(true) {
                list_models(app, None);
            }
            serde_json::to_value(&app.ui.ai.pictures.settings).map_err(|e| e.to_string())
        })()),
        PICTURE => Some((|| {
            let prompt =
                params.get("prompt").and_then(Value::as_str).ok_or("ai.picture needs {\"prompt\": \"…\", \"mode\": \"create\" | \"change\" | \"repaint\"}")?;
            let mode = match params.get("mode") {
                None => Mode::Create,
                Some(m) => serde_json::from_value::<Mode>(m.clone()).map_err(|_| "mode must be \"create\", \"change\" or \"repaint\"".to_owned())?,
            };
            if let Some(strength) = params.get("strength").and_then(Value::as_f64) {
                app.ui.ai.pictures.settings.strength = (strength as f32).clamp(0.1, 1.0);
            }
            #[cfg(not(target_arch = "wasm32"))]
            start(app, mode, prompt, None)?;
            #[cfg(target_arch = "wasm32")]
            let _ = (mode, prompt);
            Ok(json!({ "busy": app.ui.ai.pictures.busy() }))
        })()),
        _ => None,
    }
}

/// Whether an address stays on this computer or its local network (so Local only allows it).
pub fn is_local_address(url: &str) -> bool {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let host_port = rest.split('/').next().unwrap_or_default();
    let host = if let Some(v6) = host_port.strip_prefix('[') {
        v6.split(']').next().unwrap_or_default()
    } else {
        host_port.rsplit_once(':').map_or(host_port, |(h, _)| h)
    };
    let host = host.to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".local") || host.ends_with(".lan") || host.ends_with(".home.arpa") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        Ok(std::net::IpAddr::V6(ip)) => ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00,
        Err(_) => false,
    }
}

/// The area a request covers: the canvas, or for a repaint the selection and a margin around it
/// (at least 512 px a side where the canvas allows), so the model sees some surroundings.
pub fn region(mode: Mode, canvas: Rect, selection: Option<Rect>) -> Result<Rect, String> {
    if mode != Mode::Repaint {
        return Ok(canvas);
    }
    let sel = selection.map(|s| s.intersect(&canvas)).filter(|s| !s.is_empty()).ok_or("Select the area to repaint first.")?;
    let margin = (sel.width().max(sel.height()) as i32 / 4).max(32);
    let mut r = Rect::new(sel.x0 - margin, sel.y0 - margin, sel.x1 + margin, sel.y1 + margin);
    for _ in 0..2 {
        let grow_x = (512 - r.width() as i32).max(0);
        let grow_y = (512 - r.height() as i32).max(0);
        r = Rect::new(r.x0 - grow_x / 2, r.y0 - grow_y / 2, r.x1 + grow_x - grow_x / 2, r.y1 + grow_y - grow_y / 2);
        // Slide back inside the canvas before clipping, so the size is kept where it fits.
        let dx = (canvas.x0 - r.x0).max(0) - (r.x1 - canvas.x1).max(0);
        let dy = (canvas.y0 - r.y0).max(0) - (r.y1 - canvas.y1).max(0);
        r = Rect::new(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy).intersect(&canvas);
    }
    Ok(r)
}

/// The size to send: about one megapixel with the region's shape, sides a multiple of 16.
pub fn work_size(width: u32, height: u32) -> (u32, u32) {
    const PIXELS: f64 = 1024.0 * 1024.0;
    let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let scale = (PIXELS / (w * h)).sqrt();
    let side = |v: f64| ((v * scale / 16.0).round() as u32).clamp(16, 160) * 16;
    (side(w), side(h))
}

/// Resize interleaved pixels (`channels` per pixel, 8-bit), repeating the edges outward.
fn resize(bytes: &[u8], width: u32, height: u32, to_w: u32, to_h: u32, format: PixelFormat) -> Vec<u8> {
    let from = Rect::new(0, 0, width as i32, height as i32);
    let to = Rect::new(0, 0, to_w as i32, to_h as i32);
    if (width, height) == (to_w, to_h) {
        return bytes.to_vec();
    }
    let surface = Surface::from_interleaved(format, from, bytes);
    let scaled = photocraft_algo::resample::resize_surface_in_canvas(
        &surface,
        f64::from(to_w) / f64::from(width),
        f64::from(to_h) / f64::from(height),
        photocraft_algo::resample::Resample::Bicubic,
        from,
    );
    scaled.to_interleaved(to)
}

#[cfg(any(test, not(target_arch = "wasm32")))]
fn png(width: u32, height: u32, rgba: Vec<u8>) -> Result<Vec<u8>, String> {
    let image = photocraft_codecs::Image::from_u8(width, height, photocraft_codecs::ChannelLayout::Rgba, rgba).map_err(|e| e.to_string())?;
    photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).map_err(|e| e.to_string())
}

/// The ComfyUI workflow for a request. `source` and `mask` are uploaded file names.
pub fn workflow(settings: &PictureSettings, mode: Mode, prompt: &str, size: (u32, u32), seed: u64, source: Option<&str>, mask: Option<&str>) -> Value {
    let mut g = serde_json::Map::new();
    let mut node = |id: &str, class: &str, inputs: Value| {
        g.insert(id.into(), json!({ "class_type": class, "inputs": inputs }));
    };
    let link = |id: &str, out: u32| json!([id, out]);
    let (model, clip, vae, positive, negative, steps, cfg, sampler, scheduler);
    match settings.family {
        Family::Diffusion => {
            node("1", "UNETLoader", json!({ "unet_name": settings.model, "weight_dtype": "default" }));
            node("2", "CLIPLoader", json!({ "clip_name": settings.text_encoder, "type": "lumina2", "device": "default" }));
            node("3", "VAELoader", json!({ "vae_name": settings.vae }));
            node("4", "ModelSamplingAuraFlow", json!({ "model": link("1", 0), "shift": 3.0 }));
            node("5", "CLIPTextEncode", json!({ "text": prompt, "clip": link("2", 0) }));
            node("6", "ConditioningZeroOut", json!({ "conditioning": link("5", 0) }));
            (model, clip, vae, positive, negative) = (link("4", 0), link("2", 0), link("3", 0), link("5", 0), link("6", 0));
            (steps, cfg, sampler, scheduler) = (8, 1.0, "res_multistep", "simple");
        }
        Family::Checkpoint => {
            node("1", "CheckpointLoaderSimple", json!({ "ckpt_name": settings.model }));
            node("5", "CLIPTextEncode", json!({ "text": prompt, "clip": link("1", 1) }));
            node("6", "CLIPTextEncode", json!({ "text": "blurry, low quality, watermark, text", "clip": link("1", 1) }));
            (model, clip, vae, positive, negative) = (link("1", 0), link("1", 1), link("1", 2), link("5", 0), link("6", 0));
            (steps, cfg, sampler, scheduler) = (28, 6.5, "dpmpp_2m", "karras");
        }
    }
    let _ = clip;
    let (latent, denoise) = match (mode, source) {
        (Mode::Create, _) | (_, None) => {
            let empty = if settings.family == Family::Diffusion { "EmptySD3LatentImage" } else { "EmptyLatentImage" };
            node("7", empty, json!({ "width": size.0, "height": size.1, "batch_size": 1 }));
            (link("7", 0), 1.0)
        }
        (_, Some(source)) => {
            node("10", "LoadImage", json!({ "image": source }));
            node("7", "VAEEncode", json!({ "pixels": link("10", 0), "vae": vae }));
            let strength = f64::from(settings.strength.clamp(0.05, 1.0));
            match (mode, mask) {
                (Mode::Repaint, Some(mask)) => {
                    node("11", "LoadImageMask", json!({ "image": mask, "channel": "red" }));
                    node("12", "SetLatentNoiseMask", json!({ "samples": link("7", 0), "mask": link("11", 0) }));
                    (link("12", 0), strength)
                }
                _ => (link("7", 0), strength),
            }
        }
    };
    node(
        "8",
        "KSampler",
        json!({
            "model": model, "positive": positive, "negative": negative, "latent_image": latent,
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise,
        }),
    );
    node("9", "VAEDecode", json!({ "samples": link("8", 0), "vae": vae }));
    node("20", "SaveImage", json!({ "images": link("9", 0), "filename_prefix": "dalivinci" }));
    Value::Object(g)
}

/// Ask ComfyUI which models it has, on a thread.
#[cfg(not(target_arch = "wasm32"))]
pub fn list_models(app: &mut PhotocraftApp, ctx: Option<&egui::Context>) {
    let pictures = &mut app.ui.ai.pictures;
    if pictures.catalog_rx.is_some() {
        return;
    }
    let url = pictures.settings.comfy_url.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let ctx = ctx.cloned();
    std::thread::spawn(move || {
        let comfy = crate::ai_comfy::Comfy::new(&url);
        let catalog = (|| {
            let mut models: Vec<ModelChoice> =
                comfy.models("diffusion_models")?.into_iter().map(|file| ModelChoice { file, family: Family::Diffusion }).collect();
            models.extend(comfy.models("checkpoints")?.into_iter().map(|file| ModelChoice { file, family: Family::Checkpoint }));
            Ok(Catalog { models, text_encoders: comfy.models("text_encoders")?, vaes: comfy.models("vae")? })
        })();
        let _ = tx.send(catalog);
        if let Some(ctx) = ctx {
            ctx.request_repaint();
        }
    });
    pictures.catalog_rx = Some(rx);
}

/// Start a picture request for the active document.
#[cfg(not(target_arch = "wasm32"))]
pub fn start(app: &mut PhotocraftApp, mode: Mode, prompt: &str, ctx: Option<&egui::Context>) -> Result<(), String> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err("Describe the picture first.".into());
    }
    if app.ui.ai.pictures.busy() {
        return Err("A picture is already being made.".into());
    }
    let settings = app.ui.ai.pictures.settings.clone();
    if app.ui.ai.local_only && !is_local_address(&settings.comfy_url) {
        return Err(format!("{} isn't on this computer or its local network: turn off Local only to use it.", settings.comfy_url));
    }
    if settings.model.is_empty() {
        return Err("Pick a model first (Connection › List).".into());
    }
    let doc_index = app.session.active_index().ok_or("Open or create a document first.")?;
    let doc = app.session.active().map(|d| d.doc.clone()).ok_or("Open or create a document first.")?;
    let canvas = doc.bounds();
    let selection = doc.selection.clone();
    let sel_bounds = selection.as_ref().map(|s| s.content_bounds());
    let region = region(mode, canvas, sel_bounds)?;
    let size = work_size(region.width(), region.height());
    // What to send: the flattened image of the region, and for a repaint the selection as a mask.
    let source = if mode == Mode::Create {
        None
    } else {
        let flat = photocraft_compose::render(&doc, region).to_rgba8();
        Some(png(size.0, size.1, resize(&flat.pixels, flat.width, flat.height, size.0, size.1, PixelFormat::RGBA8))?)
    };
    let mask = match (&selection, mode) {
        (Some(sel), Mode::Repaint) => {
            let gray = sel.convert(PixelFormat::GRAY8).to_interleaved(region);
            let small = resize(&gray, region.width(), region.height(), size.0, size.1, PixelFormat::GRAY8);
            Some(png(size.0, size.1, small.iter().flat_map(|&v| [v, v, v, 255]).collect())?)
        }
        _ => None,
    };
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64 & 0xffff_ffff_ffff);
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = stop.clone();
    let ctx = ctx.cloned();
    let prompt_owned = prompt.to_owned();
    std::thread::spawn(move || {
        let result = (|| {
            let comfy = crate::ai_comfy::Comfy::new(&settings.comfy_url);
            let source = source.map(|png| comfy.upload_png("dalivinci-source.png", &png)).transpose()?;
            let mask = mask.map(|png| comfy.upload_png("dalivinci-mask.png", &png)).transpose()?;
            let graph = workflow(&settings, mode, &prompt_owned, size, seed, source.as_deref(), mask.as_deref());
            let id = comfy.queue(&graph)?;
            let outputs = comfy.wait(&id, &stop_worker)?;
            let bytes = comfy.fetch(&outputs[0])?;
            let image = photocraft_codecs::decode(&bytes).map_err(|e| format!("The picture from ComfyUI wasn't readable ({e})."))?;
            Ok(Made { width: image.width(), height: image.height(), rgba: image.to_rgba8() })
        })();
        let _ = tx.send(result);
        if let Some(ctx) = ctx {
            ctx.request_repaint();
        }
    });
    let pictures = &mut app.ui.ai.pictures;
    pictures.note = None;
    pictures.job =
        Some(Job { mode, label: crate::ai_panel::step_label(prompt), doc_index, region, selection: selection.filter(|_| mode == Mode::Repaint), stop, rx });
    Ok(())
}

/// Collect a finished catalog or picture. Call every frame.
pub fn poll(app: &mut PhotocraftApp) {
    let pictures = &mut app.ui.ai.pictures;
    if let Some(rx) = &pictures.catalog_rx
        && let Ok(result) = rx.try_recv()
    {
        pictures.catalog_rx = None;
        match result {
            Ok(catalog) if catalog.models.is_empty() => pictures.note = Some("ComfyUI has no checkpoints or diffusion models.".into()),
            Ok(catalog) => {
                catalog.fill(&mut pictures.settings);
                pictures.catalog = catalog;
                pictures.note = None;
            }
            Err(e) => pictures.note = Some(e),
        }
    }
    pictures.answer_overdue();
    let Some(job) = &pictures.job else { return };
    let Ok(result) = job.rx.try_recv() else { return };
    let Some(job) = pictures.job.take() else { return };
    let outcome = result.and_then(|made| place(app, &job, made)).map(|()| format!("Added as the layer \u{201c}{}\u{201d}. Ctrl+Z takes it back.", job.label));
    app.ui.ai.pictures.finish(outcome);
}

/// Put a finished picture into its document as a new layer above the active one.
fn place(app: &mut PhotocraftApp, job: &Job, made: Made) -> Result<(), String> {
    if app.session.active_index() != Some(job.doc_index) {
        return Err("The picture was made for another document; switch back to it and try again.".into());
    }
    if made.rgba.len() != made.width as usize * made.height as usize * 4 {
        return Err("The picture from ComfyUI had an unexpected size.".into());
    }
    let r = job.region;
    let pixels = resize(&made.rgba, made.width, made.height, r.width(), r.height(), PixelFormat::RGBA8);
    let mut surface = Surface::from_interleaved(PixelFormat::RGBA8, r, &pixels);
    surface.prune();
    let mode = job.mode;
    let label = job.label.clone();
    let selection = job.selection.clone();
    app.session
        .edit(&label, |doc, active| {
            let fmt = doc.pixel_format();
            let target = PixelFormat::new(fmt.mode, fmt.sample, true);
            let mut layer = Layer::raster(label.clone(), target);
            if let Some(s) = layer.surface_mut() {
                *s = if surface.format() == target { surface } else { surface.convert(target) };
            }
            if let (Mode::Repaint, Some(sel)) = (mode, selection) {
                let depth = sel.format().with_sample(doc.depth);
                layer.mask = Some(LayerMask { surface: sel.convert(depth), ..LayerMask::reveal_all() });
            }
            let id = doc.insert_above(*active, layer);
            *active = Some(id);
            Ok(id)
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// The Pictures section of the AI panel.
#[cfg(not(target_arch = "wasm32"))]
pub fn section(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &crate::theme::Tokens) {
    use egui::RichText;
    let mut list = false;
    let mut go: Option<(Mode, String)> = None;
    let local_only = app.ui.ai.local_only;
    let pictures = &mut app.ui.ai.pictures;
    egui::CollapsingHeader::new(RichText::new("Pictures (ComfyUI)").color(t.text_dim)).id_salt("ai_pictures").show(ui, |ui| {
        let settings = &mut pictures.settings;
        ui.label(RichText::new("ComfyUI address").small().color(t.text_dim));
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut settings.comfy_url).desired_width(ui.available_width() - 70.0));
            let listing = pictures.catalog_rx.is_some();
            list = ui.add_enabled(!listing, egui::Button::new(if listing { "…" } else { "List" })).on_hover_text("Ask ComfyUI which models it has").clicked();
        });
        if local_only && !is_local_address(&settings.comfy_url) {
            ui.label(RichText::new("Local only is on: this address isn't on your network.").small().color(t.text_dim));
        }
        if !pictures.catalog.models.is_empty() {
            ui.label(RichText::new("Model").small().color(t.text_dim));
            egui::ComboBox::from_id_salt("ai_picture_model").width(ui.available_width()).selected_text(settings.model.as_str()).show_ui(ui, |ui| {
                for m in &pictures.catalog.models {
                    if ui.selectable_label(m.file == settings.model, m.file.as_str()).clicked() {
                        settings.model = m.file.clone();
                        settings.family = m.family;
                    }
                }
            });
        } else if !settings.model.is_empty() {
            ui.label(RichText::new(format!("Model: {}", settings.model)).small().color(t.text_dim));
        }
        ui.horizontal(|ui| {
            for m in Mode::ALL {
                if ui.selectable_label(pictures.mode == m, RichText::new(m.label()).small()).clicked() {
                    pictures.mode = m;
                }
            }
        });
        if pictures.mode != Mode::Create {
            ui.add(egui::Slider::new(&mut settings.strength, 0.1..=1.0).text("change").fixed_decimals(2))
                .on_hover_text("How far the result may move from the original: low keeps it close, 1 repaints freely.");
        }
        ui.add(egui::TextEdit::multiline(&mut pictures.prompt).hint_text("Describe the picture…").desired_rows(2).desired_width(f32::INFINITY));
        ui.horizontal(|ui| {
            if pictures.job.is_some() {
                if crate::widgets::secondary_button(ui, "Stop", 80.0).clicked() {
                    pictures.stop();
                }
                ui.weak("Making the picture…");
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
            } else if crate::widgets::primary_button(ui, "Generate", 80.0).clicked() {
                go = Some((pictures.mode, pictures.prompt.clone()));
            }
        });
        if let Some(note) = &pictures.note {
            ui.label(RichText::new(note).small().color(t.text_dim));
        }
    });
    if list {
        list_models(app, Some(ui.ctx()));
    }
    if let Some((mode, prompt)) = go
        && let Err(e) = start(app, mode, &prompt, Some(ui.ctx()))
    {
        app.ui.ai.pictures.note = Some(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_addresses_are_recognised() {
        for url in [
            "http://localhost:8188",
            "http://127.0.0.1:8188",
            "http://10.1.0.10:7860",
            "http://192.168.1.5",
            "http://172.20.0.2:8188/",
            "http://bobross.local:7860",
            "http://[::1]:8188",
            "http://[fd00::5]:80",
        ] {
            assert!(is_local_address(url), "{url}");
        }
        for url in ["https://comfy.example.com", "http://8.8.8.8:8188", "http://172.32.0.1", "http://[2001:db8::1]:80"] {
            assert!(!is_local_address(url), "{url}");
        }
    }

    #[test]
    fn a_repaint_covers_the_selection_with_a_margin_inside_the_canvas() {
        let canvas = Rect::new(0, 0, 4000, 3000);
        assert_eq!(region(Mode::Create, canvas, None).unwrap(), canvas);
        assert!(region(Mode::Repaint, canvas, None).is_err());
        let r = region(Mode::Repaint, canvas, Some(Rect::new(1000, 1000, 1800, 1400))).unwrap();
        assert!(r.x0 <= 1000 && r.y0 <= 1000 && r.x1 >= 1800 && r.y1 >= 1400, "{r:?}");
        assert!(r.x0 < 1000 && r.width() >= 1000, "a margin around the selection: {r:?}");
        // A small selection in a corner: at least 512 px, slid inside the canvas.
        let r = region(Mode::Repaint, canvas, Some(Rect::new(0, 0, 40, 40))).unwrap();
        assert_eq!((r.x0, r.y0, r.width(), r.height()), (0, 0, 512, 512));
        // A canvas smaller than 512 is the limit.
        let small = Rect::new(0, 0, 300, 200);
        assert_eq!(region(Mode::Repaint, small, Some(Rect::new(10, 10, 50, 50))).unwrap(), small);
    }

    #[test]
    fn pictures_are_sent_at_about_a_megapixel() {
        let (w, h) = work_size(6000, 4000);
        assert!(w % 16 == 0 && h % 16 == 0);
        assert!((w * h) as f64 / (1024.0 * 1024.0) > 0.9 && (w * h) as f64 / (1024.0 * 1024.0) < 1.1, "{w}x{h}");
        assert!((f64::from(w) / f64::from(h) - 1.5).abs() < 0.05);
        assert_eq!(work_size(1024, 1024), (1024, 1024));
    }

    #[test]
    fn the_catalog_fills_in_missing_choices() {
        let catalog = Catalog {
            models: vec![
                ModelChoice { file: "sd_xl_base_1.0.safetensors".into(), family: Family::Checkpoint },
                ModelChoice { file: "z_image_turbo.safetensors".into(), family: Family::Diffusion },
            ],
            text_encoders: vec!["clip_l.safetensors".into(), "qwen_3_4b_fp8_mixed.safetensors".into()],
            vaes: vec!["sdxl_vae.safetensors".into(), "ae.safetensors".into()],
        };
        let mut s = PictureSettings::default();
        catalog.fill(&mut s);
        assert_eq!((s.model.as_str(), s.family), ("z_image_turbo.safetensors", Family::Diffusion));
        assert_eq!((s.text_encoder.as_str(), s.vae.as_str()), ("qwen_3_4b_fp8_mixed.safetensors", "ae.safetensors"));
        // A choice that's still there is kept.
        s.model = "sd_xl_base_1.0.safetensors".into();
        s.family = Family::Checkpoint;
        catalog.fill(&mut s);
        assert_eq!(s.model, "sd_xl_base_1.0.safetensors");
    }

    #[test]
    fn workflows_wire_the_model_family_and_mode() {
        let mut s = PictureSettings {
            model: "z.safetensors".into(),
            family: Family::Diffusion,
            text_encoder: "q.safetensors".into(),
            vae: "ae.safetensors".into(),
            ..Default::default()
        };
        let g = workflow(&s, Mode::Create, "a cat", (1024, 768), 7, None, None);
        assert_eq!(g["1"]["class_type"], "UNETLoader");
        assert_eq!(g["7"]["class_type"], "EmptySD3LatentImage");
        assert_eq!((g["7"]["inputs"]["width"].as_u64(), g["7"]["inputs"]["height"].as_u64()), (Some(1024), Some(768)));
        assert_eq!(g["8"]["inputs"]["denoise"], 1.0);
        assert_eq!(g["8"]["inputs"]["negative"], json!(["6", 0]));
        s.strength = 0.5;
        let g = workflow(&s, Mode::Repaint, "a hat", (512, 512), 7, Some("src.png"), Some("mask.png"));
        assert_eq!(g["10"]["inputs"]["image"], "src.png");
        assert_eq!(g["11"]["inputs"]["image"], "mask.png");
        assert_eq!(g["8"]["inputs"]["latent_image"], json!(["12", 0]));
        assert_eq!(g["8"]["inputs"]["denoise"], 0.5);
        let sdxl = PictureSettings { model: "sd_xl_base_1.0.safetensors".into(), ..Default::default() };
        let g = workflow(&sdxl, Mode::Change, "oil painting", (1024, 1024), 7, Some("src.png"), None);
        assert_eq!(g["1"]["class_type"], "CheckpointLoaderSimple");
        assert_eq!(g["7"]["class_type"], "VAEEncode");
        assert_eq!(g["7"]["inputs"]["vae"], json!(["1", 2]));
        assert!(g.get("11").is_none());
        // Every link points at a node in the graph.
        for (_, n) in g.as_object().unwrap() {
            for v in n["inputs"].as_object().unwrap().values() {
                if let Some([id, _]) = v.as_array().map(Vec::as_slice) {
                    assert!(g.get(id.as_str().unwrap()).is_some(), "{v}");
                }
            }
        }
    }

    /// A stand-in ComfyUI whose every workflow makes a solid red 64x64 picture.
    fn red_comfy() -> String {
        crate::ai_comfy::tests::mock(|path, _| match path {
            "/upload/image" => (200, br#"{"name": "dalivinci-source.png", "subfolder": "", "type": "input"}"#.to_vec()),
            "/prompt" => (200, br#"{"prompt_id": "p", "node_errors": {}}"#.to_vec()),
            "/history/p" => (
                200,
                br#"{"p": {"status": {"status_str": "success"}, "outputs": {"20": {"images": [{"filename": "red.png", "subfolder": "", "type": "output"}]}}}}"#
                    .to_vec(),
            ),
            p if p.starts_with("/view") => (200, png(64, 64, [255u8, 0, 0, 255].repeat(64 * 64)).unwrap()),
            _ => (404, b"{}".to_vec()),
        })
    }

    fn wait_for_picture(app: &mut PhotocraftApp) {
        for _ in 0..200 {
            poll(app);
            if !app.ui.ai.pictures.busy() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("the picture never arrived");
    }

    #[test]
    fn a_repaint_lands_as_one_masked_layer_over_the_selection() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
        app.run("select.rect", json!({"x": 20, "y": 10, "width": 40, "height": 30})).unwrap();
        app.ui.ai.pictures.settings = PictureSettings { comfy_url: red_comfy(), model: "sd_xl_base_1.0.safetensors".into(), ..Default::default() };
        let layers = app.session.active().unwrap().doc.layer_count();
        start(&mut app, Mode::Repaint, "a red square", None).unwrap();
        assert!(start(&mut app, Mode::Repaint, "again", None).is_err(), "one at a time");
        wait_for_picture(&mut app);
        assert_eq!(app.ui.ai.pictures.note.as_deref(), Some("Added as the layer \u{201c}AI: a red square\u{201d}. Ctrl+Z takes it back."));
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layer_count(), layers + 1);
        let layer = st.doc.layer(st.active_layer.unwrap()).unwrap();
        assert_eq!(layer.name, "AI: a red square");
        let mask = layer.mask.as_ref().expect("a repaint is masked to the selection");
        assert!(mask.surface.sample_channel(30, 20, 0) > 0.99 && mask.surface.sample_channel(150, 80, 0) < 0.01);
        assert!(st.doc.selection.is_some(), "the selection stays, to try again");
        // The picture covers the region around the selection (the whole small canvas here) and is red.
        let flat = photocraft_compose::flatten(&st.doc).to_rgba8();
        let at = |x: usize, y: usize| &flat.pixels[(y * 200 + x) * 4..][..3];
        assert_eq!(at(30, 20), [255, 0, 0]);
        assert_ne!(at(150, 80), [255, 0, 0], "outside the selection nothing changed");
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().doc.layer_count(), layers, "one undo takes it back");
    }

    #[test]
    fn a_control_request_waits_for_the_picture() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.ui.ai.pictures.settings = PictureSettings { comfy_url: red_comfy(), model: "sd.safetensors".into(), ..Default::default() };
        assert!(menu(&mut app, PICTURE, &json!({"mode": "sideways", "prompt": "x"})).unwrap().is_err());
        assert_eq!(menu(&mut app, PICTURE, &json!({"prompt": "red", "mode": "change", "strength": 0.4})).unwrap().unwrap()["busy"], true);
        assert!((app.ui.ai.pictures.settings.strength - 0.4).abs() < 1e-6);
        let (tx, rx) = std::sync::mpsc::channel();
        app.ui.ai.pictures.wait(tx);
        assert!(rx.try_recv().is_err(), "no answer before the picture is in");
        wait_for_picture(&mut app);
        let reply = rx.try_recv().expect("answered when the picture landed");
        assert_eq!(reply["ok"], true);
        assert!(reply["result"]["note"].as_str().unwrap().contains("AI: red"));
    }

    #[test]
    fn picture_settings_change_by_command_but_not_by_a_busy_ai() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let r = menu(&mut app, PICTURE_SETTINGS, &json!({"comfyUrl": "http://10.0.0.2:8188", "strength": 0.3, "bogus": 1})).unwrap().unwrap();
        assert_eq!(r["comfyUrl"], "http://10.0.0.2:8188");
        assert_eq!(app.ui.ai.pictures.settings.strength, 0.3);
        app.ui.ai.busy = true;
        assert!(menu(&mut app, PICTURE_SETTINGS, &json!({"comfyUrl": "http://elsewhere.example"})).unwrap().is_err());
        assert_eq!(app.ui.ai.pictures.settings.comfy_url, "http://10.0.0.2:8188");
    }

    #[test]
    fn requests_are_refused_with_a_reason() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.ai.pictures.settings.model = "m.safetensors".into();
        assert!(start(&mut app, Mode::Create, "a cat", None).unwrap_err().contains("document"));
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        assert!(start(&mut app, Mode::Create, "  ", None).is_err());
        assert!(start(&mut app, Mode::Repaint, "a hat", None).unwrap_err().contains("Select"));
        app.ui.ai.local_only = true;
        app.ui.ai.pictures.settings.comfy_url = "https://comfy.example.com".into();
        assert!(start(&mut app, Mode::Create, "a cat", None).unwrap_err().contains("Local only"));
    }

    #[test]
    fn a_created_picture_fills_the_canvas_without_a_mask() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 300, "height": 150})).unwrap();
        app.ui.ai.pictures.settings =
            PictureSettings { comfy_url: red_comfy(), model: "z.safetensors".into(), family: Family::Diffusion, ..Default::default() };
        start(&mut app, Mode::Create, "red", None).unwrap();
        wait_for_picture(&mut app);
        let st = app.session.active().unwrap();
        let layer = st.doc.layer(st.active_layer.unwrap()).unwrap();
        assert!(layer.mask.is_none());
        let flat = photocraft_compose::flatten(&st.doc).to_rgba8();
        assert!(flat.pixels.chunks(4).all(|p| p[..3] == [255, 0, 0]), "scaled to the whole canvas");
    }

    #[test]
    fn resizing_keeps_solid_edges() {
        let px: Vec<u8> = [200u8, 100, 50, 255].repeat(8 * 8);
        let out = resize(&px, 8, 8, 20, 12, PixelFormat::RGBA8);
        assert_eq!(out.len(), 20 * 12 * 4);
        assert!(out.chunks(4).all(|p| p[3] == 255 && p[0].abs_diff(200) <= 1), "edges stay opaque and the same colour");
    }
}

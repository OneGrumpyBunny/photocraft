//! A small client for ComfyUI's HTTP API: list models, upload an image, queue a workflow, wait
//! for it, and fetch the pictures it made.
//!
//! ComfyUI runs workflows given as its "API format" JSON (node id -> class and inputs). Uploads
//! go to its input folder; results are read back with `/view`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// How long one workflow may take before Dalivinci gives up on it.
const RUN_TIMEOUT: Duration = Duration::from_secs(600);
/// Largest image read back from ComfyUI.
const MAX_IMAGE_BYTES: u64 = 256 * 1024 * 1024;

/// A picture a workflow saved, as ComfyUI names it.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub filename: String,
    pub subfolder: String,
    pub kind: String,
}

pub struct Comfy {
    base: String,
    agent: ureq::Agent,
}

impl Comfy {
    pub fn new(base_url: &str) -> Self {
        let agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(60))).http_status_as_error(false).build().into();
        Comfy { base: base_url.trim_end_matches('/').to_owned(), agent }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn unreachable(&self, e: &ureq::Error) -> String {
        format!("Couldn't reach ComfyUI at {} ({e}). Is it running, and does the firewall allow this computer?", self.base)
    }

    /// Read a JSON reply; a non-2xx status becomes an error with ComfyUI's message.
    fn json(&self, mut response: ureq::http::Response<ureq::Body>) -> Result<Value, String> {
        let status = response.status().as_u16();
        let reply = response.body_mut().read_json::<Value>().unwrap_or(Value::Null);
        if (200..300).contains(&status) { Ok(reply) } else { Err(format!("ComfyUI answered with an error ({status}): {}", error_detail(&reply))) }
    }

    /// The model files in one of ComfyUI's model folders (`checkpoints`, `diffusion_models`, …).
    pub fn models(&self, folder: &str) -> Result<Vec<String>, String> {
        let response = self.agent.get(&self.url(&format!("/models/{folder}"))).call().map_err(|e| self.unreachable(&e))?;
        let reply = self.json(response)?;
        Ok(reply.as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_owned).collect()).unwrap_or_default())
    }

    /// Upload a PNG to ComfyUI's input folder; returns the name a `LoadImage` node uses.
    pub fn upload_png(&self, name: &str, png: &[u8]) -> Result<String, String> {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let boundary = format!("----dalivinci{nanos:x}{:x}", std::process::id());
        let body = multipart(&boundary, name, png);
        let response = self
            .agent
            .post(&self.url("/upload/image"))
            .header("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
            .send(&body[..])
            .map_err(|e| self.unreachable(&e))?;
        let reply = self.json(response)?;
        let stored = reply.get("name").and_then(Value::as_str).ok_or("ComfyUI didn't say where it stored the upload.")?;
        Ok(match reply.get("subfolder").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            Some(sub) => format!("{sub}/{stored}"),
            None => stored.to_owned(),
        })
    }

    /// Queue a workflow; returns its prompt id.
    pub fn queue(&self, workflow: &Value) -> Result<String, String> {
        let response =
            self.agent.post(&self.url("/prompt")).send_json(json!({ "prompt": workflow, "client_id": "dalivinci" })).map_err(|e| self.unreachable(&e))?;
        let reply = self.json(response)?;
        reply.get("prompt_id").and_then(Value::as_str).map(str::to_owned).ok_or_else(|| "ComfyUI didn't queue the workflow.".to_owned())
    }

    /// Wait for a queued workflow to finish and return the pictures it saved. Gives up when
    /// `stop` is set (and asks ComfyUI to stop) or after the run timeout.
    pub fn wait(&self, prompt_id: &str, stop: &AtomicBool) -> Result<Vec<Output>, String> {
        let started = Instant::now();
        loop {
            if stop.load(Ordering::Relaxed) {
                let _ = self.agent.post(&self.url("/interrupt")).send_empty();
                return Err("Stopped.".into());
            }
            if started.elapsed() > RUN_TIMEOUT {
                return Err(format!("ComfyUI didn't finish within {} minutes.", RUN_TIMEOUT.as_secs() / 60));
            }
            let response = self.agent.get(&self.url(&format!("/history/{prompt_id}"))).call().map_err(|e| self.unreachable(&e))?;
            let history = self.json(response)?;
            if let Some(entry) = history.get(prompt_id) {
                return finished(entry);
            }
            std::thread::sleep(Duration::from_millis(400));
        }
    }

    /// The bytes of a picture a workflow saved.
    pub fn fetch(&self, output: &Output) -> Result<Vec<u8>, String> {
        let mut response = self
            .agent
            .get(&self.url("/view"))
            .query("filename", &output.filename)
            .query("subfolder", &output.subfolder)
            .query("type", &output.kind)
            .call()
            .map_err(|e| self.unreachable(&e))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(format!("ComfyUI couldn't send the result ({status})."));
        }
        response.body_mut().with_config().limit(MAX_IMAGE_BYTES).read_to_vec().map_err(|e| format!("The result from ComfyUI wasn't readable ({e})."))
    }
}

/// The outcome of a history entry: its saved pictures, or why it failed.
fn finished(entry: &Value) -> Result<Vec<Output>, String> {
    let status = entry.get("status");
    if status.and_then(|s| s.get("status_str")).and_then(Value::as_str) == Some("error") {
        let why = status
            .and_then(|s| s.get("messages"))
            .and_then(Value::as_array)
            .and_then(|m| m.iter().find(|m| m.get(0).and_then(Value::as_str) == Some("execution_error")))
            .and_then(|m| m.pointer("/1/exception_message"))
            .and_then(Value::as_str)
            .unwrap_or("no details");
        return Err(format!("ComfyUI couldn't run the workflow: {}", why.trim()));
    }
    let mut outputs = Vec::new();
    if let Some(nodes) = entry.get("outputs").and_then(Value::as_object) {
        for node in nodes.values() {
            for image in node.get("images").and_then(Value::as_array).into_iter().flatten() {
                if image.get("type").and_then(Value::as_str) == Some("output") {
                    outputs.push(Output {
                        filename: image.get("filename").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        subfolder: image.get("subfolder").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        kind: "output".into(),
                    });
                }
            }
        }
    }
    if outputs.is_empty() { Err("The workflow finished without saving a picture.".into()) } else { Ok(outputs) }
}

/// ComfyUI's error message: a validation error and the first node error, if any.
fn error_detail(reply: &Value) -> String {
    let mut detail = reply.pointer("/error/message").and_then(Value::as_str).unwrap_or("no details").to_owned();
    if let Some(node) = reply.get("node_errors").and_then(Value::as_object).and_then(|n| n.values().next())
        && let Some(e) = node.pointer("/errors/0")
    {
        let message = e.get("message").and_then(Value::as_str).unwrap_or_default();
        let details = e.get("details").and_then(Value::as_str).unwrap_or_default();
        detail = format!("{detail} ({message}: {details})");
    }
    detail
}

/// A multipart form with the image file and `overwrite=true`.
fn multipart(boundary: &str, name: &str, png: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(png.len() + 512);
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n").as_bytes());
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(png);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A stand-in ComfyUI: answers each request by path, in a thread, until the test ends.
    pub(crate) fn mock(routes: fn(&str, &[u8]) -> (u16, Vec<u8>)) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut data = Vec::new();
                let mut buf = [0u8; 65536];
                // Read the head, then the body by Content-Length.
                loop {
                    let n = stream.read(&mut buf).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    data.extend_from_slice(&buf[..n]);
                    if let Some(end) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&data[..end]).to_lowercase();
                        let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
                        if data.len() >= end + 4 + len {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&data);
                let path = text.split_whitespace().nth(1).unwrap_or("/").to_owned();
                let body_at = data.windows(4).position(|w| w == b"\r\n\r\n").map_or(data.len(), |p| p + 4);
                let (status, body) = routes(&path, &data[body_at..]);
                let _ = write!(stream, "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = stream.write_all(&body);
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    #[test]
    fn a_workflow_runs_from_upload_to_result() {
        let base = mock(|path, body| {
            match path {
            "/models/checkpoints" => (200, r#"["sd_xl_base_1.0.safetensors"]"#.into()),
            "/upload/image" => {
                let text = String::from_utf8_lossy(body);
                assert!(text.contains("name=\"overwrite\"") && text.contains("filename=\"source.png\"") && text.contains("PNGDATA"));
                (200, r#"{"name": "source.png", "subfolder": "dalivinci", "type": "input"}"#.into())
            }
            "/prompt" => {
                let sent: Value = serde_json::from_slice(body).unwrap();
                assert_eq!(sent["prompt"]["1"]["inputs"]["image"], "dalivinci/source.png");
                (200, r#"{"prompt_id": "p1", "number": 0, "node_errors": {}}"#.into())
            }
            "/history/p1" => (
                200,
                r#"{"p1": {"status": {"status_str": "success", "completed": true}, "outputs": {"9": {"images": [{"filename": "out_00001_.png", "subfolder": "", "type": "output"}, {"filename": "preview.png", "subfolder": "", "type": "temp"}]}}}}"#.into(),
            ),
            p if p.starts_with("/view?filename=out_00001_.png") => (200, "IMAGEBYTES".into()),
            _ => (404, "{}".into()),
        }
        });
        let comfy = Comfy::new(&base);
        assert_eq!(comfy.models("checkpoints").unwrap(), ["sd_xl_base_1.0.safetensors"]);
        let name = comfy.upload_png("source.png", b"PNGDATA").unwrap();
        assert_eq!(name, "dalivinci/source.png");
        let id = comfy.queue(&json!({ "1": { "class_type": "LoadImage", "inputs": { "image": name } } })).unwrap();
        let outputs = comfy.wait(&id, &AtomicBool::new(false)).unwrap();
        assert_eq!(outputs.len(), 1, "previews are not results");
        assert_eq!(comfy.fetch(&outputs[0]).unwrap(), b"IMAGEBYTES");
    }

    #[test]
    fn comfy_errors_say_what_went_wrong() {
        let base = mock(|path, _| {
            match path {
            "/prompt" => (
                400,
                r#"{"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"}, "node_errors": {"4": {"errors": [{"message": "Value not in list", "details": "ckpt_name: 'x.safetensors' not in []"}]}}}"#.into(),
            ),
            "/history/p2" => (
                200,
                r#"{"p2": {"status": {"status_str": "error", "messages": [["execution_start", {}], ["execution_error", {"exception_message": "CUDA out of memory\n"}]]}, "outputs": {}}}"#.into(),
            ),
            _ => (404, "{}".into()),
        }
        });
        let comfy = Comfy::new(&base);
        let err = comfy.queue(&json!({})).unwrap_err();
        assert!(err.contains("failed validation") && err.contains("not in []"), "{err}");
        assert_eq!(comfy.wait("p2", &AtomicBool::new(false)).unwrap_err(), "ComfyUI couldn't run the workflow: CUDA out of memory");
        let stopped = AtomicBool::new(true);
        assert_eq!(comfy.wait("p3", &stopped).unwrap_err(), "Stopped.");
    }

    #[test]
    fn an_unreachable_comfy_names_the_address() {
        let comfy = Comfy::new("http://127.0.0.1:9");
        assert!(comfy.models("checkpoints").unwrap_err().contains("http://127.0.0.1:9"));
    }
}

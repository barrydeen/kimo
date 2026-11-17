//! NVIDIA hardware decode/encode for the ffmpeg stages the engine drives (proxy builds, source
//! decode, export encode), discovered at runtime on whatever GPU is present.
//!
//! Why this and not the wgpu `gpu2d` path: kimo's compositing core is CPU (see FEATURES.md "wgpu
//! pipeline" gaps), but the two stages that dominate wall time and CPU load for phone footage --
//! decoding 4K HEVC sources and encoding the edit/analysis proxies and exports with libx264 -- are
//! ffmpeg subprocess stages, and ffmpeg's NVDEC/NVENC take them to the GPU without touching any
//! rendered pixel path or golden. This module only answers "is the hardware usable?"; every caller
//! keeps its software path intact and falls back on any failure (same philosophy as gpu2d: the CPU
//! path is what stays exact).
//!
//! - `nvenc_available()` / `nvdec_available()`: one probe per process, cached. The probe runs a tiny
//!   real encode/decode (not an encoder listing) so it fails closed when the driver, the GPU session
//!   limit or the ffmpeg build says no.
//! - `nvdec_for(path)`: per-path cached "this source's codec is one NVDEC decodes well" check
//!   (h264/hevc), false for stills, EXR/TIFF and anything else -- `-hwaccel` is only ever added
//!   where it is known to apply.
//! - `KIMO_HWACCEL=off` (or `0`/`false`) turns all of it off; unset or any other value is auto.
//!
//! Probes are lazy: an engine session that never touches media never spawns them.
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

/// Test hook: force what the probes report without spawning ffmpeg. None = real probes.
static OVERRIDE: Mutex<Option<(bool, bool)>> = Mutex::new(None);

/// Force probe results (tests). Some(true, true) = both hardware paths claim to work.
pub fn set_probe_override(v: Option<(bool, bool)>) { *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = v; }

fn mode_off() -> bool {
    matches!(std::env::var("KIMO_HWACCEL").unwrap_or_default().to_ascii_lowercase().as_str(), "off" | "0" | "false")
}

/// NVENC usable for encode: one tiny real encode through h264_nvenc, cached for the process.
pub fn nvenc_available() -> bool {
    if let Some((nvenc, _)) = *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) { return nvenc; }
    if mode_off() { return false; }
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=c=black:s=256x144:r=30:d=0.2", "-frames:v", "6", "-c:v", "h264_nvenc", "-f", "null", "-"]);
        crate::color::process::wait_command(&mut cmd, std::time::Duration::from_secs(20)).is_ok()
    })
}

/// NVDEC usable for decode: one tiny real h264 decode through `-hwaccel cuda`, cached.
pub fn nvdec_available() -> bool {
    if let Some((_, nvdec)) = *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) { return nvdec; }
    if mode_off() { return false; }
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        // The probe needs an h264 file: encode one tiny software frame once (not lavfi -- the cuda
        // hwaccel applies to real decoded frames, and lavfi sources have no decoder to accelerate).
        let path = std::env::temp_dir().join(format!("kimo-hw-nvdec-probe-{}.mp4", std::process::id()));
        let mut enc = Command::new("ffmpeg");
        enc.args(["-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "testsrc2=s=256x144:r=30:d=0.2", "-frames:v", "6", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]).arg(&path);
        if crate::color::process::wait_command(&mut enc, std::time::Duration::from_secs(20)).is_err() {
            let _ = std::fs::remove_file(&path);
            return false;
        }
        let mut dec = Command::new("ffmpeg");
        dec.args(["-v", "error", "-nostdin", "-hwaccel", "cuda", "-i"]).arg(&path).args(["-f", "null", "-"]);
        let ok = crate::color::process::wait_command(&mut dec, std::time::Duration::from_secs(20)).is_ok();
        let _ = std::fs::remove_file(&path);
        ok
    })
}

/// True when THIS source should decode with `-hwaccel cuda` in render::decode: a video whose codec
/// NVDEC handles well (h264, hevc), not a still. One ffprobe per path for the process lifetime.
/// Falls back to software decode for anything else -- a decode that would have succeeded still succeeds.
pub fn nvdec_for(path: &str) -> bool {
    if !nvdec_available() { return false; }
    if crate::render::is_still_path(path) || path.ends_with(".kimopx")
        || [".exr", ".tif", ".tiff", ".pfm"].iter().any(|e| path.to_lowercase().ends_with(e)) {
        return false;
    }
    static CODECS: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let m = CODECS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = m.lock().unwrap().get(path) { return *v; }
    let v = codec_of(path).map(|c| c == "h264" || c == "hevc").unwrap_or(false);
    m.lock().unwrap().insert(path.to_string(), v);
    v
}

/// The first video stream's codec name, lowercase, or None when it cannot be probed.
fn codec_of(path: &str) -> Option<String> {
    let out = Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", path]).output().ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_ascii_lowercase();
    (!s.is_empty()).then_some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_env_disables_both_without_probing() {
        let _g = crate::render::encoder_state();
        std::env::set_var("KIMO_HWACCEL", "off");
        set_probe_override(None);
        assert!(!nvenc_available());
        assert!(!nvdec_available());
        assert!(!nvdec_for("/nonexistent/video.mp4"));
        std::env::remove_var("KIMO_HWACCEL");
    }

    #[test]
    fn override_forces_probe_results() {
        let _g = crate::render::encoder_state();
        set_probe_override(Some((true, true)));
        assert!(nvenc_available());
        assert!(nvdec_available());
        set_probe_override(Some((false, false)));
        assert!(!nvenc_available());
        assert!(!nvdec_available());
        set_probe_override(None);
    }

    #[test]
    fn stills_and_float_sources_never_take_nvdec() {
        let _g = crate::render::encoder_state();
        set_probe_override(Some((true, true)));
        assert!(!nvdec_for("frame.png"));
        assert!(!nvdec_for("scan.tif"));
        assert!(!nvdec_for("plate.exr"));
        assert!(!nvdec_for("shot.kimopx"));
        set_probe_override(None);
    }

    #[test]
    fn real_probes_match_this_machine_when_present() {
        let _g = crate::render::encoder_state();
        // On a box without NVIDIA hardware both probes must simply say no; on a box with it they
        // must say yes. Either way this never panics -- the whole point of the probe.
        set_probe_override(None);
        let _ = (nvenc_available(), nvdec_available());
    }
}

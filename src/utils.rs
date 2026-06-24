use std::path::{Path, PathBuf};
use std::process::Command;
use crate::CommandHideExt;

pub fn find_ffmpeg_path() -> Option<PathBuf> {
    // 1. Check PATH
    if Command::new("ffmpeg").hide_window().arg("-version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().map(|s| s.success()).unwrap_or(false) {
        return Some(PathBuf::from("ffmpeg"));
    }

    // 2. Check beside executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            let ffmpeg_path = dir.join("ffmpeg.exe");
            if ffmpeg_path.exists() {
                return Some(ffmpeg_path);
            }
            let ffmpeg_path_linux = dir.join("ffmpeg");
            if ffmpeg_path_linux.exists() {
                return Some(ffmpeg_path_linux);
            }
        }
    }

    // 3. Check current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let ffmpeg_path = cwd.join("ffmpeg.exe");
        if ffmpeg_path.exists() {
            return Some(ffmpeg_path);
        }
        let ffmpeg_path_linux = cwd.join("ffmpeg");
        if ffmpeg_path_linux.exists() {
            return Some(ffmpeg_path_linux);
        }
    }

    None
}

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .map(|ext| {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(ext.as_str(), "mp4" | "mkv" | "mov" | "webm" | "avi" | "m4v")
        })
        .unwrap_or(false)
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .map(|ext| {
            let ext = ext.to_string_lossy().to_lowercase();
            matches!(
                ext.as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "avif" | "gif" | "heic" | "ithmb"
                    | "tif" | "tiff" | "nef" | "cr2"
            )
        })
        .unwrap_or(false)
}

/// Camera RAW formats that need a dedicated decoder (the `image` crate can't
/// open them). Canon CR2 is handled by the pure-Rust raw pipeline; Nikon NEF
/// falls through to the ffmpeg path in `load_image_robustly`.
fn is_raw(ext: &str) -> bool {
    matches!(ext, "nef" | "cr2")
}

/// Decode a camera RAW file to RGB via the pure-Rust rawloader/imagepipe
/// pipeline. Works for Canon CR2 (and other models imagepipe supports).
fn decode_raw(path: &Path) -> anyhow::Result<image::DynamicImage> {
    let decoded = imagepipe::simple_decode_8bit(path, 0, 0)
        .map_err(|e| anyhow::anyhow!("raw decode failed: {e}"))?;
    let buf = image::RgbImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.data)
        .ok_or_else(|| anyhow::anyhow!("raw decode produced an unexpected buffer size"))?;
    Ok(image::DynamicImage::ImageRgb8(buf))
}

pub fn load_image_robustly(path: &Path) -> anyhow::Result<image::DynamicImage> {
    // Try standard image crate first (jpg, png, gif, webp, tiff, ...)
    if let Ok(img) = image::open(path) {
        return Ok(img);
    }

    let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();

    // Camera RAW: try the pure-Rust raw pipeline (handles Canon CR2 and more).
    if is_raw(&ext) {
        if let Ok(img) = decode_raw(path) {
            return Ok(img);
        }
    }

    // Fallback to ffmpeg for HEIC/AVIF and for RAW the raw pipeline couldn't
    // handle (e.g. Nikon NEF, which ffmpeg decodes but imagepipe does not).
    if matches!(ext.as_str(), "heic" | "avif") || is_raw(&ext) {
        if let Some(ffmpeg_cmd) = find_ffmpeg_path() {
            let output = Command::new(&ffmpeg_cmd)
                .hide_window()
                .arg("-i").arg(path)
                .arg("-vframes").arg("1")
                .arg("-f").arg("image2pipe")
                .arg("-vcodec").arg("png")
                .arg("-")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()?;

            if output.status.success() {
                return Ok(image::load_from_memory(&output.stdout)?);
            }
        }
    }

    anyhow::bail!("Could not decode image: {}", path.display())
}

pub fn get_video_thumbnail_path(video_path: &Path) -> PathBuf {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Use absolute path for more stable hashing if possible, 
    // but relative is fine if the workspace is moved.
    // For now, let's use the string representation as it is.
    video_path.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();
    
    crate::get_app_data_dir().join("output").join("thumbnails").join(format!("{:x}.jpg", hash))
}

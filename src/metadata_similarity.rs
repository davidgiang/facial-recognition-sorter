//! Ranks photos in the input directory by how likely they are to contain a
//! person whose face the recognition pipeline could not see (occluded, turned
//! away), using EXIF timestamp/camera proximity to already-confirmed photos of
//! that person, refined with a color-palette comparison for the closest calls.
//!
//! See the "Similar Timing" tab in the GUI, which is the only caller of
//! `rank_by_metadata`.

use image::DynamicImage;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Exponential time-decay constant, in seconds: score = exp(-delta / this).
/// ~0.37 at 5 minutes apart, ~0.14 at 10 minutes, ~0.002 at 30 minutes.
const TIME_DECAY_SECS: f64 = 300.0;
/// Flat bonus (in final-score units) when candidate and anchor share a camera model.
const CAMERA_BONUS: f32 = 0.15;
const TIME_WEIGHT: f32 = 0.65;
const CAMERA_WEIGHT: f32 = 0.15;
const COLOR_WEIGHT: f32 = 0.20;
/// Stage 1 (timestamp-only) keeps at most this many candidates before the
/// more expensive stage 2 (decode + color histogram) runs on them.
const STAGE1_TOP_N: usize = 200;
/// Side length of the RGB histogram used for color-palette comparison.
const HIST_BINS_PER_CHANNEL: usize = 8;
const HIST_SIZE: usize = HIST_BINS_PER_CHANNEL * HIST_BINS_PER_CHANNEL * HIST_BINS_PER_CHANNEL;

#[derive(Clone)]
pub struct MetaSimCandidate {
    pub path: PathBuf,
    /// Final blended score, roughly in [0, 1] (can exceed 1 slightly when
    /// time, camera and color all line up perfectly).
    pub score: f32,
    /// The confirmed photo this candidate was scored against (its nearest
    /// neighbor in time among the anchor set).
    pub anchor: PathBuf,
    pub delta_secs: i64,
    pub same_camera: bool,
    /// `None` when a color histogram couldn't be computed for either side
    /// (e.g. the candidate is a video, or the file failed to decode).
    pub color_score: Option<f32>,
}

impl MetaSimCandidate {
    /// Human-readable breakdown for a tooltip, e.g.
    /// "62% - 2m14s from IMG_0234.jpg, same camera, 71% color match".
    pub fn explain(&self) -> String {
        let anchor_name = self
            .anchor
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut parts = vec![format!(
            "{} from {}",
            humanize_delta(self.delta_secs),
            anchor_name
        )];
        parts.push(if self.same_camera {
            "same camera".to_string()
        } else {
            "different/unknown camera".to_string()
        });
        match self.color_score {
            Some(c) => parts.push(format!("{:.0}% color match", c * 100.0)),
            None => parts.push("color not compared".to_string()),
        }
        format!("{:.0}% match - {}", self.score.min(1.0) * 100.0, parts.join(", "))
    }
}

/// Rank `candidates` by metadata proximity to `anchors` (photos already
/// confirmed to contain the target person). Candidates whose nearest anchor
/// in time falls outside `window_secs` are dropped entirely.
pub fn rank_by_metadata(
    anchors: &[PathBuf],
    candidates: Vec<PathBuf>,
    window_secs: i64,
) -> Vec<MetaSimCandidate> {
    struct AnchorMeta {
        path: PathBuf,
        secs: i64,
        camera: Option<String>,
    }

    let anchor_metas: Vec<AnchorMeta> = anchors
        .par_iter()
        .filter_map(|p| {
            let (secs, camera) = read_time_and_camera(p);
            secs.map(|secs| AnchorMeta { path: p.clone(), secs, camera })
        })
        .collect();

    if anchor_metas.is_empty() {
        return Vec::new();
    }

    struct Stage1 {
        path: PathBuf,
        anchor_idx: usize,
        delta_secs: i64,
        time_score: f32,
        same_camera: bool,
    }

    let mut stage1: Vec<Stage1> = candidates
        .par_iter()
        .filter_map(|path| {
            let (secs, camera) = read_time_and_camera(path);
            let secs = secs?;

            let (anchor_idx, delta_secs) = anchor_metas
                .iter()
                .enumerate()
                .map(|(i, a)| (i, (secs - a.secs).abs()))
                .min_by_key(|&(_, d)| d)?;

            if delta_secs > window_secs {
                return None;
            }

            let time_score = (-(delta_secs as f64) / TIME_DECAY_SECS).exp() as f32;
            let same_camera = matches!(
                (&camera, &anchor_metas[anchor_idx].camera),
                (Some(a), Some(b)) if a == b
            );

            Some(Stage1 { path: path.clone(), anchor_idx, delta_secs, time_score, same_camera })
        })
        .collect();

    stage1.sort_by(|a, b| {
        let sa = a.time_score + if a.same_camera { CAMERA_BONUS } else { 0.0 };
        let sb = b.time_score + if b.same_camera { CAMERA_BONUS } else { 0.0 };
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    stage1.truncate(STAGE1_TOP_N);

    // Stage 2: decode + color-compare only the survivors, anchored to the
    // *specific* nearest-in-time confirmed photo rather than an average over
    // the whole confirmed set, since lighting drifts across a session.
    let mut anchor_hist: Vec<Option<[f32; HIST_SIZE]>> = vec![None; anchor_metas.len()];
    for entry in &stage1 {
        if anchor_hist[entry.anchor_idx].is_none() {
            anchor_hist[entry.anchor_idx] = crate::utils::load_image_robustly(&anchor_metas[entry.anchor_idx].path)
                .ok()
                .map(|img| color_signature(&img));
        }
    }

    let mut results: Vec<MetaSimCandidate> = stage1
        .par_iter()
        .map(|entry| {
            let color_score = anchor_hist[entry.anchor_idx].and_then(|anchor_sig| {
                crate::utils::load_image_robustly(&entry.path)
                    .ok()
                    .map(|img| histogram_intersection(&color_signature(&img), &anchor_sig))
            });

            let camera_component = if entry.same_camera { CAMERA_WEIGHT } else { 0.0 };
            let color_component = color_score.unwrap_or(0.0) * COLOR_WEIGHT;
            let score = entry.time_score * TIME_WEIGHT + camera_component + color_component;

            MetaSimCandidate {
                path: entry.path.clone(),
                score,
                anchor: anchor_metas[entry.anchor_idx].path.clone(),
                delta_secs: entry.delta_secs,
                same_camera: entry.same_camera,
                color_score,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Capture timestamp (seconds, Unix-scale) and camera model, preferring EXIF
/// `DateTimeOriginal`/`DateTime`/`Model` and falling back to the file's
/// modified time when EXIF is missing or unreadable (screenshots, videos,
/// formats this build of kamadak-exif doesn't parse).
///
/// EXIF timestamps are naive local time while the mtime fallback is true UTC,
/// so a delta computed between an EXIF-timed anchor and an mtime-timed
/// candidate can be off by a timezone offset. Acceptable here: this feeds a
/// soft ranking signal for human review, not a hard filter.
fn read_time_and_camera(path: &Path) -> (Option<i64>, Option<String>) {
    let exif_data = std::fs::File::open(path).ok().and_then(|file| {
        let mut reader = std::io::BufReader::new(file);
        exif::Reader::new().read_from_container(&mut reader).ok()
    });

    let mut secs = None;
    let mut camera = None;

    if let Some(exif) = &exif_data {
        secs = exif
            .get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)
            .or_else(|| exif.get_field(exif::Tag::DateTime, exif::In::PRIMARY))
            .and_then(|f| ascii_field(&f.value))
            .and_then(|s| exif::DateTime::from_ascii(s.as_bytes()).ok())
            .map(|dt| exif_datetime_to_secs(&dt));

        camera = exif
            .get_field(exif::Tag::Model, exif::In::PRIMARY)
            .and_then(|f| ascii_field(&f.value))
            .map(|s| s.trim().to_string());
    }

    if secs.is_none() {
        secs = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);
    }

    (secs, camera)
}

fn ascii_field(value: &exif::Value) -> Option<String> {
    match value {
        exif::Value::Ascii(v) if !v.is_empty() => Some(String::from_utf8_lossy(&v[0]).to_string()),
        _ => None,
    }
}

fn exif_datetime_to_secs(dt: &exif::DateTime) -> i64 {
    days_from_civil(dt.year as i64, dt.month as u32, dt.day as u32) * 86400
        + dt.hour as i64 * 3600
        + dt.minute as i64 * 60
        + dt.second as i64
}

/// Days since 1970-01-01 for a civil (year, month, day), correct across the
/// whole proleptic Gregorian calendar including leap years. Howard Hinnant's
/// well-known `days_from_civil` algorithm - see
/// http://howardhinnant.github.io/date_algorithms.html
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Coarse color-palette fingerprint: a normalized 8x8x8 RGB histogram
/// computed on a small downsample (cheap - this is what "color grading"
/// similarity is scored on, not the full-resolution image).
fn color_signature(img: &DynamicImage) -> [f32; HIST_SIZE] {
    let small = img.thumbnail(48, 48).to_rgb8();
    let mut hist = [0f32; HIST_SIZE];
    let shift = 8 - HIST_BINS_PER_CHANNEL.trailing_zeros();

    let mut total = 0f32;
    for pixel in small.pixels() {
        let r = (pixel[0] >> shift) as usize;
        let g = (pixel[1] >> shift) as usize;
        let b = (pixel[2] >> shift) as usize;
        hist[r * HIST_BINS_PER_CHANNEL * HIST_BINS_PER_CHANNEL + g * HIST_BINS_PER_CHANNEL + b] += 1.0;
        total += 1.0;
    }
    if total > 0.0 {
        for v in hist.iter_mut() {
            *v /= total;
        }
    }
    hist
}

/// Histogram intersection: sum of the per-bin minimum, in [0, 1]. Standard,
/// cheap palette-similarity metric.
fn histogram_intersection(a: &[f32; HIST_SIZE], b: &[f32; HIST_SIZE]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).sum()
}

pub fn humanize_delta(secs: i64) -> String {
    let secs = secs.max(0);
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_from_civil_matches_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 2, 29), 11016); // leap day
        assert_eq!(days_from_civil(2024, 3, 1), 19783);
    }

    #[test]
    fn humanize_delta_formats_ranges() {
        assert_eq!(humanize_delta(45), "45s");
        assert_eq!(humanize_delta(134), "2m14s");
        assert_eq!(humanize_delta(3900), "1h05m");
    }

    #[test]
    fn histogram_intersection_of_identical_histograms_is_one() {
        let mut h = [0f32; HIST_SIZE];
        h[0] = 0.5;
        h[1] = 0.5;
        assert!((histogram_intersection(&h, &h) - 1.0).abs() < 1e-6);
    }
}

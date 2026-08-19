//! Ranks photos in the input directory by how likely they are to contain a
//! person whose face the recognition pipeline could not see (occluded, turned
//! away), using proximity to already-confirmed photos of that person on
//! several metadata axes: capture time, camera, GPS location, and filename
//! sequence number.
//!
//! See the "Similar Timing" tab in the GUI, which is the only caller of
//! `rank_by_metadata`.

use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Exponential time-decay constant, in seconds: score = exp(-delta / this).
const TIME_DECAY_SECS: f64 = 300.0;
/// Exponential distance-decay constant, in km, for the GPS signal.
const GPS_DECAY_KM: f64 = 0.5;
/// Exponential decay constant for filename-sequence-number gap.
const SEQUENCE_DECAY: f64 = 5.0;

const TIME_WEIGHT: f32 = 0.55;
const CAMERA_WEIGHT: f32 = 0.10;
const GPS_WEIGHT: f32 = 0.20;
const SEQUENCE_WEIGHT: f32 = 0.15;

/// Confidence multiplier applied to a timestamp that only came from file
/// mtime (weakest source - see `TimeSource`).
const MTIME_BASE_CONFIDENCE: f32 = 0.6;
/// Confidence used instead when mtime looks like a bulk-transfer artifact
/// (see `MTIME_COLLISION_THRESHOLD`) rather than a real capture time.
const MTIME_COLLISION_CONFIDENCE: f32 = 0.05;
/// Bucket width, in seconds, used to detect "many files share almost the
/// same mtime" (a bulk copy/sync, not organic photography).
const MTIME_COLLISION_BUCKET_SECS: i64 = 600;
/// A bucket with more than this many mtime-sourced files is treated as a
/// bulk-transfer artifact.
const MTIME_COLLISION_THRESHOLD: u32 = 15;

/// Offset between the Mac/QuickTime epoch (1904-01-01) and Unix epoch
/// (1970-01-01), in seconds - used to decode MP4/MOV `mvhd` creation time.
const MAC_EPOCH_OFFSET: i64 = 2_082_844_800;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TimeSource {
    /// EXIF `DateTimeOriginal`/`DateTime` - a real camera-written capture time.
    Exif,
    /// A timestamp parsed out of the filename itself (e.g. an iOS
    /// screenshot's "Screenshot 2024-01-15 at 3.45.12 PM.png"). Trusted as
    /// much as EXIF: it's written on-device before any transfer can corrupt it.
    FilenameTimestamp,
    /// The `mvhd` box's `creation_time` field in an MP4/MOV container.
    VideoAtom,
    /// Filesystem modified-time - the only source with no origin guarantee.
    /// A bulk copy/sync can stamp thousands of unrelated files with the same
    /// mtime, so this is deliberately the least-trusted source.
    Mtime,
}

impl TimeSource {
    fn base_confidence(self) -> f32 {
        match self {
            TimeSource::Mtime => MTIME_BASE_CONFIDENCE,
            _ => 1.0,
        }
    }
}

struct PhotoMeta {
    secs: i64,
    time_source: TimeSource,
    camera: Option<String>,
    /// (latitude, longitude) in decimal degrees.
    gps: Option<(f64, f64)>,
    /// (lowercased non-numeric prefix, trailing number) parsed from the
    /// filename, e.g. "IMG_1234.jpg" -> ("img", 1234).
    sequence: Option<(String, u64)>,
}

struct Located {
    path: PathBuf,
    meta: PhotoMeta,
}

#[derive(Clone)]
pub struct MetaSimCandidate {
    pub path: PathBuf,
    /// Final blended score, roughly in [0, 1] (can exceed 1 slightly when
    /// every signal lines up perfectly).
    pub score: f32,
    /// The confirmed photo this candidate scored best against.
    pub anchor: PathBuf,
    pub delta_secs: i64,
    /// Combined trust in `delta_secs` (candidate confidence x anchor
    /// confidence) - low when either side's time only came from a
    /// bulk-transfer-tainted mtime. Surfaced so the UI can flag it.
    pub time_confidence: f32,
    pub same_camera: bool,
    pub gps_km: Option<f64>,
    pub sequence_gap: Option<u64>,
}

impl MetaSimCandidate {
    /// Human-readable breakdown for a tooltip.
    pub fn explain(&self) -> String {
        let anchor_name = self
            .anchor
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut parts = vec![format!("{} from {}", humanize_delta(self.delta_secs), anchor_name)];
        if self.time_confidence < 0.5 {
            parts.push("timestamp looks like a bulk-transfer date, weighted down".to_string());
        }
        parts.push(if self.same_camera { "same camera".to_string() } else { "different/unknown camera".to_string() });
        if let Some(km) = self.gps_km {
            parts.push(format!("{:.2} km apart", km));
        }
        if let Some(gap) = self.sequence_gap {
            parts.push(format!("{} apart in filename sequence", gap));
        }
        format!("{:.0}% match - {}", self.score.min(1.0) * 100.0, parts.join(", "))
    }
}

/// Rank `candidates` by metadata proximity to `anchors` (photos already
/// confirmed to contain the target person). A candidate whose best anchor
/// falls outside `window_secs` is dropped. Each candidate is scored against
/// every anchor and keeps whichever anchor scores best overall, not just the
/// nearest in time - a candidate can be a weak time match but a strong GPS or
/// filename-sequence match against a different anchor.
pub fn rank_by_metadata(
    anchors: &[PathBuf],
    candidates: Vec<PathBuf>,
    window_secs: i64,
) -> Vec<MetaSimCandidate> {
    let anchor_data: Vec<Located> = anchors
        .par_iter()
        .filter_map(|p| read_photo_meta(p).map(|meta| Located { path: p.clone(), meta }))
        .collect();
    if anchor_data.is_empty() {
        return Vec::new();
    }

    let cand_data: Vec<Located> = candidates
        .into_par_iter()
        .filter_map(|p| read_photo_meta(&p).map(|meta| Located { path: p, meta }))
        .collect();

    // Detect mtime values that hundreds/thousands of files share within a
    // tight window - the signature of a bulk copy/sync, not real photography.
    let mut bucket_counts: HashMap<i64, u32> = HashMap::new();
    for meta in anchor_data.iter().chain(cand_data.iter()).map(|l| &l.meta) {
        if meta.time_source == TimeSource::Mtime {
            *bucket_counts.entry(meta.secs.div_euclid(MTIME_COLLISION_BUCKET_SECS)).or_insert(0) += 1;
        }
    }
    let confidence_of = |meta: &PhotoMeta| -> f32 {
        if meta.time_source != TimeSource::Mtime {
            return 1.0;
        }
        let bucket = meta.secs.div_euclid(MTIME_COLLISION_BUCKET_SECS);
        if bucket_counts.get(&bucket).copied().unwrap_or(0) > MTIME_COLLISION_THRESHOLD {
            MTIME_COLLISION_CONFIDENCE
        } else {
            meta.time_source.base_confidence()
        }
    };

    let mut results: Vec<MetaSimCandidate> = cand_data
        .par_iter()
        .filter_map(|cand| {
            let cand_conf = confidence_of(&cand.meta);
            let mut best: Option<MetaSimCandidate> = None;

            for anchor in &anchor_data {
                let delta_secs = (cand.meta.secs - anchor.meta.secs).abs();
                if delta_secs > window_secs {
                    continue;
                }

                let combined_conf = cand_conf * confidence_of(&anchor.meta);
                let time_score = (-(delta_secs as f64) / TIME_DECAY_SECS).exp() as f32 * combined_conf;

                let same_camera = matches!(
                    (&cand.meta.camera, &anchor.meta.camera),
                    (Some(a), Some(b)) if a == b
                );

                let gps_km = match (cand.meta.gps, anchor.meta.gps) {
                    (Some(a), Some(b)) => Some(haversine_km(a, b)),
                    _ => None,
                };
                let gps_score = gps_km.map(|km| (-km / GPS_DECAY_KM).exp() as f32).unwrap_or(0.0);

                let sequence_gap = match (&cand.meta.sequence, &anchor.meta.sequence) {
                    (Some((cp, cn)), Some((ap, an))) if cp.eq_ignore_ascii_case(ap) => Some(cn.abs_diff(*an)),
                    _ => None,
                };
                let sequence_score = sequence_gap.map(|g| (-(g as f64) / SEQUENCE_DECAY).exp() as f32).unwrap_or(0.0);

                let score = time_score * TIME_WEIGHT
                    + if same_camera { CAMERA_WEIGHT } else { 0.0 }
                    + gps_score * GPS_WEIGHT
                    + sequence_score * SEQUENCE_WEIGHT;

                if best.as_ref().map(|b| score > b.score).unwrap_or(true) {
                    best = Some(MetaSimCandidate {
                        path: cand.path.clone(),
                        score,
                        anchor: anchor.path.clone(),
                        delta_secs,
                        time_confidence: combined_conf,
                        same_camera,
                        gps_km,
                        sequence_gap,
                    });
                }
            }
            best
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn read_photo_meta(path: &Path) -> Option<PhotoMeta> {
    let mut secs = None;
    let mut source = TimeSource::Mtime;
    let mut camera = None;
    let mut gps = None;

    if crate::utils::is_video(path) {
        if is_isobmff_video(path) {
            secs = read_mp4_creation_time(path);
            if secs.is_some() {
                source = TimeSource::VideoAtom;
            }
        }
    } else if let Some(exif) = read_exif(path) {
        secs = exif_capture_secs(&exif);
        if secs.is_some() {
            source = TimeSource::Exif;
        }
        camera = exif_camera(&exif);
        gps = read_gps(&exif);
    }

    if secs.is_none() {
        secs = parse_filename_timestamp(path);
        if secs.is_some() {
            source = TimeSource::FilenameTimestamp;
        }
    }
    if secs.is_none() {
        secs = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);
        source = TimeSource::Mtime;
    }

    let sequence = parse_filename_sequence(path);
    secs.map(|secs| PhotoMeta { secs, time_source: source, camera, gps, sequence })
}

fn is_isobmff_video(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("mp4" | "mov" | "m4v")
    )
}

fn read_exif(path: &Path) -> Option<exif::Exif> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    exif::Reader::new().read_from_container(&mut reader).ok()
}

fn exif_capture_secs(exif: &exif::Exif) -> Option<i64> {
    exif.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)
        .or_else(|| exif.get_field(exif::Tag::DateTime, exif::In::PRIMARY))
        .and_then(|f| ascii_field(&f.value))
        .and_then(|s| exif::DateTime::from_ascii(s.as_bytes()).ok())
        .map(|dt| exif_datetime_to_secs(&dt))
}

fn exif_camera(exif: &exif::Exif) -> Option<String> {
    exif.get_field(exif::Tag::Model, exif::In::PRIMARY)
        .and_then(|f| ascii_field(&f.value))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn read_gps(exif: &exif::Exif) -> Option<(f64, f64)> {
    let lat = gps_coord(exif, exif::Tag::GPSLatitude, exif::Tag::GPSLatitudeRef, 'S')?;
    let lon = gps_coord(exif, exif::Tag::GPSLongitude, exif::Tag::GPSLongitudeRef, 'W')?;
    Some((lat, lon))
}

fn gps_coord(exif: &exif::Exif, value_tag: exif::Tag, ref_tag: exif::Tag, negative_ref: char) -> Option<f64> {
    let field = exif.get_field(value_tag, exif::In::PRIMARY)?;
    let exif::Value::Rational(dms) = &field.value else { return None };
    if dms.len() < 3 {
        return None;
    }
    let degrees = dms[0].to_f64() + dms[1].to_f64() / 60.0 + dms[2].to_f64() / 3600.0;

    let negative = exif
        .get_field(ref_tag, exif::In::PRIMARY)
        .and_then(|f| ascii_field(&f.value))
        .map(|s| s.trim().eq_ignore_ascii_case(&negative_ref.to_string()))
        .unwrap_or(false);

    Some(if negative { -degrees } else { degrees })
}

fn haversine_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;
    let (lat1, lon1) = (a.0.to_radians(), a.1.to_radians());
    let (lat2, lon2) = (b.0.to_radians(), b.1.to_radians());
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * h.sqrt().asin()
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

/// Try to recover a real capture timestamp embedded in the filename itself -
/// written on-device, so it survives a transfer that clobbers mtime.
fn parse_filename_timestamp(path: &Path) -> Option<i64> {
    let stem = path.file_stem()?.to_str()?;
    parse_ios_screenshot_name(stem).or_else(|| parse_yyyymmdd_hhmmss(stem))
}

/// "Screenshot 2024-01-15 at 3.45.12 PM" (iOS's screenshot filename format).
fn parse_ios_screenshot_name(stem: &str) -> Option<i64> {
    let lower = stem.to_ascii_lowercase();
    let after_prefix = lower.strip_prefix("screenshot ")?;
    let (date_part, time_part) = after_prefix.split_once(" at ")?;

    let mut date_fields = date_part.split('-');
    let year: i64 = date_fields.next()?.parse().ok()?;
    let month: u32 = date_fields.next()?.parse().ok()?;
    let day: u32 = date_fields.next()?.parse().ok()?;

    let mut time_fields = time_part.split_whitespace();
    let hms = time_fields.next()?;
    let ampm = time_fields.next()?;

    let mut hms_fields = hms.split('.');
    let mut hour: u32 = hms_fields.next()?.parse().ok()?;
    let minute: u32 = hms_fields.next()?.parse().ok()?;
    let second: u32 = hms_fields.next()?.parse().ok()?;

    if ampm.starts_with('p') && hour != 12 {
        hour += 12;
    } else if ampm.starts_with('a') && hour == 12 {
        hour = 0;
    }

    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64)
}

/// A `YYYYMMDD` digit run, optionally immediately followed by `HHMMSS`
/// (contiguous, or separated by one non-digit character) - covers
/// "Screenshot_20240115-154512.png", "IMG_20240115_154512.jpg",
/// "PXL_20240115_154512332.jpg", and similar Android/messaging-app exports.
fn parse_yyyymmdd_hhmmss(stem: &str) -> Option<i64> {
    let chars: Vec<char> = stem.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i;
        while j < n && chars[j].is_ascii_digit() {
            j += 1;
        }
        let run_len = j - start;

        if run_len >= 8 {
            let date_str: String = chars[start..start + 8].iter().collect();
            if let Some((y, mo, d)) = parse_yyyymmdd(&date_str) {
                let time_digits: Option<String> = if run_len >= 14 {
                    Some(chars[start + 8..start + 14].iter().collect())
                } else {
                    let mut k = start + run_len;
                    if k < n && !chars[k].is_ascii_digit() {
                        k += 1;
                    }
                    (k + 6 <= n && chars[k..k + 6].iter().all(|c| c.is_ascii_digit()))
                        .then(|| chars[k..k + 6].iter().collect())
                };
                if let Some((h, mi, s)) = time_digits.and_then(|t| parse_hhmmss(&t)) {
                    return Some(days_from_civil(y, mo, d) * 86400 + h as i64 * 3600 + mi as i64 * 60 + s as i64);
                }
            }
        }
        i = j.max(i + 1);
    }
    None
}

fn parse_yyyymmdd(s: &str) -> Option<(i64, u32, u32)> {
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(4..6)?.parse().ok()?;
    let day: u32 = s.get(6..8)?.parse().ok()?;
    ((1990..=2035).contains(&year) && (1..=12).contains(&month) && (1..=31).contains(&day))
        .then_some((year, month, day))
}

fn parse_hhmmss(s: &str) -> Option<(u32, u32, u32)> {
    let hour: u32 = s.get(0..2)?.parse().ok()?;
    let minute: u32 = s.get(2..4)?.parse().ok()?;
    let second: u32 = s.get(4..6)?.parse().ok()?;
    (hour <= 23 && minute <= 59 && second <= 59).then_some((hour, minute, second))
}

/// Trailing digit run in the filename stem plus whatever comes before it,
/// e.g. "IMG_1234.jpg" -> ("img", 1234). Used as a weak burst/session signal
/// when two files share the same naming family and a small numeric gap.
fn parse_filename_sequence(path: &Path) -> Option<(String, u64)> {
    let stem = path.file_stem()?.to_str()?;
    let chars: Vec<char> = stem.chars().collect();
    let n = chars.len();

    let mut end = n;
    while end > 0 && !chars[end - 1].is_ascii_digit() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut start = end;
    while start > 0 && chars[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if end - start < 2 {
        return None; // require >=2 digits to cut down on noise
    }

    let number: u64 = chars[start..end].iter().collect::<String>().parse().ok()?;
    let prefix: String = chars[..start]
        .iter()
        .collect::<String>()
        .trim_end_matches(['_', '-', ' '])
        .to_ascii_lowercase();
    Some((prefix, number))
}

/// Read the `creation_time` field out of an MP4/MOV `moov/mvhd` box. Returns
/// `None` (never panics) on any malformed/unexpected structure.
fn read_mp4_creation_time(path: &Path) -> Option<i64> {
    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    let moov = find_box(&mut file, 0, file_len, b"moov")?;
    let mvhd = find_box(&mut file, moov.0, moov.0 + moov.1, b"mvhd")?;

    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(mvhd.0)).ok()?;
    let mut version = [0u8; 1];
    file.read_exact(&mut version).ok()?;

    let mac_secs = if version[0] == 1 {
        let mut buf = [0u8; 3 + 8];
        file.read_exact(&mut buf).ok()?;
        i64::from_be_bytes(buf[3..11].try_into().ok()?)
    } else {
        let mut buf = [0u8; 3 + 4];
        file.read_exact(&mut buf).ok()?;
        u32::from_be_bytes(buf[3..7].try_into().ok()?) as i64
    };
    if mac_secs == 0 {
        return None; // unset, common in some encoders
    }
    Some(mac_secs - MAC_EPOCH_OFFSET)
}

/// Search sibling ISO-BMFF boxes in byte range `[start, end)` for `fourcc`,
/// returning that box's (content_start, content_len). Bounds the number of
/// boxes walked so a malformed file can't spin forever.
fn find_box(file: &mut std::fs::File, start: u64, end: u64, fourcc: &[u8]) -> Option<(u64, u64)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut pos = start;
    let mut guard = 0u32;
    while pos + 8 <= end {
        guard += 1;
        if guard > 10_000 {
            return None;
        }
        file.seek(SeekFrom::Start(pos)).ok()?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header).ok()?;
        let mut size = u32::from_be_bytes(header[0..4].try_into().ok()?) as u64;
        let box_type = &header[4..8];
        let mut header_len = 8u64;

        if size == 1 {
            let mut ext = [0u8; 8];
            file.read_exact(&mut ext).ok()?;
            size = u64::from_be_bytes(ext);
            header_len = 16;
        } else if size == 0 {
            return None; // "extends to end of file" - not worth chasing here
        }
        if size < header_len {
            return None;
        }

        if box_type == fourcc {
            return Some((pos + header_len, size - header_len));
        }
        pos += size;
    }
    None
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
    fn ios_screenshot_name_parses_to_the_right_moment() {
        let secs = parse_ios_screenshot_name("screenshot 2024-01-15 at 3.45.12 pm").unwrap();
        assert_eq!(secs, days_from_civil(2024, 1, 15) * 86400 + 15 * 3600 + 45 * 60 + 12);
    }

    #[test]
    fn ios_screenshot_name_handles_midnight_and_noon() {
        let midnight = parse_ios_screenshot_name("screenshot 2024-01-15 at 12.00.00 am").unwrap();
        assert_eq!(midnight, days_from_civil(2024, 1, 15) * 86400);
        let noon = parse_ios_screenshot_name("screenshot 2024-01-15 at 12.00.00 pm").unwrap();
        assert_eq!(noon, days_from_civil(2024, 1, 15) * 86400 + 12 * 3600);
    }

    #[test]
    fn yyyymmdd_hhmmss_parses_common_android_and_messaging_names() {
        let expected = days_from_civil(2024, 1, 15) * 86400 + 15 * 3600 + 45 * 60 + 12;
        assert_eq!(parse_yyyymmdd_hhmmss("IMG_20240115_154512"), Some(expected));
        assert_eq!(parse_yyyymmdd_hhmmss("Screenshot_20240115-154512"), Some(expected));
        assert_eq!(parse_yyyymmdd_hhmmss("PXL_20240115_154512332"), Some(expected));
        assert_eq!(parse_yyyymmdd_hhmmss("20240115154512"), Some(expected));
    }

    #[test]
    fn yyyymmdd_hhmmss_rejects_implausible_dates() {
        assert_eq!(parse_yyyymmdd_hhmmss("IMG_99999999_999999"), None);
    }

    #[test]
    fn filename_sequence_extracts_trailing_number_and_prefix() {
        assert_eq!(
            parse_filename_sequence(Path::new("IMG_1234.jpg")),
            Some(("img".to_string(), 1234))
        );
        assert_eq!(
            parse_filename_sequence(Path::new("DSC05678.NEF")),
            Some(("dsc".to_string(), 5678))
        );
        assert_eq!(parse_filename_sequence(Path::new("photo.jpg")), None);
    }

    #[test]
    fn haversine_of_same_point_is_zero_and_scales_with_distance() {
        let sf = (37.7749, -122.4194);
        assert!(haversine_km(sf, sf) < 1e-9);
        let nyc = (40.7128, -74.0060);
        let km = haversine_km(sf, nyc);
        assert!(km > 4000.0 && km < 4200.0); // SF-NYC is ~4130 km
    }

    #[test]
    fn mtime_confidence_drops_when_many_files_share_a_bucket() {
        let mut buckets: HashMap<i64, u32> = HashMap::new();
        buckets.insert(0, 3);
        buckets.insert(1, 25);

        let sparse = PhotoMeta { secs: 100, time_source: TimeSource::Mtime, camera: None, gps: None, sequence: None };
        let bulk = PhotoMeta { secs: MTIME_COLLISION_BUCKET_SECS + 5, time_source: TimeSource::Mtime, camera: None, gps: None, sequence: None };
        let confidence_of = |meta: &PhotoMeta| -> f32 {
            if meta.time_source != TimeSource::Mtime {
                return 1.0;
            }
            let bucket = meta.secs.div_euclid(MTIME_COLLISION_BUCKET_SECS);
            if buckets.get(&bucket).copied().unwrap_or(0) > MTIME_COLLISION_THRESHOLD {
                MTIME_COLLISION_CONFIDENCE
            } else {
                meta.time_source.base_confidence()
            }
        };

        assert_eq!(confidence_of(&sparse), MTIME_BASE_CONFIDENCE);
        assert_eq!(confidence_of(&bulk), MTIME_COLLISION_CONFIDENCE);
    }

    #[test]
    fn mp4_creation_time_reads_a_minimal_synthetic_moov_mvhd_box() {
        // ftyp (8 bytes header + 4 bytes "isom") + moov > mvhd (version 0).
        let mut data = Vec::new();
        data.extend_from_slice(&12u32.to_be_bytes());
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(b"isom");

        let mut mvhd = Vec::new();
        mvhd.extend_from_slice(&0u32.to_be_bytes()); // size placeholder
        mvhd.extend_from_slice(b"mvhd");
        mvhd.push(0); // version 0
        mvhd.extend_from_slice(&[0, 0, 0]); // flags
        // creation_time: 2024-01-15 15:45:12 UTC in Unix time, converted to Mac epoch.
        let unix_secs = days_from_civil(2024, 1, 15) * 86400 + 15 * 3600 + 45 * 60 + 12;
        let mac_secs = (unix_secs + MAC_EPOCH_OFFSET) as u32;
        mvhd.extend_from_slice(&mac_secs.to_be_bytes());
        mvhd.extend_from_slice(&[0u8; 4]); // modification_time (unused)
        let mvhd_len = mvhd.len() as u32;
        mvhd[0..4].copy_from_slice(&mvhd_len.to_be_bytes());

        let mut moov = Vec::new();
        moov.extend_from_slice(&0u32.to_be_bytes());
        moov.extend_from_slice(b"moov");
        moov.extend_from_slice(&mvhd);
        let moov_len = moov.len() as u32;
        moov[0..4].copy_from_slice(&moov_len.to_be_bytes());

        data.extend_from_slice(&moov);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("metasim_test_{}.mp4", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let result = read_mp4_creation_time(&path);
        let _ = std::fs::remove_file(&path);

        assert_eq!(result, Some(unix_secs));
    }
}

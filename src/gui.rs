use std::path::{Path, PathBuf};
use eframe::egui;
use rfd::FileDialog;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use rayon::prelude::*;
use walkdir::WalkDir;
use crate::CommandHideExt;

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(PartialEq, Eq, Clone, Copy)]
enum Tab {
    Matches,
    PersonFolder,
    MetadataSimilarity,
}

/// Longest edge a decoded thumbnail keeps. Thumbnails are drawn at most a few
/// hundred pixels wide, so decoding straight to this size keeps both the decode
/// and the GPU upload cheap.
const THUMB_MAX_EDGE: u32 = 640;

/// Quality of the cached thumbnail JPEGs.
const THUMB_CACHE_QUALITY: u8 = 80;

/// Gap between thumbnails, horizontally and vertically.
const THUMB_SPACING: f32 = 4.0;

/// Longest edge the pop-out viewer decodes. Bigger than a thumbnail so a photo
/// holds up when inspected, small enough to stay a quick decode.
const PREVIEW_MAX_EDGE: u32 = 2048;

/// One row of thumbnails, scaled so the row spans the full width.
struct ThumbRow {
    start: usize,
    end: usize,
    height: f32,
}

/// Width/height of every thumbnail on the page. Items still decoding borrow the
/// average shape of the ones already loaded, so the layout barely shifts as the
/// rest arrive.
fn thumb_aspects<'a>(
    textures: impl Iterator<Item = Option<&'a Result<egui::TextureHandle, String>>>,
) -> Vec<f32> {
    let known: Vec<Option<f32>> = textures
        .map(|texture| match texture {
            Some(Ok(handle)) => {
                let size = handle.size_vec2();
                (size.y > 0.0).then(|| size.x / size.y)
            }
            _ => None,
        })
        .collect();

    let loaded: Vec<f32> = known.iter().flatten().copied().collect();
    let fallback = if loaded.is_empty() {
        1.0
    } else {
        loaded.iter().sum::<f32>() / loaded.len() as f32
    };

    known
        .into_iter()
        .map(|aspect| aspect.unwrap_or(fallback).clamp(0.2, 5.0))
        .collect()
}

/// Pack thumbnails into rows that fill `avail` end to end. Each row is laid out
/// at a common height, then scaled so it finishes flush with the right edge -
/// portrait photos would otherwise leave most of the window empty, because a
/// fixed column count has to assume every cell is as wide as it is tall. The
/// last row is never stretched, only shrunk if it would overflow.
fn pack_thumb_rows(aspects: &[f32], target_height: f32, avail: f32) -> Vec<ThumbRow> {
    let mut rows: Vec<ThumbRow> = Vec::new();
    let mut start = 0usize;
    let mut aspect_sum = 0.0f32;

    for (idx, aspect) in aspects.iter().enumerate() {
        let count = idx - start;
        let width = (aspect_sum + aspect) * target_height + THUMB_SPACING * count as f32;
        if count > 0 && width > avail {
            rows.push(justify_row(start, idx, aspect_sum, target_height, avail));
            start = idx;
            aspect_sum = *aspect;
        } else {
            aspect_sum += aspect;
        }
    }

    if start < aspects.len() {
        let last = justify_row(start, aspects.len(), aspect_sum, target_height, avail);
        rows.push(ThumbRow {
            height: last.height.min(target_height),
            ..last
        });
    }
    rows
}

fn justify_row(
    start: usize,
    end: usize,
    aspect_sum: f32,
    target_height: f32,
    avail: f32,
) -> ThumbRow {
    let gaps = THUMB_SPACING * (end - start).saturating_sub(1) as f32;
    let natural_width = aspect_sum * target_height;
    let scale = if natural_width > 0.0 {
        ((avail - gaps).max(1.0) / natural_width).clamp(0.25, 3.0)
    } else {
        1.0
    };
    ThumbRow {
        start,
        end,
        height: target_height * scale,
    }
}

/// Draw one thumbnail (or its placeholder) filling `rect`.
fn paint_thumbnail(
    ui: &egui::Ui,
    rect: egui::Rect,
    texture: &Option<Result<egui::TextureHandle, String>>,
    path: &Path,
) {
    match texture {
        Some(Ok(handle)) => egui::Image::new(handle).paint_at(ui, rect),
        Some(Err(_)) => {
            ui.painter().rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "⚠ Error",
                egui::FontId::proportional(12.0),
                ui.visuals().error_fg_color,
            );
        }
        None => {
            ui.painter().rect_filled(rect, 4.0, ui.visuals().faint_bg_color);
            egui::Spinner::new().paint_at(ui, rect);
        }
    }
    if crate::utils::is_video(path) {
        paint_video_badge(ui, rect);
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum ThumbTarget {
    Matches,
    Person,
    MetaSim,
    /// The pop-out viewer, which decodes one photo at a larger size.
    Viewer,
}

/// The photo open in the pop-out viewer window.
struct ViewerState {
    path: PathBuf,
    texture: Option<Result<egui::TextureHandle, String>>,
    /// False while the viewer is showing the small grid thumbnail as a stand-in
    /// for the larger decode still in flight.
    sharp: bool,
}

/// One decoded thumbnail on its way back from the worker pool.
struct ThumbResult {
    target: ThumbTarget,
    generation: u64,
    path: PathBuf,
    image: Result<egui::ColorImage, String>,
}

#[derive(Serialize, Deserialize, Default)]
struct AppSettings {
    input_dir: Option<PathBuf>,
    #[serde(default)]
    people_dir: Option<PathBuf>,
    #[serde(default)]
    selected_person: Option<String>,
    #[serde(default)]
    target_dir: Option<PathBuf>,
}

impl AppSettings {
    fn load() -> Self {
        let config_dir = crate::get_app_data_dir();
        let config_path = config_dir.join("settings.json");
        if let Ok(data) = fs::read_to_string(config_path) {
            if let Ok(settings) = serde_json::from_str(&data) {
                return settings;
            }
        }
        Self::default()
    }

    fn save(&self) {
        let config_dir = crate::get_app_data_dir();
        if fs::create_dir_all(&config_dir).is_ok() {
            let config_path = config_dir.join("settings.json");
            if let Ok(data) = serde_json::to_string_pretty(self) {
                let _ = fs::write(config_path, data);
            }
        }
    }
}

pub struct FaceSearchApp {
    input_dir: Option<PathBuf>,
    people_dir: Option<PathBuf>,
    selected_person: Option<String>,
    people_names: Vec<String>,
    target_dir: Option<PathBuf>,
    match_threshold_min: f32,
    match_threshold: f32,
    filter_threshold: f32,
    page_size: usize,
    thumbnail_size: f32,
    current_page: usize,
    is_processing: bool,
    status_msg: String,

    // Stats and Matches state
    processed_count: usize,
    target_image_count: usize,
    target_video_count: usize,
    all_ranked_matches: Vec<(PathBuf, f32)>,
    matched_images_cache: Vec<(PathBuf, f32, bool, Option<Result<egui::TextureHandle, String>>)>,
    last_selected_index: Option<usize>,
    show_copy_confirm: bool,
    show_rebuild_confirm: bool,
    show_new_person_modal: bool,
    new_person_name: String,
    new_person_image_path: Option<PathBuf>,

    // Target person picker search box
    person_search: String,
    person_search_focus: bool,
    /// Index into the filtered name list that the arrow keys move.
    person_search_index: usize,
    person_search_scroll: bool,

    // "Copy to Other Person" popup search box - separate from the main
    // target-person picker above so it never touches the main selection.
    other_person_search: String,
    other_person_search_focus: bool,
    other_person_search_index: usize,
    other_person_search_scroll: bool,

    // Person folder tab state
    active_tab: Tab,
    /// Photo open in the pop-out viewer, from either tab.
    viewer: Option<ViewerState>,
    person_files: Vec<PathBuf>,
    person_images_cache: Vec<(PathBuf, Option<Result<egui::TextureHandle, String>>)>,
    person_page: usize,
    person_files_loaded: bool,
    person_scroll_to_top: bool,
    /// Paths whose thumbnails still need to be handed to the worker pool.
    matches_pending_thumbs: Vec<PathBuf>,
    person_pending_thumbs: Vec<PathBuf>,
    /// Bumped whenever a page is (re)loaded so results for the page we left
    /// behind can be discarded.
    matches_thumb_gen: Arc<AtomicU64>,
    person_thumb_gen: Arc<AtomicU64>,

    // "Similar Timing" (metadata-similarity) tab state
    metasim_window_minutes: f32,
    metasim_scanning: bool,
    metasim_ranked: Vec<crate::metadata_similarity::MetaSimCandidate>,
    metasim_images_cache: Vec<(crate::metadata_similarity::MetaSimCandidate, bool, Option<Result<egui::TextureHandle, String>>)>,
    metasim_last_selected_index: Option<usize>,
    show_metasim_copy_confirm: bool,
    metasim_page: usize,
    metasim_scroll_to_top: bool,
    metasim_pending_thumbs: Vec<PathBuf>,
    metasim_thumb_gen: Arc<AtomicU64>,

    // Log history for display during processing
    log_messages: Vec<String>,

    // Scroll control
    scroll_to_top: bool,

    // Communication with background thread
    tx: Sender<UiMessage>,
    rx: Receiver<UiMessage>,
    thumb_tx: Sender<ThumbResult>,
    thumb_rx: Receiver<ThumbResult>,
    metasim_tx: Sender<MetaSimMessage>,
    metasim_rx: Receiver<MetaSimMessage>,
}

pub enum MetaSimMessage {
    Done(Vec<crate::metadata_similarity::MetaSimCandidate>),
}

pub enum UiMessage {
    Log(String),
    Done(usize, Vec<(PathBuf, f32)>), // (processed_count, sorted matches with euclidean distance)
    Error(String),
}

impl Default for FaceSearchApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let (thumb_tx, thumb_rx) = channel();
        let (metasim_tx, metasim_rx) = channel();
        let settings = AppSettings::load();
        let mut people_dir = settings.people_dir.clone();
        let mut selected_person = settings.selected_person.clone();
        let target_dir = settings.target_dir.clone();

        // Backward compatibility with older settings that only had target_dir.
        if people_dir.is_none() {
            if let Some(td) = &target_dir {
                people_dir = td.parent().map(|p| p.to_path_buf());
                selected_person = td.file_name().map(|n| n.to_string_lossy().to_string());
            }
        }

        Self {
            input_dir: settings.input_dir,
            people_dir,
            selected_person,
            people_names: Vec::new(),
            target_dir,
            match_threshold_min: 0.0,
            match_threshold: 0.65,
            filter_threshold: 0.2,
            page_size: 100,
            thumbnail_size: 300.0,
            current_page: 0,
            is_processing: false,
            status_msg: "Ready".to_string(),
            processed_count: 0,
            target_image_count: 0,
            target_video_count: 0,
            all_ranked_matches: Vec::new(),
            matched_images_cache: Vec::new(),
            last_selected_index: None,
            show_copy_confirm: false,
            show_rebuild_confirm: false,
            show_new_person_modal: false,
            new_person_name: String::new(),
            new_person_image_path: None,
            person_search: String::new(),
            person_search_focus: false,
            person_search_index: 0,
            person_search_scroll: false,
            other_person_search: String::new(),
            other_person_search_focus: false,
            other_person_search_index: 0,
            other_person_search_scroll: false,
            active_tab: Tab::Matches,
            viewer: None,
            person_files: Vec::new(),
            person_images_cache: Vec::new(),
            person_page: 0,
            person_files_loaded: false,
            person_scroll_to_top: false,
            matches_pending_thumbs: Vec::new(),
            person_pending_thumbs: Vec::new(),
            matches_thumb_gen: Arc::new(AtomicU64::new(0)),
            person_thumb_gen: Arc::new(AtomicU64::new(0)),
            metasim_window_minutes: 60.0,
            metasim_scanning: false,
            metasim_ranked: Vec::new(),
            metasim_images_cache: Vec::new(),
            metasim_last_selected_index: None,
            show_metasim_copy_confirm: false,
            metasim_page: 0,
            metasim_scroll_to_top: false,
            metasim_pending_thumbs: Vec::new(),
            metasim_thumb_gen: Arc::new(AtomicU64::new(0)),
            log_messages: Vec::new(),
            scroll_to_top: false,
            tx,
            rx,
            thumb_tx,
            thumb_rx,
            metasim_tx,
            metasim_rx,
        }
    }
}

fn get_unique_path(dir: &std::path::Path, file_name: &std::ffi::OsStr) -> PathBuf {
    let mut path = dir.join(file_name);
    let mut counter = 1;
    let stem = std::path::Path::new(file_name).file_stem().unwrap_or_default().to_string_lossy();
    let ext = std::path::Path::new(file_name).extension().unwrap_or_default().to_string_lossy();

    while path.exists() {
        let new_name = if ext.is_empty() {
            format!("{}_{}", stem, counter)
        } else {
            format!("{}_{}.{}", stem, counter, ext)
        };
        path = dir.join(new_name);
        counter += 1;
    }
    path
}

/// True if `path`'s contents byte-for-byte match a file already sorted into
/// the people library. `target_filesizes` groups already-sorted files by size
/// for a cheap first filter before falling back to a full read - the same
/// "already seen" check `process_directory` uses for the Matches tab in
/// `main.rs`, so a photo doesn't get re-suggested once it's been sorted.
fn is_already_sorted(path: &Path, target_filesizes: &HashMap<u64, Vec<PathBuf>>) -> bool {
    let size = match fs::metadata(path) {
        Ok(m) => m.len(),
        Err(_) => return false,
    };
    let Some(candidates) = target_filesizes.get(&size) else { return false; };
    let Ok(bytes) = fs::read(path) else { return false; };

    let canon_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    candidates.iter().any(|target| {
        let canon_target = fs::canonicalize(target).unwrap_or_else(|_| target.clone());
        if canon_path == canon_target {
            return false;
        }
        fs::read(target).map(|target_bytes| target_bytes == bytes).unwrap_or(false)
    })
}

/// Open Explorer with `path` selected inside its containing folder.
/// Command line Explorer needs to open a folder with `path` selected.
///
/// Explorer does not parse its command line the standard way: the switch and
/// the path must be one token with the quotes around the path only. Quoting the
/// whole `/select,<path>` token - which `Command::arg` does as soon as the path
/// contains a space - makes Explorer ignore the switch and open the default
/// folder instead, so this has to be passed through as a raw argument.
fn select_argument(path: &Path) -> String {
    format!("/select,\"{}\"", path.display())
}

fn reveal_in_explorer(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;

        // Explorer needs a real file to select; without one it opens the
        // default folder, so fall back to just opening the parent directory.
        if !path.exists() {
            if let Some(parent) = path.parent() {
                open_in_explorer(parent);
            }
            return;
        }

        let _ = std::process::Command::new("explorer")
            .raw_arg(select_argument(path))
            .spawn();
    }
    #[cfg(not(target_os = "windows"))]
    let _ = path;
}

fn open_in_explorer(path: &Path) {
    let _ = std::process::Command::new("explorer").arg(path).spawn();
}

/// Hand a file to whatever application Windows opens it with.
fn open_with_default_app(path: &Path) {
    let _ = std::process::Command::new("explorer").arg(path).spawn();
}

/// `get_unique_path` appends `_1`, `_2`, ... when a name is already taken, so a
/// copy may not carry the exact original file name. Returns the de-suffixed
/// name when there is one. The counter is capped at three digits to avoid
/// treating names like `IMG_2024.jpg` as a numbered copy.
fn strip_copy_suffix(file_name: &str) -> Option<String> {
    let path = Path::new(file_name);
    let stem = path.file_stem()?.to_string_lossy().to_string();
    let (base, suffix) = stem.rsplit_once('_')?;
    if base.is_empty()
        || suffix.is_empty()
        || suffix.len() > 3
        || !suffix.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    Some(match path.extension() {
        Some(ext) => format!("{}.{}", base, ext.to_string_lossy()),
        None => base.to_string(),
    })
}

/// Look for the file a person-folder photo was copied from by walking the input
/// directory. Candidates are ranked: an exact name with matching bytes wins,
/// then a de-suffixed name with matching bytes, then name matches alone.
fn find_original(input_dir: &Path, copy_path: &Path, size: u64) -> Option<PathBuf> {
    let name = copy_path.file_name()?.to_string_lossy().to_lowercase();
    let alt_name = strip_copy_suffix(&name);

    let mut alt_and_size: Option<PathBuf> = None;
    let mut name_only: Option<PathBuf> = None;
    let mut alt_only: Option<PathBuf> = None;

    for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let entry_name = entry.file_name().to_string_lossy().to_lowercase();
        let exact_name = entry_name == name;
        if !exact_name && Some(&entry_name) != alt_name.as_ref() {
            continue;
        }
        let path = entry.path().to_path_buf();
        if path == copy_path {
            continue;
        }

        let same_size = size == 0 || fs::metadata(&path).map(|m| m.len()).unwrap_or(0) == size;
        match (exact_name, same_size) {
            (true, true) => return Some(path),
            (false, true) => {
                alt_and_size.get_or_insert(path);
            }
            (true, false) => {
                name_only.get_or_insert(path);
            }
            (false, false) => {
                alt_only.get_or_insert(path);
            }
        }
    }
    alt_and_size.or(name_only).or(alt_only)
}

/// Grab a single frame from a video into the thumbnail cache. Videos that were
/// never processed (e.g. clips already sitting in a person folder) have no
/// cached thumbnail, and would otherwise render as an error tile.
fn ensure_quick_video_thumbnail(video_path: &Path) -> Option<PathBuf> {
    let thumb_path = crate::utils::get_video_thumbnail_path(video_path);
    if crate::utils::video_thumbnail_exists(&thumb_path) {
        return Some(thumb_path);
    }
    let ffmpeg = crate::utils::find_ffmpeg_path()?;
    fs::create_dir_all(thumb_path.parent()?).ok()?;

    let status = std::process::Command::new(&ffmpeg)
        .hide_window()
        .arg("-y")
        .arg("-i")
        .arg(video_path)
        .arg("-vf")
        .arg("thumbnail")
        .arg("-frames:v")
        .arg("1")
        .arg(&thumb_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;

    (status.success() && crate::utils::video_thumbnail_exists(&thumb_path)).then_some(thumb_path)
}

/// Where a file's downscaled thumbnail is cached. Keyed by path, size and
/// modification time, so an edited or replaced photo gets a fresh thumbnail.
fn thumb_cache_path(path: &Path) -> Option<PathBuf> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let meta = fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();

    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    meta.len().hash(&mut hasher);
    modified.hash(&mut hasher);
    THUMB_MAX_EDGE.hash(&mut hasher);

    Some(
        crate::get_app_data_dir()
            .join("output")
            .join("thumb_cache")
            .join(format!("{:x}.jpg", hasher.finish())),
    )
}

fn to_color_image(img: &image::DynamicImage) -> egui::ColorImage {
    let size = [img.width() as usize, img.height() as usize];
    let image_buffer = img.to_rgba8();
    let pixels = image_buffer.as_flat_samples();
    egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice())
}

/// Decode `path`, or a video's extracted frame, no larger than `max_edge`.
fn decode_scaled(path: &Path, max_edge: u32) -> Result<egui::ColorImage, String> {
    let load_path = if crate::utils::is_video(path) {
        ensure_quick_video_thumbnail(path)
            .unwrap_or_else(|| crate::utils::get_video_thumbnail_path(path))
    } else {
        path.to_path_buf()
    };

    let img = crate::utils::load_image_robustly(&load_path).map_err(|e| e.to_string())?;
    Ok(to_color_image(&img.thumbnail(max_edge, max_edge)))
}

/// Decode `path` down to thumbnail size. Runs on a worker thread; the UI thread
/// only uploads the result. Downscaled copies are cached on disk, so revisiting
/// a folder skips the expensive full-size decode.
fn decode_thumbnail(path: &Path) -> Result<egui::ColorImage, String> {
    let cache_path = thumb_cache_path(path);
    if let Some(cached) = &cache_path {
        if let Ok(img) = image::open(cached) {
            return Ok(to_color_image(&img));
        }
    }

    let load_path = if crate::utils::is_video(path) {
        ensure_quick_video_thumbnail(path)
            .unwrap_or_else(|| crate::utils::get_video_thumbnail_path(path))
    } else {
        path.to_path_buf()
    };

    let img = crate::utils::load_image_robustly(&load_path).map_err(|e| e.to_string())?;
    let img = img.thumbnail(THUMB_MAX_EDGE, THUMB_MAX_EDGE);

    if let Some(cached) = &cache_path {
        write_thumb_cache(cached, &img);
    }
    Ok(to_color_image(&img))
}

fn write_thumb_cache(cache_path: &Path, img: &image::DynamicImage) {
    let Some(parent) = cache_path.parent() else { return };
    if fs::create_dir_all(parent).is_err() {
        return;
    }
    // Write to a temporary name first so a half-written file is never picked up
    // as a valid cache entry by another run.
    let temp_path = cache_path.with_extension("tmp");
    let Ok(file) = fs::File::create(&temp_path) else { return };
    let mut writer = std::io::BufWriter::new(file);
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, THUMB_CACHE_QUALITY);
    if img.to_rgb8().write_with_encoder(encoder).is_ok() {
        use std::io::Write;
        if writer.flush().is_ok() {
            drop(writer);
            let _ = fs::rename(&temp_path, cache_path);
            return;
        }
    }
    drop(writer);
    let _ = fs::remove_file(&temp_path);
}

/// Small film icon in the corner of a video thumbnail.
fn paint_video_badge(ui: &egui::Ui, rect: egui::Rect) {
    ui.painter().rect_filled(
        egui::Rect::from_min_size(
            egui::pos2(rect.right() - 24.0, rect.top() + 4.0),
            egui::vec2(20.0, 20.0),
        ),
        4.0,
        egui::Color32::from_black_alpha(128),
    );
    ui.painter().text(
        egui::pos2(rect.right() - 4.0, rect.top() + 4.0),
        egui::Align2::RIGHT_TOP,
        "\u{1F3AC}",
        egui::FontId::proportional(14.0),
        egui::Color32::WHITE,
    );
}

impl FaceSearchApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let mut app = Self::default();
        app.refresh_people_names();
        app.sync_target_dir_from_selection();
        app.update_target_count();
        app
    }

    fn save_settings(&self) {
        let settings = AppSettings {
            input_dir: self.input_dir.clone(),
            people_dir: self.people_dir.clone(),
            selected_person: self.selected_person.clone(),
            target_dir: self.target_dir.clone(),
        };
        settings.save();
    }

    fn refresh_people_names(&mut self) {
        self.people_names.clear();
        if let Some(people_dir) = &self.people_dir {
            if let Ok(entries) = std::fs::read_dir(people_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(name) = path.file_name().map(|n| n.to_string_lossy().to_string()) {
                            self.people_names.push(name);
                        }
                    }
                }
            }
        }
        self.people_names.sort_unstable_by_key(|name| name.to_lowercase());

        if let Some(selected) = &self.selected_person {
            if !self.people_names.iter().any(|name| name == selected) {
                self.selected_person = None;
            }
        }
    }

    fn sync_target_dir_from_selection(&mut self) {
        self.target_dir = match (&self.people_dir, &self.selected_person) {
            (Some(people_dir), Some(person)) => Some(people_dir.join(person)),
            _ => None,
        };
        self.invalidate_person_files();
    }

    fn update_target_count(&mut self) {
        self.target_image_count = 0;
        self.target_video_count = 0;
        if let Some(dir) = &self.target_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_file() {
                        if crate::utils::is_image(&path) {
                            self.target_image_count += 1;
                        } else if crate::utils::is_video(&path) {
                            self.target_video_count += 1;
                        }
                    }
                }
            }
        }
    }

    fn total_pages(&self) -> usize {
        if self.page_size == 0 { return 1; }
        let total = self.all_ranked_matches.len();
        if total == 0 { return 1; }
        (total + self.page_size - 1) / self.page_size
    }

    fn load_page(&mut self, page: usize) {
        self.matched_images_cache.clear();
        let start = page * self.page_size;
        let end = (start + self.page_size).min(self.all_ranked_matches.len());
        for (path, dist) in &self.all_ranked_matches[start..end] {
            self.matched_images_cache.push((path.clone(), *dist, false, None));
        }
        self.current_page = page;
        self.last_selected_index = None;
        self.scroll_to_top = true;
        self.matches_pending_thumbs = self
            .matched_images_cache
            .iter()
            .map(|(path, ..)| path.clone())
            .collect();
        self.matches_thumb_gen.fetch_add(1, Ordering::Relaxed);
    }

    fn person_total_pages(&self) -> usize {
        if self.page_size == 0 { return 1; }
        let total = self.person_files.len();
        if total == 0 { return 1; }
        (total + self.page_size - 1) / self.page_size
    }

    fn load_person_page(&mut self, page: usize) {
        self.person_images_cache.clear();
        let start = page * self.page_size;
        let end = (start + self.page_size).min(self.person_files.len());
        if start < end {
            for path in &self.person_files[start..end] {
                self.person_images_cache.push((path.clone(), None));
            }
        }
        self.person_page = page;
        self.person_scroll_to_top = true;
        self.person_pending_thumbs = self
            .person_images_cache
            .iter()
            .map(|(path, _)| path.clone())
            .collect();
        self.person_thumb_gen.fetch_add(1, Ordering::Relaxed);
    }

    fn metasim_total_pages(&self) -> usize {
        if self.page_size == 0 { return 1; }
        let total = self.metasim_ranked.len();
        if total == 0 { return 1; }
        (total + self.page_size - 1) / self.page_size
    }

    fn load_metasim_page(&mut self, page: usize) {
        self.metasim_images_cache.clear();
        let start = page * self.page_size;
        let end = (start + self.page_size).min(self.metasim_ranked.len());
        for candidate in &self.metasim_ranked[start..end] {
            self.metasim_images_cache.push((candidate.clone(), false, None));
        }
        self.metasim_page = page;
        self.metasim_last_selected_index = None;
        self.metasim_scroll_to_top = true;
        self.metasim_pending_thumbs = self
            .metasim_images_cache
            .iter()
            .map(|(candidate, _, _)| candidate.path.clone())
            .collect();
        self.metasim_thumb_gen.fetch_add(1, Ordering::Relaxed);
    }

    /// Decode a page worth of thumbnails on the rayon pool. Results stream back
    /// through `thumb_rx`, so the grid fills in progressively instead of
    /// blocking the UI thread on every file in turn.
    fn spawn_thumbnail_loader(&mut self, ctx: &egui::Context, target: ThumbTarget) {
        let paths = match target {
            ThumbTarget::Matches => std::mem::take(&mut self.matches_pending_thumbs),
            ThumbTarget::Person => std::mem::take(&mut self.person_pending_thumbs),
            ThumbTarget::MetaSim => std::mem::take(&mut self.metasim_pending_thumbs),
            ThumbTarget::Viewer => Vec::new(),
        };
        if paths.is_empty() {
            return;
        }

        let counter = match target {
            ThumbTarget::Matches => self.matches_thumb_gen.clone(),
            ThumbTarget::Person => self.person_thumb_gen.clone(),
            ThumbTarget::MetaSim => self.metasim_thumb_gen.clone(),
            ThumbTarget::Viewer => return,
        };
        let generation = counter.load(Ordering::Relaxed);
        let tx = self.thumb_tx.clone();
        let ctx = ctx.clone();

        thread::spawn(move || {
            paths.into_par_iter().for_each(|path| {
                // Bail out if the user already moved to another page.
                if counter.load(Ordering::Relaxed) != generation {
                    return;
                }
                let image = decode_thumbnail(&path);
                if tx
                    .send(ThumbResult { target, generation, path, image })
                    .is_ok()
                {
                    ctx.request_repaint();
                }
            });
        });
    }

    /// Hand decoded thumbnails to the GPU and drop anything from a stale page.
    fn drain_thumbnails(&mut self, ctx: &egui::Context) {
        while let Ok(result) = self.thumb_rx.try_recv() {
            let current = match result.target {
                ThumbTarget::Matches => self.matches_thumb_gen.load(Ordering::Relaxed),
                ThumbTarget::Person => self.person_thumb_gen.load(Ordering::Relaxed),
                ThumbTarget::MetaSim => self.metasim_thumb_gen.load(Ordering::Relaxed),
                // The viewer holds one photo, so the open path is the only check.
                ThumbTarget::Viewer => result.generation,
            };
            if result.generation != current {
                continue;
            }

            let texture = result.image.map(|image| {
                ctx.load_texture(
                    match result.target {
                        ThumbTarget::Viewer => format!("viewer:{}", result.path.display()),
                        _ => result.path.display().to_string(),
                    },
                    image,
                    egui::TextureOptions::LINEAR,
                )
            });

            match result.target {
                ThumbTarget::Matches => {
                    for entry in self.matched_images_cache.iter_mut() {
                        if entry.0 == result.path && entry.3.is_none() {
                            entry.3 = Some(texture.clone());
                        }
                    }
                }
                ThumbTarget::Person => {
                    for entry in self.person_images_cache.iter_mut() {
                        if entry.0 == result.path && entry.1.is_none() {
                            entry.1 = Some(texture.clone());
                        }
                    }
                }
                ThumbTarget::MetaSim => {
                    for entry in self.metasim_images_cache.iter_mut() {
                        if entry.0.path == result.path && entry.2.is_none() {
                            entry.2 = Some(texture.clone());
                        }
                    }
                }
                ThumbTarget::Viewer => {
                    if let Some(viewer) = &mut self.viewer {
                        if viewer.path == result.path {
                            viewer.texture = Some(texture.clone());
                            viewer.sharp = true;
                        }
                    }
                }
            }
        }
    }

    /// Drop the cached listing of the person folder so it is re-read the next
    /// time the tab is shown.
    fn invalidate_person_files(&mut self) {
        self.person_files_loaded = false;
        self.person_files.clear();
        self.person_images_cache.clear();
        self.person_page = 0;
        self.viewer = None;
        self.person_pending_thumbs.clear();
        self.person_thumb_gen.fetch_add(1, Ordering::Relaxed);
    }

    fn refresh_person_files(&mut self) {
        self.person_files.clear();
        self.person_files_loaded = true;

        let Some(dir) = self.target_dir.clone() else {
            self.person_images_cache.clear();
            return;
        };

        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file()
                    && (crate::utils::is_image(&path) || crate::utils::is_video(&path))
                {
                    self.person_files.push(path);
                }
            }
        }
        self.person_files.sort_unstable_by_key(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().to_lowercase())
                .unwrap_or_default()
        });
        self.update_target_count();
        let page = self.person_page.min(self.person_total_pages() - 1);
        self.load_person_page(page);
    }

    /// Hand a photo to whatever application Windows opens it with.
    fn open_in_default_app(&mut self, path: &Path) {
        if !path.exists() {
            self.status_msg = format!("{} is no longer there.", path.display());
            return;
        }
        open_with_default_app(path);
        self.status_msg = format!(
            "Opened {} in your default app.",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    /// Open a photo in the pop-out viewer, or close it if it is already open.
    /// The grid thumbnail is shown straight away while a larger decode runs in
    /// the background, so the window never opens empty.
    fn toggle_viewer(&mut self, ctx: &egui::Context, path: PathBuf) {
        if self.viewer.as_ref().map(|v| &v.path) == Some(&path) {
            self.viewer = None;
            return;
        }

        let placeholder = self
            .person_images_cache
            .iter()
            .find(|(p, _)| p == &path)
            .and_then(|(_, texture)| texture.clone())
            .or_else(|| {
                self.matched_images_cache
                    .iter()
                    .find(|(p, ..)| p == &path)
                    .and_then(|(.., texture)| texture.clone())
            });

        self.viewer = Some(ViewerState {
            path: path.clone(),
            texture: placeholder,
            sharp: false,
        });

        let tx = self.thumb_tx.clone();
        let ctx = ctx.clone();
        thread::spawn(move || {
            let image = decode_scaled(&path, PREVIEW_MAX_EDGE);
            let _ = tx.send(ThumbResult {
                target: ThumbTarget::Viewer,
                generation: 0,
                path,
                image,
            });
            ctx.request_repaint();
        });
    }

    /// Floating window for inspecting one photo, kept out of the grid so the
    /// thumbnails never reflow under the pointer.
    fn show_viewer_window(&mut self, ctx: &egui::Context) {
        let Some(viewer) = &self.viewer else { return };

        let title = viewer
            .path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Photo".to_string());
        let screen = ctx.screen_rect();
        let default_size = egui::vec2(screen.width() * 0.6, screen.height() * 0.8);

        let mut open = true;
        let mut reveal = false;
        let mut open_external = false;

        egui::Window::new(title)
            .id(egui::Id::new("photo_viewer"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(default_size)
            .default_pos(screen.center() - default_size / 2.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("📂 Show in Explorer").clicked() {
                        reveal = true;
                    }
                    if ui.button("🖼 Open in default app").clicked() {
                        open_external = true;
                    }
                    if !viewer.sharp {
                        ui.spinner();
                        ui.label(egui::RichText::new("sharpening…").weak().size(11.0));
                    }
                    ui.label(
                        egui::RichText::new(viewer.path.display().to_string())
                            .weak()
                            .size(11.0),
                    );
                });
                ui.separator();

                // The photo is painted into the space the window gives us
                // rather than laid out as a widget: a widget's size would feed
                // back into the window's size and fight the resize handle.
                let avail = (ui.available_size() - egui::vec2(0.0, 4.0)).max(egui::vec2(32.0, 32.0));
                let (rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
                match &viewer.texture {
                    Some(Ok(texture)) => {
                        let size = texture.size_vec2();
                        let scale = (rect.width() / size.x).min(rect.height() / size.y);
                        let fitted = egui::Rect::from_center_size(rect.center(), size * scale);
                        egui::Image::new(texture).paint_at(ui, fitted);
                    }
                    Some(Err(err)) => {
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            format!("Could not open this file: {err}"),
                            egui::FontId::proportional(13.0),
                            ui.visuals().error_fg_color,
                        );
                    }
                    None => {
                        egui::Spinner::new().paint_at(
                            ui,
                            egui::Rect::from_center_size(rect.center(), egui::vec2(48.0, 48.0)),
                        );
                    }
                }
            });

        if let Some(viewer) = &self.viewer {
            if reveal {
                reveal_in_explorer(&viewer.path);
            }
            if open_external {
                open_with_default_app(&viewer.path);
            }
        }
        if !open || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.viewer = None;
        }
    }

    fn ensure_person_files_loaded(&mut self) {
        if !self.person_files_loaded {
            self.person_page = 0;
            self.refresh_person_files();
        }
    }

    /// Reveal a photo from the person folder at the location it was copied
    /// from, by searching the input directory for a matching source file.
    fn reveal_original_location(&mut self, copy_path: &Path) {
        let file_name = copy_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let Some(input) = self.input_dir.clone() else {
            reveal_in_explorer(copy_path);
            self.status_msg =
                "No input directory selected - opened the copy in the person folder.".to_string();
            return;
        };

        let size = fs::metadata(copy_path).map(|m| m.len()).unwrap_or(0);
        let copy_path = copy_path.to_path_buf();
        let tx = self.tx.clone();
        self.status_msg = format!(
            "Searching {} for the original of {}...",
            input.display(),
            file_name
        );
        thread::spawn(move || {
            let msg = match find_original(&input, &copy_path, size) {
                Some(found) => {
                    reveal_in_explorer(&found);
                    format!("Opened original location: {}", found.display())
                }
                None => {
                    reveal_in_explorer(&copy_path);
                    format!(
                        "No original found under {} - opened the copy in the person folder instead.",
                        input.display()
                    )
                }
            };
            let _ = tx.send(UiMessage::Log(msg));
        });
    }

    /// Target-person picker with a type-to-filter search box. Built on a raw
    /// popup rather than `egui::ComboBox` because a combo popup closes on any
    /// click inside it, which would dismiss the search field.
    fn person_selector_ui(&mut self, ui: &mut egui::Ui) {
        let popup_id = ui.make_persistent_id("target_person_popup");
        let enabled = self.people_dir.is_some();
        let label = self
            .selected_person
            .clone()
            .unwrap_or_else(|| "Select person...".to_string());

        let button_resp = ui.add_enabled(
            enabled,
            egui::Button::new(format!("{}  \u{23F7}", label)).min_size(egui::vec2(240.0, 0.0)),
        );
        if button_resp.clicked() {
            ui.memory_mut(|m| m.toggle_popup(popup_id));
            self.person_search.clear();
            self.person_search_focus = true;
            self.person_search_index = self
                .selected_person
                .as_ref()
                .and_then(|selected| self.people_names.iter().position(|n| n == selected))
                .unwrap_or(0);
            self.person_search_scroll = true;
        }

        let mut chosen: Option<String> = None;
        egui::popup_below_widget(
            ui,
            popup_id,
            &button_resp,
            egui::PopupCloseBehavior::CloseOnClickOutside,
            |ui| {
                ui.set_min_width(280.0);
                let search = ui.add(
                    egui::TextEdit::singleline(&mut self.person_search)
                        .hint_text("Type to search people...")
                        .desired_width(f32::INFINITY),
                );
                if self.person_search_focus {
                    search.request_focus();
                    self.person_search_focus = false;
                }
                if search.changed() {
                    self.person_search_index = 0;
                }

                let needle = self.person_search.trim().to_lowercase();
                let filtered: Vec<&String> = self
                    .people_names
                    .iter()
                    .filter(|name| needle.is_empty() || name.to_lowercase().contains(&needle))
                    .collect();

                // Arrow keys walk the filtered list, Enter takes the highlighted
                // name, so narrowing down and picking never needs the mouse.
                let (up, down, enter) = ui.input(|i| {
                    (
                        i.key_pressed(egui::Key::ArrowUp),
                        i.key_pressed(egui::Key::ArrowDown),
                        i.key_pressed(egui::Key::Enter),
                    )
                });
                let scroll_to_highlight = up || down || self.person_search_scroll;
                if filtered.is_empty() {
                    self.person_search_index = 0;
                } else {
                    self.person_search_index = self.person_search_index.min(filtered.len() - 1);
                    if down {
                        self.person_search_index = (self.person_search_index + 1) % filtered.len();
                    }
                    if up {
                        self.person_search_index =
                            (self.person_search_index + filtered.len() - 1) % filtered.len();
                    }
                    if enter {
                        chosen = Some(filtered[self.person_search_index].clone());
                    }
                }

                ui.label(
                    egui::RichText::new(format!(
                        "{} of {} shown",
                        filtered.len(),
                        self.people_names.len()
                    ))
                    .weak()
                    .size(11.0),
                );
                ui.separator();

                if filtered.is_empty() {
                    ui.label(if self.people_names.is_empty() {
                        "No person folders found in the library."
                    } else {
                        "No people match that search."
                    });
                }

                egui::ScrollArea::vertical()
                    .id_source("person_search_list")
                    .max_height(260.0)
                    .show(ui, |ui| {
                        for (idx, name) in filtered.iter().enumerate() {
                            let is_current =
                                self.selected_person.as_deref() == Some(name.as_str());
                            let text = if is_current {
                                egui::RichText::new(*name).strong()
                            } else {
                                egui::RichText::new(*name)
                            };
                            let highlighted = idx == self.person_search_index;
                            let item = ui.selectable_label(highlighted, text);
                            if item.clicked() {
                                chosen = Some((*name).clone());
                            }
                            if highlighted && scroll_to_highlight {
                                item.scroll_to_me(Some(egui::Align::Center));
                            }
                        }
                    });
                self.person_search_scroll = false;

                if chosen.is_some() {
                    ui.memory_mut(|m| m.close_popup());
                }
            },
        );

        if let Some(name) = chosen {
            if self.selected_person.as_ref() != Some(&name) {
                self.selected_person = Some(name);
                self.sync_target_dir_from_selection();
                self.update_target_count();
                self.save_settings();
            }
        }
    }

    /// Button + search popup for copying the current selection to a person
    /// other than the main target above, without changing that selection.
    /// Returns the picked name the frame it's chosen, so the caller can act
    /// on it - mirrors `person_selector_ui`'s search/arrow-key UX but reports
    /// its pick instead of writing to `selected_person`.
    fn other_person_picker_button(&mut self, ui: &mut egui::Ui, id_source: &str) -> Option<String> {
        let popup_id = ui.make_persistent_id(id_source);
        let button_resp = ui.button("Copy to Other Person...");
        if button_resp.clicked() {
            ui.memory_mut(|m| m.toggle_popup(popup_id));
            self.other_person_search.clear();
            self.other_person_search_focus = true;
            self.other_person_search_index = 0;
            self.other_person_search_scroll = true;
        }

        let mut chosen: Option<String> = None;
        egui::popup_below_widget(
            ui,
            popup_id,
            &button_resp,
            egui::PopupCloseBehavior::CloseOnClickOutside,
            |ui| {
                ui.set_min_width(280.0);
                let search = ui.add(
                    egui::TextEdit::singleline(&mut self.other_person_search)
                        .hint_text("Type to search people...")
                        .desired_width(f32::INFINITY),
                );
                if self.other_person_search_focus {
                    search.request_focus();
                    self.other_person_search_focus = false;
                }
                if search.changed() {
                    self.other_person_search_index = 0;
                }

                let needle = self.other_person_search.trim().to_lowercase();
                let filtered: Vec<&String> = self
                    .people_names
                    .iter()
                    .filter(|name| needle.is_empty() || name.to_lowercase().contains(&needle))
                    .collect();

                let (up, down, enter) = ui.input(|i| {
                    (
                        i.key_pressed(egui::Key::ArrowUp),
                        i.key_pressed(egui::Key::ArrowDown),
                        i.key_pressed(egui::Key::Enter),
                    )
                });
                let scroll_to_highlight = up || down || self.other_person_search_scroll;
                if filtered.is_empty() {
                    self.other_person_search_index = 0;
                } else {
                    self.other_person_search_index = self.other_person_search_index.min(filtered.len() - 1);
                    if down {
                        self.other_person_search_index = (self.other_person_search_index + 1) % filtered.len();
                    }
                    if up {
                        self.other_person_search_index =
                            (self.other_person_search_index + filtered.len() - 1) % filtered.len();
                    }
                    if enter {
                        chosen = Some(filtered[self.other_person_search_index].clone());
                    }
                }

                ui.label(
                    egui::RichText::new(format!("{} of {} shown", filtered.len(), self.people_names.len()))
                        .weak()
                        .size(11.0),
                );
                ui.separator();

                if filtered.is_empty() {
                    ui.label(if self.people_names.is_empty() {
                        "No person folders found in the library."
                    } else {
                        "No people match that search."
                    });
                }

                egui::ScrollArea::vertical()
                    .id_source(format!("{id_source}_list"))
                    .max_height(260.0)
                    .show(ui, |ui| {
                        for (idx, name) in filtered.iter().enumerate() {
                            let highlighted = idx == self.other_person_search_index;
                            let item = ui.selectable_label(highlighted, egui::RichText::new(*name));
                            if item.clicked() {
                                chosen = Some((*name).clone());
                            }
                            if highlighted && scroll_to_highlight {
                                item.scroll_to_me(Some(egui::Align::Center));
                            }
                        }
                    });
                self.other_person_search_scroll = false;

                if chosen.is_some() {
                    ui.memory_mut(|m| m.close_popup());
                }
            },
        );

        chosen
    }

    /// Copy the selected Matches-tab candidates into `person_name`'s folder
    /// (a person other than the main target above, which is left untouched).
    fn copy_matched_selection_to_person(&mut self, person_name: &str) {
        let Some(people_dir) = self.people_dir.clone() else { return };
        let dest_dir = people_dir.join(person_name);
        let mut copy_count = 0;
        for (path, _, selected, _) in &self.matched_images_cache {
            if *selected {
                if let Some(file_name) = path.file_name() {
                    let destination = get_unique_path(&dest_dir, file_name);
                    if std::fs::copy(path, &destination).is_ok() {
                        copy_count += 1;
                    }
                }
            }
        }

        let removed: HashSet<PathBuf> = self.matched_images_cache.iter()
            .filter(|(_, _, s, _)| *s)
            .map(|(p, _, _, _)| p.clone())
            .collect();
        self.all_ranked_matches.retain(|(p, _)| !removed.contains(p));
        self.matched_images_cache.retain(|(_, _, s, _)| !*s);

        let max_page = if self.all_ranked_matches.is_empty() { 0 } else { self.total_pages() - 1 };
        let reload_page = self.current_page.min(max_page);
        self.load_page(reload_page);

        self.status_msg = format!("Copied {} image(s) to '{}'.", copy_count, person_name);
    }

    /// Copy the selected Similar-Timing candidates into `person_name`'s
    /// folder (a person other than the main target above, left untouched).
    fn copy_metasim_selection_to_person(&mut self, person_name: &str) {
        let Some(people_dir) = self.people_dir.clone() else { return };
        let dest_dir = people_dir.join(person_name);
        let mut copy_count = 0;
        for (candidate, selected, _) in &self.metasim_images_cache {
            if *selected {
                let path = &candidate.path;
                if let Some(file_name) = path.file_name() {
                    let destination = get_unique_path(&dest_dir, file_name);
                    if std::fs::copy(path, &destination).is_ok() {
                        copy_count += 1;
                    }
                }
            }
        }

        let removed: HashSet<PathBuf> = self.metasim_images_cache.iter()
            .filter(|(_, s, _)| *s)
            .map(|(c, ..)| c.path.clone())
            .collect();
        self.metasim_ranked.retain(|c| !removed.contains(&c.path));
        self.metasim_images_cache.retain(|(c, ..)| !removed.contains(&c.path));

        let max_page = if self.metasim_ranked.is_empty() { 0 } else { self.metasim_total_pages() - 1 };
        let reload_page = self.metasim_page.min(max_page);
        self.load_metasim_page(reload_page);

        self.status_msg = format!("Copied {} image(s) to '{}'.", copy_count, person_name);
    }
}

impl eframe::App for FaceSearchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_thumbnails(ctx);

        // Handle incoming messages from background thread
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                UiMessage::Log(txt) => {
                    self.status_msg = txt.clone();
                    self.log_messages.push(txt);
                    if self.log_messages.len() > 200 {
                        self.log_messages.remove(0);
                    }
                }
                UiMessage::Done(processed, ranked) => {
                    self.is_processing = false;
                    self.processed_count = processed;
                    let total = ranked.len();
                    self.all_ranked_matches = ranked;
                    self.load_page(0);
                    let pages = self.total_pages();
                    self.log_messages.clear();
                    if total > 0 {
                        self.status_msg = format!(
                            "Found {} candidates from {} images scanned. Showing page 1 of {}.",
                            total, processed, pages
                        );
                    } else {
                        self.status_msg = format!(
                            "Finished processing {} images. No candidates found within threshold.",
                            processed
                        );
                    }
                }
                UiMessage::Error(err) => {
                    self.is_processing = false;
                    self.log_messages.clear();
                    self.status_msg = format!("Error: {}", err);
                }
            }
        }

        while let Ok(msg) = self.metasim_rx.try_recv() {
            match msg {
                MetaSimMessage::Done(ranked) => {
                    self.metasim_scanning = false;
                    let total = ranked.len();
                    self.metasim_ranked = ranked;
                    self.load_metasim_page(0);
                    if total > 0 {
                        self.status_msg = format!("Found {} candidates by timing/camera/color similarity.", total);
                    } else {
                        self.status_msg = "No candidates found within the time window.".to_string();
                    }
                }
            }
        }

        // Handle drag and drop for target person directory.
        ctx.input(|i| {
            for file in &i.raw.dropped_files {
                if let Some(path) = &file.path {
                    let target_path = if path.is_dir() {
                        path.clone()
                    } else if let Some(parent) = path.parent() {
                        parent.to_path_buf()
                    } else {
                        continue;
                    };

                    self.people_dir = target_path.parent().map(|p| p.to_path_buf());
                    self.selected_person = target_path.file_name().map(|n| n.to_string_lossy().to_string());
                    self.refresh_people_names();
                    self.sync_target_dir_from_selection();
                    self.update_target_count();
                    self.save_settings();
                    break;
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!("Facial Recognition Sorter v{}", APP_VERSION));
            ui.separator();

            // --- Directories Input ---
            ui.horizontal(|ui| {
                if ui.button("Select Input Directory").clicked() {
                    if let Some(path) = FileDialog::new().pick_folder() {
                        self.input_dir = Some(path);
                        self.save_settings();
                    }
                }
                if let Some(p) = &self.input_dir {
                    if ui.button("📂 Open").clicked() {
                        open_in_explorer(p);
                    }
                }
                ui.label(match &self.input_dir {
                    Some(p) => p.display().to_string(),
                    None => "No directory selected".to_string(),
                });
            });

            ui.separator();

            // --- People Library Input ---
            ui.horizontal(|ui| {
                if ui.button("Select People Library").clicked() {
                    let mut dialog = FileDialog::new();
                    if let Some(sd) = &self.people_dir {
                        dialog = dialog.set_directory(sd);
                    }
                    if let Some(path) = dialog.pick_folder() {
                        self.people_dir = Some(path);
                        self.refresh_people_names();
                        if self.selected_person.is_none() {
                            self.selected_person = self.people_names.first().cloned();
                        }
                        self.sync_target_dir_from_selection();
                        self.update_target_count();
                        self.save_settings();
                    }
                }
                if let Some(p) = &self.people_dir {
                    if ui.button("📂 Open").clicked() {
                        open_in_explorer(p);
                    }
                    if ui.button("⟳ Refresh").clicked() {
                        self.refresh_people_names();
                        if self.selected_person.is_none() {
                            self.selected_person = self.people_names.first().cloned();
                        }
                        self.sync_target_dir_from_selection();
                        self.update_target_count();
                        self.save_settings();
                    }
                }
                ui.label(match &self.people_dir {
                    Some(p) => format!("Library: {}", p.display()),
                    None => "No people library selected".to_string(),
                });
            });

            ui.horizontal(|ui| {
                ui.label("Target person:");
                self.person_selector_ui(ui);

                ui.label(match &self.target_dir {
                    Some(p) => format!("{} ({} images, {} videos)", p.display(), self.target_image_count, self.target_video_count),
                    None => "No person selected".to_string(),
                });
            });

            if self.target_dir.is_some() {
                ui.horizontal(|ui| {
                    ui.label("Distance range:");
                    ui.add(
                        egui::DragValue::new(&mut self.match_threshold_min)
                            .range(0.0..=2.0)
                            .speed(0.01)
                            .fixed_decimals(2)
                            .prefix("Min: "),
                    );
                    ui.label("–");
                    ui.add(
                        egui::DragValue::new(&mut self.match_threshold)
                            .range(0.0..=2.0)
                            .speed(0.01)
                            .fixed_decimals(2)
                            .prefix("Max: "),
                    );
                    ui.add_space(16.0);
                    ui.label("Per page:");
                    ui.add(
                        egui::DragValue::new(&mut self.page_size)
                            .range(10..=200)
                            .speed(1.0),
                    );
                    ui.add_space(16.0);
                    ui.label("Thumbnail:");
                    ui.add(
                        egui::DragValue::new(&mut self.thumbnail_size)
                            .range(60.0..=300.0)
                            .speed(1.0)
                            .suffix(" px"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Target rejection:");
                    ui.add(
                        egui::DragValue::new(&mut self.filter_threshold)
                            .range(0.0..=1.0)
                            .speed(0.01)
                            .fixed_decimals(2)
                    );
                    ui.label("(Higher = stricter. Faces below this similarity to the dominant identity are rejected as bystanders.)");
                });
                ui.label("(Lower distance = stricter. Results within the range are shown, ranked best-match-first.)");
            }

            ui.separator();

            // --- Start Button ---
            let can_start =
                !self.is_processing && self.input_dir.is_some() && self.target_dir.is_some();
            ui.horizontal(|ui| {
                ui.add_enabled_ui(can_start, |ui| {
                    if ui.button("Start Processing").clicked() {
                        self.update_target_count();
                        self.is_processing = true;
                        self.status_msg = "Starting...".to_string();
                        self.all_ranked_matches.clear();
                        self.matched_images_cache.clear();
                        self.current_page = 0;

                        let tx_clone = self.tx.clone();
                        let input = self.input_dir.clone().unwrap();
                        let target_dir = self.target_dir.clone();
                        let people_dir = self.people_dir.clone();
                        let threshold_min = self.match_threshold_min.min(self.match_threshold);
                        let threshold_max = self.match_threshold;
                        let filter_threshold = self.filter_threshold;

                        thread::spawn(move || {
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                crate::process_directory(input, target_dir, people_dir, threshold_min, threshold_max, filter_threshold, tx_clone.clone())
                            }));
                            match result {
                                Ok(Err(e)) => {
                                    let _ = tx_clone.send(UiMessage::Error(e.to_string()));
                                }
                                Err(panic_info) => {
                                    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                                        format!("Internal error (panic): {}", s)
                                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                                        format!("Internal error (panic): {}", s)
                                    } else {
                                        "Internal error (panic): unknown".to_string()
                                    };
                                    let _ = tx_clone.send(UiMessage::Error(msg));
                                }
                                Ok(Ok(())) => {}
                            }
                        });
                    }
                    if ui.button("Rebuild Database").clicked() {
                        self.show_rebuild_confirm = true;
                    }
                });

                if let Some(target_dir) = self.target_dir.clone() {
                    ui.add_space(16.0);
                    if ui.button("⟳ Refresh Person Folder").clicked() {
                        self.invalidate_person_files();
                    }
                    if ui.button("📂 Open Person Folder").clicked() {
                        open_in_explorer(&target_dir);
                    }
                }
            });

            if self.input_dir.is_some() && self.target_dir.is_none() {
                ui.label("Select a people library and target person to enable processing.");
            }

            // --- Status ---
            // Buttons are pinned to the right edge so a long status message on
            // the left never shifts them around. A right-to-left layout places
            // the first widget added furthest right, so these are added in
            // reverse of how they read on screen.
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if !self.is_processing && !self.all_ranked_matches.is_empty() && self.target_dir.is_some() {
                        if ui.button("Open Debug Rejected").clicked() {
                            let rejected_dir = crate::get_app_data_dir().join("output").join("debug_rejected");
                            if rejected_dir.exists() {
                                open_in_explorer(&rejected_dir);
                            }
                        }
                        if ui.button("Open Debug Targets").clicked() {
                            let debug_dir = crate::get_app_data_dir().join("output").join("debug_targets");
                            if debug_dir.exists() {
                                open_in_explorer(&debug_dir);
                            }
                        }
                        if ui.button("Open Output Folder").clicked() {
                            let output = crate::get_app_data_dir().join("output").join("target_matches");
                            if output.exists() {
                                open_in_explorer(&output);
                            }
                        }
                        if ui.button("Export All to Output Folder").clicked() {
                            let out_dir = crate::get_app_data_dir().join("output").join("target_matches");
                            let _ = std::fs::create_dir_all(&out_dir);
                            for (src_path, _) in &self.all_ranked_matches {
                                if let Some(name) = src_path.file_name() {
                                    let dest = out_dir.join(name);
                                    if !dest.exists() {
                                        let _ = std::fs::copy(src_path, &dest);
                                    }
                                }
                            }
                            self.status_msg = format!("Exported {} candidates to output folder.", self.all_ranked_matches.len());
                        }
                        let selected_count = self.matched_images_cache.iter().filter(|(_, _, s, _)| *s).count();
                        if selected_count > 0 {
                            if let Some(name) = self.other_person_picker_button(ui, "matches_other_person_popup") {
                                self.copy_matched_selection_to_person(&name);
                            }
                            if ui.button(format!("Copy {} Selected to Person Folder", selected_count)).clicked() {
                                self.show_copy_confirm = true;
                            }
                        }
                    }

                    // Whatever width the buttons left over belongs to the status
                    // text, truncated rather than allowed to push them off.
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add(
                            egui::Label::new(egui::RichText::new(&self.status_msg).strong())
                                .truncate(),
                        )
                        .on_hover_text(&self.status_msg);
                    });
                });
            });

            // --- Copy Confirmation Modal ---
            if self.show_copy_confirm {
                let selected_indices: Vec<usize> = self.matched_images_cache.iter()
                    .enumerate()
                    .filter(|(_, (_, _, s, _))| *s)
                    .map(|(i, _)| i)
                    .collect();

                let mut do_copy = false;
                let mut do_cancel = false;

                egui::Window::new("Confirm Copy")
                    .collapsible(false)
                    .resizable(true)
                    .default_size([600.0, 450.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading(format!("Copy {} selected photo(s) to the selected person folder?", selected_indices.len()));
                        ui.separator();

                        let cell_size = 100.0_f32;
                        let available_w = ui.available_width();
                        let cols = ((available_w / cell_size).floor() as usize).max(1);

                        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                            egui::Grid::new("confirm_preview_grid")
                                .num_columns(cols)
                                .spacing([4.0, 4.0])
                                .show(ui, |ui| {
                                    for (col_i, &idx) in selected_indices.iter().enumerate() {
                                        let (img_path, _, _, texture_res_opt) = &self.matched_images_cache[idx];
                                        match texture_res_opt {
                                            Some(Ok(texture)) => {
                                                let image = egui::Image::new(&*texture)
                                                    .fit_to_exact_size(egui::vec2(88.0, 88.0))
                                                    .maintain_aspect_ratio(true);
                                                ui.add(image);
                                            }
                                            Some(Err(_)) => {
                                                ui.add_sized([88.0, 88.0], egui::Label::new("⚠ Error"));
                                            }
                                            None => {
                                                ui.add_sized([88.0, 88.0], egui::Label::new(img_path.file_name()
                                                    .map(|n| n.to_string_lossy().to_string())
                                                    .unwrap_or_default()));
                                            }
                                        }
                                        if (col_i + 1) % cols == 0 {
                                            ui.end_row();
                                        }
                                    }
                                });
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button(egui::RichText::new("✔ Confirm Copy").color(egui::Color32::from_rgb(80, 200, 100))).clicked() {
                                do_copy = true;
                            }
                            ui.add_space(12.0);
                            if ui.button(egui::RichText::new("✖ Cancel").color(egui::Color32::from_rgb(220, 80, 80))).clicked() {
                                do_cancel = true;
                            }
                        });
                    });

                if do_copy {
                    let target_dest = self.target_dir.clone().unwrap();
                    let mut copy_count = 0;
                    for (path, _, selected, _) in &self.matched_images_cache {
                        if *selected {
                            if let Some(file_name) = path.file_name() {
                                let destination = get_unique_path(&target_dest, file_name);
                                if std::fs::copy(path, &destination).is_ok() {
                                    copy_count += 1;
                                }
                            }
                        }
                    }

                    // Remove copied items from master ranked list
                    let removed: HashSet<PathBuf> = self.matched_images_cache.iter()
                        .filter(|(_, _, s, _)| *s)
                        .map(|(p, _, _, _)| p.clone())
                        .collect();
                    self.all_ranked_matches.retain(|(p, _)| !removed.contains(p));
                    self.matched_images_cache.retain(|(_, _, s, _)| !*s);

                    // Reload: if current page is now past the end, go back one page
                    let max_page = if self.all_ranked_matches.is_empty() { 0 } else { self.total_pages() - 1 };
                    let reload_page = self.current_page.min(max_page);
                    self.load_page(reload_page);

                    self.status_msg = format!("Successfully copied {} images to the selected person folder!", copy_count);
                    self.update_target_count();
                    self.invalidate_person_files();
                    self.show_copy_confirm = false;
                }
                if do_cancel {
                    self.show_copy_confirm = false;
                }
            }

            // --- Similar Timing Copy Confirmation Modal ---
            if self.show_metasim_copy_confirm {
                let selected_indices: Vec<usize> = self.metasim_images_cache.iter()
                    .enumerate()
                    .filter(|(_, (_, s, _))| *s)
                    .map(|(i, _)| i)
                    .collect();

                let mut do_copy = false;
                let mut do_cancel = false;

                egui::Window::new("Confirm Copy")
                    .collapsible(false)
                    .resizable(true)
                    .default_size([600.0, 450.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading(format!("Copy {} selected photo(s) to the selected person folder?", selected_indices.len()));
                        ui.separator();

                        let cell_size = 100.0_f32;
                        let available_w = ui.available_width();
                        let cols = ((available_w / cell_size).floor() as usize).max(1);

                        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                            egui::Grid::new("metasim_confirm_preview_grid")
                                .num_columns(cols)
                                .spacing([4.0, 4.0])
                                .show(ui, |ui| {
                                    for (col_i, &idx) in selected_indices.iter().enumerate() {
                                        let (candidate, _, texture_res_opt) = &self.metasim_images_cache[idx];
                                        match texture_res_opt {
                                            Some(Ok(texture)) => {
                                                let image = egui::Image::new(&*texture)
                                                    .fit_to_exact_size(egui::vec2(88.0, 88.0))
                                                    .maintain_aspect_ratio(true);
                                                ui.add(image);
                                            }
                                            Some(Err(_)) => {
                                                ui.add_sized([88.0, 88.0], egui::Label::new("⚠ Error"));
                                            }
                                            None => {
                                                ui.add_sized([88.0, 88.0], egui::Label::new(candidate.path.file_name()
                                                    .map(|n| n.to_string_lossy().to_string())
                                                    .unwrap_or_default()));
                                            }
                                        }
                                        if (col_i + 1) % cols == 0 {
                                            ui.end_row();
                                        }
                                    }
                                });
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button(egui::RichText::new("✔ Confirm Copy").color(egui::Color32::from_rgb(80, 200, 100))).clicked() {
                                do_copy = true;
                            }
                            ui.add_space(12.0);
                            if ui.button(egui::RichText::new("✖ Cancel").color(egui::Color32::from_rgb(220, 80, 80))).clicked() {
                                do_cancel = true;
                            }
                        });
                    });

                if do_copy {
                    let target_dest = self.target_dir.clone().unwrap();
                    let mut copy_count = 0;
                    for (candidate, selected, _) in &self.metasim_images_cache {
                        if *selected {
                            let path = &candidate.path;
                            if let Some(file_name) = path.file_name() {
                                let destination = get_unique_path(&target_dest, file_name);
                                if std::fs::copy(path, &destination).is_ok() {
                                    copy_count += 1;
                                }
                            }
                        }
                    }

                    let removed: HashSet<PathBuf> = self.metasim_images_cache.iter()
                        .filter(|(_, s, _)| *s)
                        .map(|(c, ..)| c.path.clone())
                        .collect();
                    self.metasim_ranked.retain(|c| !removed.contains(&c.path));
                    self.metasim_images_cache.retain(|(c, ..)| !removed.contains(&c.path));

                    let max_page = if self.metasim_ranked.is_empty() { 0 } else { self.metasim_total_pages() - 1 };
                    let reload_page = self.metasim_page.min(max_page);
                    self.load_metasim_page(reload_page);

                    self.status_msg = format!("Successfully copied {} images to the selected person folder!", copy_count);
                    self.update_target_count();
                    self.invalidate_person_files();
                    self.show_metasim_copy_confirm = false;
                }
                if do_cancel {
                    self.show_metasim_copy_confirm = false;
                }
            }

            // --- Rebuild Database Confirmation Modal ---
            if self.show_rebuild_confirm {
                let mut do_rebuild = false;
                let mut do_cancel = false;

                egui::Window::new("Confirm Rebuild Database")
                    .collapsible(false)
                    .resizable(false)
                    .default_size([450.0, 150.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading("Are you sure?");
                        ui.separator();
                        ui.label("This will delete the face database (faces_db.bin and faces_db.json).");
                        ui.label("All images will need to be re-processed from scratch on the next run.");
                        ui.label(
                            egui::RichText::new("This cannot be undone.")
                                .color(egui::Color32::from_rgb(220, 80, 80))
                                .strong()
                        );
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button(
                                egui::RichText::new("Confirm Rebuild")
                                    .color(egui::Color32::from_rgb(220, 80, 80))
                            ).clicked() {
                                do_rebuild = true;
                            }
                            ui.add_space(12.0);
                            if ui.button("Cancel").clicked() {
                                do_cancel = true;
                            }
                        });
                    });

                if do_rebuild {
                    let db_path = crate::get_db_file();
                    if db_path.exists() {
                        let _ = std::fs::remove_file(db_path);
                    }
                    let db_json_path = crate::get_db_file_json();
                    if db_json_path.exists() {
                        let _ = std::fs::remove_file(db_json_path);
                    }
                    self.status_msg = "Database deleted. Click 'Start Processing' to rebuild from scratch.".to_string();
                    self.show_rebuild_confirm = false;
                }
                if do_cancel {
                    self.show_rebuild_confirm = false;
                }
            }

            // --- New Person Modal ---
            if self.show_new_person_modal {
                let mut do_create = false;
                let mut do_cancel = false;

                let people_dir = self.people_dir.clone();

                egui::Window::new("Create New Person")
                    .collapsible(false)
                    .resizable(false)
                    .default_size([400.0, 200.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading("Create a new person directory");
                        ui.separator();

                        if let Some(pd) = &people_dir {
                            ui.label(format!("Parent: {}", pd.display()));
                        }

                        ui.horizontal(|ui| {
                            ui.label("Person name:");
                            ui.text_edit_singleline(&mut self.new_person_name);
                        });

                        let name_trimmed = self.new_person_name.trim();
                        let is_valid = !name_trimmed.is_empty()
                            && !name_trimmed.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']);
                        let already_exists = people_dir.as_ref()
                            .map(|pd| pd.join(name_trimmed).exists())
                            .unwrap_or(false);

                        if !name_trimmed.is_empty() && !is_valid {
                            ui.colored_label(egui::Color32::from_rgb(220, 80, 80),
                                "Name contains invalid characters.");
                        }
                        if already_exists {
                            ui.colored_label(egui::Color32::from_rgb(220, 180, 0),
                                "Directory already exists. Image will be added to it.");
                        }

                        if let Some(img_path) = &self.new_person_image_path {
                            ui.label(format!("Image: {}", img_path.file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default()));
                        }

                        ui.separator();
                        ui.horizontal(|ui| {
                            let can_create = is_valid && people_dir.is_some();
                            if ui.add_enabled(can_create,
                                egui::Button::new(
                                    egui::RichText::new("Create + Copy")
                                        .color(egui::Color32::from_rgb(80, 200, 100))
                                )).clicked() {
                                do_create = true;
                            }
                            ui.add_space(12.0);
                            if ui.button(
                                egui::RichText::new("Cancel")
                                    .color(egui::Color32::from_rgb(220, 80, 80))
                            ).clicked() {
                                do_cancel = true;
                            }
                        });
                    });

                if do_create {
                    if let (Some(people_dir), Some(img_path)) = (people_dir, &self.new_person_image_path.clone()) {
                        let person_dir = people_dir.join(self.new_person_name.trim());
                        let _ = std::fs::create_dir_all(&person_dir);
                        if let Some(file_name) = img_path.file_name() {
                            let dest = get_unique_path(&person_dir, file_name);
                            if std::fs::copy(img_path, &dest).is_ok() {
                                self.status_msg = format!(
                                    "Created '{}' and copied image.",
                                    self.new_person_name.trim()
                                );
                                self.refresh_people_names();
                                self.selected_person = Some(self.new_person_name.trim().to_string());
                                self.sync_target_dir_from_selection();
                                self.update_target_count();
                                self.save_settings();
                            } else {
                                self.status_msg = "Failed to copy image.".to_string();
                            }
                        }
                    }
                    self.show_new_person_modal = false;
                    self.new_person_image_path = None;
                }
                if do_cancel {
                    self.show_new_person_modal = false;
                    self.new_person_image_path = None;
                }
            }

            if self.is_processing {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(egui::RichText::new("Processing...").italics());
                });
                if !self.log_messages.is_empty() {
                    egui::ScrollArea::vertical()
                        .id_source("log_scroll")
                        .max_height(150.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui: &mut egui::Ui| {
                            for line in &self.log_messages {
                                ui.label(egui::RichText::new(line).monospace().size(11.0));
                            }
                        });
                }
                ctx.request_repaint();
            }

            // --- Result Tabs ---
            if !self.is_processing && self.target_dir.is_some() {
                ui.separator();
                ui.horizontal(|ui| {
                    let matches_label =
                        format!("🔍 Matches ({})", self.all_ranked_matches.len());
                    let person_label = format!(
                        "👤 Person Folder ({})",
                        self.target_image_count + self.target_video_count
                    );
                    let metasim_label =
                        format!("🕒 Similar Timing ({})", self.metasim_ranked.len());
                    ui.selectable_value(&mut self.active_tab, Tab::Matches, matches_label);
                    ui.selectable_value(&mut self.active_tab, Tab::PersonFolder, person_label);
                    ui.selectable_value(&mut self.active_tab, Tab::MetadataSimilarity, metasim_label);
                });

                match self.active_tab {
                    Tab::Matches => self.show_matches_tab(ui, ctx),
                    Tab::PersonFolder => self.show_person_tab(ui, ctx),
                    Tab::MetadataSimilarity => self.show_metadata_similarity_tab(ui, ctx),
                }
            }
        });

        self.show_viewer_window(ctx);
    }
}

impl FaceSearchApp {
    /// Candidates found in the input directory, ranked best-match-first.
    fn show_matches_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.matched_images_cache.is_empty() {
            ui.label("No candidates to show. Click 'Start Processing' to search the input directory.");
            return;
        }

        let selected_count = self.matched_images_cache.iter().filter(|(_, _, s, _)| *s).count();
        let total_pages = self.total_pages();
        let total_matches = self.all_ranked_matches.len();
        let page_start = self.current_page * self.page_size + 1;
        let page_end = ((self.current_page + 1) * self.page_size).min(total_matches);

        ui.label(format!(
            "Page {}/{} — {} total candidates, showing {}-{} ({} selected on this page)",
            self.current_page + 1, total_pages, total_matches, page_start, page_end, selected_count
        ));

        // Pagination controls
        ui.horizontal(|ui| {
            let on_first = self.current_page == 0;
            let on_last = self.current_page + 1 >= total_pages;
            if ui.add_enabled(!on_first, egui::Button::new("< Prev")).clicked() {
                let prev = self.current_page - 1;
                self.load_page(prev);
            }
            if ui.add_enabled(!on_last, egui::Button::new("Next >")).clicked() {
                let next = self.current_page + 1;
                self.load_page(next);
            }
            ui.label(
                egui::RichText::new("Click to select · double-click to open in your photo app · right-click for more")
                    .weak()
                    .size(11.0),
            );
        });

        self.spawn_thumbnail_loader(ctx, ThumbTarget::Matches);

        let mut clicked_idx: Option<usize> = None;
        let mut new_person_trigger: Option<PathBuf> = None;
        let mut trash_trigger: Option<PathBuf> = None;
        let mut view_trigger: Option<PathBuf> = None;
        let mut open_trigger: Option<PathBuf> = None;

        let mut scroll_area = egui::ScrollArea::vertical().id_source("matches_scroll");
        if self.scroll_to_top {
            scroll_area = scroll_area.vertical_scroll_offset(0.0);
            self.scroll_to_top = false;
        }
        scroll_area.show(ui, |ui| {
            let avail = (ui.available_width() - 2.0).max(THUMB_SPACING);
            let aspects = thumb_aspects(
                self.matched_images_cache
                    .iter()
                    .map(|(_, _, _, texture)| texture.as_ref()),
            );
            let rows = pack_thumb_rows(&aspects, self.thumbnail_size, avail);

            ui.spacing_mut().item_spacing = egui::vec2(THUMB_SPACING, THUMB_SPACING);
            for row in &rows {
                ui.horizontal(|ui| {
                    for idx in row.start..row.end {
                        let (img_path, dist, selected, texture) = &self.matched_images_cache[idx];
                        let size = egui::vec2(aspects[idx] * row.height, row.height);
                        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());

                        if ui.is_rect_visible(rect) {
                            paint_thumbnail(ui, rect, texture, img_path);

                            if *selected {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgba_unmultiplied(0, 150, 255, 60),
                                );
                                ui.painter().rect_stroke(
                                    rect,
                                    4.0,
                                    egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 150, 255)),
                                );
                            }

                            // Distance is painted over the image so it costs no layout space.
                            ui.painter().text(
                                egui::pos2(rect.left() + 3.0, rect.bottom() - 14.0),
                                egui::Align2::LEFT_TOP,
                                format!("d={:.3}", dist),
                                egui::FontId::proportional(10.0),
                                egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200),
                            );
                        }

                        let resp = if let Some(Err(err)) = texture {
                            resp.on_hover_text(format!("⚠ Could not preview this file: {}", err))
                        } else {
                            resp
                        };

                        if resp.clicked() {
                            clicked_idx = Some(idx);
                        }

                        resp.context_menu(|ui| {
                            if ui.button("📂 Open in Explorer").clicked() {
                                reveal_in_explorer(img_path);
                                ui.close_menu();
                            }
                            if ui.button("🖼 Open in default app").clicked() {
                                open_trigger = Some(img_path.clone());
                                ui.close_menu();
                            }
                            if ui.button("🔍 Open in built-in viewer").clicked() {
                                view_trigger = Some(img_path.clone());
                                ui.close_menu();
                            }
                            ui.separator();
                            if self.people_dir.is_some() {
                                if ui.button("➕ Create New Person + Add Image").clicked() {
                                    new_person_trigger = Some(img_path.clone());
                                    ui.close_menu();
                                }
                            } else {
                                ui.add_enabled(false, egui::Button::new("➕ Create New Person (set people library first)"));
                            }
                            ui.separator();
                            if ui.button("🗑 Move to Recycle Bin").clicked() {
                                trash_trigger = Some(img_path.clone());
                                ui.close_menu();
                            }
                        });
                    }
                });
            }
        });

        // Process shift-click logic outside the grid
        if let Some(idx) = clicked_idx {
            if ctx.input(|i| i.modifiers.shift) {
                if let Some(last_idx) = self.last_selected_index {
                    let min_idx = std::cmp::min(last_idx, idx);
                    let max_idx = std::cmp::max(last_idx, idx);
                    let current_selection_state = self.matched_images_cache[idx].2;

                    for i in min_idx..=max_idx {
                        self.matched_images_cache[i].2 = !current_selection_state;
                    }
                }
            } else {
                self.matched_images_cache[idx].2 = !self.matched_images_cache[idx].2;
            }
            self.last_selected_index = Some(idx);
        }

        if let Some(path) = open_trigger {
            self.open_in_default_app(&path);
        }

        if let Some(path) = view_trigger {
            self.toggle_viewer(ctx, path);
        }

        // Handle "Create New Person" trigger from context menu
        if let Some(path) = new_person_trigger {
            self.new_person_image_path = Some(path);
            self.new_person_name.clear();
            self.show_new_person_modal = true;
        }

        if let Some(path) = trash_trigger {
            match trash::delete(&path) {
                Ok(()) => {
                    self.all_ranked_matches.retain(|(p, _)| p != &path);
                    self.matched_images_cache.retain(|(p, ..)| p != &path);
                    if self.viewer.as_ref().map(|v| &v.path) == Some(&path) {
                        self.viewer = None;
                    }
                    self.status_msg = format!(
                        "Moved {} to the Recycle Bin.",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    );
                }
                Err(e) => {
                    self.status_msg = format!("Could not move {} to the Recycle Bin: {}", path.display(), e);
                }
            }
        }
    }

    /// Kick off a background scan ranking every photo in the input directory
    /// by EXIF timestamp/camera/color proximity to the confirmed photos
    /// already in this person's folder. See `metadata_similarity::rank_by_metadata`.
    fn start_metasim_scan(&mut self, ctx: &egui::Context) {
        let Some(input_dir) = self.input_dir.clone() else { return; };

        let anchors: Vec<PathBuf> = self.person_files.clone();
        if anchors.is_empty() {
            self.status_msg =
                "No confirmed photos in this person's folder to anchor the scan on.".to_string();
            return;
        }

        let people_dir = self.people_dir.clone();
        self.metasim_scanning = true;
        self.status_msg = "Scanning input directory for similar timing…".to_string();
        let window_secs = (self.metasim_window_minutes.max(1.0) as i64) * 60;
        let tx = self.metasim_tx.clone();
        let ctx = ctx.clone();

        thread::spawn(move || {
            // Photos already sorted into *any* person's folder shouldn't show up
            // as "new" candidates. Mirrors the "already seen" dedup
            // `process_directory` uses for the Matches tab (main.rs): a
            // byte-content comparison against everything already sitting in the
            // people library, not just what the `.origins.json` sidecar happens
            // to have recorded - this also catches photos added to a person
            // folder outside this app, and copies that end up back in view
            // because `target_dir` is nested inside `input_dir`.
            //
            // Built here on the background thread, not the click handler, so a
            // large people library doesn't freeze the UI before scanning even
            // shows as in progress.
            let mut target_filesizes: HashMap<u64, Vec<PathBuf>> = HashMap::new();
            if let Some(people_dir) = &people_dir {
                for entry in WalkDir::new(people_dir).into_iter().filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_file() && (crate::utils::is_image(path) || crate::utils::is_video(path)) {
                        let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                        target_filesizes.entry(size).or_default().push(path.to_path_buf());
                    }
                }
            }

            let walked: Vec<PathBuf> = WalkDir::new(&input_dir)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
                .map(|e| e.path().to_path_buf())
                .filter(|p| crate::utils::is_image(p) || crate::utils::is_video(p))
                .collect();

            // `is_already_sorted` reads whole files on a size collision, so run
            // it in parallel rather than one file at a time.
            let candidates: Vec<PathBuf> = walked
                .into_par_iter()
                .filter(|p| !is_already_sorted(p, &target_filesizes))
                .collect();

            let ranked = crate::metadata_similarity::rank_by_metadata(&anchors, candidates, window_secs);
            let _ = tx.send(MetaSimMessage::Done(ranked));
            ctx.request_repaint();
        });
    }

    /// Candidates ranked by timestamp/camera/GPS/filename-sequence proximity
    /// to photos already confirmed in this person's folder — a fallback for
    /// photos the face pipeline can't see into (covered or turned-away faces).
    fn show_metadata_similarity_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.ensure_person_files_loaded();

        if self.person_files.is_empty() {
            ui.label(
                "Add at least one confirmed photo to this person's folder first (via the Matches tab), then scan here.",
            );
            return;
        }

        ui.horizontal(|ui| {
            ui.label("Time window:");
            ui.add(
                egui::DragValue::new(&mut self.metasim_window_minutes)
                    .range(1.0..=1440.0)
                    .suffix(" min"),
            );
            ui.add_enabled_ui(!self.metasim_scanning && self.input_dir.is_some(), |ui| {
                if ui.button("🔎 Scan for Similar Timing").clicked() {
                    self.start_metasim_scan(ctx);
                }
            });
            if self.metasim_scanning {
                ui.spinner();
                ui.label("Scanning…");
            }
        });
        ui.label(
            egui::RichText::new(
                "Ranks input-directory photos by how close their timestamp, camera, GPS location, and \
                 filename sequence number are to photos already confirmed for this person - useful when \
                 their face is covered or turned away. Hover a photo for the full breakdown.",
            )
            .weak()
            .size(11.0),
        );

        if self.metasim_ranked.is_empty() {
            if !self.metasim_scanning {
                ui.label("No results yet. Click 'Scan for Similar Timing' above.");
            }
            return;
        }

        let total = self.metasim_ranked.len();
        let total_pages = self.metasim_total_pages();
        let page_start = self.metasim_page * self.page_size + 1;
        let page_end = ((self.metasim_page + 1) * self.page_size).min(total);
        let selected_count = self.metasim_images_cache.iter().filter(|(_, s, _)| *s).count();

        ui.separator();
        ui.horizontal(|ui| {
            ui.label(format!(
                "Page {}/{} — {} total candidates, showing {}-{} ({} selected on this page)",
                self.metasim_page + 1, total_pages, total, page_start, page_end, selected_count
            ));
            if selected_count > 0 {
                if ui.button(format!("Copy {} Selected to Person Folder", selected_count)).clicked() {
                    self.show_metasim_copy_confirm = true;
                }
                if let Some(name) = self.other_person_picker_button(ui, "metasim_other_person_popup") {
                    self.copy_metasim_selection_to_person(&name);
                }
            }
        });

        ui.horizontal(|ui| {
            let on_first = self.metasim_page == 0;
            let on_last = self.metasim_page + 1 >= total_pages;
            if ui.add_enabled(!on_first, egui::Button::new("< Prev")).clicked() {
                let prev = self.metasim_page - 1;
                self.load_metasim_page(prev);
            }
            if ui.add_enabled(!on_last, egui::Button::new("Next >")).clicked() {
                let next = self.metasim_page + 1;
                self.load_metasim_page(next);
            }
            ui.label(
                egui::RichText::new("Click to select · double-click to open in your photo app · right-click for more")
                    .weak()
                    .size(11.0),
            );
        });

        self.spawn_thumbnail_loader(ctx, ThumbTarget::MetaSim);

        let mut clicked_idx: Option<usize> = None;
        let mut open_trigger: Option<PathBuf> = None;
        let mut view_trigger: Option<PathBuf> = None;
        let mut new_person_trigger: Option<PathBuf> = None;
        let mut trash_trigger: Option<PathBuf> = None;

        let mut scroll_area = egui::ScrollArea::vertical().id_source("metasim_scroll");
        if self.metasim_scroll_to_top {
            scroll_area = scroll_area.vertical_scroll_offset(0.0);
            self.metasim_scroll_to_top = false;
        }
        scroll_area.show(ui, |ui| {
            let avail = (ui.available_width() - 2.0).max(THUMB_SPACING);
            let aspects = thumb_aspects(
                self.metasim_images_cache
                    .iter()
                    .map(|(_, _, texture)| texture.as_ref()),
            );
            let rows = pack_thumb_rows(&aspects, self.thumbnail_size, avail);

            ui.spacing_mut().item_spacing = egui::vec2(THUMB_SPACING, THUMB_SPACING);
            for row in &rows {
                ui.horizontal(|ui| {
                    for idx in row.start..row.end {
                        let (candidate, selected, texture) = &self.metasim_images_cache[idx];
                        let img_path = &candidate.path;
                        let size = egui::vec2(aspects[idx] * row.height, row.height);
                        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());

                        if ui.is_rect_visible(rect) {
                            paint_thumbnail(ui, rect, texture, img_path);

                            if *selected {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgba_unmultiplied(0, 150, 255, 60),
                                );
                                ui.painter().rect_stroke(
                                    rect,
                                    4.0,
                                    egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 150, 255)),
                                );
                            }

                            let camera_icon = if candidate.same_camera { "📷" } else { "" };
                            ui.painter().text(
                                egui::pos2(rect.left() + 3.0, rect.bottom() - 14.0),
                                egui::Align2::LEFT_TOP,
                                format!(
                                    "{:.0}% · Δ{} {}",
                                    candidate.score.min(1.0) * 100.0,
                                    crate::metadata_similarity::humanize_delta(candidate.delta_secs),
                                    camera_icon,
                                ),
                                egui::FontId::proportional(10.0),
                                egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200),
                            );
                        }

                        let hover = match texture {
                            Some(Err(err)) => format!("⚠ Could not preview this file: {}\n\n{}", err, candidate.explain()),
                            _ => candidate.explain(),
                        };
                        let resp = resp.on_hover_text(hover);

                        if resp.clicked() {
                            clicked_idx = Some(idx);
                        }

                        resp.context_menu(|ui| {
                            if ui.button("📂 Open in Explorer").clicked() {
                                reveal_in_explorer(img_path);
                                ui.close_menu();
                            }
                            if ui.button("🖼 Open in default app").clicked() {
                                open_trigger = Some(img_path.clone());
                                ui.close_menu();
                            }
                            if ui.button("🔍 Open in built-in viewer").clicked() {
                                view_trigger = Some(img_path.clone());
                                ui.close_menu();
                            }
                            ui.separator();
                            if self.people_dir.is_some() {
                                if ui.button("➕ Create New Person + Add Image").clicked() {
                                    new_person_trigger = Some(img_path.clone());
                                    ui.close_menu();
                                }
                            } else {
                                ui.add_enabled(false, egui::Button::new("➕ Create New Person (set people library first)"));
                            }
                            ui.separator();
                            if ui.button("🗑 Move to Recycle Bin").clicked() {
                                trash_trigger = Some(img_path.clone());
                                ui.close_menu();
                            }
                        });
                    }
                });
            }
        });

        // Process shift-click logic outside the grid
        if let Some(idx) = clicked_idx {
            if ctx.input(|i| i.modifiers.shift) {
                if let Some(last_idx) = self.metasim_last_selected_index {
                    let min_idx = std::cmp::min(last_idx, idx);
                    let max_idx = std::cmp::max(last_idx, idx);
                    let current_selection_state = self.metasim_images_cache[idx].1;

                    for i in min_idx..=max_idx {
                        self.metasim_images_cache[i].1 = !current_selection_state;
                    }
                }
            } else {
                self.metasim_images_cache[idx].1 = !self.metasim_images_cache[idx].1;
            }
            self.metasim_last_selected_index = Some(idx);
        }

        if let Some(path) = open_trigger {
            self.open_in_default_app(&path);
        }

        if let Some(path) = view_trigger {
            self.toggle_viewer(ctx, path);
        }

        if let Some(path) = new_person_trigger {
            self.new_person_image_path = Some(path);
            self.new_person_name.clear();
            self.show_new_person_modal = true;
        }

        if let Some(path) = trash_trigger {
            match trash::delete(&path) {
                Ok(()) => {
                    self.metasim_ranked.retain(|c| c.path != path);
                    self.metasim_images_cache.retain(|(c, ..)| c.path != path);
                    if self.viewer.as_ref().map(|v| &v.path) == Some(&path) {
                        self.viewer = None;
                    }
                    self.status_msg = format!(
                        "Moved {} to the Recycle Bin.",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    );
                }
                Err(e) => {
                    self.status_msg = format!("Could not move {} to the Recycle Bin: {}", path.display(), e);
                }
            }
        }
    }

    /// Photos already sitting in the selected person's folder. "Open in
    /// Explorer" here reveals the file where it was copied *from*, not the copy
    /// in the person folder.
    fn show_person_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.ensure_person_files_loaded();

        let Some(person_dir) = self.target_dir.clone() else { return; };
        let total = self.person_files.len();

        ui.label(format!("{} file(s) in {}", total, person_dir.display()));

        if total == 0 {
            ui.label("This person folder is empty. Copy candidates into it from the Matches tab.");
            return;
        }

        // Re-slice the page if the page size changed since it was cached.
        let total_pages = self.person_total_pages();
        let page = self.person_page.min(total_pages - 1);
        let expected = ((page + 1) * self.page_size).min(total) - page * self.page_size;
        if page != self.person_page || self.person_images_cache.len() != expected {
            self.load_person_page(page);
        }

        let page_start = self.person_page * self.page_size + 1;
        let page_end = ((self.person_page + 1) * self.page_size).min(total);
        ui.label(format!(
            "Page {}/{} — showing {}-{} of {}",
            self.person_page + 1, total_pages, page_start, page_end, total
        ));

        ui.horizontal(|ui| {
            let on_first = self.person_page == 0;
            let on_last = self.person_page + 1 >= total_pages;
            if ui.add_enabled(!on_first, egui::Button::new("< Prev")).clicked() {
                let prev = self.person_page - 1;
                self.load_person_page(prev);
            }
            if ui.add_enabled(!on_last, egui::Button::new("Next >")).clicked() {
                let next = self.person_page + 1;
                self.load_person_page(next);
            }
            ui.label(
                egui::RichText::new("Click to open in your photo app · right-click for the built-in viewer, original location or delete")
                    .weak()
                    .size(11.0),
            );
        });

        self.spawn_thumbnail_loader(ctx, ThumbTarget::Person);

        let mut reveal_original: Option<PathBuf> = None;
        let mut reveal_copy: Option<PathBuf> = None;
        let mut trash_trigger: Option<PathBuf> = None;
        let mut view_trigger: Option<PathBuf> = None;
        let mut open_trigger: Option<PathBuf> = None;

        let mut scroll_area = egui::ScrollArea::vertical().id_source("person_folder_scroll");
        if self.person_scroll_to_top {
            scroll_area = scroll_area.vertical_scroll_offset(0.0);
            self.person_scroll_to_top = false;
        }
        scroll_area.show(ui, |ui| {
            let avail = (ui.available_width() - 2.0).max(THUMB_SPACING);
            let aspects = thumb_aspects(
                self.person_images_cache
                    .iter()
                    .map(|(_, texture)| texture.as_ref()),
            );
            let rows = pack_thumb_rows(&aspects, self.thumbnail_size, avail);

            ui.spacing_mut().item_spacing = egui::vec2(THUMB_SPACING, THUMB_SPACING);
            for row in &rows {
                ui.horizontal(|ui| {
                    for idx in row.start..row.end {
                        let (path, texture) = &self.person_images_cache[idx];
                        let size = egui::vec2(aspects[idx] * row.height, row.height);
                        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());

                        if ui.is_rect_visible(rect) {
                            paint_thumbnail(ui, rect, texture, path);
                        }

                        let file_name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        let mut hover = format!(
                            "{}\nRight-click \u{2192} \"Open in Explorer (original location)\" to search the input directory for where this came from.",
                            file_name
                        );
                        if let Some(Err(err)) = texture {
                            hover = format!("⚠ Could not preview this file: {}\n\n{}", err, hover);
                        }

                        resp.on_hover_text(hover).context_menu(|ui| {
                            if ui.button("📂 Open in Explorer (original location)").clicked() {
                                reveal_original = Some(path.clone());
                                ui.close_menu();
                            }
                            if ui.button("📁 Show this copy in the person folder").clicked() {
                                reveal_copy = Some(path.clone());
                                ui.close_menu();
                            }
                            if ui.button("🖼 Open in default app").clicked() {
                                open_trigger = Some(path.clone());
                                ui.close_menu();
                            }
                            if ui.button("🔍 Open in built-in viewer").clicked() {
                                view_trigger = Some(path.clone());
                                ui.close_menu();
                            }
                            ui.separator();
                            if ui.button("🗑 Move this copy to Recycle Bin").clicked() {
                                trash_trigger = Some(path.clone());
                                ui.close_menu();
                            }
                        });
                    }
                });
            }
        });

        if let Some(path) = open_trigger {
            self.open_in_default_app(&path);
        }

        if let Some(path) = view_trigger {
            self.toggle_viewer(ctx, path);
        }

        if let Some(path) = reveal_copy {
            reveal_in_explorer(&path);
            self.status_msg = format!("Opened {} in the person folder.", path.display());
        }
        if let Some(path) = reveal_original {
            self.reveal_original_location(&path);
        }

        if let Some(path) = trash_trigger {
            match trash::delete(&path) {
                Ok(()) => {
                    if self.viewer.as_ref().map(|v| &v.path) == Some(&path) {
                        self.viewer = None;
                    }
                    self.status_msg = format!(
                        "Moved {} to the Recycle Bin.",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    );
                    self.refresh_person_files();
                }
                Err(e) => {
                    self.status_msg = format!("Could not move {} to the Recycle Bin: {}", path.display(), e);
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("frs_{}_{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, contents: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    /// Width a row actually occupies once packed.
    fn row_width(row: &ThumbRow, aspects: &[f32]) -> f32 {
        let images: f32 = aspects[row.start..row.end].iter().sum::<f32>() * row.height;
        images + THUMB_SPACING * (row.end - row.start).saturating_sub(1) as f32
    }

    #[test]
    fn portrait_thumbnails_still_fill_the_width() {
        // 9:19.5 phone screenshots: the old fixed-column grid left more than
        // half the window empty for these.
        let aspects = vec![0.46_f32; 40];
        let avail = 1600.0;
        let rows = pack_thumb_rows(&aspects, 300.0, avail);

        assert!(rows.len() > 1, "40 narrow thumbnails should need several rows");
        for row in &rows[..rows.len() - 1] {
            let width = row_width(row, &aspects);
            assert!(
                (width - avail).abs() < 1.0,
                "row spans {width} of {avail} available"
            );
        }
        // Every thumbnail lands in exactly one row, in order.
        assert_eq!(rows[0].start, 0);
        assert_eq!(rows.last().unwrap().end, aspects.len());
        for pair in rows.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
    }

    #[test]
    fn mixed_orientations_fill_the_width_too() {
        let aspects = vec![0.46, 1.33, 0.75, 1.78, 0.46, 1.0, 0.56, 1.5, 0.46, 0.46];
        let avail = 1200.0;
        let rows = pack_thumb_rows(&aspects, 250.0, avail);

        for row in &rows[..rows.len() - 1] {
            assert!((row_width(row, &aspects) - avail).abs() < 1.0);
        }
        // The last row keeps its natural size instead of being blown up.
        assert!(row_width(rows.last().unwrap(), &aspects) <= avail + 1.0);
    }

    #[test]
    fn a_panorama_wider_than_the_window_is_scaled_down_to_fit() {
        let aspects = vec![8.0_f32];
        let rows = pack_thumb_rows(&aspects, 300.0, 1000.0);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].height * aspects[0] <= 1000.0);
    }

    #[test]
    fn a_short_last_row_keeps_its_natural_height() {
        // Two portrait photos on a wide screen must not be blown up to fill it.
        let aspects = vec![0.46_f32; 2];
        let rows = pack_thumb_rows(&aspects, 300.0, 4000.0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].height, 300.0);
    }

    #[test]
    fn select_argument_quotes_only_the_path() {
        assert_eq!(
            select_argument(Path::new(r"C:\Facial Recognition Sorter\a.jpg")),
            "/select,\"C:\\Facial Recognition Sorter\\a.jpg\""
        );
        // The switch must never end up inside the quotes.
        assert!(select_argument(Path::new(r"C:\a b\c.jpg")).starts_with("/select,\""));
    }

    #[test]
    fn copy_suffix_is_stripped_only_for_short_counters() {
        assert_eq!(strip_copy_suffix("img_1.jpg").as_deref(), Some("img.jpg"));
        assert_eq!(strip_copy_suffix("a_b_12.png").as_deref(), Some("a_b.png"));
        assert_eq!(strip_copy_suffix("noext_3").as_deref(), Some("noext"));
        // Four digits reads as part of the name, not a dedup counter.
        assert_eq!(strip_copy_suffix("img_2024.jpg"), None);
        assert_eq!(strip_copy_suffix("photo.jpg"), None);
        assert_eq!(strip_copy_suffix("_1.jpg"), None);
        assert_eq!(strip_copy_suffix("img_x.jpg"), None);
    }

    #[test]
    fn unique_path_avoids_clobbering_an_existing_file() {
        let dir = scratch("unique");
        write(&dir.join("a.jpg"), "one");
        let first = get_unique_path(&dir, std::ffi::OsStr::new("a.jpg"));
        assert_eq!(first.file_name().unwrap(), "a_1.jpg");

        write(&dir.join("a_1.jpg"), "two");
        let second = get_unique_path(&dir, std::ffi::OsStr::new("a.jpg"));
        assert_eq!(second.file_name().unwrap(), "a_2.jpg");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn find_original_prefers_exact_name_and_matching_size() {
        let root = scratch("find");
        let input = root.join("input");
        let person = root.join("person");

        write(&input.join("2019").join("img.jpg"), "different size entirely");
        write(&input.join("2020").join("img_1.jpg"), "exact bytes");
        write(&person.join("img_1.jpg"), "exact bytes");

        let copy = person.join("img_1.jpg");
        let size = fs::metadata(&copy).unwrap().len();
        assert_eq!(
            find_original(&input, &copy, size).unwrap(),
            input.join("2020").join("img_1.jpg")
        );

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn find_original_falls_back_to_the_de_suffixed_name() {
        let root = scratch("find_alt");
        let input = root.join("input");
        let person = root.join("person");

        write(&input.join("holiday").join("img.jpg"), "the original bytes");
        write(&person.join("img_1.jpg"), "the original bytes");

        let copy = person.join("img_1.jpg");
        let size = fs::metadata(&copy).unwrap().len();
        assert_eq!(
            find_original(&input, &copy, size).unwrap(),
            input.join("holiday").join("img.jpg")
        );

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn find_original_reports_nothing_when_the_input_tree_has_no_candidate() {
        let root = scratch("find_none");
        let input = root.join("input");
        let person = root.join("person");

        write(&input.join("unrelated.jpg"), "bytes");
        write(&person.join("img_1.jpg"), "bytes");

        let copy = person.join("img_1.jpg");
        assert!(find_original(&input, &copy, fs::metadata(&copy).unwrap().len()).is_none());

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn find_original_never_returns_the_copy_itself() {
        let root = scratch("find_self");
        let person = root.join("person");
        write(&person.join("img.jpg"), "bytes");

        let copy = person.join("img.jpg");
        // Person folder nested inside the searched tree: the copy must not be
        // mistaken for its own original.
        assert!(find_original(&root, &copy, fs::metadata(&copy).unwrap().len()).is_none());

        fs::remove_dir_all(&root).unwrap();
    }
}

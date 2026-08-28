mod camera;
mod collision;
mod history;

use camera::Camera;
use eframe::egui;
use glam::{EulerRot, Mat4, Quat, Vec3, Vec4};
use history::UndoHistory;
use image::RgbaImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ==========================================
// 1. СТРУКТУРЫ BLOCKBENCH И НОВЫЙ ФОРМАТ OBB
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawModel {
    #[serde(default)]
    pub credit: Option<String>,
    #[serde(default)]
    pub texture_size: Option<[f32; 2]>,
    #[serde(default)]
    pub textures: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub elements: Vec<RawElement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawRotation {
    pub angle: f32,
    pub axis: String,
    pub origin: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawElement {
    pub from: [f32; 3],
    pub to: [f32; 3],
    #[serde(default)]
    pub rotation: Option<RawRotation>,
    #[serde(default)]
    pub faces: Option<HashMap<String, RawFace>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawFace {
    #[serde(default)]
    pub uv: Option<[f32; 4]>,
    #[serde(default)]
    pub texture: Option<String>,
    #[serde(default)]
    pub rotation: Option<u16>,
}

// Collision metadata v2.  Coordinates are local to the model and rotations are
// always stored as pitch (X), yaw (Y), roll (Z), in degrees.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObbBox {
    pub center: [f32; 3],
    #[serde(rename = "halfSize", alias = "extents")]
    pub half_size: [f32; 3],
    pub rotation: [f32; 3],
}

impl ObbBox {
    pub fn center_v(&self) -> Vec3 {
        Vec3::from_slice(&self.center)
    }
    pub fn half_size_v(&self) -> Vec3 {
        Vec3::from_slice(&self.half_size)
    }

    pub fn rotation_quat(&self) -> Quat {
        Quat::from_euler(
            EulerRot::XYZ,
            self.rotation[0].to_radians(),
            self.rotation[1].to_radians(),
            self.rotation[2].to_radians(),
        )
    }

    pub fn volume(&self) -> f32 {
        let e = self.half_size_v();
        (e.x * 2.0) * (e.y * 2.0) * (e.z * 2.0)
    }

    /// Получить 8 ориентированных углов OBB
    pub fn get_corners(&self) -> [Vec3; 8] {
        let c = self.center_v();
        let e = self.half_size_v();
        let q = self.rotation_quat();

        let local = [
            Vec3::new(-e.x, -e.y, -e.z),
            Vec3::new(e.x, -e.y, -e.z),
            Vec3::new(e.x, -e.y, e.z),
            Vec3::new(-e.x, -e.y, e.z),
            Vec3::new(-e.x, e.y, -e.z),
            Vec3::new(e.x, e.y, -e.z),
            Vec3::new(e.x, e.y, e.z),
            Vec3::new(-e.x, e.y, e.z),
        ];

        let mut out = [Vec3::ZERO; 8];
        for i in 0..8 {
            out[i] = c + q.mul_vec3(local[i]);
        }
        out
    }

    /// Преобразование в осевой габарит AABB для быстрых тестов
    pub fn to_aabb(&self) -> (Vec3, Vec3) {
        let corners = self.get_corners();
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for pt in &corners {
            min = min.min(*pt);
            max = max.max(*pt);
        }
        (min, max)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CollisionData {
    pub version: u32,
    pub boxes: Vec<ObbBox>,
}

/// Reads v2 OBB metadata and legacy AABB files (`min`/`max`).  The legacy data
/// is converted only in memory; every export is the canonical v2 form.
fn parse_collision_data(content: &str) -> Result<CollisionData, String> {
    #[derive(Deserialize)]
    struct Input {
        #[serde(default)]
        version: u32,
        #[serde(default)]
        boxes: Vec<serde_json::Value>,
    }
    let input: Input = serde_json::from_str(content).map_err(|e| e.to_string())?;
    if input.version > 2 {
        return Err(format!("Unsupported collision version: {}", input.version));
    }
    let mut boxes = Vec::with_capacity(input.boxes.len());
    for raw in input.boxes {
        if raw.get("min").is_some() || raw.get("max").is_some() {
            let min: [f32; 3] =
                serde_json::from_value(raw.get("min").cloned().ok_or("Legacy box has no min")?)
                    .map_err(|_| "Legacy min must contain three numbers".to_string())?;
            let max: [f32; 3] =
                serde_json::from_value(raw.get("max").cloned().ok_or("Legacy box has no max")?)
                    .map_err(|_| "Legacy max must contain three numbers".to_string())?;
            let lo = Vec3::from_array(min).min(Vec3::from_array(max));
            let hi = Vec3::from_array(min).max(Vec3::from_array(max));
            boxes.push(ObbBox {
                center: ((lo + hi) * 0.5).to_array(),
                half_size: ((hi - lo) * 0.5).to_array(),
                rotation: [0.0; 3],
            });
        } else {
            boxes.push(serde_json::from_value(raw).map_err(|e| format!("Invalid OBB: {e}"))?);
        }
    }
    Ok(CollisionData { version: 2, boxes })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaData {
    pub uid: String,
    pub name: String,
    pub category: String,
    pub author: String,
    pub description: String,
    pub version: u32,
    pub material: String,
}

// ==========================================
// 2. ОСНОВНОЕ ПРИЛОЖЕНИЕ
// ==========================================

struct CollisionEditorApp {
    raw_model_json: String,
    model: Option<RawModel>,
    meta: MetaData,
    collision: CollisionData,
    texture_path: Option<PathBuf>,
    loaded_texture: Option<RgbaImage>,
    preview_texture: Option<egui::TextureHandle>,
    /// Source images keyed by Blockbench texture slot (e.g. `0`, `metal`).
    texture_sources: HashMap<String, PathBuf>,
    atlas_regions: HashMap<String, [f32; 4]>,
    texture_dimensions: HashMap<String, [f32; 2]>,

    camera: Camera,
    show_model_faces: bool,
    show_model_wire: bool,
    show_collision: bool,
    show_grid: bool,
    selected_box: Option<usize>,
    status_message: String,
    generation_mode: u8,
    /// Number of merge-search passes. Higher values find more combinations;
    /// the cap keeps dense Blockbench models responsive.
    optimization_steps: u8,
    merge_tolerance: f32,
    target_box_count: usize,
    min_detail_size: f32,
    approximation_fill: f32,
    undo_history: UndoHistory<UndoAction>,
    gizmo_drag: Option<(usize, Vec3, [f32; 3])>,
}

#[derive(Clone)]
struct TextureState {
    sources: HashMap<String, PathBuf>,
    texture_path: Option<PathBuf>,
    model_textures: Option<HashMap<String, serde_json::Value>>,
}

enum UndoAction {
    Collision(CollisionData),
    Textures(TextureState),
}

impl Default for CollisionEditorApp {
    fn default() -> Self {
        Self {
            raw_model_json: String::new(),
            model: None,
            meta: MetaData {
                uid: "model".to_string(),
                name: "model".to_string(),
                category: "Decoration".to_string(),
                author: "MDayZ".to_string(),
                description: "Decoration model".to_string(),
                version: 1,
                material: "Wood".to_string(),
            },
            collision: CollisionData {
                version: 2,
                boxes: Vec::new(),
            },
            texture_path: None,
            loaded_texture: None,
            preview_texture: None,
            texture_sources: HashMap::new(),
            atlas_regions: HashMap::new(),
            texture_dimensions: HashMap::new(),
            camera: Camera::default(),
            show_model_faces: true,
            show_model_wire: true,
            show_collision: true,
            show_grid: true,
            selected_box: None,
            status_message: "Перетащите .json модель в окно программы.".to_string(),
            generation_mode: 2,
            optimization_steps: 8,
            merge_tolerance: 0.015,
            target_box_count: 0,
            min_detail_size: 0.0,
            approximation_fill: 0.15,
            undo_history: UndoHistory::new(100),
            gizmo_drag: None,
        }
    }
}

impl CollisionEditorApp {
    fn save_undo_point(&mut self) {
        self.undo_history
            .push(UndoAction::Collision(self.collision.clone()));
    }

    fn undo_collision_edit(&mut self) {
        if let Some(previous) = self.undo_history.pop() {
            match previous {
                UndoAction::Collision(collision) => {
                    self.collision = collision;
                    self.selected_box = None;
                }
                UndoAction::Textures(state) => {
                    self.texture_sources = state.sources;
                    self.texture_path = state.texture_path;
                    if let Some(model) = &mut self.model {
                        model.textures = state.model_textures;
                    }
                    self.rebuild_texture_atlas();
                }
            }
            self.status_message = "Действие отменено.".to_string();
        }
    }

    fn save_texture_undo_point(&mut self) {
        self.undo_history.push(UndoAction::Textures(TextureState {
            sources: self.texture_sources.clone(),
            texture_path: self.texture_path.clone(),
            model_textures: self.model.as_ref().and_then(|model| model.textures.clone()),
        }));
    }

    fn remove_texture_slot(&mut self, slot: &str) {
        if !self.texture_sources.contains_key(slot) {
            return;
        }
        self.save_texture_undo_point();
        self.texture_sources.remove(slot);
        if let Some(textures) = self
            .model
            .as_mut()
            .and_then(|model| model.textures.as_mut())
        {
            textures.remove(slot);
        }
        self.texture_path = self.texture_sources.values().next().cloned();
        self.rebuild_texture_atlas();
        self.status_message = format!("Текстура #{slot} удалена. Ctrl+Z — вернуть.");
    }

    fn convert_bb_coords(v: [f32; 3]) -> Vec3 {
        Vec3::new(
            (v[0] - 8.0_f32) / 16.0_f32,
            v[1] / 16.0_f32,
            (v[2] - 8.0_f32) / 16.0_f32,
        )
    }

    fn clear_texture(&mut self) {
        self.texture_path = None;
        self.loaded_texture = None;
        self.preview_texture = None;
        self.texture_sources.clear();
        self.atlas_regions.clear();
        self.texture_dimensions.clear();
    }

    fn load_texture_from_path(&mut self, path: PathBuf) {
        self.load_texture_files(vec![path]);
    }

    fn single_texture_slots(&self) -> Option<Vec<String>> {
        if let Some(textures) = self
            .model
            .as_ref()
            .and_then(|model| model.textures.as_ref())
        {
            let unique_paths: std::collections::HashSet<_> = textures
                .values()
                .filter_map(|value| value.as_str())
                .collect();
            if unique_paths.len() <= 1 {
                return Some(textures.keys().cloned().collect());
            }
            return None;
        }
        (self.texture_sources.len() <= 1).then(|| {
            if self.texture_sources.is_empty() {
                vec!["0".to_string()]
            } else {
                self.texture_sources.keys().cloned().collect()
            }
        })
    }

    fn replace_single_texture(&mut self, path: PathBuf, slots: Vec<String>) {
        self.texture_sources.clear();
        let slots = if slots.is_empty() {
            vec!["0".to_string()]
        } else {
            slots
        };
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("texture.png")
            .to_string();
        for slot in &slots {
            self.texture_sources.insert(slot.clone(), path.clone());
        }
        if let Some(model) = &mut self.model {
            let textures = model.textures.get_or_insert_with(HashMap::new);
            for slot in slots {
                textures.insert(slot, serde_json::Value::String(file_name.clone()));
            }
        }
        self.texture_path = Some(path);
    }

    fn slot_for_path(&self, path: &Path) -> String {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        self.model
            .as_ref()
            .and_then(|model| model.textures.as_ref())
            .and_then(|textures| {
                textures.iter().find_map(|(slot, value)| {
                    value
                        .as_str()
                        .filter(|reference| {
                            Path::new(reference)
                                .file_name()
                                .and_then(|name| name.to_str())
                                == Some(file_name)
                        })
                        .map(|_| slot.clone())
                })
            })
            .unwrap_or_else(|| self.texture_sources.len().to_string())
    }

    fn rebuild_texture_atlas(&mut self) {
        // Several Minecraft aliases (`all`, `particle`, `0`) often point to
        // exactly the same PNG. Keep one physical copy and let every slot share
        // its region; duplicate copies made UV errors much easier to see.
        let mut unique: Vec<(PathBuf, RgbaImage)> = Vec::new();
        for (_slot, path) in &self.texture_sources {
            if let Ok(image) = image::open(path) {
                if !unique.iter().any(|(known, _)| known == path) {
                    unique.push((path.clone(), image.to_rgba8()));
                }
            }
        }
        if unique.is_empty() {
            self.loaded_texture = None;
            self.preview_texture = None;
            self.atlas_regions.clear();
            self.texture_dimensions.clear();
            return;
        }
        // A single physical image must remain byte-for-byte equivalent to the
        // original renderer. Do not route it through atlas scaling/padding:
        // this is also the common legacy Blockbench case.
        if unique.len() == 1 {
            let (_, image) = unique.pop().unwrap();
            self.loaded_texture = Some(image.clone());
            self.atlas_regions.clear();
            self.texture_dimensions.clear();
            for (slot, source) in &self.texture_sources {
                if image::open(source).is_ok() {
                    self.atlas_regions
                        .insert(slot.clone(), [0.0, 0.0, 1.0, 1.0]);
                    self.texture_dimensions
                        .insert(slot.clone(), [image.width() as f32, image.height() as f32]);
                }
            }
            self.preview_texture = None;
            return;
        }
        unique.sort_by(|a, b| a.0.cmp(&b.0));
        const PADDING: u32 = 2;
        let atlas_width: u32 = unique
            .iter()
            .map(|(_, image)| image.width() + PADDING)
            .sum();
        let atlas_height = unique
            .iter()
            .map(|(_, image)| image.height() + PADDING)
            .max()
            .unwrap_or(1);
        let mut atlas = RgbaImage::new(atlas_width, atlas_height);
        self.atlas_regions.clear();
        let mut x = 0;
        for (path, image) in unique {
            let left = x + PADDING / 2;
            image::imageops::replace(&mut atlas, &image, left.into(), (PADDING / 2).into());
            let region = [
                left as f32 / atlas_width as f32,
                (PADDING / 2) as f32 / atlas_height as f32,
                (left + image.width()) as f32 / atlas_width as f32,
                (PADDING / 2 + image.height()) as f32 / atlas_height as f32,
            ];
            for (slot, source) in &self.texture_sources {
                if source == &path {
                    self.atlas_regions.insert(slot.clone(), region);
                    self.texture_dimensions
                        .insert(slot.clone(), [image.width() as f32, image.height() as f32]);
                }
            }
            x += image.width() + PADDING;
        }
        self.loaded_texture = Some(atlas);
        self.preview_texture = None;
    }

    fn load_texture_files(&mut self, paths: Vec<PathBuf>) {
        // Replacing the skin of an ordinary one-texture model must stay a
        // replacement. The previous behaviour kept inventing slots 1, 2, … .
        if paths.len() == 1 {
            if let Some(slots) = self.single_texture_slots() {
                let path = paths.into_iter().next().unwrap();
                if image::open(&path).is_ok() {
                    self.save_texture_undo_point();
                    self.replace_single_texture(path, slots);
                    self.rebuild_texture_atlas();
                    self.status_message = "Текстура модели заменена.".to_string();
                }
                return;
            }
        }
        if paths.iter().any(|path| image::open(path).is_ok()) {
            self.save_texture_undo_point();
        }
        let mut loaded = 0;
        for path in paths {
            if image::open(&path).is_ok() {
                let slot = self.slot_for_path(&path);
                self.texture_sources.insert(slot.clone(), path.clone());
                // A model imported without `textures` becomes a valid Minecraft
                // multi-texture model on export. Face references such as #0/#1
                // therefore work without manual JSON editing.
                if let Some(model) = &mut self.model {
                    let textures = model.textures.get_or_insert_with(HashMap::new);
                    textures.entry(slot).or_insert_with(|| {
                        serde_json::Value::String(
                            path.file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("texture.png")
                                .to_string(),
                        )
                    });
                }
                self.texture_path.get_or_insert(path);
                loaded += 1;
            }
        }
        self.rebuild_texture_atlas();
        if loaded > 0 {
            self.status_message =
                format!("Загружено текстур в atlas: {}", self.texture_sources.len());
        }
    }

    fn load_model_textures(&mut self, model_path: &Path) {
        let Some(textures) = self
            .model
            .as_ref()
            .and_then(|model| model.textures.as_ref())
        else {
            return;
        };
        let mut resolved = Vec::new();
        for (slot, value) in textures {
            let Some(reference) = value.as_str() else {
                continue;
            };
            let reference = reference.trim_start_matches('#');
            if let Some(png_candidate) =
                Self::resolve_resource_path(model_path, reference, "textures", "png")
            {
                resolved.push((slot.clone(), png_candidate));
            }
        }
        for (slot, path) in resolved {
            self.texture_path.get_or_insert(path.clone());
            self.texture_sources.insert(slot, path);
        }
        if !self.texture_sources.is_empty() {
            self.rebuild_texture_atlas();
            self.status_message =
                format!("Загружено текстур в atlas: {}", self.texture_sources.len());
        }
    }

    /// Resolves both ordinary relative paths and Minecraft resource locations,
    /// e.g. `stalker_decorations:block/ammo`.
    fn resolve_resource_path(
        origin: &Path,
        reference: &str,
        folder: &str,
        extension: &str,
    ) -> Option<PathBuf> {
        let reference = reference.trim_start_matches('#');
        if let Some((namespace, resource_path)) = reference.split_once(':') {
            for ancestor in origin.ancestors() {
                let candidate = ancestor
                    .join("assets")
                    .join(namespace)
                    .join(folder)
                    .join(resource_path)
                    .with_extension(extension);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
            return None;
        }
        let base = origin.parent().unwrap_or_else(|| Path::new("."));
        let candidate = base.join(reference);
        let candidate = if candidate.extension().is_none() {
            candidate.with_extension(extension)
        } else {
            candidate
        };
        candidate.exists().then_some(candidate)
    }

    fn merge_parent_model(
        mut parent: serde_json::Value,
        child: serde_json::Value,
    ) -> serde_json::Value {
        let (Some(parent_object), Some(child_object)) = (parent.as_object_mut(), child.as_object())
        else {
            return child;
        };
        for (key, child_value) in child_object {
            if key == "parent" {
                continue;
            }
            if key == "textures" {
                let parent_textures = parent_object
                    .entry(key.clone())
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                if let (Some(parent_map), Some(child_map)) =
                    (parent_textures.as_object_mut(), child_value.as_object())
                {
                    for (texture_key, texture_value) in child_map {
                        parent_map.insert(texture_key.clone(), texture_value.clone());
                    }
                    continue;
                }
            }
            parent_object.insert(key.clone(), child_value.clone());
        }
        parent
    }

    fn resolve_inherited_model(path: &Path, depth: u8) -> Result<serde_json::Value, String> {
        if depth > 16 {
            return Err("Слишком глубокая цепочка parent-моделей".to_string());
        }
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Не удалось прочитать {}: {e}", path.display()))?;
        let mut child: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Ошибка JSON в {}: {e}", path.display()))?;
        let parent_reference = child
            .get("parent")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let Some(parent_reference) = parent_reference else {
            return Ok(child);
        };
        let parent_path = Self::resolve_resource_path(path, &parent_reference, "models", "json")
            .ok_or_else(|| format!("Parent не найден: {parent_reference}"))?;
        let parent = Self::resolve_inherited_model(&parent_path, depth + 1)?;
        child.as_object_mut().map(|object| object.remove("parent"));
        Ok(Self::merge_parent_model(parent, child))
    }

    fn load_model_from_string(&mut self, content: String, opt_path: Option<PathBuf>) {
        // Очищаем старую текстуру при смене модели
        self.clear_texture();

        let resolved_content = match &opt_path {
            Some(path) => match Self::resolve_inherited_model(path, 0)
                .and_then(|value| serde_json::to_string_pretty(&value).map_err(|e| e.to_string()))
            {
                Ok(value) => value,
                Err(error) => {
                    self.status_message = format!("Ошибка импорта parent: {error}");
                    return;
                }
            },
            None => content,
        };
        match serde_json::from_str::<RawModel>(&resolved_content) {
            Ok(parsed) => {
                self.raw_model_json = resolved_content;
                self.model = Some(parsed);
                if let Some(path) = &opt_path {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        self.meta.uid = stem.to_string();
                        self.meta.name = stem.to_string();
                    }
                    self.load_model_textures(path);
                    // Backward compatibility: one PNG with the model's name.
                    if self.texture_sources.is_empty() {
                        let potential_tex = path.with_extension("png");
                        if potential_tex.exists() {
                            self.load_texture_from_path(potential_tex);
                        }
                    }
                }
                self.regenerate_collision(); // Merge по умолчанию
                self.status_message = "Модель успешно загружена!".to_string();
            }
            Err(e) => {
                self.status_message = format!("Ошибка JSON: {}", e);
            }
        }
    }

    /// Преобразует Blockbench элемент в точный OBB
    fn element_to_obb(elem: &RawElement) -> ObbBox {
        let p_min = Self::convert_bb_coords(elem.from);
        let p_max = Self::convert_bb_coords(elem.to);

        let local_center = (p_min + p_max) * 0.5_f32;
        let half_size = ((p_max - p_min).abs() * 0.5_f32).to_array();

        if let Some(rot) = &elem.rotation {
            let pivot = Self::convert_bb_coords(rot.origin);
            let angle_rad = rot.angle.to_radians();

            let (quat, euler_deg) = match rot.axis.to_lowercase().as_str() {
                "x" => (
                    Quat::from_axis_angle(Vec3::X, angle_rad),
                    [rot.angle, 0.0, 0.0],
                ),
                "y" => (
                    Quat::from_axis_angle(Vec3::Y, angle_rad),
                    [0.0, rot.angle, 0.0],
                ),
                "z" => (
                    Quat::from_axis_angle(Vec3::Z, angle_rad),
                    [0.0, 0.0, rot.angle],
                ),
                _ => (Quat::IDENTITY, [0.0, 0.0, 0.0]),
            };

            let world_center = pivot + quat.mul_vec3(local_center - pivot);

            ObbBox {
                center: world_center.to_array(),
                half_size,
                rotation: euler_deg,
            }
        } else {
            ObbBox {
                center: local_center.to_array(),
                half_size,
                rotation: [0.0, 0.0, 0.0],
            }
        }
    }

    fn get_raw_obbs(&self) -> Vec<ObbBox> {
        let Some(model) = &self.model else {
            return Vec::new();
        };
        model.elements.iter().map(Self::element_to_obb).collect()
    }

    /// Попытка слияния двух OBB без заполнения полостей
    fn try_merge_obbs(a: &ObbBox, b: &ObbBox, eps: f32) -> Option<ObbBox> {
        // Merge in the local axes of A. Comparing world X/Y/Z corrupts boxes
        // that are yawed or tilted, even when their rotations are identical.
        let q = a.rotation_quat();
        if q.dot(b.rotation_quat()).abs() < 0.999_999 {
            return None;
        }

        let ac = Vec3::ZERO;
        let bc = q.inverse().mul_vec3(b.center_v() - a.center_v());
        let ae = a.half_size_v();
        let be = b.half_size_v();

        // Проверяем слияние вдоль осей X, Y, Z
        let same_x = (ac.x - bc.x).abs() < eps && (ae.x - be.x).abs() < eps;
        let same_y = (ac.y - bc.y).abs() < eps && (ae.y - be.y).abs() < eps;
        let same_z = (ac.z - bc.z).abs() < eps && (ae.z - be.z).abs() < eps;

        let mut candidate = None;

        // 1. Слияние по оси Y (стойки)
        if same_x && same_z {
            let touch_y = ((ac.y + ae.y) - (bc.y - be.y)).abs() < eps
                || ((bc.y + be.y) - (ac.y - ae.y)).abs() < eps;
            if touch_y {
                let min_y = (ac.y - ae.y).min(bc.y - be.y);
                let max_y = (ac.y + ae.y).max(bc.y + be.y);
                candidate = Some(ObbBox {
                    center: (a.center_v()
                        + q.mul_vec3(Vec3::new(ac.x, (min_y + max_y) * 0.5_f32, ac.z)))
                    .to_array(),
                    half_size: [ae.x, (max_y - min_y) * 0.5_f32, ae.z],
                    rotation: a.rotation,
                });
            }
        }

        // 2. Слияние по оси X (горизонтальные рейки)
        if candidate.is_none() && same_y && same_z {
            let touch_x = ((ac.x + ae.x) - (bc.x - be.x)).abs() < eps
                || ((bc.x + be.x) - (ac.x - ae.x)).abs() < eps;
            if touch_x {
                let min_x = (ac.x - ae.x).min(bc.x - be.x);
                let max_x = (ac.x + ae.x).max(bc.x + be.x);
                candidate = Some(ObbBox {
                    center: (a.center_v()
                        + q.mul_vec3(Vec3::new((min_x + max_x) * 0.5_f32, ac.y, ac.z)))
                    .to_array(),
                    half_size: [(max_x - min_x) * 0.5_f32, ae.y, ae.z],
                    rotation: a.rotation,
                });
            }
        }

        // 3. Слияние по оси Z
        if candidate.is_none() && same_x && same_y {
            let touch_z = ((ac.z + ae.z) - (bc.z - be.z)).abs() < eps
                || ((bc.z + be.z) - (ac.z - ae.z)).abs() < eps;
            if touch_z {
                let min_z = (ac.z - ae.z).min(bc.z - be.z);
                let max_z = (ac.z + ae.z).max(bc.z + be.z);
                candidate = Some(ObbBox {
                    center: (a.center_v()
                        + q.mul_vec3(Vec3::new(ac.x, ac.y, (min_z + max_z) * 0.5_f32)))
                    .to_array(),
                    half_size: [ae.x, ae.y, (max_z - min_z) * 0.5_f32],
                    rotation: a.rotation,
                });
            }
        }

        // Защита от заполнения пустот: объем объединённого бокса не должен превышать сумму объемов
        if let Some(cand) = candidate {
            let v_sum = a.volume() + b.volume();
            let v_cand = cand.volume();
            if (v_cand - v_sum).abs() <= v_sum * 0.05_f32 + 1e-5_f32 {
                return Some(cand);
            }
        }

        None
    }

    /// Deliberate lossy merge for gameplay collisions. Unlike `try_merge_obbs`,
    /// it may round a curved silhouette into a coarser AABB, but only when the
    /// caller explicitly permits the resulting empty-space percentage.
    fn try_approximate_merge(&self, a: &ObbBox, b: &ObbBox) -> Option<(ObbBox, f32)> {
        let (amin, amax) = a.to_aabb();
        let (bmin, bmax) = b.to_aabb();
        let gap = (amin - bmax).max(bmin - amax).max(Vec3::ZERO).max_element();
        if gap > self.merge_tolerance {
            return None;
        }
        let min = amin.min(bmin);
        let max = amax.max(bmax);
        let candidate = ObbBox {
            center: ((min + max) * 0.5).to_array(),
            half_size: ((max - min) * 0.5).to_array(),
            rotation: [0.0; 3],
        };
        let source_volume = a.volume() + b.volume();
        let fill = ((candidate.volume() - source_volume) / source_volume.max(0.000_001)).max(0.0);
        (fill <= self.approximation_fill).then_some((candidate, fill))
    }

    fn simplify_collision(&self, raw_obbs: Vec<ObbBox>) -> Vec<ObbBox> {
        let mut current: Vec<ObbBox> = raw_obbs
            .into_iter()
            .filter(|b| b.half_size_v().max_element() * 2.0 >= self.min_detail_size)
            .collect();
        if self.target_box_count == 0 {
            return current;
        }
        for _ in 0..self.optimization_steps {
            if current.len() <= self.target_box_count {
                break;
            }
            let mut best: Option<(usize, usize, ObbBox, f32)> = None;
            for i in 0..current.len() {
                for j in (i + 1)..current.len() {
                    if let Some((merged, fill)) =
                        self.try_approximate_merge(&current[i], &current[j])
                    {
                        if best.as_ref().map_or(true, |candidate| fill < candidate.3) {
                            best = Some((i, j, merged, fill));
                        }
                    }
                }
            }
            let Some((i, j, merged, _)) = best else { break };
            current[i] = merged;
            current.remove(j);
        }
        current
    }

    fn regenerate_collision(&mut self) {
        let raw_obbs = self.get_raw_obbs();
        if raw_obbs.is_empty() {
            return;
        }

        self.collision.boxes.clear();

        match self.generation_mode {
            // Low (1 общий монолитный AABB)
            0 => {
                let mut min = Vec3::splat(f32::MAX);
                let mut max = Vec3::splat(f32::MIN);
                for obb in &raw_obbs {
                    let (bmin, bmax) = obb.to_aabb();
                    min = min.min(bmin);
                    max = max.max(bmax);
                }
                let center = (min + max) * 0.5_f32;
                let half_size = (max - min) * 0.5_f32;
                self.collision.boxes.push(ObbBox {
                    center: center.to_array(),
                    half_size: half_size.to_array(),
                    rotation: [0.0, 0.0, 0.0],
                });
            }

            // Medium (Кластеризация крыльев/сегментов с сохранением OBB наклонов)
            1 => {
                let mut unrotated = Vec::new();
                let mut rotated = Vec::new();

                for obb in raw_obbs {
                    let rot_len = Vec3::from_slice(&obb.rotation).length();
                    if rot_len > 0.01_f32 {
                        rotated.push(obb);
                    } else {
                        unrotated.push(obb);
                    }
                }

                // Кластеризация прямых сегментов
                let mut groups: Vec<Vec<ObbBox>> = Vec::new();
                for b in unrotated {
                    let mut placed = false;
                    for grp in &mut groups {
                        if let Some(first) = grp.first() {
                            let (amin, amax) = b.to_aabb();
                            let (bmin, bmax) = first.to_aabb();
                            let gap = self.merge_tolerance;
                            let touch = (amin.x <= bmax.x + gap && amax.x >= bmin.x - gap)
                                && (amin.y <= bmax.y + gap && amax.y >= bmin.y - gap)
                                && (amin.z <= bmax.z + gap && amax.z >= bmin.z - gap);
                            if touch {
                                grp.push(b.clone());
                                placed = true;
                                break;
                            }
                        }
                    }
                    if !placed {
                        groups.push(vec![b]);
                    }
                }

                for grp in groups {
                    let mut min = Vec3::splat(f32::MAX);
                    let mut max = Vec3::splat(f32::MIN);
                    for b in grp {
                        let (bmin, bmax) = b.to_aabb();
                        min = min.min(bmin);
                        max = max.max(bmax);
                    }
                    self.collision.boxes.push(ObbBox {
                        center: ((min + max) * 0.5_f32).to_array(),
                        half_size: ((max - min) * 0.5_f32).to_array(),
                        rotation: [0.0, 0.0, 0.0],
                    });
                }

                // Повёрнутые детали сохраняются в идеальном OBB
                self.collision.boxes.extend(rotated);
            }

            // Merge (Умное слияние без объединения пустот)
            2 => {
                let mut current = raw_obbs;
                let mut merged_any = true;

                let mut passes = 0;
                while merged_any
                    && passes < self.optimization_steps
                    && (self.target_box_count == 0 || current.len() > self.target_box_count)
                {
                    merged_any = false;
                    passes += 1;
                    let mut next = Vec::new();
                    let mut used = vec![false; current.len()];

                    for i in 0..current.len() {
                        if used[i] {
                            continue;
                        }
                        let mut merged_box = current[i].clone();

                        for j in (i + 1)..current.len() {
                            if used[j] {
                                continue;
                            }
                            if let Some(res) =
                                Self::try_merge_obbs(&merged_box, &current[j], self.merge_tolerance)
                            {
                                merged_box = res;
                                used[j] = true;
                                merged_any = true;
                            }
                        }
                        used[i] = true;
                        next.push(merged_box);
                    }
                    current = next;
                }
                // A box budget stops the search once the requested density is
                // reached. If it cannot be reached safely, preserve accuracy and
                // report the actual count instead of filling a hollow volume.
                self.collision.boxes = current;
            }

            // Gameplay simplification: filter tiny decorative details and merge
            // neighbouring boxes with a user-controlled error budget.
            3 => self.collision.boxes = self.simplify_collision(raw_obbs),

            // Exact (1-в-1 OBB)
            _ => {
                self.collision.boxes = raw_obbs;
            }
        }

        let budget_note =
            if self.target_box_count > 0 && self.collision.boxes.len() > self.target_box_count {
                format!(
                    " (безопасный предел {}, цель {})",
                    self.collision.boxes.len(),
                    self.target_box_count
                )
            } else {
                String::new()
            };
        self.status_message = format!(
            "Сгенерировано OBB боксов: {}{}",
            self.collision.boxes.len(),
            budget_note
        );
    }

    fn face_uv(&self, elem: &RawElement, face_name: &str) -> Option<[egui::Pos2; 4]> {
        let faces = elem.faces.as_ref()?;
        let face = faces.get(face_name)?;
        let uv = face.uv.unwrap_or([0.0, 0.0, 16.0, 16.0]);
        let requested_slot = face
            .texture
            .as_deref()
            .unwrap_or("0")
            .trim_start_matches('#');
        // Blockbench's omitted `texture_size` is explicitly 16×16 logical
        // pixels, even when the PNG itself is a 128/256px HD image. Using the
        // physical PNG dimensions here samples a tiny corner and stretches it
        // over the face (the characteristic broken preview).
        let texture_size = self.model.as_ref()?.texture_size.unwrap_or([16.0, 16.0]);
        if texture_size[0] <= 0.0 || texture_size[1] <= 0.0 {
            return None;
        }
        let u0 = (uv[0] / texture_size[0]).clamp(0.0, 1.0);
        let v0 = (uv[1] / texture_size[1]).clamp(0.0, 1.0);
        let u1 = (uv[2] / texture_size[0]).clamp(0.0, 1.0);
        let v1 = (uv[3] / texture_size[1]).clamp(0.0, 1.0);
        let mut corners = [
            egui::pos2(u0, v0),
            egui::pos2(u1, v0),
            egui::pos2(u1, v1),
            egui::pos2(u0, v1),
        ];
        // Blockbench stores quarter-turn face rotations in texture space.
        let turns = (face.rotation.unwrap_or(0) / 90) % 4;
        corners.rotate_right(turns as usize);
        // A face texture (`#0`, `#metal`) occupies just one region of the
        // combined atlas. Untextured / legacy faces use the first region.
        let region = if face.texture.is_some() {
            self.atlas_regions.get(requested_slot)?
        } else {
            self.atlas_regions.values().next()?
        };
        for corner in &mut corners {
            corner.x = region[0] + corner.x * (region[2] - region[0]);
            corner.y = region[1] + corner.y * (region[3] - region[1]);
        }
        Some(corners)
    }

    fn sync_preview_texture(&mut self, ctx: &egui::Context) {
        if self.preview_texture.is_none() {
            if let Some(image) = &self.loaded_texture {
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width() as usize, image.height() as usize],
                    image.as_raw(),
                );
                self.preview_texture = Some(ctx.load_texture(
                    "model_texture",
                    color_image,
                    egui::TextureOptions::NEAREST,
                ));
            }
        }
    }

    fn pick_obb(&self, view_proj: Mat4, rect: egui::Rect, pointer: egui::Pos2) -> Option<usize> {
        let ndc_x = ((pointer.x - rect.min.x) / rect.width()) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((pointer.y - rect.min.y) / rect.height()) * 2.0;
        let inverse = view_proj.inverse();
        let near = inverse * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inverse * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let origin = near.truncate() / near.w;
        let direction = (far.truncate() / far.w - origin).normalize_or_zero();
        self.collision
            .boxes
            .iter()
            .enumerate()
            .filter_map(|(index, box_)| {
                collision::ray_hit_distance(box_, origin, direction).map(|d| (index, d))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
    }

    fn copy_texture_safely(source: &Path, destination: &Path) -> Result<(), String> {
        // `fs::copy(a, a)` truncates the source on some platforms.  A temporary
        // sibling also means an interrupted export never leaves a half PNG.
        if source == destination
            || (source.exists()
                && destination.exists()
                && source.canonicalize().ok() == destination.canonicalize().ok())
        {
            return Ok(());
        }
        let temporary = destination.with_extension("png.swrmc-tmp");
        fs::copy(source, &temporary)
            .map_err(|e| format!("Не удалось скопировать текстуру: {e}"))?;
        if let Err(e) = fs::rename(&temporary, destination) {
            // Windows cannot replace an existing file with rename.
            let _ = fs::remove_file(destination);
            fs::rename(&temporary, destination).map_err(|rename_error| {
                format!("Не удалось сохранить текстуру: {e}; {rename_error}")
            })?;
        }
        Ok(())
    }

    fn png_file_name(path: &Path) -> String {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("texture");
        if Path::new(name).extension().is_some() {
            name.to_string()
        } else {
            format!("{name}.png")
        }
    }

    fn exported_model_json(&self) -> Result<Vec<u8>, String> {
        let mut json: serde_json::Value = serde_json::from_str(&self.raw_model_json)
            .map_err(|e| format!("Некорректный model.json: {e}"))?;
        let object = json
            .as_object_mut()
            .ok_or_else(|| "Корень model.json должен быть объектом".to_string())?;
        match self
            .model
            .as_ref()
            .and_then(|model| model.textures.as_ref())
        {
            Some(model_textures) => {
                object.insert(
                    "textures".to_string(),
                    serde_json::to_value(model_textures).map_err(|e| e.to_string())?,
                );
            }
            None => {
                object.remove("textures");
            }
        }
        if !self.texture_sources.is_empty() {
            let textures = object
                .entry("textures")
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
                .as_object_mut()
                .ok_or_else(|| "Поле textures должно быть объектом".to_string())?;
            for (slot, source) in &self.texture_sources {
                // A standalone exported package must not retain resource
                // locations pointing at the original resource pack.
                textures.insert(
                    slot.clone(),
                    serde_json::Value::String(Self::png_file_name(source)),
                );
            }
        }
        serde_json::to_vec_pretty(&json).map_err(|e| e.to_string())
    }

    fn export_package(&mut self) {
        if let Some(target_dir) = rfd::FileDialog::new().pick_folder() {
            let pkg_dir = target_dir.join(&self.meta.uid);
            if let Err(e) = fs::create_dir_all(&pkg_dir) {
                rfd::MessageDialog::new()
                    .set_title("Ошибка")
                    .set_description(&format!("Не удалось создать директорию: {}", e))
                    .show();
                return;
            }

            if self.raw_model_json.trim().is_empty() {
                self.status_message = "Экспорт отменён: сначала загрузите модель.".to_string();
                return;
            }
            let model_json = match self.exported_model_json() {
                Ok(json) => json,
                Err(error) => {
                    self.status_message = error;
                    return;
                }
            };
            if let Err(e) = fs::File::create(pkg_dir.join("meta.json"))
                .and_then(|file| {
                    serde_json::to_writer_pretty(file, &self.meta).map_err(std::io::Error::other)
                })
                .and_then(|_| {
                    fs::write(
                        pkg_dir.join("collision.json"),
                        serde_json::to_vec_pretty(&self.collision).unwrap(),
                    )
                })
                .and_then(|_| fs::write(pkg_dir.join("model.json"), &model_json))
            {
                self.status_message = format!("Ошибка экспорта: {e}");
                return;
            }
            if let Some(tex) = &self.texture_path {
                if self.texture_sources.len() <= 1 {
                    let export_name = self
                        .texture_sources
                        .keys()
                        .next()
                        .and_then(|slot| {
                            self.model
                                .as_ref()
                                .and_then(|model| model.textures.as_ref())
                                .and_then(|textures| textures.get(slot))
                        })
                        .and_then(|value| value.as_str())
                        .map(Path::new)
                        .map(Self::png_file_name)
                        .unwrap_or_else(|| "texture.png".to_string());
                    if let Err(e) = Self::copy_texture_safely(tex, &pkg_dir.join(export_name)) {
                        self.status_message = e;
                        return;
                    }
                } else {
                    for (slot, source) in &self.texture_sources {
                        let export_name = self
                            .model
                            .as_ref()
                            .and_then(|model| model.textures.as_ref())
                            .and_then(|textures| textures.get(slot))
                            .and_then(|value| value.as_str())
                            .map(Path::new)
                            .map(Self::png_file_name)
                            .unwrap_or_else(|| Self::png_file_name(source));
                        if let Err(e) =
                            Self::copy_texture_safely(source, &pkg_dir.join(export_name))
                        {
                            self.status_message = e;
                            return;
                        }
                    }
                }
            }

            self.status_message = format!("Пакет готов: {}", pkg_dir.display());

            rfd::MessageDialog::new()
                .set_title("Успех")
                .set_description(&format!("Пакет сохранён в:\n{:?}", pkg_dir))
                .show();
        }
    }
}

// ==========================================
// 4. 3D VIEWPORT
// ==========================================

struct Triangle3D {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    color: egui::Color32,
    avg_depth: f32,
}

struct TexturedQuad3D {
    positions: [Vec3; 4],
    uv: [egui::Pos2; 4],
    avg_depth: f32,
}

impl CollisionEditorApp {
    fn render_viewport(&mut self, ui: &mut egui::Ui) {
        self.sync_preview_texture(ui.ctx());
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        // Камера
        if response.dragged_by(egui::PointerButton::Primary) && self.gizmo_drag.is_none() {
            let delta = response.drag_delta();
            self.camera.yaw += delta.x * 0.01_f32;
            self.camera.pitch = (self.camera.pitch - delta.y * 0.01_f32).clamp(-1.55_f32, 1.55_f32);
        }
        if response.dragged_by(egui::PointerButton::Middle)
            || (response.dragged_by(egui::PointerButton::Primary)
                && self.gizmo_drag.is_none()
                && ui.input(|i| i.modifiers.shift))
        {
            let delta = response.drag_delta();
            let right = Vec3::new(-self.camera.yaw.sin(), 0.0_f32, self.camera.yaw.cos());
            self.camera.target -= right * delta.x * 0.005_f32;
            self.camera.target += Vec3::Y * delta.y * 0.005_f32;
        }

        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0_f32 {
            self.camera.distance =
                (self.camera.distance - scroll * 0.005_f32).clamp(0.3_f32, 30.0_f32);
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0_f32, egui::Color32::from_rgb(22, 24, 29));

        let aspect = rect.width() / rect.height();
        let view = self.camera.view_matrix();
        let proj = self.camera.proj_matrix(aspect);
        let view_proj = proj * view;

        let project = |pos: Vec3| -> Option<(egui::Pos2, f32)> {
            let clip = view_proj * Vec4::new(pos.x, pos.y, pos.z, 1.0_f32);
            if clip.w <= 0.05_f32 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            if ndc.x < -1.8_f32 || ndc.x > 1.8_f32 || ndc.y < -1.8_f32 || ndc.y > 1.8_f32 {
                return None;
            }
            Some((
                egui::pos2(
                    rect.min.x + (ndc.x + 1.0_f32) * 0.5_f32 * rect.width(),
                    rect.min.y + (1.0_f32 - ndc.y) * 0.5_f32 * rect.height(),
                ),
                clip.w,
            ))
        };

        // Viewport picking and a compact translate gizmo for the selected OBB.
        // The gizmo takes priority over camera rotation and stores one undo point
        // per drag, matching editors users already know.
        let mut gizmo_pressed = false;
        if let (Some(index), Some(pointer)) = (self.selected_box, response.interact_pointer_pos()) {
            if let Some(center) = self.collision.boxes.get(index).map(|box_| box_.center_v()) {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    if let (Some((from, _)), Some((to, _))) =
                        (project(center), project(center + axis * 0.35))
                    {
                        if pointer.distance(to) < 13.0 && ui.input(|i| i.pointer.primary_pressed())
                        {
                            self.save_undo_point();
                            self.gizmo_drag = Some((index, axis, center.to_array()));
                            gizmo_pressed = true;
                        }
                        // Keep the compiler aware `from` is intentional: it is
                        // the start of the visible axis and used during dragging.
                        let _ = from;
                    }
                }
            }
        }
        if let Some((index, axis, start_center)) = self.gizmo_drag {
            if !ui.input(|i| i.pointer.primary_down()) {
                self.gizmo_drag = None;
            } else if let Some(box_) = self.collision.boxes.get_mut(index) {
                if let (Some((from, _)), Some((to, _))) = (
                    project(Vec3::from_array(start_center)),
                    project(Vec3::from_array(start_center) + axis * 0.35),
                ) {
                    let screen_axis = (to - from).normalized();
                    let world_delta =
                        response.drag_delta().dot(screen_axis) * self.camera.distance / 260.0;
                    box_.center = (Vec3::from_array(start_center) + axis * world_delta).to_array();
                }
            }
        } else if response.clicked_by(egui::PointerButton::Primary) && !gizmo_pressed {
            if let Some(pointer) = response.interact_pointer_pos() {
                self.selected_box = self.pick_obb(view_proj, rect, pointer);
            }
        }

        // 1. Сетка пола
        if self.show_grid {
            let grid_size = 8;
            let step = 0.25_f32;
            for i in -grid_size..=grid_size {
                let fi = i as f32 * step;
                let c = if i == 0 {
                    egui::Color32::from_rgb(80, 85, 95)
                } else {
                    egui::Color32::from_rgb(35, 38, 45)
                };

                if let (Some((p1, _)), Some((p2, _))) = (
                    project(Vec3::new(fi, 0.0_f32, -(grid_size as f32) * step)),
                    project(Vec3::new(fi, 0.0_f32, (grid_size as f32) * step)),
                ) {
                    painter.line_segment([p1, p2], egui::Stroke::new(1.0_f32, c));
                }

                if let (Some((p1, _)), Some((p2, _))) = (
                    project(Vec3::new(-(grid_size as f32) * step, 0.0_f32, fi)),
                    project(Vec3::new((grid_size as f32) * step, 0.0_f32, fi)),
                ) {
                    painter.line_segment([p1, p2], egui::Stroke::new(1.0_f32, c));
                }
            }
        }

        // 2. Model faces. `egui` is a 2D painter, so we explicitly cull back
        // faces and tessellate textured quads for stable perspective UVs.
        let light_dir = Vec3::new(0.5_f32, 0.8_f32, 0.4_f32).normalize();
        let camera_eye = self.camera.target
            + Vec3::new(
                self.camera.distance * self.camera.yaw.cos() * self.camera.pitch.cos(),
                self.camera.distance * self.camera.pitch.sin(),
                self.camera.distance * self.camera.yaw.sin() * self.camera.pitch.cos(),
            );
        let mut triangles: Vec<Triangle3D> = Vec::new();
        let mut textured_quads: Vec<TexturedQuad3D> = Vec::new();

        if self.show_model_faces {
            if let Some(model) = &self.model {
                for elem in &model.elements {
                    let obb = Self::element_to_obb(elem);
                    let c = obb.get_corners();

                    let quads = [
                        (4, 5, 6, 7, "up"),
                        (3, 2, 1, 0, "down"),
                        (7, 6, 2, 3, "south"),
                        (5, 4, 0, 1, "north"),
                        (6, 5, 1, 2, "east"),
                        (4, 7, 3, 0, "west"),
                    ];

                    for (i0, i1, i2, i3, face_name) in quads {
                        // Minecraft elements may intentionally omit faces (thin
                        // cut-outs, open frames, transparent constructions).
                        // Drawing all six cube sides created the large bogus
                        // planes visible in the broken preview.
                        if !elem
                            .faces
                            .as_ref()
                            .is_some_and(|faces| faces.contains_key(face_name))
                        {
                            continue;
                        }
                        let p0 = c[i0];
                        let p1 = c[i1];
                        let p2 = c[i2];
                        let p3 = c[i3];

                        let norm = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                        // Corner order produces an inward normal; its negative is
                        // the Minecraft face's outward normal.
                        let outward = -norm;
                        let face_center = (p0 + p1 + p2 + p3) * 0.25;
                        if outward.dot(camera_eye - face_center) <= 0.000_001 {
                            continue;
                        }
                        let dot = norm.dot(light_dir).abs();
                        let intensity = (0.45_f32 + 0.55_f32 * dot).clamp(0.0_f32, 1.0_f32);

                        let face_color = egui::Color32::from_rgb(
                            (160.0_f32 * intensity) as u8,
                            (140.0_f32 * intensity) as u8,
                            (110.0_f32 * intensity) as u8,
                        );

                        let d1 = (view * Vec4::new(p0.x, p0.y, p0.z, 1.0_f32)).z;
                        let d2 = (view * Vec4::new(p2.x, p2.y, p2.z, 1.0_f32)).z;

                        if self.preview_texture.is_some() {
                            if let Some(uv) = self.face_uv(elem, face_name) {
                                const SUBDIVISIONS: usize = 4;
                                let bilerp3 =
                                    |u: f32, v: f32| p0.lerp(p1, u).lerp(p3.lerp(p2, u), v);
                                let bilerp2 = |u: f32, v: f32| {
                                    uv[0].lerp(uv[1], u).lerp(uv[3].lerp(uv[2], u), v)
                                };
                                for y in 0..SUBDIVISIONS {
                                    for x in 0..SUBDIVISIONS {
                                        let u0 = x as f32 / SUBDIVISIONS as f32;
                                        let u1 = (x + 1) as f32 / SUBDIVISIONS as f32;
                                        let v0 = y as f32 / SUBDIVISIONS as f32;
                                        let v1 = (y + 1) as f32 / SUBDIVISIONS as f32;
                                        let positions = [
                                            bilerp3(u0, v0),
                                            bilerp3(u1, v0),
                                            bilerp3(u1, v1),
                                            bilerp3(u0, v1),
                                        ];
                                        let cell_uv = [
                                            bilerp2(u0, v0),
                                            bilerp2(u1, v0),
                                            bilerp2(u1, v1),
                                            bilerp2(u0, v1),
                                        ];
                                        let avg_depth = positions
                                            .iter()
                                            .map(|point| {
                                                (view * Vec4::new(point.x, point.y, point.z, 1.0)).z
                                            })
                                            .sum::<f32>()
                                            * 0.25;
                                        textured_quads.push(TexturedQuad3D {
                                            positions,
                                            uv: cell_uv,
                                            avg_depth,
                                        });
                                    }
                                }
                                continue;
                            }
                        }
                        triangles.push(Triangle3D {
                            p0,
                            p1,
                            p2,
                            color: face_color,
                            avg_depth: (d1 + d2) * 0.5_f32,
                        });
                        triangles.push(Triangle3D {
                            p0,
                            p1: p2,
                            p2: p3,
                            color: face_color,
                            avg_depth: (d1 + d2) * 0.5_f32,
                        });
                    }
                }
            }
        }

        triangles.sort_by(|a, b| {
            a.avg_depth
                .partial_cmp(&b.avg_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for tri in triangles {
            if let (Some((s0, _)), Some((s1, _)), Some((s2, _))) =
                (project(tri.p0), project(tri.p1), project(tri.p2))
            {
                painter.add(egui::Shape::convex_polygon(
                    vec![s0, s1, s2],
                    tri.color,
                    egui::Stroke::NONE,
                ));
            }
        }

        // egui meshes retain the face UVs instead of reducing every face to one
        // sampled colour.  This keeps texture atlases aligned with Blockbench UV.
        if let Some(texture) = &self.preview_texture {
            textured_quads.sort_by(|a, b| {
                a.avg_depth
                    .partial_cmp(&b.avg_depth)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for quad in textured_quads {
                let projected: Option<Vec<egui::Pos2>> = quad
                    .positions
                    .iter()
                    .map(|&p| project(p).map(|(s, _)| s))
                    .collect();
                if let Some(points) = projected {
                    let mut mesh = egui::epaint::Mesh::with_texture(texture.id());
                    for (position, uv) in points.into_iter().zip(quad.uv) {
                        mesh.vertices.push(egui::epaint::Vertex {
                            pos: position,
                            uv,
                            color: egui::Color32::WHITE,
                        });
                    }
                    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
                    painter.add(egui::Shape::mesh(mesh));
                }
            }
        }

        let draw_box_wireframe =
            |painter: &egui::Painter, corners: &[Vec3; 8], stroke: egui::Stroke| {
                let edges = [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                ];

                for (start, end) in edges {
                    if let (Some((p1, _)), Some((p2, _))) =
                        (project(corners[start]), project(corners[end]))
                    {
                        painter.line_segment([p1, p2], stroke);
                    }
                }
            };

        // 3. Тонкий контур модели
        if self.show_model_wire {
            if let Some(model) = &self.model {
                for elem in &model.elements {
                    let obb = Self::element_to_obb(elem);
                    let corners = obb.get_corners();
                    draw_box_wireframe(
                        &painter,
                        &corners,
                        egui::Stroke::new(
                            1.0_f32,
                            egui::Color32::from_rgba_unmultiplied(40, 50, 60, 180),
                        ),
                    );
                }
            }
        }

        // 4. Отрисовка OBB коллизий
        if self.show_collision {
            for (idx, b) in self.collision.boxes.iter().enumerate() {
                let is_selected = self.selected_box == Some(idx);
                let color = if is_selected {
                    egui::Color32::from_rgb(255, 220, 0)
                } else {
                    egui::Color32::from_rgb(0, 255, 120)
                };

                let corners = b.get_corners();
                draw_box_wireframe(
                    &painter,
                    &corners,
                    egui::Stroke::new(if is_selected { 2.5_f32 } else { 1.5_f32 }, color),
                );
            }
        }

        if let Some(index) = self.selected_box {
            if let Some(box_) = self.collision.boxes.get(index) {
                let center = box_.center_v();
                for (axis, color, label) in [
                    (Vec3::X, egui::Color32::from_rgb(230, 84, 84), "X"),
                    (Vec3::Y, egui::Color32::from_rgb(92, 210, 130), "Y"),
                    (Vec3::Z, egui::Color32::from_rgb(85, 145, 235), "Z"),
                ] {
                    if let (Some((from, _)), Some((to, _))) =
                        (project(center), project(center + axis * 0.35))
                    {
                        painter.line_segment([from, to], egui::Stroke::new(3.0_f32, color));
                        painter.circle_filled(to, 8.0, color);
                        painter.text(
                            to,
                            egui::Align2::CENTER_CENTER,
                            label,
                            egui::FontId::proportional(11.0),
                            egui::Color32::WHITE,
                        );
                    }
                }
            }
        }
    }
}

// ==========================================
// 5. ИНТЕРФЕЙС И DRAG & DROP
// ==========================================

impl eframe::App for CollisionEditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Neutral dark studio palette: contrast is reserved for selected OBBs
        // and the export action, so large projects remain readable.
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(25, 27, 31);
        visuals.window_fill = egui::Color32::from_rgb(29, 31, 36);
        visuals.extreme_bg_color = egui::Color32::from_rgb(18, 20, 24);
        visuals.selection.bg_fill = egui::Color32::from_rgb(70, 105, 92);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(61, 68, 78);
        ctx.set_visuals(visuals);

        if ctx.input(|i| (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(egui::Key::Z)) {
            self.undo_collision_edit();
        }
        // Drag & Drop
        if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
            let dropped = ctx.input(|i| i.raw.dropped_files.clone());
            for file in dropped {
                if let Some(path) = file.path {
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if ext == "json" {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if path.file_stem().and_then(|s| s.to_str()) == Some("collision") {
                                match parse_collision_data(&content) {
                                    Ok(collision) => {
                                        self.collision = collision;
                                        self.selected_box = None;
                                        self.status_message = format!(
                                            "Загружено OBB боксов: {}",
                                            self.collision.boxes.len()
                                        );
                                    }
                                    Err(error) => {
                                        self.status_message =
                                            format!("Ошибка collision.json: {error}")
                                    }
                                }
                            } else {
                                self.load_model_from_string(content, Some(path));
                            }
                        }
                    } else if ext == "png" {
                        self.load_texture_from_path(path);
                    }
                }
            }
        }

        egui::SidePanel::left("workflow_panel")
            .resizable(false)
            .default_width(190.0)
            .show(ctx, |ui| {
                ui.add_space(10.0);
                ui.heading("SWRMC");
                ui.label(egui::RichText::new("Decor Editor").weak());
                ui.add_space(18.0);
                ui.label(egui::RichText::new("РАБОЧИЙ ПРОЦЕСС").small().weak());
                ui.separator();
                let steps = [
                    ("01", "Модель", self.model.is_some()),
                    ("02", "Текстура", self.texture_path.is_some()),
                    ("03", "Коллизии", !self.collision.boxes.is_empty()),
                    ("04", "Экспорт", false),
                ];
                for (number, label, complete) in steps {
                    let marker = if complete { "●" } else { "○" };
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(marker).color(if complete {
                            egui::Color32::from_rgb(90, 210, 145)
                        } else {
                            egui::Color32::GRAY
                        }));
                        ui.label(format!("{number}  {label}"));
                    });
                    ui.add_space(6.0);
                }
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.separator();
                    ui.label(
                        egui::RichText::new("VBO • OBB collision • v2")
                            .small()
                            .weak(),
                    );
                });
            });

        egui::SidePanel::right("control_panel")
            .resizable(true)
            .default_width(380.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("Пакет");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("📂 Загрузить модель").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Blockbench Model / JSON", &["json"])
                            .pick_file()
                        {
                            if let Ok(content) = fs::read_to_string(&path) {
                                self.load_model_from_string(content, Some(path));
                            }
                        }
                    }
                    if ui.button("🖼 Текстуры").clicked() {
                        if let Some(paths) = rfd::FileDialog::new()
                            .add_filter("Image", &["png"])
                            .pick_files()
                        {
                            self.load_texture_files(paths);
                        }
                    }
                    if ui
                        .button("📥 Коллизия")
                        .on_hover_text("Загрузить v2 OBB или старый min/max collision.json")
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Collision JSON", &["json"])
                            .pick_file()
                        {
                            match fs::read_to_string(&path)
                                .map_err(|e| e.to_string())
                                .and_then(|s| parse_collision_data(&s))
                            {
                                Ok(collision) => {
                                    self.collision = collision;
                                    self.selected_box = None;
                                    self.status_message = format!(
                                        "Загружено OBB боксов: {}",
                                        self.collision.boxes.len()
                                    );
                                }
                                Err(error) => {
                                    self.status_message = format!("Ошибка collision.json: {error}")
                                }
                            }
                        }
                    }
                });

                if !self.texture_sources.is_empty() {
                    ui.label(format!("🖼 Atlas: {} текстур", self.texture_sources.len()));
                    let mut texture_rows: Vec<_> = self
                        .texture_sources
                        .iter()
                        .map(|(slot, path)| (slot.clone(), path.clone()))
                        .collect();
                    texture_rows.sort_by(|a, b| a.0.cmp(&b.0));
                    let mut remove_slot = None;
                    for (slot, path) in texture_rows {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "#{slot}  {}",
                                    path.file_name().unwrap_or_default().to_string_lossy()
                                ))
                                .small()
                                .weak(),
                            );
                            if ui
                                .small_button("✕")
                                .on_hover_text("Убрать texture-slot из пререндера и экспорта")
                                .clicked()
                            {
                                remove_slot = Some(slot);
                            }
                        });
                    }
                    if let Some(slot) = remove_slot {
                        self.remove_texture_slot(&slot);
                    }
                } else {
                    ui.label("🖼 Текстура не назначена (шейдинг)");
                }

                ui.add_space(8.0);
                ui.heading("Метаданные");
                egui::Grid::new("meta_grid")
                    .num_columns(2)
                    .spacing([10.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("UID:");
                        ui.text_edit_singleline(&mut self.meta.uid);
                        ui.end_row();

                        ui.label("Name:");
                        ui.text_edit_singleline(&mut self.meta.name);
                        ui.end_row();

                        ui.label("Category:");
                        ui.text_edit_singleline(&mut self.meta.category);
                        ui.end_row();

                        ui.label("Author:");
                        ui.text_edit_singleline(&mut self.meta.author);
                        ui.end_row();

                        ui.label("Material:");
                        ui.text_edit_singleline(&mut self.meta.material);
                        ui.end_row();

                        ui.label("Desc:");
                        ui.text_edit_singleline(&mut self.meta.description);
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.heading("Генерация OBB-коллизий");

                ui.horizontal_wrapped(|ui| {
                    for (mode, label, hint) in [
                        (0u8, "Габарит", "Один AABB"),
                        (1, "Кластеры", "Соседние секции"),
                        (2, "Оптимизация", "Без заполнения пустот"),
                        (3, "Упростить", "Контролируемое упрощение формы"),
                        (4, "Точно", "Один OBB на элемент"),
                    ] {
                        if ui
                            .selectable_label(self.generation_mode == mode, label)
                            .on_hover_text(hint)
                            .clicked()
                        {
                            self.generation_mode = mode;
                            self.regenerate_collision();
                        }
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Шагов поиска");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.optimization_steps, 1..=16)
                                .suffix(" проходов"),
                        )
                        .changed()
                        && matches!(self.generation_mode, 2 | 3)
                    {
                        self.regenerate_collision();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Допуск стыка");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.merge_tolerance, 0.001..=0.05)
                                .step_by(0.001),
                        )
                        .changed()
                        && matches!(self.generation_mode, 1 | 2 | 3)
                    {
                        self.regenerate_collision();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Цель по боксам");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.target_box_count)
                                .clamp_range(0..=999)
                                .prefix("до "),
                        )
                        .changed()
                        && matches!(self.generation_mode, 2 | 3)
                    {
                        self.regenerate_collision();
                    }
                });
                if self.generation_mode == 3 {
                    ui.horizontal(|ui| {
                        ui.label("Не учитывать детали <");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.min_detail_size, 0.0..=0.5)
                                    .step_by(0.01)
                                    .suffix(" блока"),
                            )
                            .changed()
                        {
                            self.regenerate_collision();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Допустимое скругление");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.approximation_fill, 0.0..=1.0)
                                    .step_by(0.01)
                                    .suffix(" × объёма"),
                            )
                            .changed()
                        {
                            self.regenerate_collision();
                        }
                    });
                    ui.label(
                        egui::RichText::new(
                            "Режим «Упростить» намеренно сглаживает углы. Укажите цель 6–8 OBB для грубой игровой формы.",
                        )
                        .small()
                        .weak(),
                    );
                }
                ui.label(
                    egui::RichText::new(
                        "0 — без ограничения. В «Упростить» цель нужна для объединения; мелкие детали фильтруются отдельно.",
                    )
                    .small()
                    .weak(),
                );

                ui.add_space(8.0);
                ui.separator();
                ui.heading(format!(
                    "🛠 Ручная правка OBB ({})",
                    self.collision.boxes.len()
                ));

                if ui.button("➕ Добавить OBB бокс").clicked() {
                    self.save_undo_point();
                    self.collision.boxes.push(ObbBox {
                        center: [0.0_f32, 0.25_f32, 0.0_f32],
                        half_size: [0.25_f32, 0.25_f32, 0.25_f32],
                        rotation: [0.0_f32, 0.0_f32, 0.0_f32],
                    });
                    self.selected_box = Some(self.collision.boxes.len() - 1);
                }

                let collision_before_edit = self.collision.clone();
                let mut to_delete = None;
                egui::ScrollArea::vertical()
                    .max_height(240.0)
                    .show(ui, |ui| {
                        for (idx, b) in self.collision.boxes.iter_mut().enumerate() {
                            let is_selected = self.selected_box == Some(idx);
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    if ui
                                        .selectable_label(is_selected, format!("OBB Box #{}", idx))
                                        .clicked()
                                    {
                                        self.selected_box = Some(idx);
                                    }
                                    if ui.button("🗑").clicked() {
                                        to_delete = Some(idx);
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Center:");
                                    ui.add(
                                        egui::DragValue::new(&mut b.center[0])
                                            .speed(0.005_f32)
                                            .prefix("X:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut b.center[1])
                                            .speed(0.005_f32)
                                            .prefix("Y:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut b.center[2])
                                            .speed(0.005_f32)
                                            .prefix("Z:"),
                                    );
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Half size:");
                                    ui.add(
                                        egui::DragValue::new(&mut b.half_size[0])
                                            .speed(0.005_f32)
                                            .prefix("X:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut b.half_size[1])
                                            .speed(0.005_f32)
                                            .prefix("Y:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut b.half_size[2])
                                            .speed(0.005_f32)
                                            .prefix("Z:"),
                                    );
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Rotation:");
                                    ui.add(
                                        egui::DragValue::new(&mut b.rotation[0])
                                            .speed(0.5_f32)
                                            .prefix("Pitch:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut b.rotation[1])
                                            .speed(0.5_f32)
                                            .prefix("Yaw:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut b.rotation[2])
                                            .speed(0.5_f32)
                                            .prefix("Roll:"),
                                    );
                                });
                            });
                        }
                    });

                if self.collision != collision_before_edit {
                    self.undo_history
                        .push(UndoAction::Collision(collision_before_edit));
                }

                if let Some(idx) = to_delete {
                    self.save_undo_point();
                    self.collision.boxes.remove(idx);
                    self.selected_box = None;
                }

                ui.add_space(12.0);
                ui.separator();

                if ui
                    .add_sized(
                        [ui.available_width(), 35.0],
                        egui::Button::new("💾 Экспортировать пакет"),
                    )
                    .clicked()
                {
                    self.export_package();
                }
                if ui.button("↶ Отменить (Ctrl+Z)").clicked() {
                    self.undo_collision_edit();
                }
            });

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_model_faces, "Тело модели");
                ui.checkbox(&mut self.show_model_wire, "Каркас модели");
                ui.checkbox(&mut self.show_collision, "Коллизия (OBB)");
                ui.checkbox(&mut self.show_grid, "Сетка");
                ui.separator();
                ui.label(&self.status_message);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_viewport(ui);
        });
    }
}

// ==========================================
// 6. ТОЧКА ВХОДА
// ==========================================

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true)
            .with_title("SWRMC — Decor Editor"),
        ..Default::default()
    };

    eframe::run_native(
        "SWRMC — Decor Editor",
        native_options,
        Box::new(|_cc| Box::new(CollisionEditorApp::default())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_v2_serializes_with_required_field_names() {
        let data = CollisionData {
            version: 2,
            boxes: vec![ObbBox {
                center: [0.0, 0.5, 0.0],
                half_size: [0.5, 0.5, 0.5],
                rotation: [0.0, 35.0, 0.0],
            }],
        };
        assert_eq!(
            serde_json::to_value(data).unwrap(),
            serde_json::json!({
                "version": 2,
                "boxes": [{"center": [0.0, 0.5, 0.0], "halfSize": [0.5, 0.5, 0.5], "rotation": [0.0, 35.0, 0.0]}]
            })
        );
    }

    #[test]
    fn legacy_min_max_is_loaded_as_an_unrotated_obb() {
        let data = parse_collision_data(r#"{"boxes":[{"min":[-1,0,2],"max":[3,4,6]}]}"#).unwrap();
        assert_eq!(data.version, 2);
        assert_eq!(data.boxes[0].center, [1.0, 2.0, 4.0]);
        assert_eq!(data.boxes[0].half_size, [2.0, 2.0, 2.0]);
        assert_eq!(data.boxes[0].rotation, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn exporting_texture_onto_itself_preserves_its_bytes() {
        let file = std::env::temp_dir().join(format!(
            "swrmc-texture-{}-{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let original = vec![137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3];
        fs::write(&file, &original).unwrap();
        CollisionEditorApp::copy_texture_safely(&file, &file).unwrap();
        assert_eq!(fs::read(&file).unwrap(), original);
        fs::remove_file(file).unwrap();
    }

    #[test]
    fn gameplay_simplifier_removes_tiny_parts_and_respects_box_goal() {
        let mut app = CollisionEditorApp::default();
        app.target_box_count = 1;
        app.optimization_steps = 4;
        app.min_detail_size = 0.1;
        app.approximation_fill = 0.01;
        let box_at = |x, half| ObbBox {
            center: [x, 0.0, 0.0],
            half_size: half,
            rotation: [0.0; 3],
        };
        let result = app.simplify_collision(vec![
            box_at(-0.25, [0.25, 0.25, 0.25]),
            box_at(0.25, [0.25, 0.25, 0.25]),
            box_at(2.0, [0.02, 0.02, 0.02]),
        ]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].half_size, [0.5, 0.25, 0.25]);
    }

    #[test]
    fn export_adds_texture_slots_without_dropping_model_fields() {
        let mut app = CollisionEditorApp::default();
        app.raw_model_json = r#"{"parent":"block/cube","elements":[]}"#.to_string();
        app.texture_sources
            .insert("0".to_string(), PathBuf::from("bed.png"));
        app.texture_sources
            .insert("1".to_string(), PathBuf::from("metal.png"));
        let json: serde_json::Value =
            serde_json::from_slice(&app.exported_model_json().unwrap()).unwrap();
        assert_eq!(json["parent"], "block/cube");
        assert_eq!(json["textures"]["0"], "bed.png");
        assert_eq!(json["textures"]["1"], "metal.png");
    }

    #[test]
    fn parent_resource_model_is_flattened_and_child_textures_override() {
        let root = std::env::temp_dir().join(format!("swrmc-parent-{}", std::process::id()));
        let parent_dir = root.join("assets/example/models/custom");
        let child_dir = root.join("assets/example/models/item");
        fs::create_dir_all(&parent_dir).unwrap();
        fs::create_dir_all(&child_dir).unwrap();
        let parent_path = parent_dir.join("ammo.json");
        let child_path = child_dir.join("child.json");
        fs::write(&parent_path, r#"{"textures":{"0":"example:block/base"},"elements":[{"from":[0,0,0],"to":[16,16,16]}]}"#).unwrap();
        fs::write(&child_path, r#"{"parent":"example:custom/ammo","textures":{"0":"example:block/child"},"render_type":"solid"}"#).unwrap();
        let value = CollisionEditorApp::resolve_inherited_model(&child_path, 0).unwrap();
        assert!(value.get("parent").is_none());
        assert_eq!(value["textures"]["0"], "example:block/child");
        assert_eq!(value["elements"].as_array().unwrap().len(), 1);
        assert_eq!(value["render_type"], "solid");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn single_texture_replacement_does_not_create_extra_slots() {
        let mut app = CollisionEditorApp::default();
        app.model = Some(RawModel {
            credit: None,
            texture_size: Some([16.0, 16.0]),
            textures: Some(HashMap::from([(
                "0".to_string(),
                serde_json::Value::String("old.png".to_string()),
            )])),
            elements: vec![],
        });
        let slots = app.single_texture_slots().unwrap();
        app.replace_single_texture(PathBuf::from("new.png"), slots);
        assert_eq!(app.texture_sources.len(), 1);
        assert_eq!(app.texture_sources["0"], PathBuf::from("new.png"));
        assert_eq!(app.model.unwrap().textures.unwrap()["0"], "new.png");
    }

    #[test]
    fn removed_texture_slot_disappears_from_export_and_undo_restores_it() {
        let mut app = CollisionEditorApp::default();
        app.raw_model_json = r#"{"textures":{"0":"old.png"},"elements":[]}"#.to_string();
        app.model = Some(RawModel {
            credit: None,
            texture_size: Some([16.0, 16.0]),
            textures: Some(HashMap::from([(
                "0".to_string(),
                serde_json::Value::String("old.png".to_string()),
            )])),
            elements: vec![],
        });
        app.texture_sources
            .insert("0".to_string(), PathBuf::from("old.png"));
        app.remove_texture_slot("0");
        let without_slot: serde_json::Value =
            serde_json::from_slice(&app.exported_model_json().unwrap()).unwrap();
        assert!(without_slot["textures"].get("0").is_none());
        app.undo_collision_edit();
        assert_eq!(app.texture_sources["0"], PathBuf::from("old.png"));
        assert_eq!(app.model.unwrap().textures.unwrap()["0"], "old.png");
    }

    #[test]
    fn omitted_texture_size_uses_blockbench_sixteen_pixel_grid() {
        let mut app = CollisionEditorApp::default();
        app.model = Some(RawModel {
            credit: None,
            texture_size: None,
            textures: None,
            elements: vec![],
        });
        app.texture_dimensions
            .insert("0".to_string(), [256.0, 256.0]);
        app.atlas_regions
            .insert("0".to_string(), [0.0, 0.0, 1.0, 1.0]);
        let element = RawElement {
            from: [0.0, 0.0, 0.0],
            to: [1.0, 1.0, 1.0],
            rotation: None,
            faces: Some(HashMap::from([(
                "north".to_string(),
                RawFace {
                    uv: Some([0.0, 0.0, 8.0, 8.0]),
                    texture: None,
                    rotation: None,
                },
            )])),
        };
        let uv = app.face_uv(&element, "north").unwrap();
        assert!((uv[1].x - 0.5).abs() < 0.0001);
        assert!((uv[2].y - 0.5).abs() < 0.0001);
    }
}

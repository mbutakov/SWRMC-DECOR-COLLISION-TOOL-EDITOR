//! Model light contract. Editor state is independent of generated collision boxes.
use eframe::egui;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Profile {
    pub id: String,
    pub direction: [f32; 3],
    pub enabled: bool,
    pub spot: bool,
    pub color: String,
    pub intensity: f32,
    pub radius: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
    pub source_size: f32,
    pub falloff: f32,
    pub volumetric: bool,
    pub flicker: String,
    pub flicker_depth: f32,
    pub flicker_speed: f32,
    pub duty: f32,
    pub seed: i64,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}
impl Default for Profile {
    fn default() -> Self {
        Self {
            id: String::new(),
            direction: [0., -1., 0.],
            enabled: true,
            spot: false,
            color: "#FFD6A0".into(),
            intensity: 1.8,
            radius: 12.,
            inner_angle: 20.,
            outer_angle: 35.,
            source_size: 0.08,
            falloff: 2.,
            volumetric: false,
            flicker: "NONE".into(),
            flicker_depth: 0.5,
            flicker_speed: 2.,
            duty: 0.5,
            seed: 0,
            extra: Default::default(),
        }
    }
}
impl Profile {
    pub fn rgb(&self) -> Option<[u8; 3]> {
        if self.color.len() != 7
            || !self.color.starts_with('#')
            || !self.color[1..].bytes().all(|b| b.is_ascii_hexdigit())
        {
            return None;
        }
        let n = u32::from_str_radix(&self.color[1..], 16).ok()?;
        Some([(n >> 16) as u8, (n >> 8) as u8, n as u8])
    }
    pub fn validate(&self) -> Result<(), String> {
        let fail = |s: &str| Err(format!("Свет «{}»: {s}", self.id));
        if self.id.is_empty()
            || self.id.len() > 64
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
        {
            return fail("ID: 1–64 символа A–Z, a–z, 0–9, _, ., -");
        }
        if self.rgb().is_none() {
            return fail("цвет должен быть #RRGGBB");
        }
        let d = Vec3::from_array(self.direction);
        if !d.is_finite() || !d.length_squared().is_finite() || d.length_squared() < 1e-12 {
            return fail("направление должно быть конечным ненулевым вектором");
        }
        for (name, value, min, max) in [
            ("intensity", self.intensity, 0., 20.),
            ("radius", self.radius, 0.1, 96.),
            ("innerAngle", self.inner_angle, 0., 88.),
            ("outerAngle", self.outer_angle, 0.1, 89.),
            ("sourceSize", self.source_size, 0.001, 8.),
            ("falloff", self.falloff, 0.25, 8.),
            ("flickerDepth", self.flicker_depth, 0., 1.),
            ("flickerSpeed", self.flicker_speed, 0.05, 30.),
            ("duty", self.duty, 0.01, 0.99),
        ] {
            if !value.is_finite() || value < min || value > max {
                return fail(&format!("{name}: допустимо {min}–{max}"));
            }
        }
        if self.inner_angle >= self.outer_angle {
            return fail("innerAngle должен быть строго меньше outerAngle");
        }
        if !["NONE", "SOFT", "BROKEN", "PULSE", "STROBE"].contains(&self.flicker.as_str()) {
            return fail("неизвестный режим мерцания");
        }
        Ok(())
    }
    pub fn flicker_at(&self, time: f32) -> f32 {
        let hash = self.id.bytes().fold(self.seed as u32, |h, b| {
            h.wrapping_mul(31).wrapping_add(b as u32)
        });
        let t = time * self.flicker_speed + (hash % 1000) as f32 / 1000.;
        let wave = match self.flicker.as_str() {
            "SOFT" => 0.5 + 0.5 * (t * std::f32::consts::TAU).sin(),
            "PULSE" => 0.5 - 0.5 * (t * std::f32::consts::TAU).cos(),
            "STROBE" => {
                if t.fract() < self.duty {
                    0.
                } else {
                    1.
                }
            }
            "BROKEN" => {
                let n = (t.floor() as u32)
                    .wrapping_add(hash)
                    .wrapping_mul(1664525)
                    .wrapping_add(1013904223);
                if (n ^ (n >> 13)) as f64 / (u32::MAX as f64) < self.duty as f64 {
                    1.
                } else {
                    0.
                }
            }
            _ => 0.,
        };
        if self.enabled {
            1. - self.flicker_depth * wave
        } else {
            0.
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub element: Option<usize>,
    pub position: [f32; 3],
    pub profile: Profile,
}
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Document {
    pub entries: Vec<Entry>,
}
impl Document {
    pub fn read(json: &Value) -> Result<Self, String> {
        let mut entries = Vec::new();
        if let Some(elements) = json.get("elements").and_then(Value::as_array) {
            for (i, e) in elements.iter().enumerate() {
                if let Some(light) = e.get("light") {
                    let profile = serde_json::from_value(light.clone())
                        .map_err(|e| format!("elements[{i}].light: {e}"))?;
                    entries.push(Entry {
                        element: Some(i),
                        position: [0.; 3],
                        profile,
                    });
                }
            }
        }
        if let Some(lights) = json.get("lights") {
            for (i, light) in lights
                .as_array()
                .ok_or("lights должен быть массивом")?
                .iter()
                .enumerate()
            {
                let position = serde_json::from_value(
                    light
                        .get("position")
                        .cloned()
                        .ok_or(format!("lights[{i}]: отсутствует position"))?,
                )
                .map_err(|e| format!("lights[{i}].position: {e}"))?;
                let mut profile: Profile = serde_json::from_value(light.clone())
                    .map_err(|e| format!("lights[{i}]: {e}"))?;
                profile.extra.remove("position");
                entries.push(Entry {
                    element: None,
                    position,
                    profile,
                });
            }
        }
        let doc = Self { entries };
        doc.validate()?;
        Ok(doc)
    }
    pub fn validate(&self) -> Result<(), String> {
        let mut ids = HashSet::new();
        let mut elements = HashSet::new();
        for e in &self.entries {
            e.profile.validate()?;
            if !ids.insert(&e.profile.id) {
                return Err(format!("Повторяющийся ID света: {}", e.profile.id));
            }
            if let Some(i) = e.element {
                if !elements.insert(i) {
                    return Err(format!("Два источника на боксе {i}"));
                }
            }
            if !Vec3::from_array(e.position).is_finite() {
                return Err(format!("{}: некорректная position", e.profile.id));
            }
        }
        Ok(())
    }
    pub fn next_id(&self) -> String {
        (1..)
            .map(|i| format!("bulb.{i}"))
            .find(|id| self.entries.iter().all(|e| e.profile.id != *id))
            .unwrap()
    }
    pub fn write(&self, json: &mut Value) -> Result<(), String> {
        self.validate()?;
        if let Some(elements) = json.get_mut("elements").and_then(Value::as_array_mut) {
            for e in elements {
                e.as_object_mut()
                    .ok_or("Некорректный элемент модели")?
                    .remove("light");
            }
        }
        let mut roots = Vec::new();
        for e in &self.entries {
            let mut p = e.profile.clone();
            p.direction = Vec3::from_array(p.direction).normalize().to_array();
            let mut light = serde_json::to_value(p).map_err(|e| e.to_string())?;
            if let Some(i) = e.element {
                let element = json
                    .get_mut("elements")
                    .and_then(Value::as_array_mut)
                    .and_then(|a| a.get_mut(i))
                    .and_then(Value::as_object_mut)
                    .ok_or("Исходный бокс источника не найден")?;
                element.insert("light".into(), light);
            } else {
                light["position"] = serde_json::json!(e.position);
                roots.push(light);
            }
        }
        let root = json.as_object_mut().ok_or("Некорректная модель")?;
        if roots.is_empty() {
            root.remove("lights");
        } else {
            root.insert("lights".into(), Value::Array(roots));
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct Editor {
    pub direction_drag: bool,
    pub document: Option<Document>,
    pub tab: bool,
    pub selected: Option<usize>,
    pub active_element: Option<usize>,
    pub reveal_element: bool,
    pub selected_elements: HashSet<usize>,
    pub only_sources: bool,
    pub night: bool,
    pub paused: bool,
    pub time: f32,
    pub play_until: Option<f32>,
    pub environment: bool,
    pub guides: bool,
}
fn slider(ui: &mut egui::Ui, label: &str, value: &mut f32, min: f32, max: f32, log: bool) {
    ui.add(
        egui::Slider::new(value, min..=max)
            .text(label)
            .logarithmic(log),
    );
}
impl Editor {
    pub fn panel(&mut self, ui: &mut egui::Ui, raw: &str) {
        let Some(doc) = &mut self.document else {
            ui.label("Загрузите модель для настройки света.");
            return;
        };
        ui.heading(format!("Свет · {} источников", doc.entries.len()));
        ui.label("16 пикселей модели = 1 блок. Источники независимы от OBB.");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.night, "Ночь");
            ui.checkbox(&mut self.environment, "Пол и стена");
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.guides, "Радиус / конусы");
            ui.checkbox(&mut self.paused, "Пауза");
        });
        if ui.button("Проиграть 10 секунд").clicked() {
            self.time = 0.;
            self.play_until = Some(10.);
            self.paused = false;
        }
        ui.small("Приближённый свет без теней; объёмный эффект проверяйте в игре.");
        ui.small("Для гирлянд объединяйте близкие LED. Source size — не area-light.");
        ui.separator();
        ui.checkbox(&mut self.only_sources, "Только источники");
        let json: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
        egui::ScrollArea::vertical()
            .id_source("light_tree")
            .max_height(160.)
            .show(ui, |ui| {
                if let Some(elements) = json.get("elements").and_then(Value::as_array) {
                    for (i, e) in elements.iter().enumerate() {
                        let source = doc.entries.iter().position(|e| e.element == Some(i));
                        if self.only_sources && source.is_none() {
                            continue;
                        }
                        ui.horizontal(|ui| {
                            let mut checked = self.selected_elements.contains(&i);
                            if ui.checkbox(&mut checked, "").changed() {
                                if checked {
                                    self.selected_elements.insert(i);
                                    self.active_element = Some(i);
                                    self.selected = source;
                                } else {
                                    self.selected_elements.remove(&i);
                                    if self.active_element == Some(i) {
                                        self.active_element = None;
                                        self.selected = None;
                                    }
                                }
                            }
                            let name = e
                                .get("name")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                                .unwrap_or(format!("Бокс {}", i + 1));
                            let row = ui.selectable_label(
                                self.active_element == Some(i)
                                    || self.selected_elements.contains(&i),
                                format!("{} {name}", if source.is_some() { "💡" } else { "□" }),
                            );
                            if self.reveal_element && self.active_element == Some(i) {
                                row.scroll_to_me(Some(egui::Align::Center));
                                self.reveal_element = false;
                            }
                            if row.clicked() {
                                if !ui.input(|input| {
                                    input.modifiers.shift
                                        || input.modifiers.command
                                        || input.modifiers.ctrl
                                }) {
                                    self.selected_elements.clear();
                                }
                                self.selected_elements.insert(i);
                                self.active_element = Some(i);
                                self.selected = source;
                            }
                        });
                    }
                }
                for (i, e) in doc
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.element.is_none())
                {
                    if ui
                        .selectable_label(
                            self.selected == Some(i),
                            format!("💡 {} · LED", e.profile.id),
                        )
                        .clicked()
                    {
                        self.selected = Some(i);
                        self.active_element = None;
                        self.selected_elements.clear();
                    }
                }
            });
        if ui
            .add_enabled(
                !self.selected_elements.is_empty(),
                egui::Button::new("Сделать выделенные источниками"),
            )
            .clicked()
        {
            let mut indices: Vec<_> = self.selected_elements.iter().copied().collect();
            indices.sort_unstable();
            for i in indices {
                if !doc.entries.iter().any(|e| e.element == Some(i)) {
                    let profile = Profile {
                        id: doc.next_id(),
                        ..Default::default()
                    };
                    doc.entries.push(Entry {
                        element: Some(i),
                        position: [0.; 3],
                        profile,
                    });
                    self.selected = Some(doc.entries.len() - 1);
                    self.active_element = Some(i);
                }
            }
        }
        if ui.button("+ Отдельный LED").clicked() {
            let profile = Profile {
                id: doc.next_id(),
                ..Default::default()
            };
            doc.entries.push(Entry {
                element: None,
                position: [8., 8., 8.],
                profile,
            });
            self.selected = Some(doc.entries.len() - 1);
            self.active_element = None;
            self.selected_elements.clear();
        }
        if let Some(i) = self.selected.filter(|&i| i < doc.entries.len()) {
            ui.separator();
            let e = &mut doc.entries[i];
            ui.label(format!("ID: {}", e.profile.id)).on_hover_text(
                "Постоянный ID: сохраняется при переименовании и изменении настроек.",
            );
            if e.element.is_none() {
                ui.label("Положение · пиксели модели");
                ui.horizontal(|ui| {
                    for (v, axis) in e.position.iter_mut().zip(["X", "Y", "Z"]) {
                        ui.add(egui::DragValue::new(v).speed(0.1).prefix(axis));
                    }
                });
            }
            egui::ComboBox::from_id_source("light_preset")
                .selected_text("Библиотека пресетов")
                .show_ui(ui, |ui| {
                    for (name, color, flicker, radius) in [
                        ("Тёплая лампа", "#FFD6A0", "NONE", 12.),
                        ("Холодный LED", "#BBDDFF", "NONE", 4.),
                        ("Аварийный красный", "#FF3030", "STROBE", 8.),
                        ("Неисправная люминесцентная", "#D8FFEB", "BROKEN", 12.),
                    ] {
                        if ui.selectable_label(false, name).clicked() {
                            e.profile = Profile {
                                id: e.profile.id.clone(),
                                direction: e.profile.direction,
                                color: color.into(),
                                flicker: flicker.into(),
                                radius,
                                extra: e.profile.extra.clone(),
                                ..Default::default()
                            };
                        }
                    }
                });
            let p = &mut e.profile;
            ui.heading("Основное");
            ui.checkbox(&mut p.enabled, "Включён");
            ui.checkbox(&mut p.spot, "Направленный конус (spot)");
            ui.horizontal(|ui| {
                let mut rgb = p.rgb().unwrap_or([255, 214, 160]);
                if ui.color_edit_button_srgb(&mut rgb).changed() {
                    p.color = format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]);
                }
                ui.text_edit_singleline(&mut p.color);
            });
            ui.horizontal(|ui| {
                for color in ["#FFD6A0", "#FFFFFF", "#55AAFF", "#FF3030", "#88FFAA"] {
                    let swatch = Profile {
                        color: color.into(),
                        ..Default::default()
                    }
                    .rgb()
                    .unwrap();
                    if ui
                        .add(
                            egui::Button::new("  ")
                                .fill(egui::Color32::from_rgb(swatch[0], swatch[1], swatch[2])),
                        )
                        .on_hover_text(color)
                        .clicked()
                    {
                        p.color = color.into();
                    }
                }
            });
            slider(ui, "Интенсивность", &mut p.intensity, 0., 20., false);
            slider(ui, "Радиус · блоки", &mut p.radius, 0.1, 96., true);
            slider(
                ui,
                "Размер источника · блоки",
                &mut p.source_size,
                0.001,
                8.,
                true,
            );
            slider(ui, "Затухание", &mut p.falloff, 0.25, 8., false);
            if p.spot {
                ui.heading("Конус");
                ui.small("Углы — половина раскрытия: 35° = полный угол 70°.");
                slider(ui, "Внутренний · °", &mut p.inner_angle, 0., 88., false);
                slider(ui, "Внешний · °", &mut p.outer_angle, 0.1, 89., false);
                ui.label("Направление (локальные координаты)");
                ui.horizontal(|ui| {
                    for (v, axis) in p.direction.iter_mut().zip(["X", "Y", "Z"]) {
                        ui.add(egui::DragValue::new(v).speed(0.01).prefix(axis));
                    }
                });
                ui.small("Стрелку выбранного источника можно перетаскивать в сцене.");
            }
            ui.heading("Атмосфера");
            ui.checkbox(&mut p.volumetric, "Объёмное рассеяние");
            egui::ComboBox::from_label("Мерцание")
                .selected_text(&p.flicker)
                .show_ui(ui, |ui| {
                    for mode in ["NONE", "SOFT", "BROKEN", "PULSE", "STROBE"] {
                        ui.selectable_value(&mut p.flicker, mode.into(), mode);
                    }
                });
            if p.flicker != "NONE" {
                slider(ui, "Глубина", &mut p.flicker_depth, 0., 1., false);
                slider(ui, "Частота · Гц", &mut p.flicker_speed, 0.05, 30., true);
                if p.flicker == "BROKEN" || p.flicker == "STROBE" {
                    slider(
                        ui,
                        if p.flicker == "BROKEN" {
                            "Вероятность провала"
                        } else {
                            "Доля включения"
                        },
                        &mut p.duty,
                        0.01,
                        0.99,
                        false,
                    );
                }
            }
            ui.add(egui::DragValue::new(&mut p.seed).prefix("Seed: "));
            if ui.button("Удалить источник").clicked() {
                doc.entries.remove(i);
                self.selected = None;
            }
        }
        if let Err(error) = doc.validate() {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
    }
}

/// Element light positions use pivot, rotation and rescale, then model pixels → blocks.
pub fn transform(entry: &Entry, raw: &Value) -> (Vec3, Vec3, Quat) {
    let mut pos = Vec3::from_array(entry.position);
    let mut rotation = Quat::IDENTITY;
    if let Some(element) = entry.element.and_then(|i| raw.get("elements")?.get(i)) {
        let vec = |key: &str| {
            serde_json::from_value::<[f32; 3]>(element[key].clone())
                .map(Vec3::from_array)
                .unwrap_or(Vec3::ZERO)
        };
        pos = (vec("from") + vec("to")) * 0.5;
        if let Some(r) = element.get("rotation") {
            let axis = match r["axis"].as_str().unwrap_or("") {
                "x" => Vec3::X,
                "y" => Vec3::Y,
                _ => Vec3::Z,
            };
            let angle = r["angle"].as_f64().unwrap_or(0.) as f32;
            rotation = Quat::from_axis_angle(axis, angle.to_radians());
            let origin = serde_json::from_value::<[f32; 3]>(r["origin"].clone())
                .map(Vec3::from_array)
                .unwrap_or(Vec3::ZERO);
            let mut offset = pos - origin;
            if r["rescale"].as_bool().unwrap_or(false) {
                offset *= axis.abs()
                    + (Vec3::ONE - axis.abs()) / angle.to_radians().cos().abs().max(0.001);
            }
            pos = origin + rotation * offset;
        }
    }
    (
        (pos - Vec3::new(8., 0., 8.)) / 16.,
        rotation * Vec3::from_array(entry.profile.direction).normalize_or_zero(),
        rotation,
    )
}

pub struct PreviewSource {
    pub position: Vec3,
    pub direction: Vec3,
    pub profile: Profile,
    pub strength: f32,
}
pub fn preview_sources(editor: &Editor, raw: &Value) -> Vec<PreviewSource> {
    if !editor.tab {
        return Vec::new();
    }
    editor
        .document
        .iter()
        .flat_map(|d| &d.entries)
        .filter(|e| e.profile.validate().is_ok())
        .map(|e| {
            let (position, direction, _) = transform(e, raw);
            PreviewSource {
                position,
                direction,
                profile: e.profile.clone(),
                strength: e.profile.flicker_at(editor.time),
            }
        })
        .collect()
}
pub fn shade(pos: Vec3, normal: Vec3, lights: &[PreviewSource], ambient: f32) -> Vec3 {
    let mut result = Vec3::splat(ambient);
    for l in lights {
        let offset = pos - l.position;
        let distance = offset.length();
        if distance >= l.profile.radius || l.strength <= 0. {
            continue;
        }
        let mut amount = (1. - distance / l.profile.radius).powf(l.profile.falloff)
            * l.profile.intensity
            * l.strength;
        if l.profile.spot {
            let cos = offset.normalize_or_zero().dot(l.direction);
            let outer = l.profile.outer_angle.to_radians().cos();
            let inner = l.profile.inner_angle.to_radians().cos();
            amount *= ((cos - outer) / (inner - outer).max(0.00001)).clamp(0., 1.);
        }
        amount *= normal.dot(-offset.normalize_or_zero()).max(0.);
        if let Some(rgb) = l.profile.rgb() {
            result += Vec3::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32) / 255. * amount;
        }
    }
    result.clamp(Vec3::ZERO, Vec3::ONE)
}
type Project<'a> = dyn Fn(Vec3) -> Option<(egui::Pos2, f32)> + 'a;
pub fn draw_environment(
    painter: &egui::Painter,
    project: &Project<'_>,
    lights: &[PreviewSource],
    night: bool,
) {
    let mut tiles = Vec::new();
    for wall in [false, true] {
        for x in -8..8 {
            for z in -8..8 {
                let x = x as f32 * 0.5;
                let z = z as f32 * 0.5;
                let points = if wall {
                    [
                        Vec3::new(x, z + 4., -4.),
                        Vec3::new(x + 0.5, z + 4., -4.),
                        Vec3::new(x + 0.5, z + 4.5, -4.),
                        Vec3::new(x, z + 4.5, -4.),
                    ]
                } else {
                    [
                        Vec3::new(x, -0.01, z),
                        Vec3::new(x + 0.5, -0.01, z),
                        Vec3::new(x + 0.5, -0.01, z + 0.5),
                        Vec3::new(x, -0.01, z + 0.5),
                    ]
                };
                let projected: Option<Vec<_>> = points.iter().map(|&p| project(p)).collect();
                if let Some(projected) = projected {
                    let color = shade(
                        points.iter().sum::<Vec3>() / 4.,
                        if wall { Vec3::Z } else { Vec3::Y },
                        lights,
                        if night { 0.04 } else { 0.35 },
                    );
                    tiles.push((
                        projected.iter().map(|p| p.1).sum::<f32>(),
                        projected.iter().map(|p| p.0).collect::<Vec<_>>(),
                        egui::Color32::from_rgb(
                            (color.x * 190.) as u8,
                            (color.y * 190.) as u8,
                            (color.z * 190.) as u8,
                        ),
                    ));
                }
            }
        }
    }
    tiles.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, points, color) in tiles {
        painter.add(egui::Shape::convex_polygon(
            points,
            color,
            egui::Stroke::NONE,
        ));
    }
}
pub fn draw_guides(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    project: &Project<'_>,
    editor: &mut Editor,
    raw: &Value,
    yaw: f32,
    pitch: f32,
) {
    editor.direction_drag = false;
    let Some(doc) = &mut editor.document else {
        return;
    };
    for (i, entry) in doc.entries.iter_mut().enumerate() {
        let (pos, direction, rotation) = transform(entry, raw);
        if !pos.is_finite() || direction.length_squared() < 0.5 {
            continue;
        }
        let rgb = entry.profile.rgb().unwrap_or([255, 0, 255]);
        let color = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        let selected = editor.selected == Some(i);
        let line = |a, b, color| {
            if let (Some((a, _)), Some((b, _))) = (project(a), project(b)) {
                painter.line_segment([a, b], egui::Stroke::new(1.0_f32, color));
            }
        };
        if let Some((screen, _)) = project(pos) {
            painter.circle_filled(
                screen,
                if selected { 6. } else { 3. },
                if entry.profile.enabled {
                    color
                } else {
                    egui::Color32::GRAY
                },
            );
            if ui
                .interact(
                    egui::Rect::from_center_size(screen, egui::vec2(14., 14.)),
                    ui.id().with(("source", i)),
                    egui::Sense::click(),
                )
                .clicked()
            {
                editor.selected = Some(i);
                editor.active_element = entry.element;
                editor.selected_elements.clear();
                if let Some(element) = entry.element {
                    editor.selected_elements.insert(element);
                }
                editor.reveal_element = true;
            }
            if selected {
                painter.text(
                    screen + egui::vec2(9., -9.),
                    egui::Align2::LEFT_BOTTOM,
                    &entry.profile.id,
                    egui::FontId::proportional(12.),
                    color,
                );
            }
        }
        if !selected {
            continue;
        }
        let p = &entry.profile;
        if editor.guides {
            if p.spot {
                let tangent = direction.any_orthonormal_vector();
                let bitangent = direction.cross(tangent);
                for angle in [p.inner_angle, p.outer_angle] {
                    let angle = angle.to_radians();
                    let center = pos + direction * p.radius * angle.cos();
                    let radius = p.radius * angle.sin();
                    for step in 0..64 {
                        let point = |n: i32| {
                            let a = n as f32 * std::f32::consts::TAU / 64.;
                            center + radius * (tangent * a.cos() + bitangent * a.sin())
                        };
                        line(point(step), point(step + 1), color);
                        if step % 16 == 0 {
                            line(pos, point(step), color);
                        }
                    }
                }
            } else {
                for (a, b) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::X, Vec3::Z)] {
                    for step in 0..64 {
                        let point = |n: i32| {
                            let theta = n as f32 * std::f32::consts::TAU / 64.;
                            pos + p.radius * (a * theta.cos() + b * theta.sin())
                        };
                        line(point(step), point(step + 1), color);
                    }
                }
            }
        }
        if p.spot {
            let tip = pos + direction * 0.7;
            if let (Some((base, _)), Some((end, _))) = (project(pos), project(tip)) {
                painter.arrow(base, end - base, egui::Stroke::new(3.0_f32, color));
                painter.circle_filled(end, 7., color);
                let response = ui.interact(
                    egui::Rect::from_center_size(end, egui::vec2(24., 24.)),
                    ui.id().with("light_direction"),
                    egui::Sense::drag(),
                );
                editor.direction_drag = response.is_pointer_button_down_on() || response.dragged();
                if response.dragged() {
                    let delta = ui.input(|i| i.pointer.delta());
                    let right = Vec3::new(-yaw.sin(), 0., yaw.cos());
                    let up = Vec3::new(
                        -yaw.cos() * pitch.sin(),
                        pitch.cos(),
                        -yaw.sin() * pitch.sin(),
                    );
                    let next =
                        (direction + (right * delta.x - up * delta.y) * 0.01).normalize_or_zero();
                    if next.length_squared() > 0. {
                        entry.profile.direction = (rotation.inverse() * next).to_array();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_and_round_trip_preserve_geometry_and_extensions() {
        let mut json = serde_json::json!({"custom":42,"elements":[{"from":[0,0,0],"to":[2,2,2],"faces":{"up":{"texture":"#0"}},"light":{"id":"main","extension":7}}],"lights":[{"id":"led","position":[4,12,8]}]});
        let geometry = json["elements"][0]["faces"].clone();
        let doc = Document::read(&json).unwrap();
        assert_eq!(doc.entries[0].profile.radius, 12.);
        doc.write(&mut json).unwrap();
        assert_eq!(json["elements"][0]["faces"], geometry);
        assert_eq!(json["custom"], 42);
        assert_eq!(json["elements"][0]["light"]["extension"], 7);
        assert_eq!(Document::read(&json).unwrap(), doc);
    }
    #[test]
    fn invalid_contract_is_rejected() {
        for patch in [
            serde_json::json!({"id":""}),
            serde_json::json!({"radius":97}),
            serde_json::json!({"innerAngle":35}),
            serde_json::json!({"color":"#oops"}),
            serde_json::json!({"direction":[0,0,0]}),
            serde_json::json!({"flicker":"INVALID"}),
            serde_json::json!({"seed":1.5}),
        ] {
            let mut p = serde_json::to_value(Profile {
                id: "main".into(),
                ..Default::default()
            })
            .unwrap();
            for (key, value) in patch.as_object().unwrap() {
                p[key] = value.clone();
            }
            assert!(Document::read(&serde_json::json!({"lights":[{ "id":"placeholder", "position":[0,0,0]}], "elements":[{"light":p}]})).is_err());
        }
        assert!(Document::read(&serde_json::json!({"elements":[{"light":{"id":"same"}}],"lights":[{"id":"same","position":[0,0,0]}]})).is_err());
    }
    #[test]
    fn three_hundred_sources_keep_ids_and_locations() {
        let mut doc = Document::default();
        for i in 0..300 {
            doc.entries.push(Entry {
                element: None,
                position: [i as f32, 0., 0.],
                profile: Profile {
                    id: doc.next_id(),
                    ..Default::default()
                },
            });
        }
        doc.entries.reverse();
        let mut json = serde_json::json!({});
        doc.write(&mut json).unwrap();
        assert_eq!(Document::read(&json).unwrap(), doc);
        assert_eq!(doc.next_id(), "bulb.301");
    }
    #[test]
    fn rotation_origin_and_rescale_are_applied() {
        let entry = Entry {
            element: Some(0),
            position: [0.; 3],
            profile: Profile {
                direction: [1., 0., 0.],
                ..Default::default()
            },
        };
        let raw = serde_json::json!({"elements":[{"from":[16,0,8],"to":[16,0,8],"rotation":{"origin":[8,0,8],"axis":"y","angle":45,"rescale":true}}]});
        let (position, direction, _) = transform(&entry, &raw);
        assert!((position - Vec3::new(0.5, 0., -0.5)).length() < 0.0001);
        assert!((direction.length() - 1.).abs() < 0.0001);
    }
    #[test]
    fn flicker_is_bounded_repeatable_and_can_be_disabled() {
        for mode in ["NONE", "SOFT", "BROKEN", "PULSE", "STROBE"] {
            let mut p = Profile {
                id: "main".into(),
                flicker: mode.into(),
                ..Default::default()
            };
            for i in 0..1000 {
                let t = i as f32 / 100.;
                let v = p.flicker_at(t);
                assert!((0.5..=1.).contains(&v));
                assert_eq!(v, p.flicker_at(t));
            }
            p.enabled = false;
            assert_eq!(p.flicker_at(0.), 0.);
        }
    }
}

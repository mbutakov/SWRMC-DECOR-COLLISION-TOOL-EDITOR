//! Adapter between the editor and the source-independent collision contract.
use crate::collision_asset::{Asset, Composition, Kind, Primitive};
use glam::{EulerRot, Quat};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub kind: Kind,
    pub radius: f32,
    pub half_segment: f32,
    pub flags: u32,
    pub mask: u32,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            kind: Kind::Obb,
            radius: 0.25,
            half_segment: 0.25,
            flags: 0,
            mask: u32::MAX,
        }
    }
}
impl Settings {
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
}
impl Asset {
    pub fn from_editor(asset: &str, boxes: &[crate::ObbBox]) -> Result<Self, String> {
        let primitives: Vec<_> = boxes.iter().map(editor_primitive).collect();
        let result = Self {
            format: "collision_asset".into(),
            version: 1,
            asset: asset.into(),
            units: "blocks".into(),
            composition: match primitives.len() {
                0 => Composition::None,
                1 => Composition::Single,
                _ => Composition::Compound,
            },
            primitives,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn into_editor(self) -> Result<crate::CollisionData, String> {
        self.validate()?;
        let mut boxes = Vec::with_capacity(self.primitives.len());
        for p in self.primitives {
            let (center, extent, q, s) = match p {
                Primitive::Aabb {
                    center,
                    half_extents,
                    flags,
                    mask,
                } => (
                    center,
                    half_extents,
                    Quat::IDENTITY,
                    Settings {
                        kind: Kind::Aabb,
                        flags,
                        mask,
                        ..Default::default()
                    },
                ),
                Primitive::Obb {
                    center,
                    half_extents,
                    orientation,
                    flags,
                    mask,
                } => (
                    center,
                    half_extents,
                    checked_quat(orientation)?,
                    Settings {
                        flags,
                        mask,
                        ..Default::default()
                    },
                ),
                Primitive::Capsule {
                    center,
                    orientation,
                    radius,
                    half_segment,
                    flags,
                    mask,
                } => (
                    center,
                    [radius, radius + half_segment, radius],
                    checked_quat(orientation)?,
                    Settings {
                        kind: Kind::Capsule,
                        radius,
                        half_segment,
                        flags,
                        mask,
                    },
                ),
            };
            let (x, y, z) = q.to_euler(EulerRot::XYZ);
            boxes.push(crate::ObbBox {
                center,
                half_size: extent,
                rotation: [x.to_degrees(), y.to_degrees(), z.to_degrees()],
                settings: s,
            });
        }
        Ok(crate::CollisionData { version: 2, boxes })
    }
}

/// Validate a borrowed editor snapshot without building another collision asset.
pub fn validate_editor(boxes: &[crate::ObbBox]) -> Result<(), String> {
    for (index, b) in boxes.iter().enumerate() {
        editor_primitive(b)
            .validate()
            .map_err(|e| format!("Primitive {index}: {e}"))?;
    }
    Ok(())
}
fn editor_primitive(b: &crate::ObbBox) -> Primitive {
    let s = &b.settings;
    match s.kind {
        Kind::Aabb => Primitive::Aabb {
            center: b.center,
            half_extents: b.half_size,
            flags: s.flags,
            mask: s.mask,
        },
        Kind::Obb => Primitive::Obb {
            center: b.center,
            half_extents: b.half_size,
            orientation: canonical_orientation(b.rotation_quat()),
            flags: s.flags,
            mask: s.mask,
        },
        Kind::Capsule => Primitive::Capsule {
            center: b.center,
            orientation: canonical_orientation(b.rotation_quat()),
            radius: s.radius,
            half_segment: s.half_segment,
            flags: s.flags,
            mask: s.mask,
        },
    }
}

// q and -q describe the same orientation. Prefer positive w, then x/y/z
// at the 180-degree boundary; remove negative zero for reproducible output.
fn canonical_orientation(q: Quat) -> [f32; 4] {
    let mut a = q.normalize().to_array();
    if [a[3], a[0], a[1], a[2]]
        .into_iter()
        .find(|v| *v != 0.)
        .is_some_and(|v| v < 0.)
    {
        for v in &mut a {
            *v = -*v;
        }
    }
    a.map(|v| if v == 0. { 0. } else { v })
}
fn checked_quat(a: [f32; 4]) -> Result<Quat, String> {
    crate::collision_asset::checked_orientation(a)
}

//! Source-format-independent intermediate collision contract; no render/editor data.
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Kind {
    Aabb,
    #[default]
    Obb,
    Capsule,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE", deny_unknown_fields)]
pub enum Primitive {
    Aabb {
        center: [f32; 3],
        half_extents: [f32; 3],
        flags: u32,
        mask: u32,
    },
    Obb {
        center: [f32; 3],
        half_extents: [f32; 3],
        orientation: [f32; 4],
        flags: u32,
        mask: u32,
    },
    Capsule {
        center: [f32; 3],
        orientation: [f32; 4],
        radius: f32,
        half_segment: f32,
        flags: u32,
        mask: u32,
    },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Composition {
    None,
    Single,
    Compound,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub format: String,
    pub version: u32,
    pub asset: String,
    pub units: String,
    pub composition: Composition,
    pub primitives: Vec<Primitive>,
}
impl Primitive {
    /// Validation uses the represented shape only, independent of editor state.
    /// Also reject overflow in derived bounds before an offline packer sees it.
    pub fn validate(&self) -> Result<(), String> {
        let (center, extent) = match self {
            Self::Aabb {
                center,
                half_extents,
                ..
            } => (Vec3::from_array(*center), checked_extent(*half_extents)?),
            Self::Obb {
                center,
                half_extents,
                orientation,
                ..
            } => {
                let e = checked_extent(*half_extents)?;
                let rotation = Mat3::from_quat(checked_orientation(*orientation)?);
                let extent = rotation.x_axis.abs() * e.x
                    + rotation.y_axis.abs() * e.y
                    + rotation.z_axis.abs() * e.z;
                (Vec3::from_array(*center), extent)
            }
            Self::Capsule {
                center,
                orientation,
                radius,
                half_segment,
                ..
            } => {
                if !radius.is_finite()
                    || *radius <= 0.
                    || !half_segment.is_finite()
                    || *half_segment < 0.
                    || !(radius + half_segment).is_finite()
                {
                    return Err("Invalid capsule dimensions".into());
                }
                let q = checked_orientation(*orientation)?;
                let extent = (q * Vec3::Y).abs() * *half_segment + Vec3::splat(*radius);
                (Vec3::from_array(*center), extent)
            }
        };
        if !center.is_finite()
            || !extent.is_finite()
            || !(center - extent).is_finite()
            || !(center + extent).is_finite()
        {
            return Err("Non-finite collision coordinates or derived bounds overflow".into());
        }
        Ok(())
    }
}
fn checked_extent(a: [f32; 3]) -> Result<Vec3, String> {
    let e = Vec3::from_array(a);
    if !e.is_finite() || e.min_element() < 0. {
        return Err("Invalid box half extents".into());
    }
    Ok(e)
}
pub fn checked_orientation(a: [f32; 4]) -> Result<Quat, String> {
    let q = Quat::from_array(a);
    let length = q.length_squared();
    if !q.is_finite() || !length.is_finite() || (length - 1.).abs() > 0.001 {
        return Err("Orientation must be a unit quaternion [x,y,z,w]".into());
    }
    Ok(q.normalize())
}
impl Asset {
    pub fn validate(&self) -> Result<(), String> {
        if self.format != "collision_asset" || self.version != 1 || self.units != "blocks" {
            return Err("Unsupported collision asset contract".into());
        }
        if self.asset.trim().is_empty()
            || self.asset == "."
            || self.asset == ".."
            || self.asset.contains(['/', '\\', ':'])
            || self.asset.chars().any(char::is_control)
        {
            return Err("Asset must be a non-empty package UID, not a path".into());
        }
        let valid = match self.composition {
            Composition::None => self.primitives.is_empty(),
            Composition::Single => self.primitives.len() == 1,
            Composition::Compound => self.primitives.len() > 1,
        };
        if !valid {
            return Err("Composition does not match primitive count".into());
        }
        for (index, p) in self.primitives.iter().enumerate() {
            p.validate()
                .map_err(|e| format!("Primitive {index}: {e}"))?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision_editor::Settings;
    #[test]
    fn mixed_compound_roundtrip_and_determinism() {
        let boxes: Vec<_> = [Kind::Aabb, Kind::Obb, Kind::Capsule]
            .into_iter()
            .map(|kind| crate::ObbBox {
                settings: Settings {
                    kind,
                    flags: 7,
                    mask: 23,
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect();
        let a = Asset::from_editor("chair", &boxes).unwrap();
        let bytes = serde_json::to_vec(&a).unwrap();
        assert_eq!(
            bytes,
            serde_json::to_vec(&Asset::from_editor("chair", &boxes).unwrap()).unwrap()
        );
        assert!(!String::from_utf8(bytes.clone())
            .unwrap()
            .contains("vertices"));
        let b = serde_json::from_slice::<Asset>(&bytes)
            .unwrap()
            .into_editor()
            .unwrap();
        for (old, new) in boxes.iter().zip(b.boxes) {
            assert_eq!(old.settings, new.settings);
        }
    }
    #[test]
    fn none_and_invalid_dimensions() {
        let a = Asset::from_editor("empty", &[]).unwrap();
        assert!(matches!(a.composition, Composition::None));
        assert!(a.into_editor().unwrap().boxes.is_empty());
        assert!(Asset::from_editor(
            "bad",
            &[crate::ObbBox {
                half_size: [-1., 0., 0.],
                ..Default::default()
            }]
        )
        .is_err());
    }
}

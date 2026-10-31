//! Geometry helpers kept separate from the editor UI.

use glam::Vec3;

/// Returns the positive distance at which a ray first touches an OBB.
pub fn ray_hit_distance(obb: &crate::ObbBox, origin: Vec3, direction: Vec3) -> Option<f32> {
    let rotation = obb.rotation_quat();
    let local_origin = rotation.inverse().mul_vec3(origin - obb.center_v());
    let local_direction = rotation.inverse().mul_vec3(direction);
    if obb.settings.kind == crate::collision_asset::Kind::Capsule {
        return capsule_hit(
            local_origin,
            local_direction,
            obb.settings.radius,
            obb.settings.half_segment,
        );
    }
    let extent = obb.half_size_v();
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;

    for axis in 0..3 {
        let o = local_origin[axis];
        let d = local_direction[axis];
        let e = extent[axis];
        if d.abs() < 0.000_001 {
            if o < -e || o > e {
                return None;
            }
            continue;
        }
        let mut a = (-e - o) / d;
        let mut b = (e - o) / d;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        near = near.max(a);
        far = far.min(b);
        if near > far {
            return None;
        }
    }
    if far < 0.0 {
        None
    } else {
        Some(near.max(0.0))
    }
}

/// Local capsule aligned with Y; half_segment excludes the hemispherical caps.
fn capsule_hit(o: Vec3, d: Vec3, r: f32, h: f32) -> Option<f32> {
    let closest = Vec3::new(0., o.y.clamp(-h, h), 0.);
    if (o - closest).length_squared() <= r * r {
        return Some(0.);
    }
    let mut hit = f32::INFINITY;
    let a = d.x * d.x + d.z * d.z;
    let b = o.x * d.x + o.z * d.z;
    let c = o.x * o.x + o.z * o.z - r * r;
    if a > 1e-12 {
        let det = b * b - a * c;
        if det >= 0. {
            for t in [(-b - det.sqrt()) / a, (-b + det.sqrt()) / a] {
                if t >= 0. && (o.y + t * d.y).abs() <= h {
                    hit = hit.min(t);
                }
            }
        }
    }
    for y in [-h, h] {
        let p = o - Vec3::new(0., y, 0.);
        let a = d.length_squared();
        let b = p.dot(d);
        let c = p.length_squared() - r * r;
        let det = b * b - a * c;
        if a > 1e-12 && det >= 0. {
            for t in [(-b - det.sqrt()) / a, (-b + det.sqrt()) / a] {
                let py = o.y + t * d.y;
                if t >= 0. && ((y >= 0. && py >= h) || (y <= 0. && py <= -h)) {
                    hit = hit.min(t);
                }
            }
        }
    }
    hit.is_finite().then_some(hit)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capsule_ray_uses_surface_instead_of_bounding_box() {
        assert!(capsule_hit(Vec3::new(0.24, 0.74, 1.), -Vec3::Z, 0.25, 0.5).is_none());
        assert!(
            (capsule_hit(Vec3::new(0., 0., 1.), -Vec3::Z, 0.25, 0.5).unwrap() - 0.75).abs() < 1e-6
        );
        assert!(
            (capsule_hit(Vec3::new(0., 2., 0.), -Vec3::Y, 0.25, 0.5).unwrap() - 1.25).abs() < 1e-6
        );
        assert_eq!(capsule_hit(Vec3::ZERO, Vec3::Y, 0.25, 0.5), Some(0.));
    }
}

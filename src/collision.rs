//! Geometry helpers kept separate from the editor UI.

use glam::Vec3;

/// Returns the positive distance at which a ray first touches an OBB.
pub fn ray_hit_distance(obb: &crate::ObbBox, origin: Vec3, direction: Vec3) -> Option<f32> {
    let rotation = obb.rotation_quat();
    let local_origin = rotation.inverse().mul_vec3(origin - obb.center_v());
    let local_direction = rotation.inverse().mul_vec3(direction);
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

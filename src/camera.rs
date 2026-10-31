//! Orbit camera used by the 3D preview.

use glam::{Mat4, Vec3};

#[derive(Clone)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.8,
            pitch: 0.4,
            distance: 3.2,
            target: Vec3::new(0.0, 0.5, 0.0),
        }
    }
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.target
            + Vec3::new(
                self.distance * self.yaw.cos() * self.pitch.cos(),
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.sin() * self.pitch.cos(),
            );
        Mat4::look_at_rh(eye, self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(
            45.0_f32.to_radians(),
            aspect,
            0.1,
            (self.distance * 10.0).max(100.0),
        )
    }
}

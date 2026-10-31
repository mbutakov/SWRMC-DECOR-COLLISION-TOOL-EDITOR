//! Indexed OBJ geometry. Source coordinates are transformed lazily, never copied.
use glam::{Vec2, Vec3};
use std::io::Write;
use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug)]
pub struct Corner {
    pub vertex: u32,
    /// One-based index; zero niche represents a missing UV without another word.
    pub uv: Option<NonZeroU32>,
}
#[derive(Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub triangles: Vec<[Corner; 3]>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub scale: f32,
    pub origin: Vec3,
    pub flip_z: bool,
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            scale: 1.,
            origin: Vec3::ZERO,
            flip_z: false,
        }
    }
}
impl Transform {
    pub fn point(self, p: Vec3) -> Vec3 {
        let mut p = (p - self.origin) * self.scale;
        if self.flip_z {
            p.z = -p.z;
        }
        p
    }
    pub fn validate(self) -> Result<(), String> {
        if !self.scale.is_finite() || self.scale <= 0. || !self.origin.is_finite() {
            return Err("OBJ scale must be finite and positive; origin must be finite".into());
        }
        Ok(())
    }
}
fn number(s: Option<&str>) -> Result<f32, String> {
    let n: f32 = s
        .ok_or("Missing coordinate")?
        .parse()
        .map_err(|_| "Invalid coordinate")?;
    if !n.is_finite() {
        return Err("Non-finite coordinate".into());
    }
    Ok(n)
}
fn index(s: &str, count: usize) -> Result<u32, String> {
    let n: i64 = s.parse().map_err(|_| "Invalid OBJ index")?;
    let i = if n > 0 { n - 1 } else { count as i64 + n };
    if n == 0 || i < 0 || i >= count as i64 || i > u32::MAX as i64 {
        return Err("OBJ index out of range".into());
    }
    Ok(i as u32)
}
impl Mesh {
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut mesh = Self::default();
        let mut face = Vec::new();
        let mut normals = 0;
        for (line_no, line) in content.lines().enumerate() {
            let result = (|| -> Result<(), String> {
                let mut words = line.split('#').next().unwrap_or("").split_whitespace();
                match words.next() {
                    Some("v") => {
                        let mut p = Vec3::new(
                            number(words.next())?,
                            number(words.next())?,
                            number(words.next())?,
                        );
                        if let Some(w) = words.next() {
                            let w = number(Some(w))?;
                            if w == 0. {
                                return Err("Zero homogeneous coordinate".into());
                            }
                            p /= w;
                        }
                        if !p.is_finite() {
                            return Err("Vertex overflow".into());
                        }
                        mesh.vertices.push(p);
                    }
                    Some("vt") => {
                        mesh.uvs
                            .push(Vec2::new(number(words.next())?, number(words.next())?));
                    }
                    Some("vn") => {
                        number(words.next())?;
                        number(words.next())?;
                        number(words.next())?;
                        normals += 1;
                    }
                    Some("f") => {
                        face.clear();
                        for word in words {
                            let mut parts = word.split('/');
                            let vertex = index(parts.next().unwrap(), mesh.vertices.len())?;
                            let uv = parts
                                .next()
                                .filter(|s| !s.is_empty())
                                .map(|s| {
                                    let i = index(s, mesh.uvs.len())?;
                                    i.checked_add(1)
                                        .and_then(NonZeroU32::new)
                                        .ok_or_else(|| "Too many OBJ UV coordinates".to_string())
                                })
                                .transpose()?;
                            if let Some(n) = parts.next().filter(|s| !s.is_empty()) {
                                index(n, normals)?;
                            }
                            if parts.next().is_some() {
                                return Err("Invalid face corner".into());
                            }
                            face.push(Corner { vertex, uv });
                        }
                        mesh.triangulate(&face)?;
                    }
                    _ => {} // Groups/materials do not affect collision geometry.
                }
                Ok(())
            })();
            result.map_err(|e| format!("OBJ line {}: {e}", line_no + 1))?;
        }
        if mesh.triangles.is_empty() {
            return Err("OBJ has no polygon geometry".into());
        }
        Ok(mesh)
    }
    fn triangulate(&mut self, face: &[Corner]) -> Result<(), String> {
        if face.len() < 3 {
            return Err("Face needs at least three corners".into());
        }
        if face.len() == 3 {
            self.triangles.push([face[0], face[1], face[2]]);
            return Ok(());
        }
        // Project onto the dominant normal plane and ear-clip concave polygons.
        let mut normal = Vec3::ZERO;
        for i in 0..face.len() {
            let a = self.vertices[face[i].vertex as usize];
            let b = self.vertices[face[(i + 1) % face.len()].vertex as usize];
            normal += a.cross(b);
        }
        let n = normal.abs();
        let axis = if n.x >= n.y && n.x >= n.z {
            0
        } else if n.y >= n.z {
            1
        } else {
            2
        };
        let points: Vec<Vec2> = face
            .iter()
            .map(|c| {
                let p = self.vertices[c.vertex as usize];
                match axis {
                    0 => Vec2::new(p.y, p.z),
                    1 => Vec2::new(p.z, p.x),
                    _ => Vec2::new(p.x, p.y),
                }
            })
            .collect();
        let area: f32 = (0..points.len())
            .map(|i| points[i].perp_dot(points[(i + 1) % points.len()]))
            .sum();
        if area.abs() < 1e-12 {
            return Err("Degenerate polygon".into());
        }
        let sign = area.signum();
        let cross = |a: Vec2, b: Vec2, c: Vec2| (b - a).perp_dot(c - a) * sign;
        let mut ring: Vec<usize> = (0..face.len()).collect();
        while ring.len() > 3 {
            let ear = (0..ring.len())
                .find(|&i| {
                    let a = ring[(i + ring.len() - 1) % ring.len()];
                    let b = ring[i];
                    let c = ring[(i + 1) % ring.len()];
                    cross(points[a], points[b], points[c]) > 0.
                        && !ring.iter().any(|&j| {
                            j != a
                                && j != b
                                && j != c
                                && cross(points[a], points[b], points[j]) >= 0.
                                && cross(points[b], points[c], points[j]) >= 0.
                                && cross(points[c], points[a], points[j]) >= 0.
                        })
                })
                .ok_or("Cannot triangulate polygon (degenerate or self-intersecting)")?;
            self.triangles.push([
                face[ring[(ear + ring.len() - 1) % ring.len()]],
                face[ring[ear]],
                face[ring[(ear + 1) % ring.len()]],
            ]);
            ring.remove(ear);
        }
        self.triangles
            .push([face[ring[0]], face[ring[1]], face[ring[2]]]);
        Ok(())
    }
    pub fn bounds(&self, transform: Transform) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for t in &self.triangles {
            for c in t {
                let p = transform.point(self.vertices[c.vertex as usize]);
                min = min.min(p);
                max = max.max(p);
            }
        }
        (min, max)
    }
    pub fn write_obj(
        &self,
        transform: Transform,
        out: &mut impl Write,
        textured: bool,
    ) -> std::io::Result<()> {
        if textured {
            writeln!(out, "mtllib model.mtl\nusemtl preview")?;
        }
        for &p in &self.vertices {
            let p = transform.point(p);
            writeln!(out, "v {} {} {}", p.x, p.y, p.z)?;
        }
        for uv in &self.uvs {
            writeln!(out, "vt {} {}", uv.x, uv.y)?;
        }
        for t in &self.triangles {
            write!(out, "f")?;
            let order = if transform.flip_z {
                [0, 2, 1]
            } else {
                [0, 1, 2]
            };
            for i in order {
                let c = t[i];
                write!(out, " {}", c.vertex + 1)?;
                if let Some(uv) = c.uv {
                    write!(out, "/{}", uv.get())?;
                }
            }
            writeln!(out)?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn negative_indices_uv_and_concave_faces() {
        let m = Mesh::parse(
            "v 0 0 0\nv 2 0 0\nv 2 2 0\nv 1 1 0\nv 0 2 0\nvt 0 0\nf -5/1 -4/1 -3/1 -2/1 -1/1",
        )
        .unwrap();
        assert_eq!(m.vertices.len(), 5);
        assert_eq!(m.triangles.len(), 3);
        let area: f32 = m
            .triangles
            .iter()
            .map(|t| {
                let a = m.vertices[t[0].vertex as usize];
                let b = m.vertices[t[1].vertex as usize];
                let c = m.vertices[t[2].vertex as usize];
                (b - a).cross(c - a).length() / 2.
            })
            .sum();
        assert_eq!(area, 3.);
    }
    #[test]
    fn invalid_indices_and_nonfinite_rejected() {
        for s in ["v NaN 0 0", "v 0 0 0\nf 0 1 1", "v 0 0 0\nf 1 2 3"] {
            assert!(Mesh::parse(s).is_err());
        }
    }
    #[test]
    fn transformed_export_roundtrip() {
        let m = Mesh::parse("v 1 0 0\nv 2 0 0\nv 1 1 1\nf 1 2 3").unwrap();
        let t = Transform {
            scale: 2.,
            origin: Vec3::X,
            flip_z: true,
        };
        let mut bytes = Vec::new();
        m.write_obj(t, &mut bytes, false).unwrap();
        let re = Mesh::parse(std::str::from_utf8(&bytes).unwrap()).unwrap();
        assert_eq!(m.bounds(t), re.bounds(Transform::default()));
        assert_eq!(re.triangles[0][1].vertex, 2);
    }
}

# Intermediate Collision Asset v1

`collision.json` — compact UTF-8 JSON, serialized in a fixed field order with
primitives in editor order. No absolute source paths, mesh, textures, light,
selection, camera, generation settings or debug data. This is input for an
offline collision packer, not a Java runtime binary.

```json
{"format":"collision_asset","version":1,"asset":"chair","units":"blocks","composition":"COMPOUND","primitives":[{"type":"AABB","center":[0,0.5,0],"half_extents":[0.5,0.5,0.5],"flags":0,"mask":4294967295},{"type":"CAPSULE","center":[0,1,0],"orientation":[0,0,0,1],"radius":0.125,"half_segment":0.25,"flags":0,"mask":4294967295}]}
```

- `format` is the discriminator; `version` is the contract version, independent
  of legacy OBB v2 and placed-decoration/runtime versions.
- `asset` matches `meta.json.uid`, binding every primitive to its source asset.
  The sibling `model.json` or `model.obj` supplies visual geometry, independently
  of collision data. Copying the package preserves this binding.
- `units`: `blocks`; right-handed XYZ, Y up. All positions are relative to the
  asset's local origin. Source import transforms are already applied.
- `composition`: `NONE` for zero primitives, `SINGLE` for one, `COMPOUND` for two
  or more. A Compound is a union of its primitives, including mixed types.
  NONE keeps the visual asset without physics.
- AABB: `center`, non-negative `half_extents`, `flags`, `mask`. Orientation is
  implicitly identity (omitted to avoid redundant data).
- OBB: the same fields plus `orientation`, a unit quaternion `[x,y,z,w]`, mapping
  primitive-local axes into asset-local coordinates. Editor Euler angles use
  XYZ radians internally and degrees in the inspector; export uses quaternions.
- Capsule: `center` is the midpoint of the central segment; `orientation` as
  above, `radius > 0`, `half_segment >= 0`. Local endpoints are
  `(0,-half_segment,0)` and `(0,+half_segment,0)`, each extended by a hemisphere.
  Total tip-to-tip length is `2*(half_segment+radius)`; `half_segment=0` is a
  sphere. Half extents of its local bounding box are derivable as
  `[radius, half_segment+radius, radius]` and are not redundantly stored.
- `flags` and `mask` are unsigned 32-bit integers on each primitive. Defaults:
  flags `0`, mask `4294967295`. The editor preserves bits without assigning
  engine-specific meanings. The packer/runtime defines filtering semantics.

Coordinates and dimensions must be finite. Negative box dimensions, invalid
capsule dimensions, unsupported versions, non-unit quaternions and inconsistent
composition counts are rejected. The contract can be reimported through the
collision import action without the source model. Legacy min/max and OBB v2
remain readable and are converted to the intermediate contract on export.

The packer can derive primitive bounds, quantize, build BVH8 and choose an
AoSoA/SIMD representation without knowing the source model's file format.

## Validation and reproducibility details

`Asset::validate` is independent of editor types; `collision_editor.rs` is the
adapter. Validation also rejects overflow in derived axis-aligned bounds, even
when all stored numbers are finite. `asset` must be a non-blank single package
UID, excluding `.`/`..`, separators, colon and control characters. Legacy import
requires an explicit `boxes` field; an arbitrary empty JSON object is not NONE.

Export normalizes quaternion length and chooses the sign by the first nonzero
component in `(w,x,y,z)` order, making it positive. Quaternion negative zeros
are emitted as positive zero. Primitive order remains significant for stable
indices: no sorting, deduplication, lossy merging or type conversion on export.
Repeated export of the same editor state is byte-repeatable. This does not
promise identical bytes for every mathematically equivalent Euler rotation,
across floating-point implementations, or after repeated quaternion/Euler edits.
A cache should hash the actual validated asset content plus compiler settings.

`collision.json` is streamed to a sibling temporary file, flushed and replaced
by rename. Serialization/write failure leaves the previous collision file
intact. This protects one file; the entire model/texture/metadata folder is not
an atomic transaction and power-loss durability of the directory is not promised.
Import a package only after export has completed successfully.

The v1 schema and required field names are unchanged. JSON field-name overhead
is intentional for the intermediate interchange format. Runtime allocation,
spatial indexing, quantization and cache layout belong to the compiler/runtime,
not this JSON schema. See [AI migration instructions](ai-mod-migration.md).

# Cloak fabrication assets

STL coordinates are unitless. Import the linked fabrication files as **millimeters**, at 100% scale. Dimensions below are axis-aligned bounding boxes in file coordinates, not manufacturing tolerances.

| File | Source | Dimensions (mm) |
| --- | --- | --- |
| droid_camera_mount.stl | DROID hand_camera_part_2.stl, unchanged | 71.223 × 101 × 34.107 |
| droid_camera_closing_part.stl | DROID hand_camera_part_1.stl, unchanged | 49.396 × 45 × 34.107 |
| cloak_sharpa_mount.stl | Cloak Drive, unchanged | 56.650 × 231.850 × 95.784 |
| cloak_yam_mount.stl | Cloak Drive, unchanged | 35.535 × 101 × 151.434 |
| sharpa_angled_adapter_mm.stl | MuJoCo angled right-hand adapter, vertices multiplied by 1000 | 80 × 80 × 70.879 |

## Sources and use

- [DROID shopping list](https://droid-dataset.github.io/droid/hardware-setup/shopping-list.html) links the [original camera files](https://drive.google.com/drive/folders/1k56XVdlfrXCX4iOlFlTlkoTh-2Px6CyD). Downloaded directly from that source. Part 1 is the smaller closing piece. Part 2 is the main bracket.
- [Cloak camera mounts](https://drive.google.com/drive/folders/1V5i0-rHte_-eX1d9eV8kM8b6u98uumMh), supplied by the hardware owner. Robotiq and UMI reuse the DROID bracket with different installation orientations. The closing piece is shared, as confirmed by the owner.
- The adapter source is `assets/franka_fr3_sharpa_angled/meshes/sharpa_mount_franka_right.stl`. The corresponding MJCF references it without a mesh scale. Its source coordinates are meters. `sharpa_angled_adapter_model.stl` preserves the exact source for traceability, **do not import that file as millimeters for printing**. The `_mm` file changes only vertex scale; normals, triangles and topology are preserved.
- `manifest.json` records SHA-256 checksums, triangle counts, dimensions in each file's native units, and edge incidence checks. The `_model` entry is in meters; all other entries are in millimeters.

## Mesh preflight and remaining validation

All files parse as binary STL with the expected triangle count and file length. An edge-incidence check after rounding vertex coordinates to six decimals finds zero irregular edges for the DROID closing piece and Sharpa adapter; it flags 1 for the DROID bracket, 10 for the supplied Sharpa camera bracket and 169 for the supplied YAM bracket. This is a diagnostic, not a complete manifold, self-intersection or mechanical-strength assessment. No automatic mesh repair was applied. Review slicer geometry and any repair before fabrication.

The original editable CAD, confirmed print material/settings, fastener bill of materials and installation torque specifications remain outstanding. The UMI fingertip fabrication files still need identification; simulation meshes are not automatically approved manufacturing files. YAM assembly instructions and validation are pending. Retain upstream attribution; a separate CAD redistribution/license statement was not provided with these downloads and should be clarified before the final release.

## Camera fastener diameter check

The original DROID closing piece has a 3.000 mm circular bore (45 perimeter vertices at each of y=-363.0 and -366.5 mm). The main bracket has a matching 3.000 mm holder bore and two 3.000 mm wrist holes (36 perimeter vertices each at z=0 and z=3 mm, centers x=37.6875, y approximately -295.1 and -264.9 mm). These are nominal CAD dimensions, not measured printed clearances. Combined with the hardware owner confirming the same thread size at both connections, this supports M3 fasteners. DROID specifies 10 mm holder and 30 mm wrist screw lengths. An STL does not establish thread pitch, screw grade, torque or the actual robot thread condition.

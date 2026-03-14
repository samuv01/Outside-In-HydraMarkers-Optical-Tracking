# Outside-In-HydraMarkers-Optical-Tracking



A complete Python pipeline for generating, detecting, mapping, and tracking \*\*HydraMarker\*\* fiducial patterns in an outside-in optical tracking setup. Designed for medical ultrasound probe tracking and general 6-DoF pose estimation using a single calibrated camera.



\---



\## Acknowledgements



The HydraMarker concept and original generation and detection algorithms were developed by the authors of \[HydraMarker (C++, OpenCV)](https://github.com/Rinze283/Author---HydraMarker).



The \*\*marker generation, detection and identification scripts\*\* in this repository (`generator\_hydra\_marker\_SV.py`, `GenerateMarker\_SV.py`, `ReadMarker\_SV`) are a Python port of the original C++/MATLAB implementation. All \*\*mapping (SfM), and tracking algorithms\*\* were developed independently by the author of this repository.



\---



\## Overview



The system works in four main stages:



1\. \*\*Marker Generation\*\* — Design and render printable HydraMarker checkerboard patterns with embedded dot-coded identifiers.

2\. \*\*Marker Detection \& Identification\*\* — Detect checkerboard corners in camera images and identify each corner's unique ID from the dot pattern.

3\. \*\*3D Mapping (Structure from Motion)\*\* — Reconstruct the 3D positions of all marker corners from multiple camera views using incremental SfM and bundle adjustment.

4\. \*\*6-DoF Tracking\*\* — Estimate the real-time camera-to-marker pose from a known 3D map via PnP + RANSAC.



Both \*\*single-marker\*\* and \*\*multi-marker\*\* workflows are supported throughout the pipeline.



\---



\## Repository Structure



\### Marker Generation, Detection \& Identification (ported from C++/MATLAB)



| Script | Description |

|--------|-------------|

| `generator\_hydra\_marker\_SV.py` | Core HydraMarker generator class (Python port of the C++ `generator\_HydraMarker`). Implements the fast-bWFC (backtracking Wave Function Collapse) strategy to fill a marker field ensuring unique tag identifiability under all rotations. |

| `GenerateMarker\_SV.py` | CLI entry point for generating a HydraMarker. Defines the field size, tag shapes, and rendering parameters (cell size, dot radius, supersampling). Outputs a high-resolution PNG, a print-ready PDF, the binary dot mask (`.npy` / `.mat`), and a colour-coded field image. |

| `OptimizeMarkerSize.py` | Parameter search and optimization tool. Sweeps over detection parameters (sigma, radius) to find marker grid configurations that satisfy constraints on detectability, printability, working distance, image footprint, and ID count. Supports `max\_ids`, `max\_robustness`, and `balanced` (L2 utopia compromise) objectives. |

| `duplicate\_marker\_in\_PDF.py` | Utility to compose multiple marker images onto a single letter-sized PDF page for batch printing. Supports vector-based rendering with ReportLab, die-line generation for die-cut production, and automatic column/row packing. |

| `ReadMarker\_SV.py` | \*\*Core detection pipeline (single frame).\*\* Implements the full detection chain: (1) Gaussian-blurred gradient computation and corner response filtering (`pre\_filter`), (2) sub-pixel corner refinement via quadratic surface fitting (`pt\_refine`), (3) sparse-cluster filtering, (4) Delaunay-based structural edge recovery and grid fitting (`pt\_struct`), (5) dot-pattern readout for each checkerboard cell, and (6) grid-consistent corner identification (`pt\_identify`). Returns per-corner positions, IDs, and structural edges. |

| `ReadMarker\_SV\_multi.py` | Extended detection pipeline for frames containing \*\*multiple markers simultaneously\*\*. Runs the same detection stages but matches detected corners against several STA patterns, assigning separate ID namespaces per marker. |

| `ReadMarker\_FromNPZ.py` | Batch runner for a \*\*single marker\*\*. Iterates over frames stored in a `.npz` recording, runs `ReadMarker\_SV` on each frame, and saves per-frame overlays, raw images, detection arrays (for SfM), and timing statistics. |

| `ReadMarker\_FromNPZ\_multi.py` | Batch runner for \*\*multiple markers\*\* sharing the same frame sequence. Processes each frame once, running detection against all provided STA patterns, and saves separate outputs per marker. |

| `ReadMarker\_MultiMarker.py` | CLI wrapper that scans a directory for marker sub-folders (each with its own `marker\_sta.npy`), then calls `ReadMarker\_FromNPZ\_multi` to process them in a single pass. |



\### 3D Mapping — Structure from Motion (original)



| Script | Description |

|--------|-------------|

| `sfm.py` | \*\*Full incremental SfM engine.\*\* Includes: camera calibration loading, detection preprocessing and epipolar validation, bootstrap pair selection with essential-matrix decomposition, two-view triangulation, incremental pose estimation (PnP + RANSAC) with new-marker triangulation, and bundle adjustment via PyCeres. Also provides frame-of-reference alignment (PCA + user-defined origin/axes) and metric rescaling using known cell spacing. |

| `sfm\_runner.py` | CLI script that orchestrates the complete SfM pipeline: loads calibration and detections, preprocesses frames, bootstraps the reconstruction, runs incremental registration, performs bundle adjustment, aligns to an object frame, rescales to millimetres, and saves the final `marker\_map\_aligned.npz`. |

| `sfm\_graph.py` | Diagnostic / visualization tool. Loads a reconstructed marker map and generates a 3D scatter plot of marker positions and camera poses. Also computes inter-marker neighbour distances and reports grid-spacing statistics (corner, edge, internal markers). |



\### 6-DoF Pose Tracking (original)



| Script | Description |

|--------|-------------|

| `track\_from\_map.py` | \*\*Single-marker tracking engine.\*\* Implements `MapTracker`, which matches frame detections to a known 3D map and solves camera pose via `cv2.solvePnPRansac`. Features temporal consistency checks, pose-prior seeding, Euler-angle decomposition, and optional live visualization with axis overlay. Also supports batch tracking from pre-computed detection records. |

| `track\_from\_map\_multi.py` | \*\*Multi-marker tracking engine.\*\* Extends the tracker with `MultiMapTracker` to simultaneously track multiple independently-mapped objects in the same camera stream. Reports per-object 6-DoF trajectories and relative transforms. |

| `track\_runner.py` | CLI script for batch single-marker tracking across multiple trial directories. Loads a map and camera calibration, iterates over `trial\_\*` folders containing `detections\_data.npz`, runs tracking, and saves results per trial. |

| `track\_runner\_multi.py` | CLI script for batch multi-marker tracking. Loads multiple maps and their respective detection records, initialises `MultiMapTracker`, and outputs per-object pose trajectories. |



\### Orchestration \& Automation (original)



| Script | Description |

|--------|-------------|

| `orchestrator.py` | End-to-end automation script. Given a marker folder and NPZ recordings, runs detection, SfM mapping (if no map exists), and tracking in sequence. Supports both single-marker and multi-marker configurations. Includes hardcoded calibration transforms for probe-to-ultrasound chains. |

| `orchestrator\_new.py` | Updated orchestrator with non-uniform temporal filtering (Savitzky–Golay via `smooth\_nonuniform\_gorry`), extended multi-marker automation, and refined configuration dataclasses. |



\### Analysis \& Utilities (original)



| Script | Description |

|--------|-------------|

| `jitter\_analysis.py` | Analyses translational and rotational jitter from static tracking results. Computes sample standard deviation and 95% chi-squared confidence intervals for both position (mm) and orientation (degrees). |

| `timing\_analysis.py` | Loads the `timings\_summary.csv` produced by the detection pipeline and generates bar-chart visualizations of per-stage execution times with percentage breakdowns. |

| `map\_for\_slicer.py` | Converts the SfM marker map (`marker\_map\_aligned.npz`) into 3D Slicer-compatible formats: `.mrk.json` (Markups JSON), `.fcsv` (fiducial CSV), and `.tfm` (ITK identity transform). Supports both RAS and LPS coordinate systems. |



\### Data Files



| File | Description |

|------|-------------|

| `marker\_sta.npy` | Example binary dot-pattern mask for a generated HydraMarker (NumPy array). |



\---



\## Pipeline Workflow



```

┌────────────────────────────────────────────────────────────────────┐

│                     1. MARKER GENERATION                           │

│  OptimizeMarkerSize.py  →  GenerateMarker\_SV.py  →  Print marker  │

└───────────────────────────────┬────────────────────────────────────┘

                                │

                                ▼

┌────────────────────────────────────────────────────────────────────┐

│                     2. DETECTION                                   │

│  Record NPZ frames  →  ReadMarker\_FromNPZ.py  →  detections.npz   │

└───────────────────────────────┬────────────────────────────────────┘

                                │

                                ▼

┌────────────────────────────────────────────────────────────────────┐

│                     3. 3D MAPPING (SfM)                            │

│  sfm\_runner.py  →  bootstrap + incremental + BA  →  3D map (.npz) │

└───────────────────────────────┬────────────────────────────────────┘

                                │

                                ▼

┌────────────────────────────────────────────────────────────────────┐

│                     4. 6-DoF TRACKING                              │

│  track\_runner.py  →  PnP from map  →  pose trajectories           │

└────────────────────────────────────────────────────────────────────┘

```



\---



\## Dependencies



\- Python 3.10+

\- NumPy, SciPy, OpenCV (`cv2`), Matplotlib

\- \[PyCeres](https://github.com/cvg/pyceres) — for bundle adjustment in SfM

\- Pillow (`PIL`) — for image/PDF export

\- ReportLab — for vector PDF generation (used by `duplicate\_marker\_in\_PDF.py`)

\- pandas (optional) — for CSV export in optimization results

\- psutil (optional) — for memory monitoring during generation



\---



\## Quick Start



\### 1. Generate a Marker



```bash

python GenerateMarker\_SV.py --output-dir ./my\_marker --cell-size 400 --dot-radius 0.2

```



\### 2. Detect Markers in Recorded Frames



Edit paths in `ReadMarker\_FromNPZ.py` and run:



```bash

python ReadMarker\_FromNPZ.py

```



\### 3. Build a 3D Map via SfM



Edit calibration and detection paths in `sfm\_runner.py`, then:



```bash

python sfm\_runner.py

```



\### 4. Track from the Map



Edit paths in `track\_runner.py`:



```bash

python track\_runner.py

```



\### Full Automation



For end-to-end execution (detection → mapping → tracking):



```bash

python orchestrator.py

```



\---



\## License



Please refer to the original \[HydraMarker repository](https://github.com/Rinze283/Author---HydraMarker) for licensing terms related to the marker generation, detection and identification algorithms.




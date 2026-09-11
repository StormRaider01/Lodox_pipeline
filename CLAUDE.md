# YM-03 Thesis — Claude Code Context

Project context for Claude Code when working on the image-processing pipeline and CNR/SNR validation code for this thesis. This file is scoped to the software side only (not phantom fabrication/materials).

## Thesis overview

**Title:** The Development of a Low-cost X-ray Phantom for Bone Suppression and Radiographic System Calibration (Project ID YM-03)
**Student:** Saeed Solomon, 4th-year EEE4022F/S honours, UCT
**Supervisor:** Dr Yaaseen Martin (UCT EE); co-supervisor Dr Lindie Du Plessis (UCT Biomedical Engineering, manages Lodox Statscan access)

**Problem:** The Lodox Statscan is a low-dose full-body X-ray scanner used in trauma units. Suppressing bone in the image lets radiologists focus on soft-tissue findings (tumours, fractures, infections) without the bone signal obscuring them. Commercial calibration phantoms are expensive, so this project builds a low-cost tissue-equivalent phantom and validates a dual-energy bone-suppression pipeline against it.

**Three interconnected deliverables:**
1. A tissue-mimicking anthropomorphic phantom (mid-forearm shaft design) with bone- and soft-tissue-equivalent materials matched to Lodox dual-energy attenuation characteristics.
2. A **sequential dual-energy bone suppression pipeline (SS3)** using weighted log-subtraction on image pairs acquired at two kVp levels to isolate soft tissue.
3. A **validation pipeline** computing CNR/SNR metrics, benchmarked against open-source X-ray data and theoretical models.

**Key hardware constraint:** UCT's Lodox Statscan uses a CCD detector, which only supports *sequential* (not simultaneous) multi-kVp acquisition — two exposures at different kVp, one after another, rather than one simultaneous dual-energy capture. True simultaneous dual-energy needs a photon-counting detector (PCD), which this machine doesn't have. This is why the phantom is static (eliminates motion/misregistration issues that sequential acquisition would have on a live patient) and why the pipeline is framed around sequential acquisition rather than PCD-based methods.

## Image processing pipeline (SS3)

**Goal:** Given a pair of Lodox images of the same subject/phantom acquired at two different kVp settings (low-energy and high-energy), produce a soft-tissue-only image with bone signal suppressed, using weighted log-subtraction.

**Planned stages:**
1. **Load** — read Lodox DICOM image pairs (low-kVp, high-kVp) via `pydicom`.
2. **Preprocessing / alignment** — since acquisition is sequential (not simultaneous), the two exposures may need registration/alignment before subtraction if there's any subject or detector movement between shots. For a static phantom this should be minimal but should still be checked/handled in code (not assumed away).
3. **Weighted log-subtraction (SS3)** — the core bone-suppression algorithm: take the log of each image, subtract with a weighting factor tuned so bone signal cancels and soft tissue remains. (Grounded in Brody et al. 1981, the foundational weighted-subtraction reference; Lemacks et al. 2002 covers SNR tradeoffs in dual-energy subtraction — worth consulting when choosing/justifying the weighting factor.)
4. **Output** — the resulting soft-tissue-suppressed image, ready for CNR/SNR evaluation.

**Toolchain:**
- Python, currently prototyped in Jupyter Notebook, with a planned refactor to standalone scripts.
- Libraries: `pydicom` (DICOM I/O), `OpenCV`, `scikit-image`.
- Coding style preference: modular from the start — notebook cells should call out to standalone functions (not inline scratch code), to make the eventual script refactor low-friction. When writing or reviewing pipeline code, keep functions small, pure where possible, and separated by pipeline stage (load / align / subtract / metric) so they can be lifted into scripts directly.

**Benchmarking datasets (open-source, for validating the pipeline independent of the phantom):**
- JSRT + BSE-JSRT (chest, bone-suppression paired dataset)
- IEEE DataPort bone shadow suppression dataset
- NIH ChestX-ray14 / RSNA subset
- BS-LDM (chest, SS3-relevant)
- RSNA Pediatric Bone Age Challenge, RHPE, Digital Hand Atlas (hand/forearm reference — more anatomically relevant to the forearm phantom than the chest datasets above)

## CNR/SNR validation process

**Purpose:** Quantitatively validate both (a) the bone-suppression pipeline's output quality and (b) the phantom's fidelity as a calibration tool, by measuring Contrast-to-Noise Ratio and Signal-to-Noise Ratio and comparing against open-source data and theoretical models.

**What's being measured:**
- **SNR** — signal strength relative to background noise within a region of interest (ROI), typically mean pixel intensity over the ROI divided by the standard deviation of a noise/background region.
- **CNR** — contrast between two ROIs (e.g. a bone-mimicking region vs. adjacent soft-tissue-mimicking region) relative to noise, typically `|mean_ROI1 - mean_ROI2| / noise_std`.

**Where ROIs come from:** the phantom's known material regions (bone core, muscle-mimic sleeve, outer soft-tissue layer — see materials screening below) give ground-truth locations with known target HU/attenuation values, so ROI selection should be tied to the phantom's physical geometry, not picked ad hoc from the image.

**Validation logic:**
1. Compute CNR/SNR on the pipeline's output images (post bone-suppression) using ROIs corresponding to the phantom's known material zones.
2. Compare these measured values against:
   - **Theoretical models** — expected attenuation/contrast derived from the phantom's known material densities and linear attenuation coefficients at the two Lodox kVp energies (this is why attenuation accuracy, not anatomical fidelity, is the phantom's key design parameter).
   - **Open-source X-ray data** — CNR/SNR benchmarks or reference values computed the same way on the datasets listed above, to sanity-check that the pipeline behaves consistently on real clinical/reference images, not just the custom phantom.
3. Target HU ranges the phantom materials were calibrated against: cortical bone ~600–1000 HU, cancellous/trabecular bone ~100–400 HU (materials screened via an ice-cube-tray method against CaCO₃/epoxy mixes at multiple ratios — see Akhlaghi et al. 2018, O'Reilly et al. 2020).

**Open literature gap this section of the thesis needs to fill:** the lit review currently lacks a dedicated CNR/SNR validation-metrics subsection — when generating or reviewing this content, make sure the metric definitions, ROI methodology, and the theoretical-vs-empirical comparison logic above are explicit and justified, not just the formulas.

## Working conventions for this codebase

- Project scope is sequential dual-energy only; do not suggest or assume PCD/simultaneous dual-energy approaches are in scope.

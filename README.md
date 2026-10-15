# ALINA: AI/ML-Based Automatic ROI Selection for Taxiway Yellow-Line Detection

**Assignment 4 | Question 3(a) and 3(b)**

This project extends **ALINA** by automatically selecting a **region of interest (ROI)** around airport taxiway yellow lines. The original ALINA processing pipeline is retained for downstream line detection; the main experimental variable is **how the ROI is generated**.

We investigated **five AI/ML approaches**:

1. **Zero-shot vision-language model (VLM)** — Qwen2.5-VL-3B
2. **K-Means clustering** — unsupervised color clustering
3. **Gaussian Mixture Model (GMM)** — probabilistic color clustering
4. **Hough transform + RANSAC** — geometric line estimation
5. **Random Forest pixel classification** — supervised segmentation

We also investigated **fixed vs. dynamic GMM ROIs**: use the first frame's ROI for the entire video, or regenerate an ROI on each frame.

> **Evidence status:** The four supplied evaluation CSVs contain quantitative results for K-Means, GMM, Hough + RANSAC, and Random Forest. A corresponding **zero-shot VLM evaluation CSV was not supplied**. Its results are not fabricated here; add its outputs and evaluation before claiming that all five methods were quantitatively evaluated.

---

## 1. Objectives and research questions

**Question 3(a).** Can an AI/ML algorithm replace manually selected ROIs in ALINA? Which method best balances detection output frequency and processing cost?

**Question 3(b).** Does recomputing the ROI on **every frame** improve performance compared with using one fixed ROI from the first frame?

**Main contributions**

- Implement five different approaches for automated ROI estimation.
- Connect the estimated ROIs to the original ALINA processing pipeline.
- Evaluate outputs on three video-frame sequences.
- Compare labeling rate and ALINA processing time, and discuss their limitations.
- Explore dynamic ROI selection and its computational trade-off.

### Processing flow

```text
Video frames (vidd_1, vidd_2, vidd_3)
           |
           v
First frame -> AI/ML ROI estimator -> ROI corners (BL, TL, TR, BR)
                                      |
                                      v
                             Original ALINA pipeline
                                      |
                         Perspective warp + yellow detection
                                      |
                                      v
                      Annotated frames + coordinates + logs
                                      |
                                      v
                        CSV metrics + graphs + demo video
```

For the dynamic experiment, the ROI estimator runs on **each frame**, not just the first.

---

## 2. Dataset and directory structure

Three frame sequences are used: `vidd_1`, `vidd_2`, and `vidd_3`. The uploaded evaluation CSVs report **50 processed frames per sequence**, totaling **150 frames per method**.

Suggested repository layout (include only files actually present; filenames of some evaluation scripts may differ in your checkout):

```text
ALINA/
├── alina/                            # Original ALINA source
│   ├── config.py
│   ├── pipeline.py
│   ├── roi.py
│   └── ...
├── data/
│   ├── Raw_Data/
│   │   ├── vidd_1/                  # JPG frames
│   │   ├── vidd_2/
│   │   └── vidd_3/
│   ├── rf_training_masks/          # Supervised RF annotations
│   │   ├── vidd_1.png
│   │   ├── vidd_2.png
│   │   └── vidd_3.png
│   └── ground_truth_cbem/          # Optional independent evaluation masks
├── experiments/
│   ├── kmeans_roi.py
│   ├── gmm_roi.py
│   ├── hough_ransac_roi.py
│   ├── random_forest_roi.py
│   ├── run_saved_roi.py
│   ├── gmm_dynamic_roi.py
│   ├── run_dynamic_roi.py
│   └── evaluate_dynamic_vs_fixed.py
├── notebooks/
│   └── zero_shot_vlm.ipynb         # Add actual Kaggle notebook
├── outputs/                         # Generated ROIs and ALINA outputs
├── results/
│   ├── comparison.csv
│   ├── graphs/
│   └── ...
├── scripts/
│   └── plot_results.py
├── README.md
├── requirements.txt                # Generate and pin after verification
└── .gitignore
```

### Data preparation

1. Place original video frames in `data/Raw_Data/vidd_1`, `vidd_2`, and `vidd_3`.
2. Preserve original frame names and **sort filenames consistently**.
3. Use the **first frame** of each sequence for fixed-ROI estimation.
4. For Random Forest, supply the corresponding first-frame pixel masks under `data/rf_training_masks/` (white = target line, black = background).
5. Keep evaluation ground truth **separate** from any training masks.

**Dataset distribution:** If the original dataset is not publicly redistributable, provide its authorized download/access instructions in this section before publishing the repository.

### ROI format

Each estimator exports `outputs/<method>/vidd_N_roi.txt`, containing four `(x, y)` points in **original image pixel coordinates**, ordered:

```text
BL_x BL_y
TL_x TL_y
TR_x TR_y
BR_x BR_y
```

Here BL = bottom-left, TL = top-left, TR = top-right, BR = bottom-right. `run_saved_roi.py` converts these four points to the six-point format required by ALINA.

---

## 3. Installation and environment

### Local environment (Windows / PowerShell)

Run commands from the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib opencv-python scikit-learn
```

If your repository includes a **verified, pinned** `requirements.txt`, install it instead:

```powershell
python -m pip install -r requirements.txt
```

Capture the actual working environment:

```powershell
python --version
python -m pip freeze > requirements-lock.txt
```

Record your OS, Python version, CPU, RAM, GPU (if used), and exact package versions in the final report. These specifications were not included in the uploaded evaluation CSVs and must not be guessed.

**Hardware note:** Processing high-resolution `vidd_1` frames is memory-intensive. Run one ALINA experiment at a time, avoid unnecessarily copying full-resolution arrays, and ensure sufficient free RAM/pagefile space.

### Zero-shot VLM environment

The Qwen2.5-VL-3B zero-shot experiment was run as a **Kaggle notebook** workflow. To reproduce it, upload the original notebook, enable an appropriate GPU, attach the three first-frame images, and record the model revision, prompt, decoding settings, and checkpoint used. Export predicted ROIs to the same `vidd_N_roi.txt` format before running ALINA locally.

> The exact zero-shot notebook code and successful run logs were not among the four supplied evaluation CSVs; publish the real notebook and instructions rather than substituting unverified commands.

---

## 4. Reproducibility and random seeds

**Target seed: `42`.** The earlier implementations specified `random_state=42` for stochastic K-Means, GMM, RANSAC, and Random Forest estimators. Confirm the value in the **final committed scripts**, including the dynamic GMM implementation.

At the beginning of each experiment:

```python
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

Also set `random_state=SEED` explicitly on applicable scikit-learn models. For VLM experiments, set PyTorch seeds:

```python
import torch

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

For a fully reproducible submission, also document:

- Exact frame files and their processing order.
- Model settings, preprocessing thresholds, and ROI geometry rules.
- All training masks and how they were annotated (Random Forest).
- VLM checkpoint/revision, prompt, sampling/decoding settings, and any postprocessing.
- Same ALINA configuration across all methods.
- Exact Python/package versions and hardware.

**Important:** A seed improves repeatability but does not guarantee bit-for-bit identical results across different GPU hardware or dependency versions. A zero-shot model with deterministic decoding may not involve stochastic sampling; nevertheless record its generation configuration.

### Common ALINA settings

The local diagnostic configuration used for successful GMM runs was:

```python
peak_pixel_threshold = 20
min_white_pixels = 50
mask_ignore_left_columns = 0
```

All methods **must be rerun using the same configuration** before presenting their differences as a strictly controlled benchmark. The uploaded CSVs do not prove that their original runs used identical settings. Keep all other `PipelineConfig` values constant as well.

---

## 5. Five ROI-estimation techniques

### 5.1 Zero-shot VLM: Qwen2.5-VL-3B

**Type:** pretrained vision-language model, zero-shot inference.

**How it works:** The first frame and an instruction describing the yellow taxiway line are passed to the VLM. Its coordinate prediction is parsed, checked for valid image bounds, converted to a four-corner ROI, and exported to ALINA.

**Why test it?** It requires no task-specific training data and can use visual-semantic knowledge.

**Limitations:** In development, some predicted ROIs covered unrelated regions (including sky). A failed or invalid ROI should be counted as a failed ROI-generation attempt and documented, not replaced silently with a manual ROI.

**How to reproduce:** Run the actual `notebooks/zero_shot_vlm.ipynb` on Kaggle; record prompt, checkpoint, generation parameters, and seed; export `outputs/zero_shot_vlm/vidd_N_roi.txt`; then run the same ALINA runner described below. **Numerical results are pending** because no zero-shot evaluation CSV was supplied.

### 5.2 K-Means clustering

**Type:** unsupervised clustering.

**How it works:** Resize the first frame, convert colors to LAB, cluster pixels with MiniBatch K-Means, identify clusters consistent with a yellow HSV reference, then estimate robust line-region bounds and a four-corner ROI.

**Advantages:** Simple, explainable, no training masks.

**Limitations:** May confuse yellow markings with similarly colored backgrounds or fail under changing illumination.

```powershell
python experiments/kmeans_roi.py
```

Outputs: `outputs/kmeans/vidd_N_roi.txt`, masks, previews, and `results.json`.

### 5.3 Gaussian Mixture Model (GMM)

**Type:** unsupervised probabilistic clustering.

**How it works:** Convert the first frame to LAB, fit a Gaussian mixture to sampled pixel colors, use an HSV yellow reference to select candidate pixels, and generate an ROI around the detected line region.

**Reference parameters from the implementation:** `n_components=5`, `covariance_type="diag"`, `reg_covar=1e-3`, `max_iter=100`, `n_init=3`, `random_state=42`; fit on a sampled set of pixels to reduce memory use.

**Advantages:** More flexible color modeling than hard K-Means assignment.

**Limitations:** Color-based detections can be affected by shadows and background colors.

```powershell
python experiments/gmm_roi.py
```

Outputs: `outputs/gmm/vidd_N_roi.txt`, masks, previews, and `results.json`.

### 5.4 Hough transform + RANSAC

**Type:** geometric feature detection and robust model fitting.

**How it works:** Threshold candidate yellow pixels in HSV, detect edges using Canny, find line segments using probabilistic Hough transform, and fit a robust line model with RANSAC. Use the candidate geometry to determine the ROI.

**Advantages:** Interpretable geometric model, resistant to some outliers.

**Limitations:** Curved, occluded, or weak yellow markings can be challenging for a straight-line model.

```powershell
python experiments/hough_ransac_roi.py
```

Outputs: `outputs/hough_ransac/vidd_N_roi.txt`, masks, edge previews, and `results.json`.

### 5.5 Random Forest pixel classification

**Type:** supervised pixel-level classification.

**How it works:** Annotate yellow-line pixels in first-frame masks, extract color and spatial features, train a Random Forest to classify pixels as line/background, predict a mask on an input frame, and derive the ROI.

**Advantages:** Can learn nonlinear relationships between color, position, and target pixels.

**Limitations:** Requires labeled training masks; may overfit limited training scenes.

```powershell
python experiments/random_forest_roi.py
```

**Important experimental caveat:** If first-frame masks from all three sequences were used to train the classifier and those same first frames were used to estimate ROIs, this is **not an independent generalization test**. Report this limitation. A stronger study would train on separate frames or sequences and evaluate on held-out frames.

Outputs: `outputs/random_forest/vidd_N_roi.txt`, masks, previews, and `results.json`.

---

## 6. Running the fixed-ROI ALINA experiments

For each method, first generate its `vidd_N_roi.txt` files using the method-specific command above. Then execute the corresponding ALINA runs below, **one at a time**.

`run_saved_roi.py` uses the stored ROI to process the frame sequence. The existing ALINA pipeline may show a **Confirm ROI** window; confirm the ROI to proceed. Do not manually alter the ROI if you are reporting it as an automatically generated result.

### K-Means

```powershell
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/kmeans --output-root outputs/kmeans_alina
python experiments/run_saved_roi.py --video vidd_2 --roi-dir outputs/kmeans --output-root outputs/kmeans_alina
python experiments/run_saved_roi.py --video vidd_3 --roi-dir outputs/kmeans --output-root outputs/kmeans_alina
```

### GMM

```powershell
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/gmm --output-root outputs/gmm_alina
python experiments/run_saved_roi.py --video vidd_2 --roi-dir outputs/gmm --output-root outputs/gmm_alina
python experiments/run_saved_roi.py --video vidd_3 --roi-dir outputs/gmm --output-root outputs/gmm_alina
```

### Hough + RANSAC

```powershell
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/hough_ransac --output-root outputs/hough_ransac_alina
python experiments/run_saved_roi.py --video vidd_2 --roi-dir outputs/hough_ransac --output-root outputs/hough_ransac_alina
python experiments/run_saved_roi.py --video vidd_3 --roi-dir outputs/hough_ransac --output-root outputs/hough_ransac_alina
```

### Random Forest

```powershell
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/random_forest --output-root outputs/random_forest_alina
python experiments/run_saved_roi.py --video vidd_2 --roi-dir outputs/random_forest --output-root outputs/random_forest_alina
python experiments/run_saved_roi.py --video vidd_3 --roi-dir outputs/random_forest --output-root outputs/random_forest_alina
```

### Zero-shot VLM

Once the Kaggle notebook exports valid ROI files into `outputs/zero_shot_vlm/`:

```powershell
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/zero_shot_vlm --output-root outputs/zero_shot_vlm_alina
python experiments/run_saved_roi.py --video vidd_2 --roi-dir outputs/zero_shot_vlm --output-root outputs/zero_shot_vlm_alina
python experiments/run_saved_roi.py --video vidd_3 --roi-dir outputs/zero_shot_vlm --output-root outputs/zero_shot_vlm_alina
```

A missing/invalid ROI is a **failed experiment**, not a successful run. Record it explicitly.

### ALINA outputs

Each ALINA run generates files under `outputs/<method>_alina/vidd_N/`, including annotated frames, detected coordinates, and `timing.log`. The `timing.log` contains per-frame status and elapsed processing time.

---

## 7. Evaluation metrics

### 7.1 Frame labeling rate (additional metric)

\[
\text{Labeling Rate (\%)} = \frac{\text{Frames marked Labeled}}{\text{Total processed frames}}\times100
\]

This metric tells us how often ALINA returns a line label. **It does not verify that the label matches the true yellow line.** False positives can increase this rate.

### 7.2 Detection rate / recall (ALINA paper metric)

\[
\text{Recall (\%)} = \frac{TP}{TP+FN}\times100
\]

True recall requires **independently annotated ground-truth yellow-line pixels** (e.g., CBEM-style masks) and a defined pixel-matching procedure. **Recall is not present in the supplied evaluation CSVs**, so it must be reported as **N/A**, not 0% and not the frame labeling rate.

### 7.3 ALINA processing time

\[
\text{Mean ALINA Time (ms/frame)} = \frac{1000}{N}\sum_{i=1}^{N}t_i
\]

The uploaded CSVs provide mean ALINA processing time for each video. **These values exclude the cost of ROI estimation.** They should not be labeled end-to-end times.

### 7.4 End-to-end runtime

For a fixed ROI generated once:

\[
T_{\mathrm{fixed}}\approx T_{\mathrm{ALINA}}+\frac{T_{\mathrm{ROI,first}}}{N}
\]

For dynamic ROI generated every frame:

\[
T_{\mathrm{dynamic}}\approx T_{\mathrm{ALINA}}+\frac{\sum_iT_{\mathrm{ROI},i}}{N}
\]

For a fair latency comparison, use the same hardware, frame set, and timing boundaries. The formulas omit manual confirmation, repeated program startup, and disk I/O unless separately measured.

### 7.5 ROI stability (dynamic experiment)

\[
\operatorname{IoU}(ROI_t,ROI_{t-1})=\frac{|ROI_t\cap ROI_{t-1}|}{|ROI_t\cup ROI_{t-1}|}
\]

A high IoU means the ROI changes little. **High stability is not automatically high accuracy**, because the line may genuinely move during aircraft turns.

---

## 8. Quantitative results (from supplied CSV files)

**Source:** `gmm_evaluation(1).csv`, `hough_evaluation(1).csv`, `kmeans_evaluation(1).csv`, and `random_forestevaluation(1).csv` supplied with the assignment discussion. The numbers below are copied from those files; no zero-shot metrics were supplied.

### 8.1 Frames labeled / frame labeling rate

| Method | vidd_1 | vidd_2 | vidd_3 | Overall |
|---|---:|---:|---:|---:|
| **K-Means** | 50/50 (100%) | 38/50 (76%) | 7/50 (14%) | **95/150 (63.33%)** |
| **GMM** | 49/50 (98%) | 50/50 (100%) | 36/50 (72%) | **135/150 (90.00%)** |
| **Hough + RANSAC** | 49/50 (98%) | 50/50 (100%) | 36/50 (72%) | **135/150 (90.00%)** |
| **Random Forest** | 50/50 (100%) | 49/50 (98%) | 20/50 (40%) | **119/150 (79.33%)** |
| **Zero-shot VLM** | N/A | N/A | N/A | **N/A** |

**Interpretation:** GMM and Hough + RANSAC tie for the highest overall **labeling frequency** among the four methods with supplied results: 135 of 150 frames (90%). This is **not a ranking of detection recall**.

### 8.2 Mean ALINA processing time (ms/frame; lower is better)

| Method | vidd_1 | vidd_2 | vidd_3 | Overall mean* |
|---|---:|---:|---:|---:|
| **K-Means** | 4829.18 | 274.00 | 72.44 | **1725.21** |
| **GMM** | 3745.76 | 277.52 | 1510.64 | **1844.64** |
| **Hough + RANSAC** | 3201.02 | 356.74 | 2250.98 | **1936.25** |
| **Random Forest** | 3558.18 | 487.02 | 208.56 | **1417.92** |
| **Zero-shot VLM** | N/A | N/A | N/A | **N/A** |

*Each video has 50 frames, so the overall mean is the simple average of the three per-video means. These timings measure **ALINA only**, not ROI generation. Large timing differences across videos may reflect image resolution, content, and system load.*

### 8.3 Recall (requires independent annotations)

| Method | Pixel-level recall |
|---|---|
| Zero-shot VLM | Not measured |
| K-Means | Not measured |
| GMM | Not measured |
| Hough + RANSAC | Not measured |
| Random Forest | Not measured |

### 8.4 Selecting the best method

**Provisional choice: GMM.** GMM and Hough + RANSAC achieve the same overall frame labeling rate (90%). GMM has a lower mean ALINA processing time in the supplied results (1844.64 vs. 1936.25 ms/frame). This is a **provisional choice based on the measured proxy metrics only**.

The choice must be revisited if independent recall measurements, ROI generation overhead, VLM evaluation, or reruns with strictly identical ALINA settings change the comparison. Random Forest has the lowest ALINA-only mean time, but lower labeling frequency (79.33%).

**Key limitation:** The uploaded files alone cannot prove that GMM has the highest detection accuracy.

### Graphs

Run:

```powershell
python scripts/plot_results.py
```

This creates:

- `results/graphs/labeling_rate.png` — grouped frame labeling rate by video and method.
- `results/graphs/alina_time.png` — grouped ALINA processing time by video and method.
- `results/graphs/overall_summary.png` — overall labeling rate by method.

![Frame labeling rate](results/graphs/labeling_rate.png)

![ALINA processing time](results/graphs/alina_time.png)

![Overall frame labeling rate](results/graphs/overall_summary.png)

**Graph integrity:** The zero-shot VLM is omitted from numeric plots until its measured results are available. Do not plot missing results as zero.

---

## 9. Question 3(b): Fixed ROI vs. dynamic ROI (GMM)

### Fixed ROI

1. Estimate the ROI on the **first frame** using GMM.
2. Reuse that same ROI for all 50 frames.
3. Record ALINA labels and processing time.

### Dynamic ROI

1. Run GMM separately on **every frame**.
2. Save one ROI text file per frame.
3. Feed the matching ROI to ALINA when processing that frame.
4. Record per-frame ROI-generation time, ALINA time, labeling rate, and consecutive ROI IoU.

Commands:

```powershell
python experiments/gmm_dynamic_roi.py --video vidd_1
python experiments/run_dynamic_roi.py --video vidd_1
python experiments/evaluate_dynamic_vs_fixed.py
```

For all three videos, use `--video all` for ROI generation if supported by the checked-in script, then run the dynamic ALINA runner separately for each video. The evaluation script may default to multiple videos; configure it to compare `vidd_1` only when reproducing the single-video result below.

### Observed vidd_1 comparison (previously reported in this project)

| Metric | Fixed GMM | Dynamic GMM |
|---|---:|---:|
| Frames labeled | 49/50 | 50/50 |
| Frame labeling rate | 98% | 100% |
| ALINA processing time | 3745.76 ms/frame | 3527.62 ms/frame |
| ROI-generation time | Not reported | 458.84 ms/frame |
| ALINA + ROI time | Not available | 3986.46 ms/frame |
| Mean consecutive ROI IoU | 1.0000 | 0.9701 |

**Interpretation:** Dynamic GMM increased frame labeling frequency by **2 percentage points** on `vidd_1`, but required per-frame ROI estimation. Its reported combined time (3986.46 ms/frame) exceeds the fixed method's ALINA-only time (3745.76 ms/frame); the exact fixed end-to-end comparison requires the one-time fixed ROI-generation cost. The dynamic ROI was relatively stable (mean IoU 0.9701). This does **not** establish improved pixel-level recall.

**Evidence note:** The dynamic figures above were previously reported in the project discussion; a separate dynamic comparison CSV was not included in the current uploads. Commit the generated `comparison.csv`/logs and rerun with matched GMM implementations to make the claim independently reproducible.

### Experimental control for 3(b)

The fixed and dynamic ROI generators must use **the same GMM estimator and parameters**. The only difference should be whether the ROI is computed once or for every frame. If the fixed and dynamic scripts use different segmentation rules, the experiment is confounded; regenerate the fixed baseline using the dynamic estimator on frame 1.

---

## 10. Video demonstration of the end-to-end approach

Record a short demonstration (recommended: **2–4 minutes**) showing the **selected GMM approach** from raw frame to annotated ALINA output.

**Recommended recording sequence:**

1. Show the project directory and `vidd_1`, `vidd_2`, `vidd_3` input sequences.
2. Show the first input frame and the GMM-generated yellow mask and ROI preview.
3. Execute `python experiments/gmm_roi.py` and show `vidd_N_roi.txt` being saved.
4. Execute `run_saved_roi.py` using `--roi-dir outputs/gmm`; confirm the generated ROI without manually modifying it.
5. Show a sample of the ALINA annotated output frames and line-coordinate files.
6. Show the evaluation CSV and three generated graphs.
7. Show the fixed-versus-dynamic GMM comparison, including a few consecutive ROI previews.
8. Summarize the findings and limitations (no independently measured pixel-level recall; ROI overhead).

Save the recording as `demo/alina_gmm_demo.mp4` if GitHub size limits permit, or host it elsewhere and insert a **working public or instructor-accessible link** here:

**Demo video:** `ADD_VIDEO_LINK_HERE`

Do not present a video link until the recording has actually been uploaded and verified.

---

## 11. Files needed for independent verification

To make the results auditable, commit or provide links to:

- Source code for all five methods and the ALINA runner.
- The actual zero-shot Kaggle notebook and its prompt/model metadata.
- Seeded experiment settings and pinned package versions.
- Input dataset access instructions and frame ordering.
- Random Forest training masks (if redistribution is permitted).
- Each method's `vidd_N_roi.txt`, ROI preview, and failure logs.
- ALINA `timing.log` files and generated `evaluation.csv` files.
- `results/comparison.csv`, graph-generation script, and output graphs.
- Dynamic-vs-fixed evaluation CSV and matched baseline configuration.
- A working video-demonstration link.

### Suggested `.gitignore`

```gitignore
venv/
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
*.pt
*.pth
*.ckpt
# Large source frames and generated frame sequences; adjust if small samples are permitted:
data/Raw_Data/
outputs/*_alina/*/annotated/
outputs/*_alina/*/birdseye/
```

Do not ignore your evaluation CSVs, experiment configurations, or small ROI previews if they are needed to reproduce your conclusions.

---

## 12. Limitations and future improvements

- **Frame labeling rate is a proxy**, not pixel-level recall. Add independently annotated test masks and evaluate true positives/false negatives.
- **Zero-shot VLM numerical results are missing** from the supplied evidence. Complete and report its evaluation, including failed ROI-generation cases.
- **Random Forest generalization is not proven** if training and evaluation share the same initial frames.
- **Timing excludes ROI inference** for fixed-ROI comparisons. Measure full end-to-end cost before claiming the fastest deployment method.
- **High-resolution frames and system load** can significantly affect measured processing times; document hardware and repeat runs when possible.
- **Dynamic ROI** may follow turns more closely, but can add latency or instability. Test on all sequences, not just `vidd_1`, if time permits.
- **Fairness:** Verify all historical experiments use the same ALINA thresholds and frame sets.

---

## 13. References and acknowledgments

- **ALINA paper:** [ALINA: Airport Line Navigation Assistant](https://arxiv.org/abs/2406.08775). Cite the original authors in the final report and retain the repository's original license notices.
- **Qwen2.5-VL:** Include the exact model repository, revision, and license used in the zero-shot notebook.
- **scikit-learn:** K-Means, GaussianMixture, RANSACRegressor, and RandomForestClassifier.
- **OpenCV:** Image processing, HSV thresholding, edge detection, Hough transform, perspective warping.

**Submission:** Submit the URL of the actual GitHub repository after pushing the files, verifying the README renders, and checking the video link. This README does not invent a GitHub URL.

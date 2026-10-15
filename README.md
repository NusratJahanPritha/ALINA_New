# ALINA: Automatic Taxiway ROI Selection Using AI/ML

**Assignment 4 

This project automates the selection of a **region of interest (ROI)** for the ALINA taxiway yellow-line detection pipeline. We compare five ROI-generation techniques and investigate whether updating the ROI for every frame is better than selecting it once from the first frame.

## 1. Methods

| Method | Approach | Training labels needed? |
|---|---|---|
| Zero-shot VLM (Qwen2.5-VL-3B) | Ask a pretrained vision-language model to locate the taxiway line and return an ROI | No |
| K-Means | Cluster LAB image colors and select yellow-like pixels | No |
| Gaussian Mixture Model (GMM) | Model LAB color groups probabilistically and identify yellow-like pixels | No |
| Hough + RANSAC | Find yellow edges and fit robust line geometry | No |
| Random Forest | Classify pixels as yellow-line/background using color and position features | **Yes** |

**Additional experiment:** Fixed GMM ROI versus per-frame (dynamic) GMM ROI.

The first four conventional methods use yellow-color information, either as a selection cue or a preprocessing filter. Random Forest requires annotated training masks. The zero-shot VLM was tested separately on Kaggle.

## 2. Data and project layout

The evaluation uses three image sequences: `vidd_1`, `vidd_2`, and `vidd_3` (**50 frames each** in the supplied evaluation files). Preserve the original JPG filenames and their sorted order.

```text
ALINA/
├── alina/                       # Existing ALINA code
├── data/
│   ├── Raw_Data/
│   │   ├── vidd_1/              # JPG frames
│   │   ├── vidd_2/
│   │   └── vidd_3/
│   └── rf_training_masks/       # vidd_1.png, vidd_2.png, vidd_3.png
├── experiments/
│   ├── kmeans_roi.py
│   ├── gmm_roi.py
│   ├── hough_ransac_roi.py
│   ├── random_forest_roi.py
│   ├── run_saved_roi.py
│   ├── gmm_dynamic_roi.py
│   ├── run_dynamic_roi.py
│   └── evaluate_dynamic_vs_fixed.py
├── notebooks/                   # Add the actual zero-shot VLM notebook
├── outputs/                     # Generated ROI files and ALINA outputs
├── results/
│   ├── comparison.csv
│   ├── graphs/
│   └── *_evaluation.csv
├── scripts/plot_results.py
└── README.md
```

The `notebooks/` folder and other paths above describe the intended submission layout. Ensure the corresponding scripts/files are actually committed before publishing.

### ROI file format

Each method saves `outputs/<method>/vidd_N_roi.txt` with four `(x, y)` coordinates in the **original frame resolution**, ordered:

```text
bottom-left
 top-left
 top-right
bottom-right
```

`run_saved_roi.py` converts this format into the six-point ROI representation used by ALINA.

## 3. Installation

Use Python in a Windows PowerShell terminal from the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install numpy opencv-python scikit-learn pandas matplotlib
```

Also install any dependencies required by the original ALINA code. To capture the **actual working environment**, run:

```powershell
python --version
python -m pip freeze > requirements-lock.txt
```

For the zero-shot VLM, use the Kaggle GPU notebook with its own model dependencies and model weights. Record the exact checkpoint, prompt, and decoding settings in that notebook.

## 4. Reproducibility

The classical ML scripts use **random seed 42** for their stochastic estimators (`random_state=42`). Keep the same seed for every repeat:

```python
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

Pass `random_state=SEED` explicitly to K-Means, GMM, RANSAC, and Random Forest. For the GPU VLM, set PyTorch seeds and document model version and decoding settings; verify these in the submitted notebook. Seeds reduce variation but cannot guarantee identical results across different GPU hardware or library versions.

**Fair-comparison controls:** Use identical input frames, ALINA configuration, ROI coordinate convention, and evaluation scripts for all methods. The adjusted ALINA settings used in the local experiments were:

```python
peak_pixel_threshold = 20
min_white_pixels = 50
mask_ignore_left_columns = 0
```

Confirm the final runner uses these values consistently. If an older result was generated with different thresholds, rerun it before treating the comparison as controlled.

## 5. Reproduce the experiments

### Step 1 — Generate the first-frame ROIs

From the project root:

```powershell
python experiments/kmeans_roi.py
python experiments/gmm_roi.py
python experiments/hough_ransac_roi.py
python experiments/random_forest_roi.py
```

**Random Forest prerequisite:** Provide the first-frame training masks in `data/rf_training_masks/` (white = yellow-line pixel; black = background). Its evaluation on training frames should not be interpreted as an independent test of generalization.

**Zero-shot VLM:** Run the submitted Kaggle notebook on the same three first frames. Export valid ROIs as `outputs/zero_shot_vlm/vidd_1_roi.txt`, `vidd_2_roi.txt`, and `vidd_3_roi.txt`. Keep failed or invalid predictions as documented failures, rather than replacing them manually without disclosure.

### Step 2 — Run ALINA for each ROI method

Run one command at a time and confirm the ROI in the ALINA preview window:

```powershell
# K-Means
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/kmeans --output-root outputs/kmeans_alina

# GMM
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/gmm --output-root outputs/gmm_alina

# Hough + RANSAC
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/hough_ransac --output-root outputs/hough_ransac_alina

# Random Forest
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/random_forest --output-root outputs/random_forest_alina

# Zero-shot VLM (after exporting its ROI files)
python experiments/run_saved_roi.py --video vidd_1 --roi-dir outputs/zero_shot_vlm --output-root outputs/zero_shot_vlm_alina
```

Repeat each command with `vidd_2` and `vidd_3` in both the `--video` argument and the final output folder (for example, `outputs/gmm_alina` stays unchanged as the output root; the runner creates its per-video subfolder).

ALINA saves annotated images, detected coordinates, and timing logs under `outputs/<method>_alina/vidd_N/`.

### Step 3 — Evaluate and plot

The supplied per-method evaluation scripts read the ALINA timing logs and save CSV summaries. Run the versions included in your repository, for example:

```powershell
python experiments/evaluate_kmeans.py
python experiments/evaluate_gmm.py
```

For Hough + RANSAC and Random Forest, use their corresponding evaluation scripts if included, or adapt the same evaluator by changing its ALINA input directory and CSV output path. Check the paths before running.

The supplied `results/comparison.csv` contains the measurements used below. Regenerate the figures using:

```powershell
python scripts/plot_results.py
```

Figures are saved in `results/graphs/`.

## 6. Evaluation metrics

**Frame labeling rate (additional metric)** — how often ALINA returns a line label:

```text
Labeling rate (%) = 100 × labeled frames / processed frames
```

A labeled frame is **not necessarily a correctly detected line**.

**Detection recall (ALINA paper metric)** — how many actual yellow-line pixels are detected:

```text
Recall (%) = 100 × true-positive pixels / (true-positive + false-negative pixels)
```

This requires **independent, manually annotated ground-truth masks** and a consistent pixel-matching rule. No such recall values are present in the supplied CSVs; report **N/A**, not 0%.

**ALINA processing time (ms/frame)** — mean time per frame reported in the ALINA logs. It **excludes ROI generation** and should not be described as full end-to-end latency.

**Total algorithm time (ms/frame)** — for fixed ROI, add the one-time ROI generation cost divided by the number of frames; for dynamic ROI, add per-frame ROI generation time. Exclude manual confirmation time unless measuring full wall-clock execution.

## 7. Experimental results

The following values are from the four supplied evaluation CSVs; all times are **ALINA-only** averages, in milliseconds per frame.

| Method | vidd_1 labeled | vidd_2 labeled | vidd_3 labeled | Total labeled | Overall labeling rate | Mean ALINA time* |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot VLM | N/A | N/A | N/A | N/A | N/A | N/A |
| K-Means | 50/50 | 38/50 | 7/50 | 95/150 | 63.33% | 1725.21 |
| GMM | 49/50 | 50/50 | 36/50 | **135/150** | **90.00%** | 1844.64 |
| Hough + RANSAC | 49/50 | 50/50 | 36/50 | **135/150** | **90.00%** | 1936.25 |
| Random Forest | 50/50 | 49/50 | 20/50 | 119/150 | 79.33% | **1417.92** |

*Unweighted average of the three per-video mean ALINA times. Each video has 50 evaluated frames, so this also equals the frame-weighted mean for these CSVs.*

**Interpretation:** GMM and Hough + RANSAC have the highest **labeling frequency** (90%). GMM is faster than Hough + RANSAC on the reported ALINA-only timing. Random Forest has the lowest reported ALINA-only time but labels fewer frames. These data **do not establish which method has the highest detection recall**, and do not include ROI-generation time.

**Provisional choice:** **GMM**, based on its 90% labeling rate and lower ALINA-only time than Hough + RANSAC. The choice should be revisited when zero-shot VLM results, independent recall, and full ROI-generation costs are available.



## 8. Question 3(b): Fixed vs. dynamic ROI

For **fixed GMM**, generate an ROI from the first frame and reuse it for all 50 frames. For **dynamic GMM**, generate a new ROI for each frame and feed it to ALINA.

```powershell
# Generate an ROI for every frame
python experiments/gmm_dynamic_roi.py --video vidd_1

# Run ALINA using the matching per-frame ROIs
python experiments/run_dynamic_roi.py --video vidd_1

# Compare with the fixed-GMM run
python experiments/evaluate_dynamic_vs_fixed.py
```

The previous `vidd_1` comparison produced:

| Metric | Fixed GMM | Dynamic GMM |
|---|---:|---:|
| Frames labeled | 49/50 | 50/50 |
| Labeling rate | 98% | 100% |
| ALINA time (ms/frame) | 3745.76 | 3527.62 |
| Dynamic ROI generation (ms/frame) | — | 458.84 |
| Dynamic ALINA + ROI (ms/frame) | — | 3986.46 |
| Mean consecutive ROI IoU | 1.0000 | 0.9701 |

**Finding:** Dynamic ROI increased labeling frequency by **2 percentage points**, but its additional ROI-generation work increased the reported algorithmic cost relative to fixed ALINA-only time. A fair end-to-end comparison still needs the fixed method's one-time ROI generation cost. Consecutive ROI IoU of 0.9701 indicates relatively small frame-to-frame changes. Improved true detection accuracy is **not proven** without ground truth.

**Control:** Use the **same GMM estimator and ALINA settings** in both fixed and dynamic experiments. Otherwise, the observed difference cannot be attributed solely to ROI update frequency.

## 9. Limitations and reproducibility status

- **Zero-shot VLM:** The evaluation CSV was not provided. Its qualitative failures should be reported, and its numerical evaluation must be added to substantiate five fully evaluated methods.
- **Accuracy:** Frame labeling rate is not recall. Independent ground-truth line masks are needed for pixel-level detection evaluation.
- **Runtime:** Current method-comparison times exclude ROI-generation cost. Report that cost separately before claiming end-to-end speed superiority.
- **Random Forest:** Training masks and held-out test annotations must be distinguished to avoid data leakage.
- **Consistency:** Verify the same ALINA configuration was used for all stored CSVs; otherwise rerun with identical settings.
- **Dependencies:** Commit the actual VLM notebook, the working evaluation scripts, and a tested dependency lockfile for repeatable execution.

## 10. References

- ALINA paper: https://arxiv.org/abs/2406.08775
- Qwen2.5-VL model family: https://huggingface.co/Qwen
- scikit-learn documentation: https://scikit-learn.org/stable/
- OpenCV documentation: https://docs.opencv.org/



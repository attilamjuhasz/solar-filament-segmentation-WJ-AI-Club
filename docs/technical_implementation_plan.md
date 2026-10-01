**Recommended approach:** build Option B first: native-resolution tiled segmentation with a ConvNeXt-Large U-Net, explicit boundary and instance-center supervision, topology-aware losses, and conservative instance reconstruction. Benchmark Option A under the same folds, evaluator, and inference budget.

The strongest initial improvements are annotation correctness, native-resolution geometry, and instance-aware validation. Larger backbones and foundation models should follow those fixes.

I inspected the two local U-Net exports, verified that their code matches the corresponding Kaggle kernels, and recovered the requested “LB 1st” kernel through Kaggle’s public API. The additional local 0.70 Mask R-CNN/refiner notebook provides useful inference practices, but belongs to a different author and pipeline.

All proposed hyperparameters below are starting configurations. Model-performance improvements require grouped out-of-fold experiments; the available workspace contains notebooks and competition-page exports, rather than the training images and annotations.

## 1. Baseline Gap Analysis & Synthesis

### 1.1 What the three requested kernels actually implement

The kernels share the same substantive implementation:

| Kernel | Version inspected | Model | Training resolution | Epochs | Learning rate |
|---|---:|---|---:|---:|---:|
| [rokaiyasomapti / LB 1st](https://www.kaggle.com/code/rokaiyasomapti/lb-1st-solar-filament-segmentation-2026) | 7 | ImageNet ResNet34 + SMP U-Net | 512×512 | 35 | `3e-3` |
| [foysalemonshanto / solar-filament-segmentation](https://www.kaggle.com/code/foysalemonshanto/solar-filament-segmentation) | 8 | Same | 512×512 | 20 | `2e-4` |
| [pavloivanin / 0.9 baseline](https://www.kaggle.com/code/pavloivanin/0-9-solar-filament-baseline-pipeline-u-net) | 1 | Same | 512×512 | 5 | `3e-4` |

Their common pipeline uses:

- An 80/20 random filename split.
- Polygon-oriented mask parsing.
- Horizontal and vertical flips.
- ImageNet normalization.
- Equal-weight BCE and Dice losses.
- Checkpoint selection by validation loss.
- Thresholding at 0.5.
- Connected components with a 30-pixel minimum area.
- Compressed COCO RLE encoding.

These are variations of one baseline, so comparing their leaderboard labels does not establish architectural superiority.

### 1.2 Verified defects and conditional risks

| Finding | Evidence and consequence | Required correction |
|---|---|---|
| **Incorrect submission geometry** | All three encode the 512×512 model mask directly. No restoration to 2048×2048 occurs before encoding. | Restore probabilities to native coordinates, extract native-resolution instances, then encode. |
| **Annotation-schema assumption** | The dataset calls `annotations.get(filename, [])`. The parser handles coordinate lists and a limited polygon dictionary. Missing keys and parsing exceptions silently become empty masks. | Build an explicit annotation index; support the actual competition schema and compressed/uncompressed COCO RLE. Reject unresolved annotations. |
| **Instance identities disappear during target construction** | All polygons are filled into one binary mask. | Preserve per-annotator instance masks alongside the semantic union. |
| **No explicit observation grouping** | The filename split neither canonicalizes annotator prefixes nor checks duplicate observations. It also supplies no stratification labels. | Persist canonical observation groups and five stratified group folds. |
| **Thin structures lose resolution** | Full-disk reduction from 2048 to 512 compresses a two-pixel barb to half a pixel. | Train and infer on native-resolution patches with overlapping reconstruction. |
| **Loss lacks structural supervision** | BCE and Dice optimize occupancy and overlap, with no explicit spine-continuity or instance-separation objective. | Add clDice, boundary supervision, and instance-center targets. |
| **Validation does not measure the submission objective** | Training reports validation loss, without native-resolution PQ, split/merge diagnostics, or submission round trips. | Evaluate the complete inference and instance-formatting pipeline. |
| **Scale-dependent filtering** | A 30-pixel cutoff at 512 corresponds approximately to 480 native pixels under 4× enlargement. | Express filtering parameters in native pixels and calibrate them on validation data. |
| **Artificial empty-instance rows** | No detections cause insertion of an all-zero RLE as a purported filament. | Represent no detections as an empty instance set; confirm the current container’s empty-image protocol before release. |
| **Weak reproducibility** | Dependencies are installed without exact versions; checkpoints contain only model weights. | Lock the environment and save model, optimizer, scheduler, scaler, RNG, configuration, folds, and provenance. |

The annotation failure mechanism is confirmed in the code. Whether it produced entirely empty training targets depends on the competition JSON, which is absent locally. Likewise, a random filename split **can** preserve annotator variants if every physical observation has exactly one image file; leakage must be established through the manifest rather than assumed.

MAGFiLO’s published release uses COCO-style collections, making schema inspection particularly important. The competition export must be checked against that structure before adapting it. [MAGFiLO data descriptor](https://www.nature.com/articles/s41597-024-03876-y)

### 1.3 Memory assessment

There is **no demonstrated retained-autograd-graph leak** in these three kernels: they accumulate `loss.item()` and use `torch.no_grad()` for validation and inference.

The production plan should address these actual memory risks:

1. **Worker duplication:** Windows worker processes can duplicate large annotation dictionaries. Workers should receive compact image/annotation indices and decode only required masks.
2. **Dense instance stacks:** Native masks cost about 16 MiB each as float32. A metric implementation holding 100 GT and 100 prediction masks requires approximately 3.125 GiB before temporary buffers.
3. **Unbounded caches:** Keep image and mask caches bounded; avoid caching every augmented crop.
4. **Fold-to-fold retention:** Dispose of models, optimizers, EMA copies, loaders, and predictors between sequential fold jobs.
5. **Foundation-model caches:** Reset image embeddings when an observation or crop changes.
6. **Logging:** Store detached scalar summaries and compressed masks rather than GPU tensors.

Use AMP, activation checkpointing, `zero_grad(set_to_none=True)`, CPU accumulation of inference maps, and RLE-based overlap calculations. CUDA reserved memory is allocator behavior; rising allocated memory after warm-up is the relevant leak signal.

### 1.4 Metric correction that changes the engineering priorities

The competition changed leaderboard ranking from Dice to PQ in August and re-scored submissions. Historical “0.9,” “0.70,” and “1st” labels require revalidation under the current evaluator. [Competition ranking and announcements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking)

The organizer’s self-evaluation notebook, version 6, establishes a more precise local contract:

- Evaluate each **annotator–image entry** separately against the same image prediction.
- Mark every pair with **IoU strictly greater than 0.5** as a hit.
- Count predictions with no qualifying hit as FP.
- Count GT masks with no qualifying hit as FN.
- Aggregate IoU sums and counts globally.
- Return zero when the PQ denominator is zero.
- It performs **no Hungarian matching**. [Organizer self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)

For entry \(e\), define \(I_{eij}=\operatorname{IoU}(G_{ei},P_{ej})\) and:

\[
H_{eij}=\mathbf{1}[I_{eij}>0.5].
\]

Then:

\[
\operatorname{PQ}_{\mathrm{reference}}
=
\frac{\sum_{eij}H_{eij}I_{eij}}
{\sum_{eij}H_{eij}
+\frac12\sum_{ej}\mathbf{1}[\sum_iH_{eij}=0]
+\frac12\sum_{ei}\mathbf{1}[\sum_jH_{eij}=0]}.
\]

Preserve this behavior in the competition adapter. Separately report conventional one-to-one instance matching for scientific diagnosis.

A perfect merger of two equal disjoint GT instances gives each an IoU of exactly 0.5 and therefore receives no qualifying match. This makes instance separation central to the plan.

**Upgrade sequence:** correct data and geometry → grouped evaluation → topology and boundary supervision → native resolution → stronger features → calibrated instance reconstruction → selective ensembling.

## 2. Robust CV Strategy & Leakage Prevention

### 2.1 Define three separate identities

Maintain these fields throughout the data pipeline:

| Identity | Meaning |
|---|---|
| `observation_id` | Physical image, such as `20150125172714Mh` |
| `annotator_image_id` | Annotation variant, such as `010401-20150125172714Mh` |
| `instance_id` | One filament within that variant |

Keep the complete timestamp and station suffix. Grouping only by calendar date would unnecessarily combine distinct observations.

A strict canonicalizer should accept image filenames and annotation identifiers:

```python
import re
from datetime import datetime

def canonical_observation_id(value: str) -> str:
    name = re.split(r"[\\/]", str(value))[-1]
    name = re.sub(
        r"\.(?:jpe?g|png|fits?|fts)(?:\.gz)?$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"_\d+$", "", name)
    name = re.sub(r"^\d{6}-", "", name)

    if not re.fullmatch(r"\d{14}[A-Za-z]{2}", name):
        raise ValueError(f"Unrecognized observation identifier: {value}")

    datetime.strptime(name[:14], "%Y%m%d%H%M%S")
    return name
```

Thus both `010401-...` and `010402-...` map to the same observation.

Unknown identifier formats should produce an audit error. Add a documented adapter when the actual dataset requires another format.

### 2.2 Build the manifest before generating patches

Create a manifest containing:

```text
observation_id
annotator_image_id
image_path
image_height, image_width
image_file_sha256
decoded_pixels_sha256
annotation_source_sha256
instance_count
union_mask_area
disk_area
foreground_fraction
station
observation_timestamp
leakage_group
fold
```

Compute mask area from decoded masks, rather than trusting stored annotation metadata.

Use union area within each annotation variant so overlapping masks do not double-count pixels. Compute observation-level count and area statistics as the median across its annotator variants.

Additional grouping safeguards:

- Identical decoded-pixel hashes must share a leakage group even when filenames differ.
- Investigate near-duplicate images using perceptual similarity and timestamps.
- Join verified aliases before fold assignment.
- Keep all patches, augmentations, and annotation variants within their parent group.

Perceptual similarity alone should trigger review; it should not automatically merge unrelated solar observations.

### 2.3 Five-fold stratified group assignment

Construct a table with one row per independent leakage group.

Use two stratification variables:

\[
n_g=\operatorname{median}_a(\text{instance count}_{g,a}),
\]

\[
a_g=\operatorname{median}_a
\left(
\frac{\text{union mask area}_{g,a}}{\text{disk area}_g}
\right).
\]

Starting stratification:

1. Separate zero-filament observations when they exist.
2. Divide positive filament counts into three quantile bins.
3. Divide `log1p(mask_area)` into three quantile bins.
4. Form the joint label `count_bin:area_bin`.
5. Coarsen adjacent bins until every retained stratum contains at least five independent groups.

Apply:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026,
)
```

Persist assignments once. Reuse the exact folds for every ablation. Group isolation takes priority over perfect class balance. [StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

Validate:

- Each observation and image hash appears in exactly one validation fold.
- No training/validation group intersection exists.
- Each observation receives exactly one OOF prediction.
- All annotation variants inherit the same fold.
- Fold distributions of count, area, station, year, and annotator multiplicity are reported.

### 2.4 Handle annotator disagreement without creating artificial labels

For training:

- Sample observations with approximately equal probability.
- Select one annotation variant for that observation.
- Retain its original instance identities.
- Avoid treating every annotator record as an independent image.
- Avoid unioning all annotators’ masks: disagreement can create false bridges and inflate foreground.

Optional consensus experiments should first match instances across annotators. Ambiguous merges and splits should remain separate supervision cases or receive reduced confidence.

For evaluation, generate one prediction per observation and compare it with every annotator variant, as the reference notebook does.

Report both the reference weighting and an observation-balanced diagnostic average.

### 2.5 Prevent calibration and ensemble leakage

For final confirmation, choose thresholds and post-processing settings using an inner grouped calibration split within the outer training fold. Reserve the outer validation fold for the final prediction.

Development experiments may use ordinary validation-driven selection, but label those results as development CV.

A particularly important ensemble rule:

> For OOF image \(x\) in fold \(f\), ensemble only models trained with fold \(f\) excluded.

Averaging all five fold checkpoints on an OOF image leaks training exposure through four of those checkpoints. Averaging all five on the test set is appropriate.

Also run a temporal robustness experiment with day/block grouping and a short exclusion buffer. Exact-observation grouping removes annotation duplication; temporal blocking tests persistence of similar physical filaments across observations.

## 3. Domain-Specific Preprocessing & Augmentations

### 3.1 Separate the JPEG and raw-FITS paths

MAGFiLO JPEG conversion already removes large-scale intensity variations through Fourier-based processing. Consequently, aggressive limb correction applied again may damage useful contrast. [MAGFiLO conversion procedure](https://www.nature.com/articles/s41597-024-03876-y)

Use these branches:

| Input | Default treatment |
|---|---|
| Supplied competition JPEG | Robust normalization; further limb correction and CLAHE are controlled ablations |
| Supplied raw FITS | Physical-value loading, disk detection, robust limb/background correction |
| External raw counterpart of a test JPEG | Excluded from competition inference |

The competition requires inference from the supplied test H-alpha images and restricts additional ground-truth metadata. Derive boxes, centers, boundaries, and skeletons from permitted segmentation masks. [Competition evaluation conditions](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking)

### 3.2 Establish native geometry

For each image:

1. Decode to a two-dimensional intensity array.
2. Check shape, finite values, clipping, and bit depth.
3. Estimate disk center and radius from valid FITS headers when available.
4. Otherwise fit a circle or ellipse to the image’s limb.
5. Create a disk/validity mask.
6. Preserve the original pixel coordinate system.

Keep a small limb tolerance rather than clipping everything beyond \(0.98R\); that would remove near-limb filaments.

FITS loading must apply scaling correctly and close file handles. Use a context manager and materialize a bounded float32 image. Verify any FITS/JPEG orientation mapping with an asymmetric registration fixture before transferring labels.

### 3.3 Robust limb correction for raw images

Let:

\[
r(x,y)=\frac{\sqrt{(x-c_x)^2+(y-c_y)^2}}{R},
\qquad
\mu=\sqrt{\max(0,1-r^2)}.
\]

Estimate a smooth intensity profile using robust annular statistics:

\[
L(\mu)=a_0+a_1\mu+a_2\mu^2.
\]

Starting procedure:

- Annuli approximately 16 native pixels wide.
- Robust medians with intensity outliers suppressed.
- Fit primarily inside \(r<0.95\).
- Extrapolate conservatively into the outer ring.
- Bound the correction gain to avoid amplifying limb noise.

With estimated background \(b\):

\[
I_{\mathrm{corr}}(x,y)
=
\frac{I(x,y)-b}
{\max(L(\mu)-b,\epsilon)}
\operatorname{median}(L-b).
\]

Estimate the profile from image intensities alone in both training and inference.

Retain the raw view. A preprocessing transform that accidentally removes broad filament absorption should not erase the only available representation of that signal.

### 3.4 Background homogenization and CLAHE

After limb correction, optionally estimate a slowly varying residual illumination field with a robust smoother at approximately 64–128 native-pixel scale.

Prefer an additional normalized view over aggressive replacement of the original intensity image.

CLAHE starting configuration:

```text
clipLimit = 2.0
tileGridSize = (16, 16) on the native 2048×2048 image
```

Apply CLAHE to the full observation before patch extraction. Independent per-patch contrast adjustment creates seams and train/inference inconsistencies.

Compare:

1. Replicated original grayscale.
2. Replicated corrected grayscale.
3. Three views: original, corrected, CLAHE-adjusted.

Normalize intensities using robust on-disk percentiles, such as the 1st and 99th percentiles. Then apply the chosen pretrained backbone’s normalization consistently. Any dataset-level statistics must come from the training fold.

### 3.5 Native patches plus global context

Use native patches for morphology and a global view for context.

**Starting local configuration**

```text
Training patch: 768×768 native pixels
Inference tile: 768×768 native pixels
Inference stride: 512 pixels
Minimum overlap: 256 pixels
Output canvas: 2048×2048
```

For a 2048-pixel axis, use starts:

```text
[0, 512, 1024, 1280]
```

This produces 16 tiles with complete coverage.

Blend probabilities before thresholding:

\[
P(x,y)
=
\frac{\sum_t W_t(x,y)P_t(x,y)}
{\sum_t W_t(x,y)+\epsilon},
\]

where \(W_t\) is a Hann/cosine window with a small positive floor.

Use the same reconstruction for boundaries, center heatmaps, and offset fields.

**Global configuration**

- One 1024×1024 full-disk view.
- Supervise broad semantic context and instance-center localization.
- Use native local views for barb and topology losses.
- Start with 80% local training steps and 20% global steps, using homogeneous batch shapes.

Large filaments can extend beyond a tile. Stitch the full probability and marker maps before watershed or final connected-components analysis.

### 3.6 Patch sampling

Starting local sampler:

| Sample type | Probability |
|---|---:|
| Filament-centered | 50% |
| Boundary/barb-centered | 20% |
| Hard background: network, sunspots, limb | 20% |
| Uniform on-disk | 10% |

Derive barb/spine sampling locations from the permitted masks.

For validation, scan the full disk uniformly. Use model predictions rather than GT locations to select inference regions.

Preserve global instance IDs through cropping. Recompute auxiliary targets after spatial transformation and ignore padded regions.

### 3.7 Physics-consistent augmentation policy

Prioritize rigid transformations, which preserve pixel topology most reliably.

| Transformation | Starting setting | Constraint |
|---|---|---|
| Rotations by multiples of 90° | Uniform D4 sampling | Exact mask geometry |
| Horizontal/vertical reflections | Included in D4 | Segmentation labels only |
| Small arbitrary rotations | Approximately ±15°, probability 0.25 | Joint image/mask transform |
| Brightness/contrast | Brightness ±0.10; contrast ±0.15 | Preserve dark-filament polarity |
| Gamma | Approximately 0.85–1.15 | Avoid saturation |
| Mild seeing blur | Native-pixel \(\sigma\) approximately 0.2–1.0 | Preserve discernible small structures |
| Mild noise | Approximately 0.5–2% intensity range | Image only |
| Elastic deformation | Low probability; smooth, about ≤2-pixel displacement | No folding or instance disappearance |
| Grid distortion | Low probability; small displacement | Positive local Jacobian; joint labels |

Implement with a pinned Albumentations version. Use image interpolation appropriate to intensities and nearest-neighbor interpolation for discrete instance IDs.

Validate augmentation geometry on one- to three-pixel branched structures. Regenerate skeleton and boundary targets after deformation.

Reflections are acceptable for this segmentation objective, but chirality changes under reflection; chirality labels should not enter this pipeline.

Keep MixUp, CutMix, and unrestricted elastic warps disabled initially because their interaction with physical instance identity and topology complicates interpretation.

## 4. Model Architecture & Pipeline Design

### 4.1 Controlled comparison

| Criterion | Option A: detect then refine | Option B: dense or query-based segmentation |
|---|---|---|
| Instance identity | Explicit proposals | Learned markers/boundaries or mask queries |
| Recall ceiling | Limited by localization recall | Full-disk dense predictions cover every region |
| Thin-detail recovery | Strong when crops retain native detail | Strong with native patches and shallow detail features |
| Crowded filaments | Proposal suppression and overlapping crops need care | Marker quality or query separation becomes critical |
| Operational complexity | Multiple models and coordinate systems | Simpler for semantic model; higher for query model |
| Recommended role | Controlled alternative and possible proposal source | Primary development path |

### 4.2 Option B1: ConvNeXt-Large with a full-resolution U-Net

**Checkpoint**

```text
timm/convnext_large.fb_in22k_ft_in1k_384
```

**Encoder**

| Stage | Stride | Channels | Blocks |
|---|---:|---:|---:|
| Stem + stage 1 | 4 | 192 | 3 |
| Stage 2 | 8 | 384 | 3 |
| Stage 3 | 16 | 768 | 27 |
| Stage 4 | 32 | 1536 | 3 |

ConvNeXt blocks use depthwise spatial convolution, normalization, channel expansion, GELU, and residual connections. The checkpoint provides ImageNet-22K pretraining followed by ImageNet-1K fine-tuning. [ConvNeXt implementation](https://github.com/facebookresearch/ConvNeXt), [checkpoint](https://huggingface.co/timm/convnext_large.fb_in22k_ft_in1k_384)

**Shallow detail branch**

A stride-four pretrained stem alone cannot reliably preserve a one- or two-pixel barb. Add:

```text
D1: two 3×3 convolutions, 3 → 32 → 32, stride 1
D2: 3×3 stride-2 convolution, 32 → 64,
    followed by a 3×3 convolution
```

Use GroupNorm and SiLU in the new decoder/detail layers. GroupNorm avoids reliance on large training batches.

**Decoder**

| Output scale | Inputs | Output channels |
|---|---|---:|
| 1/16 | Upsampled encoder stage 4 + stage 3 | 512 |
| 1/8 | Previous output + stage 2 | 256 |
| 1/4 | Previous output + stage 1 | 128 |
| 1/2 | Previous output + D2 | 64 |
| Full resolution | Previous output + D1 | 32 |

Each block contains two 3×3 convolution–GroupNorm–SiLU operations. Use bilinear upsampling with explicitly aligned output sizes.

**Prediction heads**

Apply separate 1×1 convolutions to the full-resolution feature map:

```text
foreground_logits: [B, 1, H, W]
boundary_logits:   [B, 1, H, W]
center_logits:     [B, 1, H, W]
offset_vectors:   [B, 2, H, W]
```

The offset vectors point toward an instance anchor in global image coordinates.

Define the anchor by projecting the mask centroid onto its skeleton. A geometric centroid can fall outside a curved filament.

Use the global view to stabilize center proposals for long filaments. Local patches should refine morphology; independently estimating a new centroid inside every tile would create inconsistent instance markers.

### 4.3 Option B2: Swin-Base with the same decoder

**Checkpoint**

```text
timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k
```

**Configuration**

```text
patch size: 4
stage channels: [128, 256, 512, 1024]
stage depths: [2, 2, 18, 2]
attention heads: [4, 8, 16, 32]
window size: 12
MLP ratio: 4
feature strides: [4, 8, 16, 32]
```

Retain the same detail branch, decoder, and output heads.

The adapter must explicitly support the training resolutions, handle window padding, and convert feature layouts where necessary. Do not assume a classification checkpoint configured for 384 pixels accepts 768 or 1024 unchanged. [Swin implementation and checkpoint configuration](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py)

This is the strongest first alternative when ConvNeXt-Large exceeds the available memory or runtime budget.

### 4.4 Option B3: SegFormer-B5 comparison

**Checkpoint**

```text
nvidia/segformer-b5-finetuned-ade-640-640
```

**Encoder**

```text
channels:         [64, 128, 320, 512]
depths:           [3, 6, 40, 3]
attention heads:  [1, 2, 5, 8]
patch kernels:    [7, 3, 3, 3]
stage strides:    [4, 2, 2, 2]
spatial reduction ratios: [8, 4, 2, 1]
MLP ratios:       [4, 4, 4, 4]
```

The checkpoint’s semantic decoder width is 768. Replace its category classifier and add a shallow detail decoder that reaches native output resolution. Merely upsampling a stride-four prediction leaves a representational bottleneck for fine barbs. [SegFormer-B5 configuration](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640/blob/main/config.json)

### 4.5 Option B4: Swin-Base Mask2Former

Use the COCO instance checkpoint associated with:

```text
maskformer2_swin_base_IN21k_384_bs16_50ep
model_final_83d103.pkl
```

Starting architecture:

```text
Swin-Base encoder: [2, 2, 18, 2] blocks
pixel decoder width: 256
deformable encoder layers: 6
mask features: stride 4, width 256
object queries: 100
query width: 256
attention heads: 8
feed-forward width: 2048
masked transformer decoder blocks: 9
classification output: filament + no-object
```

The official configuration uses `DEC_LAYERS=10` to include supervision on the initial query prediction in addition to nine decoder blocks. [Mask2Former configuration](https://github.com/facebookresearch/Mask2Former/blob/main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml), [model zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)

Changes for this challenge:

1. Replace the category classifier with one foreground class plus no-object.
2. Derive instance targets from one annotator variant at a time.
3. Retain Hungarian assignment **for training**.
4. Compute competition PQ using the separate reference adapter.
5. Sample training points from foreground interiors, boundaries, and skeleton neighborhoods.
6. Apply clDice to dense matched-instance crops, rather than to sparse point samples.
7. Test a full-resolution residual mask branch using shallow image features.

Keep 100 queries unless the training manifest demonstrates that this truncates valid images. Increase the query count only with evidence.

For query outputs, preserve learned instance identities through stitching and deduplication. Avoid unioning their masks and subsequently hoping watershed recovers the original separation.

### 4.6 Option A1: YOLO11 localization and coarse masks

Prefer `yolo11m-seg.pt` over detection-only weights because coarse masks provide better automatic prompts.

**Architecture**

```text
Official YOLO11-seg medium scale:
depth multiplier: 0.50
width multiplier: 1.00
maximum channels: 512

Backbone:
stride-2 convolutions
C3k2 blocks
SPPF
C2PSA

Neck/head:
PAN feature fusion at strides 8, 16, 32
Segment head with 32 mask prototypes
prototype branch width: 256
one filament class
```

Use the official architecture configuration directly and record its source revision. [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/), [segmentation configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml)

Training inputs:

- Full-disk 1024-pixel views for complete large instances.
- Native 1024-pixel tiles for small structures.
- Boxes and masks derived from permitted segmentation annotations.

Starting inference settings:

```text
candidate confidence: 0.10
box NMS IoU: 0.65
maximum detections: 300
```

These deliberately prioritize recall and require calibration.

Evaluate proposal coverage directly:

- Fraction of GT instances reached.
- Fraction of each GT mask covered by a proposed refinement crop.
- Recall by instance size and limb distance.

A missed proposal cannot be repaired by a downstream segmenter.

### 4.7 Option A2: Mask DINO localization alternative

Use the official R50 instance checkpoint:

```text
maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth
```

Starting configuration:

```text
ResNet50 stage blocks: [3, 4, 6, 3]
encoder feature channels: [256, 512, 1024, 2048]
pixel/mask feature width: 256
deformable encoder layers: 6
decoder layers: 9
object queries: 300
attention heads: 8
feed-forward width: 2048
mask-enhanced box initialization
one filament class
```

Mask DINO supplies query masks and boxes jointly. Benchmark it as an alternative proposal generator before adding another refinement model. [Official Mask DINO implementation and checkpoints](https://github.com/IDEA-Research/MaskDINO)

Keep its compiled dependencies in a separately validated environment.

### 4.8 High-resolution prompt-based refinement

**Primary foundation-model comparison**

```text
SAM 2.1 checkpoint: sam2.1_hiera_base_plus.pt
configuration: sam2.1_hiera_b+.yaml
input: 1024×1024
Hiera channels: [112, 224, 448, 896]
Hiera stage depths: [2, 3, 16, 3]
stage attention heads: [2, 4, 8, 16]
FPN width: 256
high-resolution mask features enabled
```

Use independent image inference without temporal tracking state. [SAM 2 repository](https://github.com/facebookresearch/sam2)

Refinement procedure:

1. Expand each proposal with approximately 15% context.
2. Extract a native crop, normally 512–1024 pixels across.
3. Preserve aspect ratio and record padding/resizing transforms.
4. Supply the proposal box.
5. Select positive points from confident coarse-mask interiors and skeleton locations.
6. Select negative points from nearby competing proposals and credible background.
7. Restore logits to native crop coordinates.
8. Select/refine masks using validation-calibrated quality criteria.
9. Deduplicate overlapping proposals using mask evidence.

For long instances exceeding the crop size, refine overlapping subcrops and preserve the parent instance association.

Fine-tune through the differentiable model/training APIs. An inference predictor with detached cached embeddings cannot train the image encoder.

Start by training the mask decoder, then unfreeze late encoder blocks if grouped validation supports it.

**MedSAM comparison**

Use `medsam_vit_b.pth`:

```text
ViT-B image encoder:
patch size 16
embedding width 768
12 transformer blocks
12 attention heads
1024-pixel input

prompt/mask embedding width: 256
low-resolution mask output: 256×256
```

Medical pretraining is an empirical comparison; grayscale similarity alone does not establish transfer suitability. [MedSAM implementation](https://github.com/bowang-lab/MedSAM)

**Thin-detail limitation**

Both foundation-model refiners still have internal spatial bottlenecks. A 1024-pixel input does not guarantee one-pixel barb recovery.

Test a native-resolution residual head:

```text
Inputs:
3 intensity channels
upsampled foundation-model probability
coarse proposal probability

Convolutions:
5 → 32 → 32 → 16 → 1
3×3 kernels for feature layers
1×1 final output
GroupNorm + SiLU

Final logits:
upsampled foundation-model logits + predicted residual
```

Compare this against a conventional crop U-Net refiner under identical proposals.

### 4.9 Shared training configuration

Starting configuration for the primary dense model:

```yaml
seed: 2026

model:
  encoder: convnext_large.fb_in22k_ft_in1k_384
  decoder_channels: [512, 256, 128, 64, 32]
  detail_channels: [32, 64]
  heads:
    foreground: 1
    boundary: 1
    center: 1
    offset: 2

training:
  epochs: 60
  native_patch_size: 768
  global_view_size: 1024
  local_step_fraction: 0.80
  effective_batch_size: 16
  encoder_lr: 0.00003
  new_layers_lr: 0.0003
  weight_decay: 0.01
  warmup_fraction: 0.05
  schedule: cosine
  min_lr: 0.000001
  grad_clip_norm: 1.0
  ema_decay: 0.999
```

Use BF16 where supported; otherwise FP16 with gradient scaling. Compute overlap losses and soft skeletonization in float32.

Initial memory profiles:

| GPU memory | Initial microbatch | Accumulation | Adjustment |
|---|---:|---:|---|
| 16 GB | 1 with smaller backbone first | 16 | Establish feasibility before Large |
| 24 GB | 1 | 16 | Activation checkpointing |
| 48 GB | 2 | 8 | Profile larger crops |
| 80 GB | 4 | 4 | Profile query models/refiners |

These are starting profiles, not measured capacity guarantees.

## 5. Custom Topology-Aware Loss Functions

### 5.1 Core compound loss

For logits \(z\), probabilities \(p=\sigma(z)\), target \(g\), and valid-pixel mask \(v\):

\[
\boxed{
\mathcal L_{\mathrm{seg}}
=
\alpha\mathcal L_{\mathrm{BCE}}
+
\beta\mathcal L_{\mathrm{Dice}}
+
\gamma\mathcal L_{\mathrm{clDice}}
}
\]

Starting final weights:

\[
(\alpha,\beta,\gamma)=(0.35,0.45,0.20).
\]

Use a warm-up:

- Epochs 1–5: \((0.40,0.60,0)\).
- Epochs 6–15: interpolate toward the final weights.
- Thereafter: fixed weights.

A strong topology penalty applied before reliable foreground localization can reinforce noisy predictions.

### 5.2 Class-balanced BCE

\[
\mathcal L_{\mathrm{BCE}}
=
-\frac{
\sum_x v_x
\left[
w_+g_x\log p_x
+(1-g_x)\log(1-p_x)
\right]
}{
\sum_x v_x+\epsilon
}.
\]

Compute the starting positive weight from training-fold on-disk pixels:

\[
w_+
=
\operatorname{clip}
\left(
\sqrt{\frac{N_-}{N_+}},
2,20
\right).
\]

Use a stable logits-based implementation.

The square-root ratio is intentionally less aggressive than the full imbalance ratio, which can increase false bridges and network-noise detections. Compare it with unweighted BCE and a focal alternative.

### 5.3 Per-sample Dice

\[
\mathcal L_{\mathrm{Dice}}
=
1-
\frac{
2\sum_x v_xp_xg_x+\epsilon
}{
\sum_x v_xp_x+\sum_x v_xg_x+\epsilon
}.
\]

Average across samples rather than flattening the entire batch. Otherwise large foreground examples can dominate small filament crops.

For empty-GT crops, use an explicit foreground-suppression term such as mean predicted foreground probability; keep BCE active.

### 5.4 clDice and spine continuity

Let \(S_p\) be a differentiable soft skeleton of the prediction and \(S_g\) the target skeleton.

\[
T_{\mathrm{prec}}
=
\frac{
\sum_x v_x S_p(x)g(x)+\epsilon
}{
\sum_x v_x S_p(x)+\epsilon
},
\]

\[
T_{\mathrm{sens}}
=
\frac{
\sum_x v_x S_g(x)p(x)+\epsilon
}{
\sum_x v_x S_g(x)+\epsilon
},
\]

\[
\mathcal L_{\mathrm{clDice}}
=
1-
\frac{
2T_{\mathrm{prec}}T_{\mathrm{sens}}
}{
T_{\mathrm{prec}}+T_{\mathrm{sens}}+\epsilon
}.
\]

The precision term discourages unsupported predicted centerlines. The sensitivity term penalizes missing GT spine and barb pixels. Differentiable skeletonization uses iterative morphological pooling. [clDice authors’ implementation](https://github.com/jocpae/clDice)

Implementation requirements:

- Start with 20 skeletonization iterations; compare 10, 20, and 40.
- Relate iteration count to observed filament thickness.
- Preserve all target branches.
- Compute targets after deformation.
- Skip the topology term for empty-GT crops.
- Compute skeletons with context, then exclude an approximately 32-pixel patch-edge halo from topology reductions.
- Avoid thresholding predictions before differentiable skeletonization.

Finite soft-clDice training does not guarantee perfect connectivity. Track actual fragmentation.

### 5.5 Boundary supervision

Derive boundaries independently for each instance:

\[
B_g=\bigvee_i \partial G_i.
\]

Using only the boundary of the semantic union would lose separation cues where instances touch.

Train the boundary head with BCE plus Dice:

\[
\mathcal L_{\mathrm{boundary}}
=
\mathcal L_{\mathrm{BCE}}(q,B_g)
+
\mathcal L_{\mathrm{Dice}}(q,B_g).
\]

Also test a small signed-distance boundary term:

\[
\mathcal L_{\mathrm{SDF}}
=
\frac{1}{|V|}
\sum_x v_x\phi_g(x)p(x),
\]

where \(\phi_g\) is negative inside the target and positive outside, with distances clipped and normalized. This term can be negative; its absolute value is not a segmentation score. [Boundary-loss paper](https://proceedings.mlr.press/v102/kervadec19a.html)

### 5.6 Center and offset supervision

For each instance, choose an internal anchor \(c_i\) and generate a Gaussian center target. Start with a native-pixel sigma around 6, reducing it when nearby anchors would overlap excessively.

For \(x\in G_i\):

\[
o^*(x)=\frac{c_i-x}{2048}.
\]

Use foreground-masked Smooth L1:

\[
\mathcal L_{\mathrm{offset}}
=
\frac{
\sum_i\sum_{x\in G_i}
v_x\operatorname{SmoothL1}(\hat o(x),o^*(x))
}{
\sum_i|G_i\cap V|+\epsilon
}.
\]

Derive global anchors before cropping. Offsets may point outside a local crop.

For matched query/crop predictions, an optional centroid term is:

\[
c(P_i)
=
\frac{\sum_x xp_i(x)}{\sum_xp_i(x)+\epsilon},
\]

\[
\mathcal L_{\mathrm{centroid}}
=
\frac1M
\sum_i
\frac{\|c(P_i)-c(G_i)\|_1}
{\sqrt{H^2+W^2}}.
\]

Apply this to individual matched instances. A centroid loss on the entire foreground union cannot enforce correct instance count.

### 5.7 Starting total objective

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{seg}}
+0.10\mathcal L_{\mathrm{boundary}}
+0.05\mathcal L_{\mathrm{center}}
+0.05\mathcal L_{\mathrm{offset}}.
\]

Test SDF and moment-centroid terms separately before combining them.

For Mask2Former/Mask DINO, retain classification/no-object and assignment losses. Apply topology terms to matched instances.

**Critical interaction:** semantic clDice alone can tolerate or encourage a bridge between neighboring filaments. Instance boundaries and marker supervision address that PQ failure mode.

## 6. Post-Processing & Instance Formatting Engine

### 6.1 Reconstruct fields before extracting instances

Per observation:

```text
native image
→ deterministic preprocessing
→ global prediction + overlapping native tiles
→ inverse geometric transforms
→ probability/field blending
→ semantic support
→ calibrated markers and boundaries
→ instance extraction
→ artifact filtering and overlap resolution
→ native masks
→ compressed RLE rows
```

Process observations sequentially and encode completed instances immediately.

### 6.2 Semantic support through hysteresis

Start with:

```text
high-confidence threshold: 0.55
support threshold: 0.30
```

Retain lower-confidence pixels connected to credible high-confidence foreground or a supported instance marker.

This can preserve faint barb continuations while rejecting isolated chromospheric noise. Calibrate both thresholds against reference PQ and mean Dice.

### 6.3 Conservative spine-oriented closing

Use short directional structuring elements or supported endpoint links.

Starting bridge conditions:

- Endpoint gap no larger than approximately four native pixels.
- Compatible local spine directions, within approximately 20°.
- Predicted foreground support along the connecting path.
- Low predicted separating-boundary probability.
- No conflicting instance marker or proposal identity.

Apply closing only where those conditions hold.

A broad disk-wide closing operation would readily join neighboring filaments through their barbs. Report its added area and merge-rate effects explicitly.

### 6.4 Marker-controlled watershed

Construct markers from:

1. Global-view center heatmaps.
2. Native center/offset voting.
3. Detector/query proposals when that pipeline is enabled.
4. A conservative fallback marker for each unmarked connected component.

Avoid placing a marker at every distance-transform maximum: a single long branched filament naturally contains many maxima.

A starting watershed energy is:

\[
E(x)
=
\lambda_b q(x)
-\lambda_f p(x)
-\lambda_d\widetilde{\operatorname{DT}}(M)(x).
\]

Run watershed within semantic support \(M\).

Use boundary probability as separating evidence, rather than deleting every boundary pixel; thin barbs can be almost entirely boundary.

For learned query masks, use their identities directly and apply watershed only to justified ambiguous cases.

### 6.5 Artifact filtering and final identities

Evaluate native minimum-area thresholds of 0, 8, 16, and 32 pixels.

Use area jointly with:

- Skeleton length.
- Confidence.
- Shape.
- Agreement across views/models.
- Distance from plausible disk support.

A small confident elongated component may be a true filament. Preserve thin branches attached to a larger instance.

Deduplicate proposals using mask overlap and containment. A single overlapping pixel should not force two proposals to merge.

Resolve conflicting pixels using calibrated instance confidence, boundary evidence, or query scores.

Separate unconnected physical filaments receive separate rows. Any gap repair must be supported by image/model evidence before it creates a connected instance.

### 6.6 Native compressed-COCO-RLE utility

The competition expects counts for a 2048×2048 mask, with the image ID preserved and a unique instance suffix. Store the decoded counts string directly. [Competition submission format](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking)

A strict encoder:

```python
import numpy as np
from pycocotools import mask as coco_mask

def encode_instance(mask: np.ndarray) -> str:
    value = np.asarray(mask)

    if value.shape != (2048, 2048):
        raise ValueError("Expected a native 2048×2048 instance mask")

    if not np.logical_or(value == 0, value == 1).all():
        raise ValueError("Mask must contain only binary values")

    if not value.any():
        raise ValueError("An empty mask is not a filament instance")

    binary = np.asfortranarray(value, dtype=np.uint8)
    encoded = coco_mask.encode(binary)
    counts = encoded["counts"].decode("ascii")

    if any(ch in counts for ch in "\"'\r\n,"):
        raise ValueError("Unexpected character in compressed RLE counts")

    decoded = coco_mask.decode({
        "size": [2048, 2048],
        "counts": counts.encode("ascii"),
    })

    if not np.array_equal(decoded, binary):
        raise ValueError("RLE round-trip mismatch")

    return counts
```

Use `pycocotools.mask.encode`, `decode`, and `frPyObjects` as appropriate. The library expects uint8 masks in column-major order. It also supports overlap calculations directly on compressed masks. [pycocotools source](https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)

CSV writing:

```python
import csv

with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(
        handle,
        quoting=csv.QUOTE_NONE,
        lineterminator="\n",
    )
    writer.writerow(["filament_id", "segmentation_rle"])
    writer.writerows(rows)
```

Generate IDs such as:

```text
20150125172714Mh_1
20150125172714Mh_2
20150125172714Mh_3
```

Sort deterministically by observation, then native centroid and a stable tie-breaker.

### 6.7 Submission validation

Validate the written CSV by reading it back:

- Exactly the two required columns, in order.
- Unique filament IDs.
- Only known test observation IDs.
- ASCII compressed counts, without string wrappers or `b'...'`.
- Every row decodes to 2048×2048.
- Every instance has positive area.
- Decoded area matches the source instance.
- No accidental transpose, crop shift, or padding remains.
- Every test image was processed, including images yielding zero instances.

Maintain processed-image coverage in a separate run report. Confirm the current scoring container’s handling of zero-instance observations before release; additional container edge-case checks are distinct from the public self-evaluation notebook.

### 6.8 Metric implementation and diagnostics

Implement overlaps with `coco_mask.iou` and `iscrowd=0`, then reproduce the organizer’s hit/count aggregation.

For valid nonempty masks:

\[
\operatorname{Dice}=\frac{2\operatorname{IoU}}{1+\operatorname{IoU}}.
\]

Verify numerical parity with the dense reference implementation.

Report:

| Metric | Purpose |
|---|---|
| Reference global PQ | Competition ranking proxy |
| Observation-level PQ distribution | Robustness and difficult cases |
| Foreground union mean Dice | Pixel delineation |
| GT-centric instance Dice, unmatched GT scored zero | Completeness |
| Positive-overlap pairwise Dice distribution | Organizer-style diagnostic |
| Boundary recall at 1, 2, and 4 native pixels | Fine boundaries |
| Hard clDice | Spine/barb coverage |
| One-to-many and many-to-one rates | Fragmentation and merging |
| Runtime, peak GPU memory, peak RAM | Operational cost |

The positive-overlap Dice distribution excludes missed structures and should not serve as the sole mean-Dice measure.

If using TorchMetrics, explicitly set foreground/background handling, input format, averaging, and samplewise/global aggregation. Freeze and test empty-mask behavior.

## 7. Project Structure & Execution Roadmap

### 7.1 Proposed file tree

```text
solar-filament/
├── pyproject.toml
├── requirements.txt
├── environment.lock
├── Dockerfile
├── README.md
├── train.py
├── inference.py
├── evaluate.py
├── validate_submission.py
│
├── configs/
│   ├── data.yaml
│   ├── cv.yaml
│   ├── b0_resnet34.yaml
│   ├── convnext_large_detail.yaml
│   ├── swin_base_detail.yaml
│   ├── segformer_b5_detail.yaml
│   ├── mask2former_swin_base.yaml
│   ├── yolo11_sam2.yaml
│   ├── maskdino_sam2.yaml
│   └── postprocess.yaml
│
├── src/
│   ├── contracts.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── training.py
│   ├── checkpointing.py
│   ├── model_registry.py
│   │
│   ├── data/
│   │   ├── annotations.py
│   │   ├── manifest.py
│   │   ├── folds.py
│   │   ├── preprocessing.py
│   │   ├── augmentations.py
│   │   ├── targets.py
│   │   └── sampling.py
│   │
│   ├── models/
│   │   ├── backbones.py
│   │   ├── detail_unet.py
│   │   ├── mask2former_adapter.py
│   │   ├── detector_adapter.py
│   │   └── prompt_refiner.py
│   │
│   ├── inference/
│   │   ├── tiling.py
│   │   ├── tta.py
│   │   ├── ensemble.py
│   │   ├── instances.py
│   │   └── rle.py
│   │
│   └── evaluation/
│       ├── competition_adapter.py
│       └── diagnostics.py
│
├── tests/
│   ├── test_annotation_decoding.py
│   ├── test_fold_isolation.py
│   ├── test_target_geometry.py
│   ├── test_losses.py
│   ├── test_metric_parity.py
│   ├── test_tile_reconstruction.py
│   ├── test_tta_vectors.py
│   ├── test_instance_extraction.py
│   └── test_submission_roundtrip.py
│
├── notebooks/
│   └── reproduce_pipeline.ipynb
├── docs/
│   ├── data_contract.md
│   ├── metric_contract.md
│   └── experiments.md
└── artifacts/
    ├── manifests/
    ├── folds/
    ├── checkpoints/
    ├── oof/
    ├── reports/
    └── release/
```

### 7.2 Module interfaces

Keep these contracts stable:

| Module | Interface |
|---|---|
| Annotation adapter | `load_annotations(path) -> AnnotationIndex` |
| Dataset | `SolarDataset[index] -> TrainingSample` |
| Preprocessing | `preprocess(image, config) -> image_views, valid_mask, geometry` |
| Target generation | `make_targets(instance_masks, geometry) -> TargetBundle` |
| Model | `model(images) -> PredictionFields` |
| Tiled inference | `predict_observation(image, model, config) -> NativeFields` |
| Instance extraction | `extract_instances(fields, config) -> list[InstancePrediction]` |
| Encoder | `encode_instance(mask) -> str` |
| Metric adapter | `score_reference(gt, predictions) -> PQStats` |

Define shapes, dtypes, native coordinate conventions, valid regions, and optional fields in `contracts.py`. Avoid undocumented tensor tuples shared across modules.

### 7.3 Sequential implementation tasks

**Task 1 — Establish data and evaluator contracts**

Files: `annotations.py`, `manifest.py`, `competition_adapter.py`, `rle.py`.

- [ ] Inspect actual JSON keys and segmentation representations.
- [ ] Build annotation-to-image joins without silent defaults.
- [ ] Decode polygons, compressed RLE, and uncompressed RLE.
- [ ] Compare decoded masks against the COCO reference on asymmetric fixtures.
- [ ] Implement strict native RLE and CSV round trips.
- [ ] Pin the inspected organizer evaluator version and source hash.
- [ ] Reproduce PQ threshold and empty-set edge cases.

Acceptance: unresolved references fail; geometry is exact; optimized and reference metrics agree within numerical tolerance.

**Task 2 — Freeze leakage-safe folds**

Files: `folds.py`, `sampling.py`, `dataset.py`.

- [ ] Canonicalize observations and annotator variants.
- [ ] Join verified duplicate-image aliases.
- [ ] Compute count/area strata.
- [ ] Assign and save five folds.
- [ ] Test zero group/hash overlap.
- [ ] Implement observation-balanced annotation sampling.

Acceptance: every observation has one fold and one eventual OOF prediction.

**Task 3 — Reproduce a corrected ResNet34 baseline**

Files: `b0_resnet34.yaml`, `training.py`, `train.py`, `evaluate.py`.

- [ ] Retain the original architecture and BCE/Dice objective.
- [ ] Correct annotation loading and native output restoration.
- [ ] Overfit a few fixed native patches using one fixed annotator variant.
- [ ] Train the grouped baseline.
- [ ] Evaluate native masks and instances.
- [ ] Save OOF predictions and runtime measurements.

Acceptance: establish measured baseline PQ and Dice; keep historical leaderboard labels separate.

**Task 4 — Optimize losses on a fixed pilot configuration**

Files: `losses.py`, `targets.py`, `test_losses.py`.

- [ ] Establish a native 512-pixel patch baseline for topology experiments.
- [ ] Compare unweighted and class-balanced BCE.
- [ ] Compare clDice weights 0, 0.1, 0.2, and 0.3.
- [ ] Check gradients on empty, missed, thin, and branched targets.
- [ ] Add boundary supervision.
- [ ] Add center/offset supervision.
- [ ] Measure fragmentation and merging alongside overlap.

Acceptance: finite gradients and reproducible structural improvements under the same patch/model configuration.

**Task 5 — Scale resolution and context**

Files: `preprocessing.py`, `augmentations.py`, `tiling.py`.

- [ ] Compare native 512, 768, and 1024 patches.
- [ ] Add global-view training.
- [ ] Test full-image probability reconstruction.
- [ ] Test filaments crossing tile boundaries.
- [ ] Compare original, corrected, and CLAHE views.
- [ ] Profile memory and score per unit compute.

Acceptance: complete native coverage, preserved geometry, and no seam-induced fragmentation.

**Task 6 — Benchmark stronger backbones**

Files: `backbones.py`, `detail_unet.py`, model configurations.

- [ ] Verify feature shapes and layouts.
- [ ] Add the full-resolution detail branch.
- [ ] Train ConvNeXt-Large and Swin-Base under matched folds.
- [ ] Test SegFormer-B5 and Mask2Former only after the core path is stable.
- [ ] Record parameters, GPU hours, memory, and inference time.

Acceptance: improvements survive multiple folds and are attributable to architecture changes.

**Task 7 — Benchmark Option A**

Files: `detector_adapter.py`, `prompt_refiner.py`, two-stage configurations.

- [ ] Train localization targets derived from masks.
- [ ] Measure proposal coverage.
- [ ] Compare crop U-Net, SAM 2.1, and MedSAM refinement.
- [ ] Test native residual refinement.
- [ ] Verify box, point, crop, and mask coordinate transforms.
- [ ] Compare final PQ and Dice against Option B.

Acceptance: the complete pipeline improves the measured benchmark within the declared runtime budget.

**Task 8 — Calibrate instance reconstruction**

Files: `instances.py`, `postprocess.yaml`.

- [ ] Compare connected components alone.
- [ ] Add hysteresis.
- [ ] Add learned markers and boundary-aware watershed.
- [ ] Test conservative directional gap repair.
- [ ] Calibrate artifact filtering and proposal deduplication.
- [ ] Freeze settings before outer-fold confirmation.

Acceptance: lower split/merge error with preserved thin-boundary recall.

**Task 9 — Ensemble and TTA**

Files: `tta.py`, `ensemble.py`.

- [ ] Compare no TTA with four flips.
- [ ] Invert spatial maps correctly.
- [ ] Invert offset-vector directions as well as pixel positions.
- [ ] Match query instances before fusion.
- [ ] Ensemble OOF models only when every member excluded the target fold.
- [ ] Compare cross-backbone and fold ensembles on test-time cost.

Five folds × four TTA views × sixteen tiles means **320 tile forwards per image**, or **57,600 for 180 images**, before global passes. Benchmark that cost before selecting the release configuration.

Acceptance: retained TTA/ensemble gains justify their measured overhead.

**Task 10 — Package an offline reproducible release**

Files: environment locks, registry, checkpointing, inference entry point, reproduction notebook.

- [ ] Record dependency versions, source revisions, and checkpoint SHA-256 hashes.
- [ ] Save optimizer/scaler/RNG state for resumable training.
- [ ] Package all required weights and preprocessing artifacts.
- [ ] Run inference with network access disabled.
- [ ] Read back and validate the final CSV.
- [ ] Save processed-image coverage, runtime, memory, and configuration hashes.
- [ ] Reproduce the run from a clean environment.

Acceptance: identical inputs and configuration produce equivalent native instance masks and a valid submission.

### 7.4 Prioritized ablation checklist

| Priority | Experiment | Fixed comparison |
|---:|---|---|
| P0 | Annotation and submission corrections | Corrected ResNet34 |
| P0 | Reference metric parity | Organizer evaluator |
| P0 | Grouped five-fold baseline | Frozen folds |
| P1 | BCE weighting and clDice | Fixed native-patch pilot |
| P1 | Boundary and center/offset targets | Same backbone/resolution |
| P1 | Native resolution and global context | Same loss/model |
| P2 | Preprocessing views | Same geometry/training schedule |
| P2 | ConvNeXt-Large versus Swin-Base | Same decoder/heads |
| P2 | Watershed and conservative repair | Frozen probability maps |
| P3 | Mask2Former or detect/refine | Same evaluator and budget |
| P3 | Ensembling and TTA | Correct OOF membership |
| P3 | Smaller release model or distillation | Measured accuracy/runtime tradeoff |

Compare resolutions using both fixed training exposure and measured compute cost. Report the protocol so a more expensive configuration is not presented as an efficiency gain.

### 7.5 Promotion and release criteria

Define “outperform” against the **corrected, current-metric baseline**.

Suggested final promotion gate:

- At least **+0.01 absolute reference PQ**.
- Mean foreground Dice at least equal to the corrected baseline.
- Improved thin-boundary or skeleton recall.
- No material increase in fragmentation or over-merging.
- Positive paired improvement under observation-cluster bootstrap.
- Acceptable runtime and memory on the selected execution hardware.

Treat the +0.01 margin as an engineering decision, not a predicted result.

Bootstrap physical observations while retaining all their annotator variants. For the leading configuration, repeat a second training seed to assess training variance.

I verified 17 in-memory checks covering ID normalization, native compressed-RLE/CSV round trips, PQ edge cases, fold isolation, and tile coverage. Training performance, GPU capacity, and parity with additional unpublished scoring-container checks remain implementation-stage validation requirements.
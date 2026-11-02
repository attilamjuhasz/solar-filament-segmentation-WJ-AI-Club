# Solar Filament Segmentation 2026

Code for the Kaggle [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
It finds every solar filament in a GONG H-alpha image and outputs one mask per filament.

## How it works (the big picture)

```
H-alpha image (2048x2048)
   │  preprocess.py  find the solar disk, remove limb darkening
   ▼
contrast image ── downscale to 1024 ──► U-Net (model.py)
                                          │  "how likely is each pixel a filament?" (0..1)
                                          ▼
                                   probability map ── upscale to 2048
                                          │  postprocess.py  threshold, join fragments, clean up
                                          ▼
                              one mask per filament ──► submission.py ──► submission.csv
```

The neural network only answers **"is this pixel part of a filament?"**. Deciding **which pixels belong to
the same filament** is done afterwards by `postprocess.py`. The competition metric (Panoptic Quality)
punishes splitting one filament into pieces or merging two together, so that step matters a lot, and
`tune.py` searches for its best settings automatically.

## What each file does

### The library: `filseg/`
Each file has one job. The scripts at the bottom of this list use the others.

| File | In plain words |
|---|---|
| `data.py` | Reads the annotation JSON and the images. An `AnnotatedImage` is one annotator's set of filaments for one image. It also splits the data into training and validation **by image**, so the same picture never ends up on both sides through a second annotator. |
| `rle.py` | Converts masks to and from the RLE text format the submission uses (a compact way to write "which pixels are on"). |
| `preprocess.py` | Finds the solar disk and corrects limb darkening (the Sun's edge looks darker than its centre). After this, a filament looks the same everywhere on the disk. |
| `dataset.py` | Feeds training examples to PyTorch: downscaled images plus target masks, random crops, flips and brightness changes ("augmentation") so the model generalizes. It saves processed images to `cache/` so later epochs are fast. |
| `model.py` | Defines the network: a **U-Net**, the standard architecture for pixel-wise segmentation, with an encoder pretrained on ImageNet. It also has the loss function (how wrong the model is) and saving/loading of weights. |
| `infer.py` | Runs a trained model on one full image and returns the probability map (with flip-averaging, called TTA, for a small accuracy boost). |
| `postprocess.py` | Turns the probability map into separate filaments: thresholding, removing specks, joining nearby fragments, smoothing. **Its settings strongly affect the score.** |
| `metrics.py` | Our own copy of the scoring: PQ, Dice, IoU, and counts of split and merged filaments. This lets you measure progress without uploading to Kaggle. |
| `validate.py` | Glue code: predict all validation images and score them. Used by `train.py` and `tune.py`. |
| `submission.py` | Writes and reads the competition CSV (`filament_id,segmentation_rle`). |
| `viz.py` | Draws predictions (one colour per filament) over the image, with the true outline in white. |

### The scripts you run (each is `python -m filseg.<name> ...`, add `--help` to see all options)

| Command | What it does | Output |
|---|---|---|
| `filseg.evaluate --gt-as-pred` | Sanity check: scores the true masks against themselves. Should give PQ ≈ 1. Also shows how much different human annotators agree. | printed scores |
| `filseg.train` | Trains the U-Net and keeps the version with the best validation PQ. | `weights/model.pt` |
| `filseg.tune` | Tries many post-processing settings on the validation images and keeps the best. | `weights/postprocess.json` |
| `filseg.predict` | Predicts the test images and writes the submission. | `submission.csv` |
| `filseg.evaluate --csv` | Scores any CSV you made for training/validation images. | printed scores |

### Other files
| File | Purpose |
|---|---|
| `pipeline.ipynb` | The notebook the rules require: walks through the whole pipeline with pictures and scores. |
| `requirements.txt` | Exact package versions (also required by the rules). |
| `weights/` | Trained model and tuned settings. **Must be committed to the repo** (rules), and stays under GitHub's 100 MB limit because weights are saved in half precision. |
| `tests/make_fake_data.py` | Makes a tiny fake dataset so you can check the code runs end to end in a minute, without the real data. |
| `.gitignore` | Keeps data, caches and CSVs out of git. |

## Step by step

1. **Get the data.** On the competition's Data tab: Download All, then unzip into `data/`.
2. **Install:** `pip install -r requirements.txt`. You need a GPU for training: a Kaggle notebook with a GPU
   (free, about 30 hours a week) or Google Colab works.
3. **Sanity check:**
   `python -m filseg.evaluate --data data --gt-as-pred`
4. **Train** (about 1-2 hours on a T4 GPU):
   `python -m filseg.train --data data --epochs 40`
5. **Tune post-processing:**
   `python -m filseg.tune --data data --tta`
6. **Predict and submit:**
   `python -m filseg.predict --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images`,
   then upload `submission.csv` on Kaggle.
7. **Final submission** (due **Nov 15, 2026**): push the repo with `weights/` to a public GitHub repo,
   write the 4-page report with the organizers' Overleaf template, and fill in their Google form.

Quick test without the real data:
```bash
python tests/make_fake_data.py fake_data
python -m filseg.train --data fake_data --no-pretrained --encoder resnet18 --size 512 --crop 256 --epochs 2 --batch 2 --workers 0 --eval-every 1 --out weights/test.pt
```

## Ideas to improve the score
- Tune post-processing first; it is cheap and usually the biggest gain.
- Train at full 2048 resolution with tiles, to catch thin barbs.
- Try a larger encoder (`--encoder efficientnet-b4`) or train several seeds and average them.
- Train only on one annotator per image, or average the annotators into a soft target.
- Instance-aware models (e.g. predicting filament boundaries as a second output) for better splitting
  of neighbouring filaments.

# 2-Day Learning Roadmap → YOLOv8-Poly-Dist-TF (and your ML/DL career)

**Audience:** experienced software engineer, low hands-on ML/DL, presenting this project Monday.
**Goal of the 2 days:** build a correct end-to-end mental model, be able to walk through every
module and justify each design choice, and answer hard questions. Expertise comes after, from
working on it — this also lays the long-term track.

> **How to use this:** every concept below is tied to a *specific file in this repo* and a
> *small thing to code or run*. Don't read passively. Open the file, run the test, print a
> tensor's shape. The fastest way to "get" ML is to watch shapes flow.

---

## The one-paragraph mental model (read this first, re-read it Monday morning)

This is **YOLOv8** — a single-shot object detector — reimplemented in TensorFlow and extended
with two extra prediction "heads": a **polygon** head (outlines each object, not just a box)
and a **distance** head (how far each object is, in meters). An image (672×672×3) flows through
a **backbone** (a CNN that extracts features at 3 scales), a **decoder/neck** (FPN-PAN, which
mixes those scales together), and a **head** (6 small conv branches that predict box, class,
polygon-angle, polygon-distance, polygon-confidence, and object-distance *at every grid cell*).
During **training**, a *label assigner* decides which grid cells are responsible for which
ground-truth objects, and a *loss function* scores the predictions; an *optimizer* (SGD) nudges
the weights to reduce the loss. During **inference**, a *detection generator* decodes the raw
grid outputs into actual boxes/polygons and runs **NMS** to remove duplicates. Everything is
driven by **YAML config files** so the same code runs 3 model tiers (box-only, +polygon,
+distance). That's the whole system. Every file in the repo is one of those nouns.

---

## Day 1 — Foundations + top-down read of the system

Aim: by end of Day 1 you can explain the mental model above in your own words and point to the
file behind every noun.

### Block 0 (30 min) — Orient in the repo
- Read `README.md` and `CLAUDE.md` (you basically have, skim again).
- Read all of `docs/`: `architecture.md`, `data_pipeline.md`, `losses.md`, `training.md`,
  `testing.md`, and **`design_register.md`** (this last one is gold — it lists *deliberate*
  design decisions and the reasoning; presentation-question ammunition).
- Run the tests so you see green and know the codebase is alive:
  ```bash
  pytest tests/unit/ -v        # fast, no dataset needed
  pytest tests/integration/ -v # synthetic-data end-to-end
  ```
- **Do:** in `tests/unit/test_model_forward.py`, find where the model is built and a forward
  pass is run. This is your "hello world" for the model.

### Block 1 (2 hr) — CNN fundamentals (only what this project uses)
You did a famous DL course; this is the *hands-on* you're missing. Don't go broad — go to
exactly what this repo uses: convolutions, channels, stride, feature maps, batchnorm, ReLU.

- **Read:** CS231n Convolutional Networks notes — https://cs231n.github.io/convolutional-networks/
  (the canonical, clear explanation of conv/stride/pooling/receptive field).
- **Watch (optional, excellent):** 3Blue1Brown "But what is a convolution?" for intuition.
- **Code (the important part):** TensorFlow CNN tutorial — train a tiny CNN on CIFAR-10:
  https://www.tensorflow.org/tutorials/images/cnn
  - Then **modify it**: print `model.summary()`, change a conv's `filters` and `strides`,
    watch the output spatial size halve when stride=2. *This is exactly what "stride 8/16/32"
    means in this repo* — each downsample halves resolution.
- **Tie to repo:** open `models/backbone.py`. You'll see `Conv` blocks, `C2f` blocks, and
  `SPPF`. You don't need to memorize C2f yet — just recognize: backbone = stack of convs that
  shrinks H×W and grows channels, emitting feature maps at levels "3","4","5" (strides 8/16/32).
- **Key terms to nail:** feature map, channel, stride, downsampling, receptive field,
  BatchNorm, ReLU.

### Block 2 (2 hr) — Object detection concepts (the heart of YOLO)
This is the single most important block for understanding the project. Detection = "where +
what" for many objects at once.

- **Read (best free resource):** *Dive into Deep Learning*, Computer Vision chapter —
  https://d2l.ai/chapter_computer-vision/index.html
  Specifically: Bounding Boxes, Anchor Boxes, Multiscale Object Detection, IoU, NMS, and the
  SSD section. This maps *directly* onto YOLO.
- **Read (intuition for the metric):** Jonathan Hui, "mAP for object detection" —
  https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
- **Code these two by hand (1 hour, huge payoff)** in a scratch notebook:
  1. **IoU** of two boxes (intersection area / union area). 10 lines of numpy.
  2. **Greedy NMS**: sort boxes by score, keep the top, drop others with IoU > threshold,
     repeat. ~15 lines.
  These two functions ARE the core of detection post-processing. Once you've written them,
  `models/detection_generator.py` will read like English.
- **Concepts to nail:** bounding box (xyxy vs cxcywh), IoU, **anchor-free** (this repo predicts
  box offsets directly from each cell center — 1 "anchor" per cell), grid/cell, multi-scale
  detection (small objects on the fine P3 map, big objects on the coarse P5 map), NMS, mAP/mAP50.
- **Tie to repo:** `models/detection_generator.py` (NMS, score_thresh=0.05, decode) and
  `eval/coco_metrics.py` (mAP). Anchor points = cell centers `(i+0.5)·stride` — see
  `docs/architecture.md` "Anchors & strides".

### Block 3 (2 hr) — YOLOv8 specifically, top-down through this repo
Now read the actual model in dependency order. For each file, your job is one sentence:
"this takes X and produces Y."

- **Watch/read for YOLO context:** Ultralytics YOLOv8 docs — https://docs.ultralytics.com/
  (this repo is a TF reimplementation of their model; same architecture vocabulary:
  CSPDarknet backbone, C2f, SPPF, PAN neck, decoupled head, DFL, TAL).
- **Read the repo in this order** (have `docs/architecture.md` open beside you):
  1. `models/backbone.py` — image → 3 feature maps. (CSPDarkNetV8-S; C2f = cross-stage
     partial block; SPPF = spatial pyramid pooling fast.)
  2. `models/decoder.py` — 3 feature maps → 3 *fused* feature maps (FPN top-down + PAN
     bottom-up so each scale sees the others).
  3. `models/head.py` — fused maps → 6 raw prediction tensors per scale. Look at the channel
     counts: box=64 (4 sides × 16 DFL bins), cls=39, poly_angle/dist/conf=24 each, dist=1.
     Note the **smart bias init** (`log(5/num_classes/...)`) — ask "why?" (answer: makes
     initial class predictions near the prior probability, so loss starts stable).
  4. `models/yolo_v8.py` — `build_yolov8()` glues backbone+decoder+head together.
  5. `models/detection_generator.py` — raw tensors → boxes/polygons/distances + NMS (inference
     only, `deploy=True`).
- **Do:** run `python tools/trace_shapes.py` (traces tensor shapes through the pipeline) — this
  is the single best exercise for *seeing* the data flow. Watch H×W shrink and channels change.
- **Concepts to nail:** backbone/neck/head split, FPN-PAN, decoupled head, DFL
  (Distribution Focal Loss — box edges predicted as a probability distribution over 16 bins,
  not a single number — read https://arxiv.org/abs/2006.04388 abstract + intro only).

### Block 4 (1.5 hr) — The two extensions (this project's special sauce)
This is what makes *your* project different from stock YOLOv8. Expect heavy presentation focus
here.

- **Polygon (PolyYOLO radial format):** instead of a free-form mask, each object's outline is
  encoded as **24 vertices at fixed 15° angles** around the box center. The model predicts, per
  angle bin: a **radial distance** (how far the outline is), a **sub-bin angle offset**, and a
  **confidence** (is there a vertex in this bin). Read `docs/data_pipeline.md` "Polygon
  Representation" and the "Polygon Format Reference" table in `README.md` carefully — the format
  changes shape 3 times (TFDS input → radial `[N,72]` target → Cartesian `[M,2]` for eval/viz).
  - **Read for background:** PolyYOLO paper abstract — https://arxiv.org/abs/2005.13243
  - **Open the notebook:** `notebooks/02_polygon_bins_walkthrough.ipynb` — it visually shows the
    binning. This will make the radial format *click*.
  - **Why radial?** Fixed-size output (always 24×3), no variable-length masks, cheap to predict
    with a conv head. Trade-off: can't represent concave/multi-part shapes well. Good
    presentation talking point.
- **Distance:** a separate 1-channel head predicts per-object distance in **log-scale** meters,
  trained on a *separate dataset* (`servingbot_polygon`) merged into the batch. Read
  `losses/distance_loss.py` (L1 on log-scale, sentinel -10.0 masks invalid). Range [0.5, 10] m.
  - **Why log-scale + L1?** Log makes relative error uniform across near/far; L1 is robust to
    outliers. Why a separate stream merged at batch level? Because distance labels exist on a
    different dataset than detection labels — see `data_pipeline/input_reader.py`.

**End of Day 1 self-test (do this out loud):** trace one image from pixels to final boxes,
naming each file it passes through. If you can do that, Day 1 worked.

---

## Day 2 — Training machinery, deep dives, and presentation prep

Aim: by end of Day 2 you understand *how the model learns* and you have a presentation.

### Block 5 (2 hr) — How training works (loss + assignment)
This is the conceptual peak. Take it slow.

- **Background — custom training loops in TF** (this repo uses a custom loop, not `model.fit`):
  https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
  - **Code:** adapt that tutorial to print the loss each step. Understand `GradientTape`,
    forward pass → loss → `tape.gradient` → `optimizer.apply_gradients`. *This exact pattern is
    in `train/task.py:train_step`.*
- **Loss functions — read `docs/losses.md`, then `losses/`:**
  1. `losses/tal_assigner.py` — **TaskAlignedAssigner**. The hardest idea: before computing
     loss, you must decide *which predictions are responsible for which ground-truth objects*.
     TAL scores each prediction by `score^0.5 × IoU^6` and assigns the top-10 per object.
     Background paper (skim abstract): TOOD — https://arxiv.org/abs/2108.07755
  2. `losses/tal_loss.py` — the combined loss: **CIoU** (box regression, https://arxiv.org/abs/1911.08287),
     **DFL** (box distribution), **BCE** (classification), + polygon + distance.
  3. `losses/polygon_loss.py` — angle (BCE on sub-bin offset), dist (L2+softplus), conf (BCE
     over all 24 bins). Read the `design_register.md` note on *why conf is over all bins, not
     just valid ones* (the "spiky polygon" bug) — **classic interview/presentation question.**
  4. `losses/distance_loss.py` — covered above.
- **Concepts to nail:** label assignment (why it's needed, why it's the trickiest part),
  multi-task loss (weighted sum of many losses — see the `iou_gain/cls_gain/...` in the YAML),
  CIoU vs IoU, why classification uses BCE not softmax (multi-label friendliness + the soft
  targets `one_hot × align_norm × overlap`).
- **Do:** run a 10-step training loop and watch the loss numbers move:
  ```bash
  pytest tests/smoke/test_train_10_steps.py::TestDrySmoke -v
  ```

### Block 6 (1.5 hr) — Optimizer, LR schedule, EMA
- **Read `optimizers/sgd_warmup.py`:** SGD with Nesterov momentum (0.937), decoupled weight
  decay, 3 parameter groups, linear momentum warmup. **Read `optimizers/ema.py`:** Exponential
  Moving Average of weights with dynamic decay `min(0.9999, (1+step)/(10+step))`, swapped in for
  eval.
- **Concepts to nail:** gradient descent → SGD → momentum → Nesterov; **cosine LR decay**
  (https://arxiv.org/abs/1608.03983 — SGDR, skim); **warmup** (start LR/momentum low so early
  unstable steps don't blow up); **weight decay** (regularization); **EMA** (averaged weights
  generalize better — why eval uses EMA shadows).
- **Code (15 min):** plot a cosine decay schedule with warmup in numpy/matplotlib so you *see*
  the LR curve. (initial=0.01, decays over 635,400 steps, alpha=0.01.)
- **Tie to repo:** all these values live in `configs/experiments/yolo/yolov8_poly_dist.yaml`.

### Block 7 (2 hr) — Data pipeline + the trainer (the engineering scale)
This is where your software-engineering strength shines — it's mostly `tf.data` plumbing.

- **Background — `tf.data`:** https://www.tensorflow.org/guide/data (datasets, map, batch,
  prefetch, shuffle). This repo's pipeline is a sophisticated `tf.data` graph.
- **Read `docs/data_pipeline.md` then the files in this order:**
  1. `data_pipeline/tfds_decoders.py` — raw dataset → dict of tensors.
  2. `data_pipeline/input_reader.py` — multi-dataset weighted sampling + distance-stream merge +
     `padded_batch` with the critical `-1.0` polygon padding (read the CLAUDE.md note on why
     0-padding corrupts polygons — great detail to mention).
  3. `data_pipeline/copy_paste.py` → `mosaic.py` → `augmentations.py` — data augmentation
     (Copy-Paste before Mosaic; Mosaic = stitch 4 images; random_perspective warp). Open
     `notebooks/02_augmentation_debug.ipynb` to *see* augmented images.
  4. `data_pipeline/yolo_parser.py` + `distance_parser.py` — turn labels into the radial polygon
     target and the training tensors.
- **Read `train/trainer.py` + `train/task.py`:** the custom loop — epoch math (exactly
  `steps_per_loop` = 2118 steps/epoch), checkpointing, EMA swap, auto-resume, TensorBoard. Your
  SWE brain will like this; it's systems code.
- **Concepts to nail:** data augmentation (and *why* — more effective data, regularization),
  Mosaic/MixUp/Copy-Paste, infinite repeated streams, batch-level dataset merging, checkpoint/
  resume, TensorBoard logging.

### Block 8 (1.5 hr) — Eval, export, tools (rounding out)
- `eval/coco_metrics.py` (mAP), `distance_metrics.py` (MAE/RMSE), `polygon_metrics.py`
  (mask IoU via `cv2.fillPoly`). Skim — you just need to know what each metric *means*
  (mAP = detection quality, MAE = avg distance error in meters, mIoU = polygon overlap).
- `tools/eval.py`, `tools/export_saved_model.py` (SavedModel + TFLite), `tools/checkpoint_migration.py`.
- Open `notebooks/03_training_analysis.ipynb` and `04_per_category_evaluation.ipynb` to see how
  results are read.

### Block 9 (2.5 hr) — Build your presentation
Structure it like the mental-model paragraph, then drill in:
1. **What & why** — detection + polygon + distance; the 3 config tiers; what problem it solves.
2. **Architecture** — one slide: image → backbone → decoder → 6 heads → detection generator.
   Use the diagram in `docs/architecture.md`.
3. **The two extensions** — radial polygon format (show a binning image from the notebook) +
   distance head. This is your differentiator; spend time here.
4. **Training** — assignment (TAL) → multi-task loss → SGD+cosine+EMA. One slide on the loss
   breakdown with the gains from the YAML.
5. **Data pipeline** — multi-dataset sampling, augmentation stack, batch merge of distance.
6. **Engineering** — config-driven tiers, tests, checkpoint/resume, export, TensorBoard.
7. **One "I went deep" slide** — pick ONE subtle design decision from `docs/design_register.md`
   (e.g., the polygon-conf-over-all-bins fix, or arc-length vertex resampling) and explain the
   bug and the fix. This signals you actually understand it, not just skimmed.

**Anticipate these questions** (write 2-sentence answers for each):
- Why anchor-free? Why 672×672? Why ReLU? Why log-scale distance?
- What does the assigner do and why is it needed?
- Why is the polygon format radial and what are its limits?
- What's DFL? What's CIoU? Why BCE for classification?
- Why EMA for eval? Why cosine LR + warmup?
- How does the distance dataset merge with the detection dataset?
- What would you change/improve? (Have one honest idea — e.g., polygon format struggles with
  concave shapes.)

---

## After Monday — the path from "can present it" to "expert" (your career track)

Months 1–3, in priority order (each ~1–2 weeks, hands-on):
1. **Solidify CNNs + training:** Andrej Karpathy's "Neural Networks: Zero to Hero"
   (https://karpathy.ai/zero-to-hero.html) — build backprop and a small net from scratch. This
   removes all remaining magic.
2. **Read Karpathy's "A Recipe for Training Neural Networks"** —
   http://karpathy.github.io/2019/04/25/recipe/ — how to actually debug training. Invaluable
   on the job.
3. **Detection depth:** read the papers you skimmed, fully — YOLOv1 (https://arxiv.org/abs/1506.02640),
   FPN (https://arxiv.org/abs/1612.03144), CSPNet (https://arxiv.org/abs/1911.11929),
   PANet (https://arxiv.org/abs/1803.01534), Generalized Focal Loss/DFL, TOOD.
4. **Reproduce a result:** train `yolov8_bbox` (the fast tier) end-to-end on a small dataset,
   read TensorBoard curves, debug a NaN or a low-mAP run. Nothing teaches like a real run.
5. **Contribute:** pick a `tests/` gap or a `tools/` improvement and make a real PR. Owning code
   is how you become the expert on it.
6. **Breadth later:** transformers/attention, segmentation (Mask R-CNN, SAM), and modern
   detectors (DETR) — once detection fundamentals are solid.

---

## Glossary (skim until each is one sentence in your head)

- **Backbone / neck / head** — feature extractor / feature mixer / predictor.
- **Feature map** — a conv layer's output: H×W grid × C channels.
- **Stride** — downsampling factor; stride 32 means the map is 1/32 the input size.
- **FPN-PAN** — combines fine + coarse feature maps so every scale sees context.
- **Anchor-free** — predict box directly from each cell center (no preset anchor boxes).
- **DFL** — predict each box edge as a distribution over 16 bins, not one number.
- **IoU / CIoU** — box overlap metric / a better version used as the box loss.
- **NMS** — remove duplicate detections of the same object.
- **mAP** — the standard detection accuracy metric.
- **Label assignment (TAL)** — choosing which predictions are "responsible" for each GT object.
- **Multi-task loss** — weighted sum of box + class + polygon + distance losses.
- **SGD / momentum / Nesterov / weight decay** — the optimizer and its knobs.
- **Cosine LR + warmup** — learning-rate schedule: ramp up, then smoothly decay.
- **EMA** — moving average of weights; generalizes better; used at eval time.
- **PolyYOLO radial format** — outline as 24 (distance, angle, conf) triples around the center.
- **TFDS / tf.data** — TensorFlow's dataset format / data pipeline API.
- **Mosaic / MixUp / Copy-Paste** — data augmentations that combine images.
- **Checkpoint / EMA swap / resume** — save weights / swap in averaged weights / restart training.
```

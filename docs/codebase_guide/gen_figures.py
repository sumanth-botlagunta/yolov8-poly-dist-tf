"""Generate the illustrated-guide figures by running the ACTUAL pipeline code.

Every "real-pipeline" figure here is produced by calling the project's own
functions (``data_pipeline.augmentations`` / ``data_pipeline.mosaic``), not a
re-implementation — so the guide shows what the code genuinely does. A few
schematic figures (radial encoding, DFL, LR/EMA schedules) are computed straight
from the documented formulas / optimizer code.

Run (project env with TF + matplotlib):
    PYTHONPATH=<repo> python docs/codebase_guide/gen_figures.py

Outputs PNGs into ``docs/codebase_guide/figures/``. Inputs are synthetic scenes
(the real training TFDS datasets are not needed / not local) so the transforms are
visually legible; the CODE PATHS are the real ones.
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from data_pipeline.augmentations import (
    hsv_augment,
    random_horizontal_flip,
    random_perspective,
    resample_polygons,
)
from data_pipeline.mosaic import Mosaic

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

S = 672                      # canonical input size
ANGLE_STEP_DEG = 15
NBINS = 24

# A small, friendly palette (color-blind safe-ish).
BLUE, ORANGE, GREEN, RED, PURPLE = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed"


# ===========================================================================
# Synthetic scene builders (inputs only — the transforms applied are real)
# ===========================================================================
def _draw_object(draw: ImageDraw.ImageDraw, verts_px, fill):
    draw.polygon([tuple(p) for p in verts_px], fill=fill, outline=(20, 20, 20))


def _ngon(cx, cy, r, n, rot=0.0, squash=1.0):
    return np.array([
        [cx + r * math.cos(rot + 2 * math.pi * k / n),
         cy + r * squash * math.sin(rot + 2 * math.pi * k / n)]
        for k in range(n)
    ], dtype=np.float32)


def make_scene(idx: int):
    """Return (image_uint8 HxWx3, list[ (verts_px Nx2, color) ]) for scene idx."""
    bg = [(196, 214, 232), (210, 225, 200), (232, 214, 198), (220, 210, 230)][idx % 4]
    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)
    # light grid so the geometric warp is legible
    for g in range(0, S, 56):
        d.line([(g, 0), (g, S)], fill=(255, 255, 255), width=1)
        d.line([(0, g), (S, g)], fill=(255, 255, 255), width=1)
    d.text((12, 10), f"INPUT {idx + 1}", fill=(40, 40, 40))

    objs = []
    if idx == 0:
        v = _ngon(250, 300, 130, 6, rot=0.3); _draw_object(d, v, (37, 99, 235)); objs.append((v, BLUE))
        v = _ngon(470, 470, 80, 3, rot=0.8); _draw_object(d, v, (234, 88, 12)); objs.append((v, ORANGE))
    elif idx == 1:
        v = _ngon(360, 320, 150, 5, rot=-0.4); _draw_object(d, v, (22, 163, 74)); objs.append((v, GREEN))
    elif idx == 2:
        v = _ngon(300, 360, 120, 8, rot=0.1, squash=0.7); _draw_object(d, v, (124, 58, 237)); objs.append((v, PURPLE))
        v = _ngon(500, 200, 70, 4, rot=0.4); _draw_object(d, v, (220, 38, 38)); objs.append((v, RED))
    else:
        v = _ngon(340, 340, 160, 12, rot=0.0); _draw_object(d, v, (8, 145, 178)); objs.append((v, "#0891b2"))
    return np.array(img, dtype=np.uint8), objs


def scene_to_example(idx: int, vmax: int = 24):
    """Build the pipeline example dict (image + GT boxes/polygons/side fields)."""
    img, objs = make_scene(idx)
    boxes, polys = [], []
    for verts_px, _c in objs:
        xs, ys = verts_px[:, 0], verts_px[:, 1]
        boxes.append([ys.min() / S, xs.min() / S, ys.max() / S, xs.max() / S])  # yxyx norm
        flat = np.full((vmax * 2,), -1.0, dtype=np.float32)
        n = min(len(verts_px), vmax)
        flat[:2 * n] = (verts_px[:n] / S).reshape(-1)                          # xy interleaved norm
        polys.append(flat)
    n = len(objs)
    return {
        "image": tf.constant(img),
        "groundtruth_boxes": tf.constant(np.array(boxes, np.float32)),
        "groundtruth_polygons": tf.constant(np.array(polys, np.float32)),
        "groundtruth_classes": tf.zeros([n], tf.int64),
        "groundtruth_is_crowd": tf.zeros([n], tf.bool),
        "groundtruth_area": tf.ones([n], tf.float32),
        "groundtruth_dontcare": tf.zeros([n], tf.int64),
        "groundtruth_dists": tf.fill([n], tf.constant(-1.0)),
    }


# ===========================================================================
# Overlay drawing helpers
# ===========================================================================
def _show(ax, img, boxes=None, polys=None, title=None, hw=S):
    ax.imshow(np.asarray(img))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    if polys is not None:
        P = np.asarray(polys)
        for row in P:
            pts = row.reshape(-1, 2)
            pts = pts[pts[:, 0] > -1.0]
            if len(pts) >= 2:
                ax.add_patch(MplPolygon(pts * hw, closed=True, fill=False,
                                        edgecolor=RED, lw=2.0))
    if boxes is not None:
        B = np.asarray(boxes)
        for ymin, xmin, ymax, xmax in B:
            ax.add_patch(Rectangle((xmin * hw, ymin * hw), (xmax - xmin) * hw,
                                   (ymax - ymin) * hw, fill=False, edgecolor=BLUE, lw=2.0))


# ===========================================================================
# 1. random_perspective (REAL)
# ===========================================================================
def fig_perspective():
    tf.random.set_seed(7)
    ex = scene_to_example(0)
    img_o, box_o, keep, poly_o = random_perspective(
        ex["image"], ex["groundtruth_boxes"], ex["groundtruth_polygons"],
        target_h=S, target_w=S, degrees=10.0, translate=0.1,
        scale_min=0.6, scale_max=1.2, shear=2.0, perspective=0.0, area_thresh=0.1)
    keep = keep.numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    _show(ax[0], ex["image"].numpy(), ex["groundtruth_boxes"].numpy(),
          ex["groundtruth_polygons"].numpy(), "Before — original image + GT")
    _show(ax[1], img_o.numpy(), box_o.numpy()[keep], poly_o.numpy()[keep],
          "After — random_perspective (rotate·scale·shear·translate)")
    fig.suptitle("Geometric augmentation: data_pipeline.augmentations.random_perspective",
                 fontsize=12, y=0.99)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_perspective.png", dpi=130); plt.close(fig)


# ===========================================================================
# 2. Mosaic 4-in-1 (REAL Mosaic._mosaic)
# ===========================================================================
def fig_mosaic():
    m = Mosaic(output_size=[S, S], mosaic_frequency=1.0, mosaic_center=0.12,
               aug_scale_min=0.85, aug_scale_max=1.0, degrees=6.0, shear=1.5,
               perspective=0.0, translate=0.03, area_thresh=0.05,
               group_size=4, decodes_per_output=4)
    exs = [scene_to_example(i) for i in range(4)]
    # pick the seed that fills the frame best (least gray-114 padding) so the
    # 4-quadrant stitch is clearly legible.
    best, best_gray = None, 1.0
    for s in range(40):
        tf.random.set_seed(s)
        cand = m._mosaic(exs[0], exs[1], exs[2], exs[3])
        gray = float(tf.reduce_mean(tf.cast(
            tf.reduce_all(tf.equal(cand["image"], 114), axis=-1), tf.float32)))
        if gray < best_gray:
            best_gray, best = gray, cand
    out = best

    fig = plt.figure(figsize=(11.5, 6.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.15, 2.0])
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, (r, c) in enumerate(pos):
        ax = fig.add_subplot(gs[r, c])
        _show(ax, exs[i]["image"].numpy(), title=f"source {i + 1}")
    axm = fig.add_subplot(gs[:, 3])
    _show(axm, out["image"].numpy(), out["groundtruth_boxes"].numpy(),
          out["groundtruth_polygons"].numpy(),
          "Mosaic output (2x canvas -> one warp -> 672x672)")
    fig.suptitle("Mosaic augmentation: 4 sources stitched on a 2x canvas, warped once "
                 "(data_pipeline.mosaic.Mosaic)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_mosaic.png", dpi=130); plt.close(fig)


# ===========================================================================
# 3. Horizontal flip (REAL) + 4. HSV jitter (REAL)
# ===========================================================================
def fig_flip_hsv():
    ex = scene_to_example(1)
    # force a flip by seeding random_horizontal_flip until it flips
    flipped = None
    for s in range(20):
        tf.random.set_seed(s)
        im, bx, pl = random_horizontal_flip(ex["image"], ex["groundtruth_boxes"],
                                             ex["groundtruth_polygons"])
        if not bool(tf.reduce_all(tf.equal(im, ex["image"]))):
            flipped = (im, bx, pl); break
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    _show(ax[0], ex["image"].numpy(), ex["groundtruth_boxes"].numpy(),
          ex["groundtruth_polygons"].numpy(), "original")
    im, bx, pl = flipped
    _show(ax[1], im.numpy(), bx.numpy(), pl.numpy(), "random_horizontal_flip")
    base = tf.cast(ex["image"], tf.float32) / 255.0
    for j, sd in enumerate((11, 23)):
        tf.random.set_seed(sd)
        aug = hsv_augment(base, hue=0.05, sat=0.7, val=0.4)
        _show(ax[2 + j], (aug.numpy() * 255).astype(np.uint8), title=f"hsv_augment draw {j + 1}")
    fig.suptitle("Photometric & flip augmentation (real ops; HSV runs per-batch on GPU at train time)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_flip_hsv.png", dpi=120); plt.close(fig)


# ===========================================================================
# 5. Arc-length polygon resampling (REAL resample_polygons)
# ===========================================================================
def fig_resample():
    # a 4-corner rectangle: the classic case the arc-length resample fixes
    rect = np.array([[0.25, 0.30], [0.75, 0.30], [0.75, 0.70], [0.25, 0.70]], np.float32)
    flat = np.full((24 * 2,), -1.0, np.float32); flat[:8] = rect.reshape(-1)
    poly = tf.constant(flat[None], tf.float32)
    out = resample_polygons(poly, max_points=64).numpy()[0].reshape(-1, 2)
    out = out[out[:, 0] > -1.0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.0))
    for a in ax:
        a.set_xlim(0, 1); a.set_ylim(1, 0); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    ax[0].add_patch(MplPolygon(rect, closed=True, fill=False, edgecolor=BLUE, lw=2))
    ax[0].scatter(rect[:, 0], rect[:, 1], c=RED, zorder=5, s=60)
    ax[0].set_title("Stored GT: 4 corner vertices\n(-> only ~4 angular bins occupied)")
    ax[1].add_patch(MplPolygon(out, closed=True, fill=False, edgecolor=BLUE, lw=2))
    ax[1].scatter(out[:, 0], out[:, 1], c=GREEN, zorder=5, s=18)
    ax[1].set_title(f"After resample_polygons(K=64)\n({len(out)} points ALONG edges -> all bins)")
    fig.suptitle("Arc-length resampling makes the 24-bin radial target track real shapes",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_resample.png", dpi=130); plt.close(fig)


# ===========================================================================
# 6. PolyYOLO radial encoding (schematic, faithful to _preprocess_polygons_v2)
# ===========================================================================
def fig_radial():
    # an irregular polygon around a center; encode to 24 bins (max radius per bin)
    cx, cy = 0.5, 0.5
    ang = np.array([10, 35, 70, 95, 130, 165, 200, 240, 285, 320], float) * math.pi / 180
    rad = np.array([0.34, 0.30, 0.40, 0.22, 0.33, 0.28, 0.38, 0.25, 0.42, 0.30])
    verts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)

    step = ANGLE_STEP_DEG * math.pi / 180
    bin_dist = np.zeros(NBINS); bin_conf = np.zeros(NBINS); bin_off = np.zeros(NBINS)
    for vx, vy in verts:
        a = math.atan2(vy - cy, vx - cx) % (2 * math.pi)
        b = int(a // step)
        r = math.hypot(vx - cx, vy - cy)
        if r > bin_dist[b]:
            bin_dist[b] = r; bin_conf[b] = 1.0; bin_off[b] = (a - b * step) / step

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    a0 = ax[0]
    a0.set_xlim(0, 1); a0.set_ylim(1, 0); a0.set_aspect("equal")
    a0.set_xticks([]); a0.set_yticks([])
    for i in range(NBINS):
        th = i * step
        a0.plot([cx, cx + 0.46 * math.cos(th)], [cy, cy + 0.46 * math.sin(th)],
                color="#cbd5e1", lw=0.8, zorder=1)
    a0.add_patch(MplPolygon(verts, closed=True, fill=False, edgecolor=BLUE, lw=2, zorder=2))
    a0.scatter(*verts.T, c=RED, s=40, zorder=4)
    for i in range(NBINS):
        if bin_conf[i]:
            th = (i + bin_off[i]) * step
            a0.plot([cx, cx + bin_dist[i] * math.cos(th)], [cy, cy + bin_dist[i] * math.sin(th)],
                    color=GREEN, lw=1.8, zorder=3)
    a0.scatter([cx], [cy], c="k", marker="+", s=120, zorder=5)
    a0.set_title("24 angular bins (15 deg).  green = encoded radius per occupied bin\n"
                 "origin = box center; vertex angle = (i + offset)*15 deg")

    a1 = ax[1]
    x = np.arange(NBINS)
    a1.bar(x - 0.2, bin_dist, 0.4, color=GREEN, label="dist (radius)")
    a1.bar(x + 0.2, bin_conf, 0.4, color=ORANGE, label="conf (1=occupied)")
    a1.set_xlabel("angular bin index i (0..23)")
    a1.set_title("Radial target per bin: [dist, angle_offset, conf] x 24\n"
                 "(empty bins -> dist 0, conf 0; conf is the decode gate)")
    a1.legend(fontsize=9); a1.set_xticks(x[::2])
    fig.suptitle("PolyYOLO radial polygon encoding (losses/tal_loss.py:_polygon_loss target)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_radial.png", dpi=130); plt.close(fig)


# ===========================================================================
# 7. LR + bias schedule (REAL optimizer formulas) & 8. EMA decay
# ===========================================================================
def fig_schedules():
    steps_per_epoch = 2118
    epochs = 300
    total = steps_per_epoch * epochs
    warm = 7164
    base_lr, alpha, bias_lr = 0.01, 0.01, 0.1
    t = np.linspace(0, total, 1500)

    def cosine(s):
        s = np.clip(s, 0, total)
        return base_lr * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(math.pi * s / total)))

    weight_lr = np.where(t < warm, cosine(t) * (t / warm), cosine(t))
    bias_ramp = np.where(t < warm, bias_lr + (cosine(t) - bias_lr) * (t / warm), cosine(t))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(t, weight_lr, color=BLUE, label="weight group (ramp UP from 0)")
    ax[0].plot(t, bias_ramp, color=ORANGE, label="BN/bias group (ramp DOWN from 0.1)")
    ax[0].axvline(warm, color="#94a3b8", ls="--", lw=1)
    ax[0].annotate("warmup\nends", (warm, bias_lr), (warm * 3, 0.07), fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="#94a3b8"))
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("learning rate")
    ax[0].set_title("SGD warmup + cosine decay (optimizers/sgd_warmup.py)")
    ax[0].legend(fontsize=8)

    st = np.arange(0, 4000)
    decay = np.minimum(0.9999, (1 + st) / (10 + st))
    ax[1].plot(st, decay, color=GREEN)
    ax[1].axhline(0.9999, color="#94a3b8", ls="--", lw=1)
    ax[1].set_xlabel("EMA update step"); ax[1].set_ylabel("decay")
    ax[1].set_title("EMA dynamic decay  min(0.9999, (1+t)/(10+t))\n(optimizers/ema.py)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_schedules.png", dpi=130); plt.close(fig)


# ===========================================================================
# 9. DFL box regression (schematic)
# ===========================================================================
def fig_dfl():
    bins = np.arange(16)
    logits = -((bins - 6.3) ** 2) / 5.0
    p = np.exp(logits) / np.exp(logits).sum()
    exp = (p * bins).sum()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(bins, p, color=BLUE, alpha=0.85)
    ax.axvline(exp, color=RED, lw=2, label=f"E[bin] = sum(p*bin) = {exp:.2f}")
    ax.set_xlabel("DFL bin (0..15)  -> one of the 4 box sides (l,t,r,b)")
    ax.set_ylabel("softmax prob")
    ax.set_title("Distribution Focal Loss: each side is a softmax over 16 bins;\n"
                 "the decoded distance is the expected bin value (x stride)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_dfl.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    print("Generating figures into", FIG)
    for fn in (fig_perspective, fig_mosaic, fig_flip_hsv, fig_resample,
               fig_radial, fig_schedules, fig_dfl):
        fn(); print("  ok:", fn.__name__)
    print("done.")

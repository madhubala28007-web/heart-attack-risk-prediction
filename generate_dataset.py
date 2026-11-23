"""Creates a synthetic retinal fundus image dataset (for demonstration / testing).

Real, outcome-labelled retinal datasets are restricted, so this script draws
fundus-like images whose vessel properties follow published clinical findings:
  * high risk -> narrower arterioles, wider venules, more tortuous vessels,
                 fewer small branches (rarefaction), more haemorrhages/exudates.
If you have real images, simply place them in data/retina/low_risk and
data/retina/high_risk and skip this script.
"""
import os
import argparse
import numpy as np
import cv2
from config import DATA_DIR, CLASS_NAMES, IMG_SIZE, RANDOM_STATE

SS = 2  # super-sampling factor (draw large, shrink later -> smooth thin vessels)


def _grow_vessel(canvas, start, angle, length, width, depth, rng, p, color):
    """Draw one vessel as a wiggly poly-line and recursively add child branches."""
    step = 6.0 * SS
    n = max(int(length / step), 3)
    curve = rng.normal(0, 0.012)
    amp = p["tort"] * rng.uniform(0.7, 1.3)
    freq = rng.uniform(0.35, 0.8)
    phase = rng.uniform(0, 2 * np.pi)
    pts = [np.array(start, dtype=float)]
    for i in range(n):
        heading = angle + curve * i + amp * np.sin(freq * i + phase) + rng.normal(0, 0.03)
        pts.append(pts[-1] + step * np.array([np.cos(heading), np.sin(heading)]))
    poly = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [poly], False, color, max(1, int(round(width))), cv2.LINE_AA)
    if depth > 0 and width > 2.0:
        for sign in (-1, 1):
            if rng.random() < p["branch"]:
                k = int(rng.integers(max(2, n // 4), n))
                _grow_vessel(canvas, pts[k], angle + sign * rng.uniform(0.4, 0.9),
                             length * 0.6, width * 0.62, depth - 1, rng, p, color)


def make_fundus(latent, rng, size=IMG_SIZE):
    """latent in [0,1] : hidden cardiovascular-risk level that drives the image."""
    S = size * SS
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.hypot(xx - S / 2, yy - S / 2) / (S / 2)
    tone = rng.uniform(0.85, 1.15)
    img = np.zeros((S, S, 3), np.float32)
    img[..., 2] = (200 - 70 * r ** 2) * tone    # red channel
    img[..., 1] = (85 - 35 * r ** 2) * tone     # green channel
    img[..., 0] = (35 - 15 * r ** 2) * tone     # blue channel
    img = np.clip(img, 0, 255).astype(np.uint8)

    # optic disc
    disc = (int(S * 0.66), int(S * 0.5 + rng.integers(-15, 15) * SS))
    cv2.circle(img, disc, 20 * SS, (110, 190, 240), -1, cv2.LINE_AA)
    cv2.circle(img, disc, 8 * SS, (150, 215, 250), -1, cv2.LINE_AA)

    # lesions (haemorrhages dark red, exudates yellow-white)
    for _ in range(rng.poisson(0.4 + 5 * latent ** 2)):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.75) * S / 2
        c = (int(S / 2 + d * np.cos(a)), int(S / 2 + d * np.sin(a)))
        cv2.circle(img, c, int(rng.uniform(2, 4) * SS), (25, 25, 120), -1, cv2.LINE_AA)
    for _ in range(rng.poisson(0.3 + 4 * latent ** 2)):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.75) * S / 2
        c = (int(S / 2 + d * np.cos(a)), int(S / 2 + d * np.sin(a)))
        cv2.circle(img, c, int(rng.uniform(1, 3) * SS), (130, 210, 235), -1, cv2.LINE_AA)

    # vessel tree : 2 arteries (thin, bright) + 2 veins (thick, dark)
    p = {"tort": 0.04 + 0.22 * latent + abs(rng.normal(0, 0.02)),
         "branch": float(np.clip(0.95 - 0.5 * latent + rng.normal(0, 0.05), 0.25, 0.98))}
    art_w = (3.4 - 1.5 * latent + rng.normal(0, 0.25)) * SS
    vein_w = (4.4 + 0.9 * latent + rng.normal(0, 0.25)) * SS
    angles = np.pi + np.array([-1.05, -0.5, 0.5, 1.05]) + rng.normal(0, 0.08, 4)
    kinds = ["artery", "vein", "artery", "vein"]
    for ang, kind in zip(angles, kinds):
        w, col = (art_w, (35, 50, 170)) if kind == "artery" else (vein_w, (20, 30, 120))
        _grow_vessel(img, disc, ang, rng.uniform(150, 190) * SS, w, 3, rng, p, col)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    img += rng.normal(0, 3, img.shape)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 3, 1, -1)
    img = np.clip(img, 0, 255).astype(np.uint8) * mask[..., None]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=240, help="images per class")
    args = ap.parse_args()
    rng = np.random.default_rng(RANDOM_STATE)
    means = {"low_risk": 0.30, "high_risk": 0.70}
    for cls in CLASS_NAMES:
        folder = os.path.join(DATA_DIR, cls)
        os.makedirs(folder, exist_ok=True)
        for i in range(args.per_class):
            latent = float(np.clip(rng.normal(means[cls], 0.17), 0, 1))
            cv2.imwrite(os.path.join(folder, f"{cls}_{i:04d}.jpg"), make_fundus(latent, rng))
        print(f"[OK] {args.per_class} images saved in {folder}")


if __name__ == "__main__":
    main()

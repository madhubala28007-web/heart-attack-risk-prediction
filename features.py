"""Feature extraction : turns a segmented retinal image into a numeric vector."""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, shannon_entropy
from preprocess import run_pipeline, find_optic_disc

FEATURE_NAMES = [
    "vessel_density", "skeleton_density", "mean_width", "std_width",
    "width_p25", "width_p75", "width_ratio", "tortuosity_mean", "tortuosity_p90",
    "branch_density", "end_density", "fractal_dim", "red_lesions", "bright_lesions",
    "mean_red", "mean_green", "mean_blue", "std_green", "red_green_ratio",
    "entropy", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
]

_K8 = np.ones((3, 3), int)
_K8[1, 1] = 0
_OFFS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def neighbour_count(skel):
    return ndi.convolve(skel.astype(int), _K8, mode="constant") * skel


def box_counting_dimension(binary):
    """Fractal dimension of the vessel pattern (box-counting method)."""
    S = min(binary.shape)
    b = binary[:S, :S]
    sizes = np.array([2, 4, 8, 16, 32, 64])
    counts = []
    for s in sizes:
        n = S // s
        view = b[:n * s, :n * s].reshape(n, s, n, s)
        counts.append(max(1, int(view.any(axis=(1, 3)).sum())))
    slope = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)[0]
    return float(slope)


def _trace_length(coords, start):
    """Walk along a skeleton segment and return (path length, chord length)."""
    cs = set(coords)
    path, seen, cur = [start], {start}, start
    while True:
        nxt = None
        for dy, dx in _OFFS:
            q = (cur[0] + dy, cur[1] + dx)
            if q in cs and q not in seen:
                nxt = q
                break
        if nxt is None:
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    arc = sum(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
              for i in range(len(path) - 1))
    chord = np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
    return arc, chord


def tortuosity(skel):
    """Mean and 90th-percentile of (arc length / chord length) of vessel segments."""
    nb = neighbour_count(skel)
    branch = ndi.binary_dilation(skel & (nb >= 3), structure=np.ones((3, 3)))
    segs = skel & ~branch
    lab, n = ndi.label(segs, structure=np.ones((3, 3)))
    vals = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        sub = lab[sl] == i
        if sub.sum() < 12:
            continue
        snb = neighbour_count(sub)
        ends = np.argwhere(sub & (snb == 1))
        if len(ends) < 1:
            continue
        coords = [tuple(c) for c in np.argwhere(sub)]
        arc, chord = _trace_length(coords, tuple(ends[0]))
        if chord > 5:
            vals.append(min(arc / chord, 3.0))
    if not vals:
        return 1.0, 1.0
    return float(np.mean(vals)), float(np.percentile(vals, 90))


def count_blobs(binary, min_area, max_area, max_ecc=0.85):
    """Count compact (round) blobs; long thin shapes (vessel pieces) are ignored."""
    lab = label(binary)
    return sum(1 for r in regionprops(lab)
               if min_area <= r.area <= max_area and r.eccentricity < max_ecc)


def lesion_counts(green, mask, vessels):
    """Red lesions (haemorrhages) and bright lesions (exudates)."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dark = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, se)
    vessel_zone = cv2.dilate(vessels.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    red = (dark > 25) & mask & ~vessel_zone
    bright = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, se)
    dy, dx = find_optic_disc(green, mask)
    yy, xx = np.ogrid[:green.shape[0], :green.shape[1]]
    away = (yy - dy) ** 2 + (xx - dx) ** 2 > 40 ** 2
    bri = (bright > 35) & mask & away & ~vessel_zone
    return count_blobs(red, 8, 250), count_blobs(bri, 3, 250)


def extract_features_from_result(res):
    img, mask, green = res["image"], res["mask"], res["green"]
    vessels, skel = res["vessels"], res["skeleton"]
    area = float(mask.sum())
    dist = ndi.distance_transform_edt(vessels)
    widths = 2.0 * dist[skel] if skel.any() else np.array([0.0])
    nb = neighbour_count(skel)
    skel_len = max(float(skel.sum()), 1.0)
    tort_mean, tort_p90 = tortuosity(skel)
    red_n, bright_n = lesion_counts(green, mask, vessels)
    p25, p75 = np.percentile(widths, 25), np.percentile(widths, 75)

    gq = (green // 8).astype(np.uint8)                  # 32 grey levels
    glcm = graycomatrix(gq, distances=[1, 3], angles=[0], levels=32,
                        symmetric=True, normed=True)
    b, g, r = [img[:, :, c][mask].astype(float) for c in range(3)]

    feats = [
        vessels.sum() / area, skel_len / area, widths.mean(), widths.std(),
        p25, p75, p25 / (p75 + 1e-9),
        tort_mean, tort_p90,
        (skel & (nb >= 3)).sum() / skel_len, (skel & (nb == 1)).sum() / skel_len,
        box_counting_dimension(vessels), red_n, bright_n,
        r.mean(), g.mean(), b.mean(), g.std(), r.mean() / (g.mean() + 1e-9),
        shannon_entropy(green[mask]),
        graycoprops(glcm, "contrast").mean(), graycoprops(glcm, "homogeneity").mean(),
        graycoprops(glcm, "energy").mean(), graycoprops(glcm, "correlation").mean(),
    ]
    return np.array(feats, dtype=float)


def extract_features(path):
    """Image path -> feature vector (numpy array of len(FEATURE_NAMES))."""
    return extract_features_from_result(run_pipeline(path))

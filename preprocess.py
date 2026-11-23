"""Image pre-processing and retinal blood-vessel segmentation."""
import warnings
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize
from config import IMG_SIZE

warnings.filterwarnings("ignore", category=FutureWarning)


def load_image(path, size=IMG_SIZE):
    """Read a fundus image (BGR) and resize it to size x size."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def get_mask(img, erode=8):
    """Binary mask of the circular retina area (black corners removed)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    return cv2.erode(mask, k).astype(bool)


def enhance(img, mask):
    """Green channel + CLAHE contrast enhancement + light smoothing."""
    green = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(green)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    g[~mask] = int(g[mask].mean())          # avoid strong edge at the border
    return g


def find_optic_disc(green, mask):
    """Optic disc = brightest smooth region of the retina; returns (row, col)."""
    blur = cv2.GaussianBlur(green, (31, 31), 0)
    blur[~mask] = 0
    return np.unravel_index(np.argmax(blur), blur.shape)


def segment_vessels(green_enh, mask, rel_thresh=0.10, disc_radius=26):
    """Frangi vesselness filter -> threshold -> clean -> skeleton."""
    resp = frangi(green_enh.astype(float) / 255.0, sigmas=(1.0, 1.5, 2.0, 2.5),
                  black_ridges=True)
    resp[~mask] = 0
    dy, dx = find_optic_disc(green_enh, mask)           # ignore the disc border
    yy, xx = np.ogrid[:resp.shape[0], :resp.shape[1]]
    resp[(yy - dy) ** 2 + (xx - dx) ** 2 < disc_radius ** 2] = 0
    ref = np.percentile(resp[mask], 99.5) + 1e-9
    binary = (resp / ref) > rel_thresh
    binary = remove_small_objects(binary, min_size=30)
    skeleton = skeletonize(binary)
    return binary, skeleton


def run_pipeline(path):
    """Full pre-processing chain; returns every intermediate result."""
    img = load_image(path)
    mask = get_mask(img)
    green = enhance(img, mask)
    vessels, skeleton = segment_vessels(green, mask)
    return {"image": img, "mask": mask, "green": green,
            "vessels": vessels, "skeleton": skeleton}

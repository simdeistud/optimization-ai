import subprocess
import lilyponddist
import numpy as np
from PIL import Image
from fitness.base_ff_classes.base_ff import base_ff
from algorithm.parameters import params

class score_ff(base_ff):
    def __init__(self):
        super().__init__()
        self.maximise = True

    def evaluate(self, ind, **kwargs):
        # We add additional data to complete the score
        pre = "\\version \"2.24.0\" \\paper { tagline = ##f } \\score { \\new Staff { \\relative c'' { \\key c \\major \\bar \"|\" "
        post = " \\bar \"|\" }} \\layout { }}"
        print(ind)
        # We write the score to a tmp location (lilypond doesn't work well with stdin and stdout)
        with open('/tmp/currscore.ly', "w", encoding="utf-8") as f:
            f.write(f"{pre}{ind.phenotype}{post}")
        if subprocess.call([lilyponddist.lilypondbin(), '--png', '-o', '/tmp/currscore', '/tmp/currscore.ly']) == 1:
            fit =  self.default_fitness
        else:
            fit =  common_black_pixels("/tmp/currscore.png", params['TARGET'])
        return fit

def load_rgba_to_gray(path: str) -> np.ndarray:
    """
    Load an image, composite alpha over white, convert to grayscale.
    Returns a float32 array in [0, 255].
    """
    img = Image.open(path).convert("RGBA")
    rgba = np.asarray(img, dtype=np.uint8)

    # Composite alpha over white: out = rgb*alpha + white*(1-alpha)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    comp = rgb * alpha + 255.0 * (1.0 - alpha)

    # Luma (Rec. 709): Y = 0.2126 R + 0.7152 G + 0.0722 B
    gray = 0.2126 * comp[..., 0] + 0.7152 * comp[..., 1] + 0.0722 * comp[..., 2]
    return gray.astype(np.float32)


def binarize(gray: np.ndarray, threshold: float = 128.0) -> np.ndarray:
    """
    Convert grayscale to pure B&W using a global threshold.
    Output is uint8 {0, 255}: 0 = black, 255 = white.
    """
    bw = np.where(gray < threshold, 0, 255).astype(np.uint8)
    return bw


def common_black_pixels(img1_path: str, img2_path: str, threshold: float = 128.0):
    """
    - Binarizes both images to pure B&W using `threshold`.
    - Computes count of black pixels present in both (intersection).
    - Handles different sizes by using overlapping area (top-left aligned).
    Returns: (count_common_black, bw1, bw2) where bw1/bw2 are the binarized images.
    """
    g1 = load_rgba_to_gray(img1_path)
    g2 = load_rgba_to_gray(img2_path)

    # Use overlapping region if sizes differ
    h = min(g1.shape[0], g2.shape[0])
    w = min(g1.shape[1], g2.shape[1])
    g1c = g1[:h, :w]
    g2c = g2[:h, :w]

    bw1 = binarize(g1c, threshold)
    bw2 = binarize(g2c, threshold)

    # Common black pixels = positions where both are 0
    common_black = np.sum((bw1 == 0) & (bw2 == 0))
    print(f"fitness: {common_black}")
    return common_black

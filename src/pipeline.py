import numpy as np
import paths as PATH
import cv2 as cv
from segmentation import find_iris, find_pupil
import config as CFG


def load(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img


def find_pupil_iris_boundaries(image):
    x_pupil, y_pupil, r_pupil = find_pupil(image)
    if x_pupil is None:
        raise ValueError("Nie wykryto źrenicy.")

    x_iris, y_iris, r_iris = find_iris(image, x_pupil, y_pupil, r_pupil)
    if x_iris is None:
        raise ValueError("Nie wykryto tęczówki.")

    blend = CFG.IRIS_CENTER_BLEND
    x_iris = int(round(blend * x_pupil + (1.0 - blend) * x_iris))
    y_iris = int(round(blend * y_pupil + (1.0 - blend) * y_iris))

    if r_iris < r_pupil * 1.4:
        r_iris = int(r_pupil * 2.0)

    return (x_pupil, y_pupil, r_pupil), (x_iris, y_iris, r_iris)


def normalize_iris(img, pupil_circle, iris_circle, height=None, width=None):
    """
    Normalizacja tęczówki (rubber sheet).
    Zewnętrzna granica używa środka źrenicy — stabilniejsze między sesjami.
    """
    height = height or CFG.NORMALIZED_HEIGHT
    width = width or CFG.NORMALIZED_WIDTH

    xp, yp, rp = map(float, pupil_circle)
    _, _, ri = map(float, iris_circle)

    thetas = np.linspace(0, 2 * np.pi, width, endpoint=False)
    rs = np.linspace(0, 1, height)
    theta_mat, r_mat = np.meshgrid(thetas, rs)

    x_pupil = xp + rp * np.cos(theta_mat)
    y_pupil = yp + rp * np.sin(theta_mat)
    x_iris = xp + ri * np.cos(theta_mat)
    y_iris = yp + ri * np.sin(theta_mat)

    target_x = (1 - r_mat) * x_pupil + r_mat * x_iris
    target_y = (1 - r_mat) * y_pupil + r_mat * y_iris

    normalized_iris = cv.remap(
        img,
        target_x.astype(np.float32),
        target_y.astype(np.float32),
        cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE,
    )
    return normalized_iris


def create_iris_mask(normalized_iris):
    """Maska: powieki + kolumny z wysoką wariancją (rzęsy / cienie)."""
    mask = np.ones(normalized_iris.shape, dtype=np.uint8)
    h, w = normalized_iris.shape

    mask[0 : int(h * 0.10), :] = 0
    mask[int(h * 0.78) :, :] = 0

    col_std = np.std(normalized_iris.astype(np.float32), axis=0)
    valid_cols = col_std > 0
    if np.any(valid_cols):
        limit = np.percentile(col_std[valid_cols], 88)
        mask[:, col_std > limit] = 0

    return mask


def _preprocess_strip(normalized_iris, mask):
    """Wygładzenie i wyrównanie kontrastu przed filtrami Gabora."""
    strip = normalized_iris.astype(np.float32)
    strip = cv.medianBlur(strip.astype(np.uint8), 3).astype(np.float32)
    strip = cv.bilateralFilter(strip.astype(np.uint8), d=5, sigmaColor=40, sigmaSpace=40).astype(np.float32)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(strip.astype(np.uint8)).astype(np.float32)

    valid = mask > 0
    if not np.any(valid):
        raise ValueError("Maska tęczówki jest pusta — segmentacja nieudana.")

    mean_val = float(enhanced[valid].mean())
    std_val = float(enhanced[valid].std()) + 1e-5
    return (enhanced - mean_val) / std_val


def encode_iris(normalized_iris, mask):
    """Kod Daugmana: 4 orientacje x 2 fazy kwadraturowe."""
    img_normalized = _preprocess_strip(normalized_iris, mask)

    bit_planes = []
    ksize = (CFG.GABOR_KERNEL_SIZE, CFG.GABOR_KERNEL_SIZE)

    for theta in CFG.GABOR_ORIENTATIONS:
        for psi in (0, np.pi / 2):
            kernel = cv.getGaborKernel(
                ksize=ksize,
                sigma=CFG.GABOR_SIGMA,
                theta=theta,
                lambd=CFG.GABOR_LAMBDA,
                gamma=CFG.GABOR_GAMMA,
                psi=psi,
                ktype=cv.CV_32F,
            )
            kernel -= np.mean(kernel)
            response = cv.filter2D(img_normalized, cv.CV_32F, kernel)
            bit_planes.append((response > 0).astype(np.uint8))

    return np.stack(bit_planes, axis=-1), mask


def calculate_iris_code(image_path):
    image = load(image_path)
    pupil, iris = find_pupil_iris_boundaries(image)
    normalized_iris = normalize_iris(image, pupil, iris)
    mask = create_iris_mask(normalized_iris)
    iris_code, binary_mask = encode_iris(normalized_iris, mask)
    return iris_code, binary_mask

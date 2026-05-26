import numpy as np
import config as CFG


def calculate_hamming_distance(code1, mask1, code2, mask2):
    """Odległość Hamminga na wszystkich bitach kodu (wszystkie orientacje Gabora)."""
    combined_mask = np.logical_and(mask1, mask2)
    if np.sum(combined_mask) == 0:
        return 1.0

    diff = np.logical_xor(code1, code2)
    if diff.ndim == 3:
        return float(np.mean(diff[combined_mask]))

    reliable_diffs = np.logical_and(diff, combined_mask)
    return np.sum(reliable_diffs) / np.sum(combined_mask)


def compare_codes_with_shift(code1, mask1, code2, mask2, shift_range=None):
    """Dwuetapowe przesunięcie w poziomie (kompensacja rotacji oka)."""
    shift_range = CFG.SHIFT_RANGE if shift_range is None else shift_range
    coarse_step = CFG.SHIFT_COARSE_STEP

    best_shift = 0
    min_hd = 1.0

    for s in range(-shift_range, shift_range + 1, coarse_step):
        shifted_code = np.roll(code2, s, axis=1)
        shifted_mask = np.roll(mask2, s, axis=1)
        hd = calculate_hamming_distance(code1, mask1, shifted_code, shifted_mask)
        if hd < min_hd:
            min_hd = hd
            best_shift = s

    fine_lo = max(-shift_range, best_shift - coarse_step)
    fine_hi = min(shift_range, best_shift + coarse_step)
    for s in range(fine_lo, fine_hi + 1):
        shifted_code = np.roll(code2, s, axis=1)
        shifted_mask = np.roll(mask2, s, axis=1)
        hd = calculate_hamming_distance(code1, mask1, shifted_code, shifted_mask)
        if hd < min_hd:
            min_hd = hd

    return min_hd

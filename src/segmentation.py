import cv2
import numpy as np


def find_pupil(image):
    """Zwraca (x, y, r) źrenicy lub None."""
    gray = _to_gray(image)
    h, w = gray.shape

    m = 15
    roi = gray[m:h-m, m:w-m]
    blurred = cv2.medianBlur(roi, 11)
    min_val = int(blurred.min())

    for thresh_margin in [25, 40, 60]:
        _, thresh = cv2.threshold(blurred, min_val + thresh_margin, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > roi.size * 0.15:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ > 0.55:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                candidates.append((circ * area, x + m, y + m, r))

        if candidates:
            candidates.sort(reverse=True)
            _, cx, cy, r = candidates[0]
            return int(cx), int(cy), int(r)

    return None


def find_iris(image, pupil_x, pupil_y, pupil_r):
    """
    Szuka granicy tęczówki metodą radialnego profilu jasności.

    Tęczówka jest ciemna, twardówka jasna — idąc promieniami od centrum
    źrenicy na boki (pomijamy powieki), szukamy gwałtownego skoku jasności.

    Etap 1: coarse — pełny profil radialny od centrum źrenicy, peak gradientu
            daje iris_r_est.
    Etap 2: fine — przeszukujemy środek w oknie ±14px oceniając gradient
            TYLKO w oknie promieni wokół iris_r_est (nie szukamy nowego peaka).
    Etap 3: finalny promień z najlepszego centrum.

    Zwraca: (x, y, r) lub None.
    """
    gray = _to_gray(image)
    h, w = gray.shape

    # Mocne rozmycie niszczy rzęsy i teksturę tęczówki
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=4.0)

    r_min = int(pupil_r * 1.5)
    r_max = int(pupil_r * 4.5)
    if r_min >= r_max:
        return None

    # Kąty boczne — pomijamy ±70° od pionu (powieki)
    angles_deg = list(range(-65, 66, 4)) + list(range(115, 246, 4))
    cos_a = np.cos(np.deg2rad(angles_deg))
    sin_a = np.sin(np.deg2rad(angles_deg))
    r_vals = np.arange(r_min, r_max)

    def radial_profile(cx, cy):
        profile = []
        for r in r_vals:
            xs = np.clip((cx + r * cos_a).astype(np.int32), 0, w - 1)
            ys = np.clip((cy + r * sin_a).astype(np.int32), 0, h - 1)
            profile.append(float(smooth[ys, xs].mean()))
        return np.array(profile)

    def peak_radius(profile):
        deriv = np.abs(np.diff(profile))
        k = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
        ds = np.convolve(deriv, k, mode='same') if len(deriv) >= 5 else deriv
        return int(r_vals[np.argmax(ds)])

    # ── Etap 1: coarse ─────────────────────────────────────────
    iris_r_est = peak_radius(radial_profile(pupil_x, pupil_y))

    # ── Etap 2: fine-search środka ─────────────────────────────
    # Oceniamy gradient (out - in) przy iris_r_est, nie szukamy nowego peaka
    slack = 14
    r_window = np.arange(max(r_min, iris_r_est - 6), min(r_max - 1, iris_r_est + 7))

    best_score = -1.0
    best_cx, best_cy = pupil_x, pupil_y

    for dcx in range(-slack, slack + 1, 2):
        for dcy in range(-slack, slack + 1, 2):
            cx = pupil_x + dcx
            cy = pupil_y + dcy
            scores = []
            for r in r_window:
                xs_o = np.clip((cx + (r + 4) * cos_a).astype(np.int32), 0, w - 1)
                ys_o = np.clip((cy + (r + 4) * sin_a).astype(np.int32), 0, h - 1)
                xs_i = np.clip((cx + (r - 4) * cos_a).astype(np.int32), 0, w - 1)
                ys_i = np.clip((cy + (r - 4) * sin_a).astype(np.int32), 0, h - 1)
                scores.append(float(smooth[ys_o, xs_o].mean() - smooth[ys_i, xs_i].mean()))
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_cx, best_cy = cx, cy

    # ── Etap 3: finalny promień z najlepszego centrum ──────────
    best_r = peak_radius(radial_profile(best_cx, best_cy))

    return best_cx, best_cy, best_r


def find_iris_ray_casting(image, p_x, p_y, p_r):
    return find_iris(image, p_x, p_y, p_r)


def draw_result(image, pupil, iris):
    """Rysuje wynik na obrazie do podglądu."""
    gray = _to_gray(image)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if pupil:
        px, py, pr = pupil
        cv2.circle(vis, (px, py), pr, (0, 220, 80), 2)
        cv2.drawMarker(vis, (px, py), (0, 220, 80), cv2.MARKER_CROSS, 14, 2)
    if iris:
        ix, iy, ir = iris
        cv2.circle(vis, (ix, iy), ir, (255, 100, 0), 2)
        cv2.drawMarker(vis, (ix, iy), (255, 100, 0), cv2.MARKER_CROSS, 14, 2)
    return vis


def _to_gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

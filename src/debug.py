import matplotlib.pyplot as plt
from pipeline import load, find_pupil_iris_boundaries, normalize_iris, create_iris_mask
import cv2 as cv
import paths as PATH
from segmentation import find_pupil
import os
import numpy as np
from evaluation_with_visual import run_evaluation_IMPOSTORS, run_evaluation_USERS

def debug_show_image(image):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis('off')
    plt.show()

def debug_show_segmentation(image_path):
    img = load(image_path)
    try:
        pupil, iris = find_pupil_iris_boundaries(img)
        
        output = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        
        cv.circle(output, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 2)
        cv.circle(output, (pupil[0], pupil[1]), 2, (0, 255, 0), 3)
        
        cv.circle(output, (iris[0], iris[1]), iris[2], (0, 0, 255), 2)
        cv.circle(output, (iris[0], iris[1]), 2, (0, 0, 255), 3)
        
        plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
        plt.title("Wynik Segmentacji")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Błąd segmentacji: {e}")


def debug_show_pupil_segmentation(image):
    try:
        x, y, radius = find_pupil(image)
        
        output = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        cv.circle(output, (x,y), radius, (0, 255, 0), 2)
        cv.circle(output, (x, y), radius, (0, 255, 0), 3)
        
        plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
        plt.title("Wynik Segmentacji")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Błąd segmentacji: {e}")

def debug_load_and_preprocess(image_path):
    img = load(image_path)
    debug_show_image(img)


def debug_find_pupil(image_path):
    img = load(image_path)
    output = debug_show_pupil_segmentation(img)
    debug_show_image(output)


def debug_normalize(image_path):
    img = load(image_path)
    pupil, iris = find_pupil_iris_boundaries(img)
    res = normalize_iris(img, pupil, iris)
    mask = create_iris_mask(res)

    plt.figure(figsize=(15, 5))
    plt.imshow(mask, cmap='gray')
    plt.show()


def visualize_normalization_and_mask(image_path):
    image = load(image_path)
    pupil, iris = find_pupil_iris_boundaries(image)
    normalized_iris = normalize_iris(image, pupil, iris)
    mask = create_iris_mask(normalized_iris)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f"Analiza biometryczna dla: {image_path.name if hasattr(image_path, 'name') else image_path}", fontsize=14, fontweight='bold')

    axes[0].imshow(normalized_iris, cmap='gray')
    axes[0].set_title("Znormalizowany pasek tęczówki (Rubber Sheet Model) - 64x512 px")
    axes[0].axis('on')
    
    axes[1].imshow(mask * 255, cmap='gray')
    axes[1].set_title("Wygenerowana maska (Czarne obszary = odrzucone rzęsy/powieki)")
    axes[1].axis('on')
    
    plt.tight_layout()
    plt.show()

def visualize_raw_dataset(base_data_path, max_images=12):
    """
    Wczytuje pierwsze 12 oryginalnych zdjęć z datasetu
    i wyświetla je w siatce 4 kolumny x 3 wiersze z odstępami w pionie.
    """
    image_paths = []
    user_ids = []
    
    # Sortujemy foldery, aby zachować stałą kolejność
    user_folders = sorted(os.listdir(base_data_path))
    
    # Przeszukujemy foldery użytkowników w poszukiwaniu pierwszego zdjęcia z katalogu "L"
    for user_id in user_folders:
        eye_path = os.path.join(base_data_path, user_id, "L")
        if os.path.exists(eye_path):
            images = sorted(os.listdir(eye_path))
            if images:
                image_paths.append(os.path.join(eye_path, images[0]))
                user_ids.append(user_id)
        
        if len(image_paths) >= max_images:
            break

    if not image_paths:
        print("Nie znaleziono zdjęć. Sprawdź ścieżkę do bazy danych.")
        return

    # Przygotowanie okna matplotlib (siatka 4 kolumny x 3 wiersze)
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    fig.suptitle("Przykładowe surowe zdjęcia z bazy danych (Dataset)", fontsize=16, fontweight='bold', y=0.95)
    
    axes = axes.flatten()

    for idx, img_path in enumerate(image_paths):
        ax = axes[idx]
        
        # Wczytanie oryginalnego obrazu (w skali szarości)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        if img is not None:
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Użytkownik: {user_ids[idx]}", fontsize=12, pad=10)
        else:
            ax.text(0.5, 0.5, "Błąd wczytywania", color='red', ha='center', va='center')
            
        ax.axis('off')  # Ukrywamy osie liczbowe dla lepszego wyglądu

    # Jeśli zdjęć byłoby mniej niż 12, ukrywamy puste podwykresy
    for ax in axes[len(image_paths):]:
        ax.axis('off')

    # --- TUTAJ DOPASOWUJEMY ODSTĘPY ---
    # hspace steruje odstępem w pionie (height space) między siatkami wykresów
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    
    plt.show()


def plot_far_frr_curve(user_distances, impostor_distances):
    """
    Rysuje wykres krzywych FAR i FRR w funkcji thresholda (0.0 do 1.0).
    
    :param user_distances: lista lub tablica wszystkich odległości Hamminga 
                           zmierzonej dla LEGALNYCH użytkowników (testy TP/FN)
    :param impostor_distances: lista lub tablica minimalnych odległości 
                               zmierzonych dla IMPOSTORÓW (testy TN/FP)
    """
    # Tworzymy wektor progów od 0 do 1 (np. co 0.01)
    thresholds = np.linspace(0.0, 1.0, 101)
    
    far_list = []
    frr_list = []
    
    total_users = len(user_distances)
    total_impostors = len(impostor_distances)
    
    if total_users == 0 or total_impostors == 0:
        print("Błąd: Tablice odległości nie mogą być puste!")
        return

    # Obliczamy FAR i FRR dla każdego progu
    for t in thresholds:
        # FRR: Procent legalnych użytkowników odrzuconych (dystans >= t)
        false_rejections = np.sum(np.array(user_distances) >= t)
        frr = (false_rejections / total_users) * 100
        frr_list.append(frr)
        
        # FAR: Procent impostorów wpuszczonych (dystans < t)
        false_accepts = np.sum(np.array(impostor_distances) < t)
        far = (false_accepts / total_impostors) * 100
        far_list.append(far)

    # Znajdowanie punktu przecięcia (Equal Error Rate - EER)
    # Szukamy miejsca, gdzie różnica między FAR a FRR jest najmniejsza
    idx_eer = np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))
    eer_threshold = thresholds[idx_eer]
    eer_value = (far_list[idx_eer] + frr_list[idx_eer]) / 2

    # --- RYSOWANIE WYKRESU ---
    plt.figure(figsize=(10, 6))
    
    # Rysujemy dwie funkcje
    plt.plot(thresholds, far_list, label='FAR (Włamania / False Accept)', color='#d62728', lw=2.5)
    plt.plot(thresholds, frr_list, label='FRR (Błędne odrzucenia / False Reject)', color='#ff7f0e', lw=2.5)
    
    # Oznaczamy punkt przecięcia (EER)
    plt.plot(eer_threshold, eer_value, 'ro', markersize=8, label=f'EER (Punkt optymalny)')
    
    # Dodatkowe linie pomocnicze (przerywane) wskazujące punkt przecięcia
    plt.axvline(x=eer_threshold, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=eer_value, color='gray', linestyle='--', alpha=0.7)
    
    # Adnotacja tekstowa na wykresie
    plt.text(eer_threshold + 0.02, eer_value + 3, 
             f'Optymalny Threshold: {eer_threshold:.2f}\nBłąd EER: {eer_value:.2f}%', 
             color='black', weight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Stylizacja wykresu
    plt.title('Krzywe błędu FAR vs FRR (Wyznaczanie progu optymalnego)', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Próg decyzyjny (Threshold)', fontsize=12)
    plt.ylabel('Wskaźnik błędu (%)', fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(-2, 102)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper center', frameon=True, shadow=True, fontsize=11)
    
    plt.tight_layout()
    plt.show()

# for i in range(10):
#     file_name = f"002/L/S5002L0{i}.jpg"
#     debug_normalize(PATH.DATA_DIR / file_name)
    
# file_name = f"011/L/S5011L03.jpg"
# debug_show_segmentation(PATH.DATA_DIR / file_name)
# visualize_normalization_and_mask(PATH.DATA_DIR / file_name)
# debug_load_and_preprocess(PATH.DATA_DIR / "002/L/S5002L05.jpg")
# debug_find_pupil(PATH.DATA_DIR / "000/L/S5000L02.jpg")


# visualize_raw_dataset(PATH.DATA_DIR)

user_limit = 50
# _, _, user_distances = run_evaluation_USERS(PATH.DATA_DIR, users_limit=user_limit)
# _, _, impostors_distances = run_evaluation_IMPOSTORS(PATH.DATA_DIR, users_limit=user_limit)

# Odległości Hamminga dla 50 LEGALNYCH użytkowników (testy TP / FN)
user_distances = [
    0.2398, 0.3025, 0.3800, 0.3631, 0.2953, 0.3390, 0.2653, 0.1724, 0.0000, 0.3422,
    0.3610, 0.2637, 0.3268, 0.3817, 0.4842, 0.3005, 0.3216, 0.2958, 0.3467, 0.3146,
    0.3108, 0.3440, 0.3595, 0.4089, 0.3282, 0.3858, 0.3790, 0.1943, 0.2791, 0.3304,
    0.3271, 0.2958, 0.4709, 0.2649, 0.3380, 0.4853, 0.4624, 0.3791, 0.4020, 0.3424,
    0.4789, 0.2666, 0.4487, 0.3195, 0.3175, 0.4614, 0.4484, 0.2425, 0.2372, 0.3109
]

# Minimalne odległości Hamminga dla IMPOSTORÓW (testy TN / FP, pominięto błąd 136)
impostor_distances = [
    0.4527, 0.4423, 0.4574, 0.4276, 0.4523, 0.4431, 0.4611, 0.4422, 0.4378, 0.4622,
    0.4482, 0.4451, 0.4457, 0.4541, 0.4565, 0.4405, 0.4400, 0.4538, 0.4420, 0.4534,
    0.4373, 0.4584, 0.4535, 0.4422, 0.4563, 0.4465, 0.4603, 0.4307, 0.4392, 0.4344,
    0.4351, 0.4622, 0.4492, 0.4491, 0.4529, 0.4507, 0.4412, 0.4519, 0.4549, 0.4508,
    0.4502, 0.4459, 0.4505, 0.4545, 0.4542, 0.4613, 0.4469, 0.4466, 0.4487
]
plot_far_frr_curve(user_distances=user_distances, impostor_distances=impostor_distances)

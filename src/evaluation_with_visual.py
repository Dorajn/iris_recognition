import os
import numpy as np
import matplotlib.pyplot as plt
import pipeline
import paths as PATH
import config as CFG
from matcher import compare_codes_with_shift


def run_evaluation_USERS(base_data_path, users_limit=50):
    """
    Automatyczna ewaluacja systemu (Legalni Użytkownicy).
    Zwraca: (True Positives, False Negatives)
    """
    tp_correct_authorizations = 0
    fn_false_rejections = 0
    total_attempts = 0
    
    all_folders = sorted([f for f in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, f))])
    user_folders = all_folders[:users_limit]

    print(f"Rozpoczynam ewaluację dla {len(user_folders)} LEGALNYCH użytkowników...\n")
    print(f"{'User':<10} | {'Status':<15} | {'Dystans':<10} | {'Wynik'}")
    print("-" * 55)

    database = {}
    for user_id in user_folders:
        eye_dir = os.path.join(base_data_path, user_id, "L")
        if not os.path.exists(eye_dir): continue
        
        images = sorted(os.listdir(eye_dir))
        if len(images) < 2: continue
        
        img_path = os.path.join(eye_dir, images[0])
        try:
            code, mask = pipeline.calculate_iris_code(img_path)
            database[user_id] = {"code": code, "mask": mask}
        except Exception:
            continue

    threshold = CFG.MATCH_THRESHOLD
    all_distances = []

    for user_id, template in database.items():
        eye_dir = os.path.join(base_data_path, user_id, "L")
        images = sorted(os.listdir(eye_dir))
        
        test_img_path = os.path.join(eye_dir, images[1])
        
        try:
            test_code, test_mask = pipeline.calculate_iris_code(test_img_path)
            
            dist = compare_codes_with_shift(template["code"], template["mask"], 
                                            test_code, test_mask)
            all_distances.append(all_distances)
            
            total_attempts += 1
            is_correct = dist < threshold
            
            if is_correct:
                tp_correct_authorizations += 1
                status = "OK"
                wynik = "Sukces"
            else:
                fn_false_rejections += 1
                status = "FAILED"
                wynik = "Błąd (Odrzucono)"
                
            print(f"{user_id:<10} | {status:<15} | {dist:.4f}     | {wynik}")
            
        except Exception as e:
            print(f"{user_id:<10} | ERROR           | N/A        | Pominięto")

    if total_attempts > 0:
        print("-" * 55)
        print(f"Poprawnych rozpoznawań (TP): {tp_correct_authorizations}")
        print(f"Błędnych odrzuceń (FN): {fn_false_rejections}")
    
    return tp_correct_authorizations, fn_false_rejections, all_distances


def run_evaluation_IMPOSTORS(base_data_path, users_limit=50):
    """
    Automatyczna ewaluacja systemu (Odrzucanie impostorów).
    Zwraca: (True Negatives, False Positives)
    """
    tn_correct_rejections = 0
    fp_false_accepts = 0
    total_attempts = 0
    
    threshold = CFG.MATCH_THRESHOLD + 0.01

    all_folders = sorted([f for f in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, f))])
    
    enrolled_users = all_folders[:users_limit]
    impostor_users = all_folders[users_limit : users_limit * 2]

    print(f"\nRozpoczynam ewaluację IMPOSTORÓW...")
    print(f"Baza: {len(enrolled_users)} użytkowników. Impostorzy próbujący się włamać: {len(impostor_users)}\n")

    database = {}
    for user_id in enrolled_users:
        eye_dir = os.path.join(base_data_path, user_id, "L")
        if not os.path.exists(eye_dir): continue
        
        images = sorted(os.listdir(eye_dir))
        if len(images) < 1: continue
        
        img_path = os.path.join(eye_dir, images[0])
        try:
            code, mask = pipeline.calculate_iris_code(img_path)
            database[user_id] = {"code": code, "mask": mask}
        except Exception:
            continue

    print(f"{'Impostor':<10} | {'Status':<18} | {'Min Dystans':<11} | {'Wynik'}")
    print("-" * 65)

    all_distances = []
    for impostor_id in impostor_users:
        eye_dir = os.path.join(base_data_path, impostor_id, "L")
        if not os.path.exists(eye_dir): continue
        
        images = sorted(os.listdir(eye_dir))
        if len(images) < 1: continue
        
        test_img_path = os.path.join(eye_dir, images[0]) 
        
        try:
            test_code, test_mask = pipeline.calculate_iris_code(test_img_path)
            
            min_dist = float('inf')
            matched_user = None
            
            for enrolled_id, template in database.items():
                dist = compare_codes_with_shift(template["code"], template["mask"], 
                                                test_code, test_mask)
                if dist < min_dist:
                    min_dist = dist
                    matched_user = enrolled_id

            all_distances.append(min_dist)
            
            total_attempts += 1
            is_rejected = min_dist >= threshold
            
            if is_rejected:
                tn_correct_rejections += 1
                status = "REJECTED (OK)"
                wynik = "Sukces"
            else:
                fp_false_accepts += 1
                status = f"ACCEPTED ({matched_user})"
                wynik = "BŁĄD (Włamanie!)"
            
            print(f"{impostor_id:<10} | {status:<18} | {min_dist:.4f}      | {wynik}")
            
        except Exception as e:
            print(f"{impostor_id:<10} | ERROR              | N/A         | Pominięto")

    if total_attempts > 0:
        print("-" * 65)
        print(f"Poprawne odrzucenia - True Negative (TN): {tn_correct_rejections}")
        print(f"Błędne akceptacje - False Positive (FP): {fp_false_accepts}")
        
    return tn_correct_rejections, fp_false_accepts, all_distances


def plot_biometric_metrics(tp, fn, tn, fp):
    total = tp + fn + tn + fp
    if total == 0:
        print("Brak danych do wygenerowania wykresów.")
        return

    accuracy = ((tp + tn) / total) * 100
    far = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
    frr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Raport Skuteczności Systemu Biometrycznego", fontsize=16, fontweight='bold')

    conf_matrix = np.array([[tp, fn], [fp, tn]])
    cax = ax1.matshow(conf_matrix, cmap='Blues', alpha=0.8)
    fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)

    ax1.set_title("Macierz Błędu (Confusion Matrix)", pad=20, fontsize=12)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Zaakceptowano\n(System: TAK)', 'Odrzucono\n(System: NIE)'])
    ax1.set_yticklabels(['Legalny Użytkownik\n(Rzeczywistość: TAK)', 'Impostor\n(Rzeczywistość: NIE)'])

    labels = [
        [f"True Positive\n(TP): {tp}", f"False Negative\n(FRR): {fn}"],
        [f"False Positive\n(FAR): {fp}", f"True Negative\n(TN): {tn}"]
    ]
    for i in range(2):
        for j in range(2):
            color = 'white' if conf_matrix[i, j] > (total / 3) else 'black'
            ax1.text(j, i, labels[i][j], va='center', ha='center', color=color, fontsize=11, weight='bold')

    metrics_names = ['Skuteczność ogólna\n(Accuracy)', 'Włamania\n(False Accept - FAR)', 'Błędne odrzucenia\n(False Reject - FRR)']
    values = [accuracy, far, frr]
    colors = ['#2ca02c', '#d62728', '#ff7f0e']

    bars = ax2.bar(metrics_names, values, color=colors, edgecolor='black')
    ax2.set_ylim(0, 105)
    ax2.set_title("Kluczowe wskaźniki (Wartości procentowe)", pad=20, fontsize=12)
    ax2.set_ylabel("Procent (%)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    USERS_TO_TEST = 50
    
    print("=" * 65)
    tp, fn, _ = run_evaluation_USERS(PATH.DATA_DIR, users_limit=USERS_TO_TEST)
    
    print("\n" + "=" * 65 + "\n")
    tn, fp, _ = run_evaluation_IMPOSTORS(PATH.DATA_DIR, users_limit=USERS_TO_TEST)
    
    print("\n" + "=" * 65)
    print("GENEROWANIE WYKRESÓW...")
    plot_biometric_metrics(tp, fn, tn, fp)
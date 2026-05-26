import os
import numpy as np
import pipeline
import paths as PATH
import config as CFG
from matcher import compare_codes_with_shift

def run_evaluation_USERS(base_data_path, users_limit=150):
    """
    Automatyczna ewaluacja systemu.
    Dla każdego użytkownika:
    1. Pierwsze zdjęcie (S1...L01) służy jako wzorzec (Enrollment).
    2. Drugie zdjęcie (S1...L02) służy do próby autoryzacji (Verification).
    """
    correct_authorizations = 0
    total_attempts = 0
    
    all_folders = sorted([f for f in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, f))])
    user_folders = all_folders[150:users_limit + users_limit]

    print(f"Rozpoczynam ewaluację dla {len(user_folders)} potencjalnych użytkowników...\n")
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

    for user_id, template in database.items():
        eye_dir = os.path.join(base_data_path, user_id, "L")
        images = sorted(os.listdir(eye_dir))
        
        test_img_path = os.path.join(eye_dir, images[1])
        
        try:
            test_code, test_mask = pipeline.calculate_iris_code(test_img_path)
            
            dist = compare_codes_with_shift(template["code"], template["mask"], 
                                            test_code, test_mask)
            
            total_attempts += 1
            is_correct = dist < threshold
            if is_correct:
                correct_authorizations += 1
            
            status = "OK" if is_correct else "FAILED"
            print(f"{user_id:<10} | {status:<15} | {dist:.4f}     | {'Sukces' if is_correct else 'Błąd'}")
            
        except Exception as e:
            print(f"{user_id:<10} | ERROR           | N/A        | Pominięto")

    if total_attempts > 0:
        accuracy = (correct_authorizations / total_attempts) * 100
        print("-" * 55)
        print(f"FINALNE STATYSTYKI:")
        print(f"Przetworzonych użytkowników: {total_attempts}")
        print(f"Poprawnych rozpoznawań: {correct_authorizations}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Brak danych do przeprowadzenia testu.")



def run_evaluation_IMPOSTORS(base_data_path, users_limit=100):
    """
    Automatyczna ewaluacja systemu (Odrzucanie impostorów).
    1. Rejestruje w bazie pierwszych 'users_limit' użytkowników.
    2. Próbuje zautoryzować KOLEJNYCH 'users_limit' użytkowników (impostorów).
    3. Sprawdza, czy system poprawnie odrzucił intruzów (brak dopasowania w bazie).
    """
    correct_rejections = 0
    total_attempts = 0
    threshold = CFG.MATCH_THRESHOLD + 0.01

    all_folders = sorted([f for f in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, f))])
    
    # Dzielenie na legalnych użytkowników i impostorów
    enrolled_users = all_folders[:users_limit]
    impostor_users = all_folders[users_limit : users_limit * 2]

    print(f"Rozpoczynam ewaluację impostorów...")
    print(f"Baza: {len(enrolled_users)} użytkowników. Impostorzy próbujący się włamać: {len(impostor_users)}\n")

    # ENROLLMENT (Dodawanie legalnych użytkowników)
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

    print(f"Zarejestrowano {len(database)} użytkowników w bazie.\n")
    print(f"{'Impostor':<10} | {'Status':<18} | {'Min Dystans':<11} | {'Wynik'}")
    print("-" * 65)

    # PRÓBY WŁAMANIA (Impostorzy testują system)
    for impostor_id in impostor_users:
        eye_dir = os.path.join(base_data_path, impostor_id, "L")
        if not os.path.exists(eye_dir): continue
        
        images = sorted(os.listdir(eye_dir))
        if len(images) < 1: continue
        
        # Bierzemy pierwsze zdjęcie impostora do ataku
        test_img_path = os.path.join(eye_dir, images[0]) 
        
        try:
            test_code, test_mask = pipeline.calculate_iris_code(test_img_path)
            
            min_dist = float('inf')
            matched_user = None
            
            # Porównujemy impostora ze wszystkimi wzorcami w bazie (1:N)
            for enrolled_id, template in database.items():
                dist = compare_codes_with_shift(template["code"], template["mask"], 
                                                test_code, test_mask)
                if dist < min_dist:
                    min_dist = dist
                    matched_user = enrolled_id
            
            total_attempts += 1
            
            is_rejected = min_dist >= threshold
            
            if is_rejected:
                correct_rejections += 1
                status = "REJECTED (OK)"
                wynik = "Sukces"
            else:
                status = f"ACCEPTED ({matched_user})"
                wynik = "BŁĄD (Włamanie!)"
            
            print(f"{impostor_id:<10} | {status:<18} | {min_dist:.4f}      | {wynik}")
            
        except Exception as e:
            print(f"{impostor_id:<10} | ERROR              | N/A         | Pominięto")

    if total_attempts > 0:
        accuracy = (correct_rejections / total_attempts) * 100
        print("-" * 65)
        print(f"FINALNE STATYSTYKI ODRZUCEŃ:")
        print(f"Liczba prób włamania: {total_attempts}")
        print(f"Poprawne odrzucenia (True Rejection): {correct_rejections}")
        print(f"Błędne akceptacje (False Accept): {total_attempts - correct_rejections}")
        print(f"Accuracy (True Rejection Rate): {accuracy:.2f}%")
    else:
        print("Brak danych do przeprowadzenia testu.")


if __name__ == "__main__":
    # run_evaluation_USERS(PATH.DATA_DIR)
    # print("\n" + "=" * 65 + "\n")
    run_evaluation_IMPOSTORS(PATH.DATA_DIR)
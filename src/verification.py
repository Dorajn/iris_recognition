import json
import numpy as np
import pipeline
import paths as PATH
import config as CFG
from matcher import compare_codes_with_shift

def verify_user(image_path):
    # Wczytaj bazę danych
    with open(PATH.DATABASE_MAP_JSON, "r") as f:
        db = json.load(f)
    
    # Wygeneruj kod dla "nowego" oka
    print("Przetwarzanie nowego zdjęcia...")
    new_code, new_mask = pipeline.calculate_iris_code(image_path)
    
    threshold = db["threshold"]
    best_match = None
    min_dist = 1.0

    # Szukaj w bazie (1:N matching)
    for user in db["authorized_users"]:
        # Wczytaj wzorzec
        template_data = np.load(PATH.IRIS_CODES_DIR / user["template_file"], allow_pickle=True).item()
        
        dist = compare_codes_with_shift(new_code, new_mask, 
                                        template_data["code"], template_data["mask"])
        
        print(f"Użytkownik: {user['user_id']} | Odległość: {dist:.4f}")
        
        if dist < min_dist:
            min_dist = dist
            best_match = user

            
    # Decyduj, do którego użytkownika należy oko
    print("-" * 30)
    if min_dist < threshold:
        print(f"DOSTĘP PRZYZNANY: Witaj {best_match['user_id']}!")
        print(f"Wynik: {min_dist:.4f} (Próg: {threshold})")
    else:
        print("DOSTĘP ODCIĘTY: Nie rozpoznano użytkownika.")
        print(f"Najlepszy wynik: {min_dist:.4f}")

if __name__ == "__main__":
    USER = 3
    file_name = "00" + str(USER) + "/L/S500" + str(USER) + "L01.jpg" 
    test_img = PATH.DATA_DIR / file_name
    verify_user(test_img)
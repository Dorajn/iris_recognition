import os
import json
import numpy as np
import paths as PATH
import pipeline
import config as CFG


def run_enrollment(base_data_path, iris_code_output_dir, database_dir, users_num=100, threshold=None):
    ''' Metoda budująca bazę dancyh użytkowników, licząć ich
    kody tęczówek i zapizujać w pliku .json '''

    if threshold is None:
        threshold = CFG.MATCH_THRESHOLD
    database_info = {
        "authorized_users": [],
        "threshold": threshold
    }

    user_folders = sorted(os.listdir(base_data_path))[:users_num]

    for user_id in user_folders:
        eye_path = os.path.join(base_data_path, user_id, "L")
        
        if not os.path.exists(eye_path):
            continue
            
        images = sorted(os.listdir(eye_path))
        if not images:
            continue
            
        first_img_path = os.path.join(eye_path, images[0])
        
        # --- BIOMETRIC PROCESS ---
        print(f"User registration {user_id}...")
        iris_code, mask = pipeline.calculate_iris_code(first_img_path)
        
        template_name = f"{user_id}_L.npy"
        template_path = os.path.join(iris_code_output_dir, template_name)
        np.save(template_path, {"code": iris_code, "mask": mask})
        
        database_info["authorized_users"].append({
            "user_id": user_id,
            "template_file": template_name,
            "original_image": first_img_path
        })

    with open(database_dir, "w") as f:
        json.dump(database_info, f, indent=4)

    print("\nEnrollment succeded!")

if __name__ == '__main__':
    run_enrollment(PATH.DATA_DIR, PATH.IRIS_CODES_DIR, PATH.DATABASE_MAP_JSON)
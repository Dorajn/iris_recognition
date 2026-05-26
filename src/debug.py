import matplotlib.pyplot as plt
from pipeline import load, find_pupil_iris_boundaries, normalize_iris, create_iris_mask
import cv2 as cv
import paths as PATH
from segmentation import find_pupil

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


# for i in range(10):
#     file_name = f"002/L/S5002L0{i}.jpg"
#     debug_normalize(PATH.DATA_DIR / file_name)
    
file_name = f"000/L/S5000L00.jpg"
debug_show_segmentation(PATH.DATA_DIR / file_name)
# debug_load_and_preprocess(PATH.DATA_DIR / "002/L/S5002L05.jpg")
# debug_find_pupil(PATH.DATA_DIR / "000/L/S5000L02.jpg")
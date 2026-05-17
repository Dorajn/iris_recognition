import matplotlib.pyplot as plt
import cv2 as cv

def show(image):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis('off')
    plt.show()
import cv2
import numpy as np
import os

def digit_recognition(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Cannot load image from '{input_path}'. Please check the file format and path.")
        return

    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noiseless_image_bw = cv2.fastNlMeansDenoising(gray_image, None, 110, 7, 50)

        divide = cv2.divide(gray_image, noiseless_image_bw, scale=255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_image = cv2.morphologyEx(divide, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(morph_image, 100, 200, apertureSize=7, L2gradient=True)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 24 and h > 45:
                points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Output image saved to '{output_path}'")
        else:
            print(f"Error: Unable to save image to '{output_path}'")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

digit_recognition("input.png", "output.png")

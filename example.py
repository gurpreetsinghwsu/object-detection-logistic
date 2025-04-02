import cv2
import numpy as np
import joblib

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    haralick = cv2.calcHist([gray], [0], None, [8], [0,256]).flatten()
    haralick = haralick / np.sum(haralick)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour) / (image.shape[0] * image.shape[1])
        perimeter = cv2.arcLength(largest_contour, True)
    else:
        area = perimeter = 0
    return np.concatenate([mean_color, std_color, [edge_density], haralick, [area, perimeter]])

def detect_object(image_path):
    model = joblib.load('object_detection_model.joblib')
    scaler = joblib.load('object_detection_scaler.joblib')
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    features = extract_features(image)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    
    return "Object detected" if prediction == 1 else "No object detected"

if __name__ == "__main__":
    result = detect_object("path_to_your_image.jpg")
    print(result) 
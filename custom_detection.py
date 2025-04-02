import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import joblib

def create_dataset():
    os.makedirs('dataset/objects', exist_ok=True)
    os.makedirs('dataset/background', exist_ok=True)
    
    object_images = []
    for i in range(50):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 255
        x, y = np.random.randint(10, 54, 2)
        w, h = np.random.randint(10, 30, 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), -1)
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        path = f'dataset/objects/object_{i}.png'
        cv2.imwrite(path, img)
        object_images.append((path, 1))
    
    background_images = []
    for i in range(50):
        img = np.random.normal(128, 30, (64, 64, 3)).astype(np.uint8)
        img = np.clip(img, 0, 255)
        path = f'dataset/background/background_{i}.png'
        cv2.imwrite(path, img)
        background_images.append((path, 0))
    
    return object_images + background_images

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mean_color = np.mean(img, axis=(0, 1))
    std_color = np.std(img, axis=(0, 1))
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    haralick = cv2.calcHist([gray], [0], None, [8], [0,256]).flatten()
    haralick = haralick / np.sum(haralick)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour) / (img.shape[0] * img.shape[1])
        perimeter = cv2.arcLength(largest_contour, True)
    else:
        area = perimeter = 0
    
    return np.concatenate([mean_color, std_color, [edge_density], haralick, [area, perimeter]])

def main():
    dataset = create_dataset()
    X = [extract_features(path) for path, _ in dataset]
    y = [label for _, label in dataset]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted (0: Background, 1: Object)')
    plt.ylabel('Actual (0: Background, 1: Object)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    joblib.dump(model, 'object_detection_model.joblib')
    joblib.dump(scaler, 'object_detection_scaler.joblib')

if __name__ == "__main__":
    main() 
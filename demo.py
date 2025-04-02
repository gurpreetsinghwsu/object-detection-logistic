import cv2
import numpy as np
import joblib
from custom_detection import extract_features

def create_demo_image(has_object=True):
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255
    if has_object:
        x, y = np.random.randint(10, 54, 2)
        w, h = np.random.randint(10, 30, 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), -1)
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return img

def main():
    model = joblib.load('object_detection_model.joblib')
    scaler = joblib.load('object_detection_scaler.joblib')
    
    out = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2, (256, 64))
    
    for _ in range(10):
        has_object = np.random.choice([True, False])
        img = create_demo_image(has_object)
        
        features = extract_features(img)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        display_img = cv2.resize(img, (64, 64))
        
        # Check if prediction matches ground truth
        correct_prediction = (prediction == 1 and has_object) or (prediction == 0 and not has_object)
        
        # Create result image
        result_img = np.ones((64, 256, 3), dtype=np.uint8) * 255
        result_img[:, :64] = display_img
        
        # Add prediction text
        pred_text = "Object Detected" if prediction == 1 else "No Object"
        pred_color = (0, 255, 0) if correct_prediction else (0, 0, 255)  # Green if correct, Red if wrong
        cv2.putText(result_img, pred_text, (74, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        
        # Add ground truth text
        truth_text = "Truth: Object Present" if has_object else "Truth: No Object"
        cv2.putText(result_img, truth_text, (74, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        out.write(result_img)
        cv2.imshow('Demo', result_img)
        cv2.waitKey(500)
    
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
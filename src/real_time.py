import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# ==========================================
# 1. Setup & Configuration
# ==========================================

# Check if model exists
MODEL_PATH = '../model/emotion_model.h5'
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file '{MODEL_PATH}' not found!")
    print("Please download 'emotion_model.h5' from Colab and place it in this folder.")
    exit()

# Load the trained model
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# Define emotion labels (Must match the training order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Face Detector (Haar Cascade)
# OpenCV usually includes this file automatically
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("[ERROR] Could not find Haar Cascade XML. Please download 'haarcascade_frontalface_default.xml'")
    exit()

# ==========================================
# 2. Filter Logic (The Creative Part)
# ==========================================
def apply_emotion_filter(face_image, emotion):
    """
    Applies a specific visual filter based on the detected emotion.
    Input: face_image (Color BGR), emotion (String)
    Output: filtered_face (Color BGR)
    """
    # Create a copy to avoid modifying original array directly initially
    output = face_image.copy()
    
    if emotion == 'Happy':
        # TASK REQUIREMENT: Blur the image if laughing/happy
        output = cv2.GaussianBlur(output, (35, 35), 0)

    elif emotion == 'Angry':
        # Filter: Red Tint (Increase Red channel)
        # BGR format -> Channel 2 is Red
        output[:, :, 2] = np.clip(output[:, :, 2] + 60, 0, 255)

    elif emotion == 'Sad':
        # Filter: Grayscale effect (Black & White)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR so we can paste it into the color frame
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif emotion == 'Fear':
        # Filter: Invert Colors (Negative effect - looks spooky)
        output = cv2.bitwise_not(output)

    elif emotion == 'Surprise':
        # Filter: Canny Edge Detection (Sketch effect)
        edges = cv2.Canny(output, 100, 200)
        # Convert edges to color format (White lines on black background)
        output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif emotion == 'Disgust':
        # Filter: Green Tint (Increase Green channel)
        # BGR format -> Channel 1 is Green
        output[:, :, 1] = np.clip(output[:, :, 1] + 60, 0, 255)

    else:
        # Neutral: No filter, or maybe just a slight brightness boost
        pass
        
    return output

# ==========================================
# 3. Main Video Loop
# ==========================================
# Open Webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Camera started. Press 'q' to exit.")

while True:
    # 1. Read Frame
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Convert to Grayscale (For Face Detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect Faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # 4. Process Each Face
    for (x, y, w, h) in faces:
        # --- A. Preprocessing for Model ---
        # Crop the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Resize to 48x48 (Model input size)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Normalize (0 to 1)
        roi_gray = roi_gray.astype("float") / 255.0
        
        # Expand dims to fit model input: (1, 48, 48, 1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # --- B. Prediction ---
        preds = model.predict(roi_gray, verbose=0)
        label_index = np.argmax(preds)
        predicted_emotion = EMOTION_LABELS[label_index]
        confidence = preds[0][label_index]

        # --- C. Apply Filters ---
        # Extract the color face region
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Apply the filter function defined above
        filtered_face = apply_emotion_filter(face_roi_color, predicted_emotion)
        
        # Put the filtered face back into the main frame
        frame[y:y+h, x:x+w] = filtered_face

        # --- D. Visualization (Box & Text) ---
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Prepare label text
        text = f"{predicted_emotion} ({confidence*100:.1f}%)"
        
        # Put text above the box
        cv2.putText(frame, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5. Show Result
    cv2.imshow('Emotion Recognition Project', frame)

    # 6. Exit Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
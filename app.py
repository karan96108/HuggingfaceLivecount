import gradio as gr
import cv2
from deepface import DeepFace
import time
import numpy as np

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def process_video():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Count faces and emotions
        person_count = len(faces)
        happy_count = 0
        not_happy_count = 0
        
        for (x, y, w, h) in faces:
            try:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = preprocess_face(face_roi)
                
                analysis = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name, confidence = dominant_emotion
                
                # Draw rectangles and labels
                if emotion_name in ['happy', 'surprise'] and confidence > 50:
                    happy_count += 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Happy {confidence:.0f}%", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    not_happy_count += 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Not Happy {confidence:.0f}%", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
        # Display metrics
        cv2.putText(frame, f"Total: {person_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Happy: {happy_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Not Happy: {not_happy_count}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        yield frame

demo = gr.Interface(
    fn=process_video,
    inputs=None,
    outputs=gr.Image(label="Live Detection"),
    live=True,
    title="Live Emotion Detection",
    description="Real-time emotion detection with people counting"
)

if __name__ == "__main__":
    demo.queue().launch(share=True)  # Added queue() and share=True
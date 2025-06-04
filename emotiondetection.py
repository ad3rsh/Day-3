import cv2
from deepface import DeepFace
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Load video from webcam
cap = cv2.VideoCapture(0)
last_emotion = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotion
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = results[0]['dominant_emotion']

    # Display emotion on frame
    cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Emotion Recognition", frame)

    # Speak the emotion if it changes
    if emotion != last_emotion:
        engine.say(f"You look {emotion}")
        engine.runAndWait()
        last_emotion = emotion

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

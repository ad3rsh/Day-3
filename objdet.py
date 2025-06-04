import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lower red range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    # Upper red range
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2  # Combine both masks

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    cv2.imshow('Red Detection', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

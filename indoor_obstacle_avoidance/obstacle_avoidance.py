import cv2
import numpy as np

# Load input video
cap = cv2.VideoCapture("test_video1.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Find contours (obstacles)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    decision = "FORWARD"

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 3000:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            center_x = x + w // 2

            # Control heuristics
            if width//3 < center_x < 2*width//3:
                decision = "TURN LEFT"

            elif center_x <= width//3:
                decision = "TURN RIGHT"

            # Safety check
            if w > width * 0.5:
                decision = "STOP"

    cv2.putText(
        frame,
        f"Decision: {decision}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("AI Indoor Obstacle Avoidance", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
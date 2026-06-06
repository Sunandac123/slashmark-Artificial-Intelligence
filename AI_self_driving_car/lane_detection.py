import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (800, 600))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        50,
        minLineLength=100,
        maxLineGap=50
    )

    # Draw lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Show output
    cv2.imshow("Self Driving Car Lane Detection", frame)

    # Exit on Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
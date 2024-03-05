import cv2

# Initialize video capture
cap  = cv2.VideoCapture(0)

# Read the first frame
ret, frame = cap.read()

# Set up tracker
tracker = cv2.TrackerMOSSE_create()

# Select ROI (Region of Interest)
roi = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ret = tracker.init(frame, roi)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    ret, roi = tracker.update(frame)

    # Draw ROI Rectangle if tracking is successful
    if ret:
        (x, y, w, h) = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        # In case of tracking failure
        cv2.putText(frame, "Failure to detect tracking!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display result
    cv2.imshow('Tracking', frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
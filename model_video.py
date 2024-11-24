import cv2
from ultralytics import YOLO

# Load the YOLOv8n trained model
model = YOLO("precrime.pt")  # Replace with your model's path

# Input and output video paths
input_video_path = "phone_thief.mp4"
output_video_path = "phone_thief_detected.mp4"

# Initialize video capture and writer
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)
    
    # Draw the detection results on the frame
    annotated_frame = results[0].plot()  # Annotated frame with detections

    # Write the frame to the output video
    out.write(annotated_frame)

    # Display the frame (optional)
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

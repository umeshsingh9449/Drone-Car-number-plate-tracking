import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO



# load YOLOv8 nano model

model = YOLO("yolov8n.pt")


# Configure depth + color streams

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)

#Start Streaming
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

         # Convert to numpy array
        frame = np.asanyarray(color_frame.get_data())

        #Run YOLO inference

        results = model(frame)

        # Draw Detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])


                # Class 0 = person

                if cls == 0 and conf > 0.4:
                    x1 , y1 , x2  , y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2 , y2), (0, 255 , 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 , 0), 2)

        cv2.imshow("RealSense People Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    #Stop  streaming
    pipeline.stop()
    cv2.destroyAllWindows() 





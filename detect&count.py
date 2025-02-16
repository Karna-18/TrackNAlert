import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture("Cars.mp4")
vehicle_counts = {}
tracked_vehicles = set()  # To track counted vehicles

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    detections = []
    
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            class_name = result.names[int(class_id)]
            
            if class_name not in vehicle_counts:
                vehicle_counts[class_name] = 0
            
            detections.append([[x1, y1, x2, y2], score, class_name])
    
    # Track vehicles
    tracks = tracker.update_tracks(detections, frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()
        class_name = track.get_det_class()
        track_id = track.track_id
        
        # Count vehicle only once
        if track_id not in tracked_vehicles:
            tracked_vehicles.add(track_id)
            vehicle_counts[class_name] += 1
        
        # Reduce bounding box size to approximately car size
        shrink_factor = 10  # Adjust this to control box size
        x1, y1, x2, y2 = (
            int(bbox[0]) + shrink_factor,
            int(bbox[1]) + shrink_factor,
            int(bbox[2]) - shrink_factor,
            int(bbox[3]) - shrink_factor,
        )
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Display count (small size, top-left corner)
    count_text = " | ".join([f"{k}: {v}" for k, v in vehicle_counts.items()])
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Vehicle Detection & Counting", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











# import cv2
# import torch
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")
# tracker = DeepSort(max_age=30)

# # Load video
# cap = cv2.VideoCapture("Cars.mp4")
# vehicle_counts = {}
# tracked_vehicles = set()  # To track counted vehicles

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Run YOLO detection
#     results = model(frame)
#     detections = []
    
#     for result in results:
#         for box in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = box
#             class_name = result.names[int(class_id)]
            
#             if class_name not in vehicle_counts:
#                 vehicle_counts[class_name] = 0
            
#             width, height = x2 - x1, y2 - y1
#             x1 += width * 0.01  # Reduce bounding box width
#             x2 -= width * 0.01
#             y1 += height * 0.01  # Reduce bounding box height
#             y2 -= height * 0.01
            
#             detections.append([[x1, y1, x2, y2], score, class_name])
    
#     # Track vehicles
#     tracks = tracker.update_tracks(detections, frame=frame)
    
#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         bbox = track.to_tlbr()
#         class_name = track.get_det_class()
#         track_id = track.track_id
        
#         # Count vehicle only once
#         if track_id not in tracked_vehicles:
#             tracked_vehicles.add(track_id)
#             vehicle_counts[class_name] += 1
        
#         # Draw bounding box
#         x1, y1, x2, y2 = map(int, bbox)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
#     # Display count (small size, top-left corner)
#     count_text = " | ".join([f"{k}: {v}" for k, v in vehicle_counts.items()])
#     cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
#     cv2.imshow("Vehicle Detection & Counting", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

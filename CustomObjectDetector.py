import torch
import numpy as np
import cv2
from time import time


class BottleDetection:

   def __init__(self, capture_index, model_name):
        
      self.capture_index = capture_index                                                                  # Specifies where the input is coming from (webcam, mp4 or other)
      self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)     # Using custom model 
      self.classes = self.model.names
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                                        # Uses CUDA GPU acceleration if available
      print("Using Device: ", self.device)           


   def score_frame(self, frame):                                                                          # This function takes each frame and scores it using the custom YOLOv5 model

      self.model.to(self.device)
      frame = [frame]
      results = self.model(frame)
      labels, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
      return labels, coord


   def plot_boxes(self, results, frame):                                                                  # Draws boundary boxes and labels on each frame
        
      labels, cord = results
      n = len(labels)
      x_shape, y_shape = frame.shape[1], frame.shape[0]
      
      for i in range(n):
         row = cord[i]
         
         if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, self.classes[int((labels[i]))], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

      return frame


   def run(self):

      cap = cv2.VideoCapture(self.capture_index)                                                         # Gets the feed
      
      while True:
         _, frame = cap.read()
            
         frame = cv2.resize(frame, (640,640))                                                            # Resizing the frame to 640, since the model was trained on that
            
         start_time = time()
         results = self.score_frame(frame)
         frame = self.plot_boxes(results, frame)  
         end_time = time()
         
         fps = 1/np.round(end_time - start_time, 2)
             
         cv2.putText(frame, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
         cv2.imshow('YOLOv5 Custom Bottle Detection', frame)
 
         if cv2.waitKey(1) & 0xFF == 27:
            break
      
      cap.release()
      cv2.destroyAllWindows()
        

# Driver Code
detector = BottleDetection(capture_index='F:\\OpenCV\\Custom Object Detection\\bottle_vid1.mp4', model_name='F:\\OpenCV\\Custom Object Detection\\best.pt')
detector.run()
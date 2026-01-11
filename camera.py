import cv2
import numpy as np

class VideoCamera:
    def __init__(self):
        pass

    def get_frame(self):
        # create block
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (25, 25, 25) # Dark neutral grey
        
        # static text
        cv2.putText(frame, "CAMERA FEED PLACEHOLDER", (350, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        
        # encode to jpeg
        ret, buffer = cv2.imencode('.jpg', frame)
        
        # return bytes, status, score
        return buffer.tobytes(), "FOCUSED", 100

    def release(self):
        pass
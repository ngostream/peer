import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request

class VideoCamera:
    def __init__(self):
        # setup camera
        # we force 1080p to capture peripheral vision
        # this helps catch phones held lower in the frame than standard webcams show
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not self.camera.isOpened():
            print("Camera failed to open.")

        self._setup_models()

        self.status = "FOCUSED"
        self.focus_score = 100
        self.distracted_frames = 0
        self.DISTRACTION_THRESHOLD = 15

    def _setup_models(self):
        # auto-download mediapipe models to local /models folder
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)

        def download(url, filename):
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                print(f"downloading {filename}...")
                urllib.request.urlretrieve(url, path)
            return path

        # pose model for head tilt/slouch
        # using the 'full' model for better accuracy on shoulder tracking
        pose_path = download(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            "pose_landmarker_full.task"
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_path),
                running_mode=vision.RunningMode.IMAGE
            )
        )

        # object moddel for phones
        det_path = download(
            "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite",
            "efficientdet_lite0.tflite"
        )
        self.detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=det_path),
                score_threshold=0.4, # 40% confidence required to trigger
                category_allowlist=["cell phone", "mobile phone"] # ignore cups, laptops, etc.
            )
        )

    def get_frame(self):
        success, frame = self.camera.read()
        if not success: return b'', "ERROR", 0

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self.pose_landmarker.detect(mp_image)

        det_result = self.detector.detect(mp_image)

        is_distracted = False
        reason = ""

        # phone detection (priority #1)
        if det_result.detections:
            for detection in det_result.detections:
                is_distracted = True
                reason = "PHONE"
                
                # draw red box around the phone
                bbox = detection.bounding_box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 4)
                
                # label
                cv2.putText(frame, "NO PHONES", (bbox.origin_x, bbox.origin_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # posture detection (priority #2)
        if not is_distracted and pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            
            # extract normalized coordinates (0.0 is top, 1.0 is bottom)
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # calculate the average height of the user's shoulders
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # logic: "high water mark"
            # we subtract 0.28 from the shoulder height to create a limit line near the chin.
            # if the nose drops below this line (y value gets larger), it's a slouch.
            threshold_val = shoulder_y - 0.24

            if nose.y > threshold_val:
                is_distracted = True
                reason = "POSTURE"
                
                # visual debugging: draw the invisible threshold line
                line_y_px = int(threshold_val * h)
                cv2.line(frame, (0, line_y_px), (w, line_y_px), (0, 0, 255), 1)
                
                # ensure text stays on screen even if line is at the very top
                text_y = line_y_px - 10
                if text_y < 20: text_y = line_y_px + 20 # draw below line if too high
                
                cv2.putText(frame, "THRESHOLD LIMIT", (10, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # score update logic
        if is_distracted:
            self.distracted_frames += 1
        else:
            # reset the counter to zero immediately when distraction is gone
            self.distracted_frames = 0

        # hysteresis buffer to prevent flickering
        if self.distracted_frames > self.DISTRACTION_THRESHOLD:
            self.status = "DISTRACTED" 
            self.focus_score = max(0, self.focus_score - 0.5)
            cv2.putText(frame, f"WARNING: {reason}", (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            self.status = "FOCUSED"
            self.focus_score = min(100, self.focus_score + 0.1)
            cv2.putText(frame, "VISION ACTIVE", (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes(), self.status, int(self.focus_score)

    def release(self):
        self.camera.release()

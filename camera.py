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
        self.frame_timestamp_ms = 0  # timestamp for video mode

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
                running_mode=vision.RunningMode.VIDEO
            )
        )

        # object model for phones and study material
        det_path = download(
            "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite",
            "efficientdet_lite0.tflite"
        )
        self.detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=det_path),
                score_threshold=0.25, # lower threshold to catch phones more reliably
                category_allowlist=["cell phone", "mobile phone", "book", "laptop"] # ignore cups, etc.
            )
        )

    def get_frame(self):
        success, frame = self.camera.read()
        if not success: return b'', "ERROR", 0

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # run ai inference
        # video mode requires timestamp (increment by 33ms per frame for ~30fps)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        self.frame_timestamp_ms += 33  # increment timestamp for next frame
        
        det_result = self.detector.detect(mp_image)

        # state flags
        has_phone = False
        has_study_material = False
        study_box = None

        # object analysis loop
        # phones are blacklisted (always bad)
        # books/laptops/tablets are whitelisted (allow looking down to study)
        if det_result.detections:
            for detection in det_result.detections:
                category = detection.categories[0].category_name
                score = detection.categories[0].score
                bbox = detection.bounding_box
                
                # blacklist: phones (always trigger distraction)
                if category in ["cell phone", "mobile phone"]:
                    has_phone = True
                    # draw red box immediately on phone
                    cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                                 (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 0, 255), 4)
                    cv2.putText(frame, f"NO PHONES ({score:.2f})", (bbox.origin_x, bbox.origin_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                               
                # whitelist: books, laptops, tablets (iPads usually detected as "laptop")
                elif category in ["book", "laptop"]:
                    has_study_material = True
                    study_box = bbox

        # logic decision tree
        is_distracted = False
        reason = ""

        # priority #1: phone detection (blacklist - always triggers distraction)
        # phones override everything, even if study materials are present
        if has_phone:
            is_distracted = True
            reason = "PHONE"

        # priority #2: study materials whitelist (allow looking down to study)
        # books, laptops, tablets (detected as "laptop") allow posture violations
        # but only if no phone is detected (phone takes priority)
        elif has_study_material:
            is_distracted = False
            reason = "STUDYING"
            
            # draw blue box to indicate safe zone
            if study_box:
                cv2.rectangle(frame, (study_box.origin_x, study_box.origin_y), 
                             (study_box.origin_x + study_box.width, study_box.origin_y + study_box.height), (255, 0, 0), 2)
                cv2.putText(frame, "WORK DETECTED", (study_box.origin_x, study_box.origin_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # priority #3: posture detection (fallback)
        # only check this if no phone was found and no work materials are visible
        elif pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            
            # extract normalized coordinates (0.0 is top, 1.0 is bottom)
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # calculate the average height of the user's shoulders
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # logic: "high water mark"
            # we subtract 0.20 from the shoulder height to create a limit line near the chin.
            # if the nose drops below this line (y value gets larger), it's a slouch.
            threshold_val = shoulder_y - 0.20

            if nose.y > threshold_val:
                is_distracted = True
                reason = "POSTURE"
                
                # visual debugging: draw the invisible threshold line (red)
                line_y_px = int(threshold_val * h)
                cv2.line(frame, (0, line_y_px), (w, line_y_px), (0, 0, 255), 1)
                
                # ensure text stays on screen even if line is at the very top
                text_y = line_y_px - 10
                if text_y < 20: text_y = line_y_px + 20 
                
                cv2.putText(frame, "EYES UP", (10, text_y), 
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
            
            # update hud based on context
            status_text = "STUDY MODE" if has_study_material else "VISION ACTIVE"
            color = (255, 0, 0) if has_study_material else (0, 255, 0)
            
            cv2.putText(frame, status_text, (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes(), self.status, int(self.focus_score)

    def release(self):
        self.camera.release()
    
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import time
import uuid

class VideoCamera:
    def __init__(self):
        # setup camera
        # we force 1080p to capture peripheral vision
        # this helps catch phones held lower in the frame than standard webcams show
        # use AVFoundation backend on macOS to avoid crashes
        self.camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # reduce buffer size to avoid crashes on macOS
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.camera.isOpened():
            print("Camera failed to open.")

        self._setup_models()

        self.status = "FOCUSED"
        self.focus_score = 100
        self.distracted_frames = 0
        self.DISTRACTION_THRESHOLD = 15
        self.start_time_ms = int(time.time() * 1000)  # start time in milliseconds
        
        # calibration state
        # default fallback is 0.22, which is forgiving but effective
        # user can override this by clicking calibrate in the ui
        self.baseline_dist = 0.22
        self.is_calibrating = False
        
        # session history tracking
        self.history = []
        self.sessions = []
        self.is_currently_distracted = False
        self.distraction_start_time = None
        self.distraction_reason = ""
        self.distraction_snapshot_filename = None
        
        # current session tracking
        self.current_session_start = None
        self.current_session_history_index = None

    def calibrate(self):
        # trigger flag to capture posture on next frame
        self.is_calibrating = True

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
        # video mode requires timestamp - use actual time relative to start
        current_time_ms = int(time.time() * 1000) - self.start_time_ms
        pose_result = self.pose_landmarker.detect_for_video(mp_image, current_time_ms)
        
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

            # calibration check
            # if user requested calibration, capture current posture as the new 100%
            # we add a 0.05 buffer so they can move slightly without triggering
            if self.is_calibrating:
                current_dist = shoulder_y - nose.y
                self.baseline_dist = current_dist - 0.05
                self.is_calibrating = False
                print(f"calibrated: new threshold offset is {self.baseline_dist}")
            
            # logic: "high water mark"
            # we subtract the baseline (either default 0.20 or calibrated) from shoulder height
            # to create a limit line near the chin.
            threshold_val = shoulder_y - self.baseline_dist

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
            else:
                 # visual debugging: draw the safe line (green)
                 # this helps users calibrate manually if needed
                 line_y_px = int(threshold_val * h)
                 cv2.line(frame, (0, line_y_px), (w, line_y_px), (0, 255, 0), 1)

        # score update logic
        if is_distracted:
            self.distracted_frames += 1
        else:
            # reset the counter to zero immediately when distraction is gone
            self.distracted_frames = 0

        # state transition detection for history
        if not self.is_currently_distracted and self.distracted_frames > self.DISTRACTION_THRESHOLD:
            # distraction starts
            self.is_currently_distracted = True
            self.distraction_start_time = time.time()
            self.distraction_reason = reason
            
            # capture screenshot
            snapshot_filename = f"shame/{uuid.uuid4()}.jpg"
            snapshot_path = os.path.join(os.path.dirname(__file__), 'static', snapshot_filename)
            cv2.imwrite(snapshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.distraction_snapshot_filename = snapshot_filename
            print(f"distraction started: {reason}, snapshot: {snapshot_filename}")

        elif self.is_currently_distracted and self.distracted_frames <= self.DISTRACTION_THRESHOLD:
            # distraction ends
            self.is_currently_distracted = False
            if self.distraction_start_time:
                duration = time.time() - self.distraction_start_time
                self.history.append({
                    "id": str(uuid.uuid4()),
                    "timestamp": int(self.distraction_start_time),
                    "reason": self.distraction_reason,
                    "duration": duration,
                    "snapshot_url": self.distraction_snapshot_filename
                })
                print(f"distraction ended: {self.distraction_reason}, duration: {duration:.2f}s")
            self.distraction_start_time = None
            self.distraction_reason = ""
            self.distraction_snapshot_filename = None

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

    def start_session(self):
        # start a new session
        self.current_session_start = time.time()
        self.current_session_history_index = len(self.history)
        
    def stop_session(self):
        # end current session and create summary
        if self.current_session_start is None:
            return
            
        session_end = time.time()
        session_duration = session_end - self.current_session_start
        
        # get all events for this session
        session_events = self.history[self.current_session_history_index:] if self.current_session_history_index is not None else self.history
        
        # calculate stats
        distraction_count = len(session_events)
        total_distraction_time = sum(event.get("duration", 0) for event in session_events)
        avg_focus_score = 100  # placeholder - we don't track average score yet
        
        session_summary = {
            "id": str(uuid.uuid4()),
            "start_time": int(self.current_session_start),
            "end_time": int(session_end),
            "duration": session_duration,
            "distraction_count": distraction_count,
            "total_distraction_time": total_distraction_time,
            "avg_focus_score": avg_focus_score
        }
        
        self.sessions.append(session_summary)
        self.current_session_start = None
        self.current_session_history_index = None
        
        return session_summary
    
    def get_history(self):
        # return history list (most recent first)
        return list(reversed(self.history))
    
    def get_sessions(self):
        # return session summaries (most recent first)
        return list(reversed(self.sessions))

    def release(self):
        self.camera.release()

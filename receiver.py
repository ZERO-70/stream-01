import cv2
import socket
import struct
import pickle
import numpy as np
import sys
import time
import logging
import io
import os
import base64
import threading
import queue
from threading import Lock
from transformers import pipeline
from PIL import Image
from inference_sdk import InferenceHTTPClient

# Set up logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Use stdout for better encoding
        logging.FileHandler('nsfw_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class VideoReceiver:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Initialize NSFW detection with detailed logging
        logger.info("Initializing NSFW detection model...")
        try:
            logger.info("Loading Falconsai/nsfw_image_detection model...")
            logger.info("This may take a few minutes on first run as the model will be downloaded.")
            
            # Enable verbose logging for transformers
            import transformers
            transformers.logging.set_verbosity_info()
            
            # Remove verbose parameter as it's not supported
            self.nsfw_pipeline = pipeline(
                "image-classification", 
                model="Falconsai/nsfw_image_detection"
            )
            
            logger.info("NSFW detection model loaded successfully!")
            self.nsfw_detection_available = True
            
            # Log model info
            logger.info(f"Model loaded: {self.nsfw_pipeline.model.name_or_path}")
            logger.info(f"Pipeline type: {type(self.nsfw_pipeline).__name__}")
            
        except Exception as e:
            logger.error(f"NSFW detection not available: {e}")
            logger.error("The system will continue without NSFW detection.")
            self.nsfw_detection_available = False
        
        # Initialize Gun detection
        logger.info("Initializing Gun detection model...")
        try:
            logger.info("Setting up Roboflow Gun Detection API client...")
            self.gun_client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key="Nxa6kPAzjbqdE8Xh9pne"
            )
            logger.info("Gun detection API client initialized successfully!")
            self.gun_detection_available = True
        except Exception as e:
            logger.error(f"Gun detection not available: {e}")
            logger.error("The system will continue without gun detection.")
            self.gun_detection_available = False
        
        # NSFW detection settings
        self.last_nsfw_check_time = time.time()
        self.nsfw_time_interval = 0.333  # Check 3 times per second (every 0.333 seconds)
        self.nsfw_model_interval = 3.0  # Model inference limited to once per 3 seconds
        self.last_nsfw_model_time = 0
        self.current_nsfw_status = False
        self.nsfw_confidence = 0.0
        self.detection_count = 0
        self.frame_count_since_last_nsfw_check = 0
        
        # Gun detection settings
        self.gun_detection_count = 0
        self.current_gun_status = False
        self.gun_confidence = 0.0
        self.gun_objects = []
        self.last_gun_api_call_time = 0  # Track time of last gun API call
        self.gun_api_time_interval = 2.0  # Limit gun API calls to once per 2 seconds
        
        # Threading support for NSFW detection
        self.nsfw_detection_queue = queue.Queue(maxsize=1)  # Queue for frames to process
        self.nsfw_results_lock = Lock()  # Lock for thread-safe access to detection results
        self.nsfw_detection_thread = None
        self.nsfw_detection_running = False
        
        # Threading support for gun detection
        self.gun_detection_queue = queue.Queue(maxsize=1)  # Queue for frames to process
        self.gun_results_lock = Lock()  # Lock for thread-safe access to detection results
        self.gun_detection_thread = None
        self.gun_detection_running = False
        
        # Start the NSFW detection thread if detection is available
        if self.nsfw_detection_available:
            self.nsfw_detection_running = True
            self.nsfw_detection_thread = threading.Thread(target=self._nsfw_detection_worker, daemon=True)
            self.nsfw_detection_thread.start()
            logger.info("Started NSFW detection in background thread")
            
        # Start the gun detection thread if detection is available
        if self.gun_detection_available:
            self.gun_detection_running = True
            self.gun_detection_thread = threading.Thread(target=self._gun_detection_worker, daemon=True)
            self.gun_detection_thread.start()
            logger.info("Started gun detection in background thread")
        
    def connect_to_server(self):
        """Connect to the video streaming server"""
        try:
            logger.info(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            logger.info("Connected to server!")
            return True
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            logger.error("Make sure the sender.py is running first!")
            return False
            
    def _nsfw_detection_worker(self):
        """Background worker thread for NSFW detection processing"""
        logger.info("NSFW detection worker thread started")
        
        while self.nsfw_detection_running:
            try:
                # Get a frame from the queue with a timeout
                # This allows the thread to check nsfw_detection_running periodically
                try:
                    frame = self.nsfw_detection_queue.get(timeout=1.0)
                except queue.Empty:
                    # No frame to process, continue waiting
                    continue
                
                # Check if we need to limit model inference (once per 3 seconds)
                current_time = time.time()
                time_since_last_model = current_time - self.last_nsfw_model_time
                
                if time_since_last_model < self.nsfw_model_interval:
                    logger.info(f"Skipping NSFW model inference - too soon since last inference ({time_since_last_model:.1f}s < {self.nsfw_model_interval}s)")
                    # Mark task as done and continue
                    self.nsfw_detection_queue.task_done()
                    # Add short sleep to prevent CPU spinning
                    time.sleep(0.1)
                    continue
                
                # Process the frame (NSFW detection)
                logger.info("Background thread processing NSFW detection")
                self.last_nsfw_model_time = current_time
                nsfw_status, confidence = self._process_nsfw_detection(frame)
                
                # Update the results with thread safety
                with self.nsfw_results_lock:
                    self.current_nsfw_status = nsfw_status
                    self.nsfw_confidence = confidence
                
                logger.info(f"Background NSFW detection completed: detected={nsfw_status}, confidence={confidence:.2f}")
                
                # Mark task as done
                self.nsfw_detection_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in NSFW detection thread: {e}")
    
    def _gun_detection_worker(self):
        """Background worker thread for gun detection processing"""
        logger.info("Gun detection worker thread started")
        
        while self.gun_detection_running:
            try:
                # Get a frame from the queue with a timeout
                # This allows the thread to check gun_detection_running periodically
                try:
                    frame = self.gun_detection_queue.get(timeout=1.0)
                except queue.Empty:
                    # No frame to process, continue waiting
                    continue
                
                # Check if we need to skip API call due to rate limiting
                current_time = time.time()
                time_since_last_call = current_time - self.last_gun_api_call_time
                
                if time_since_last_call < self.gun_api_time_interval:
                    logger.info(f"Skipping gun API call - too soon since last call ({time_since_last_call:.1f}s < {self.gun_api_time_interval:.1f}s)")
                    self.gun_detection_queue.task_done()
                    continue
                
                # Process the frame (gun detection)
                logger.info("Background thread processing gun detection")
                gun_status, confidence, objects = self._process_gun_detection(frame)
                
                # Update the last API call time for rate limiting
                self.last_gun_api_call_time = time.time()
                
                # Update the results with thread safety
                with self.gun_results_lock:
                    self.current_gun_status = gun_status
                    self.gun_confidence = confidence
                    self.gun_objects = objects
                
                logger.info(f"Background gun detection completed: detected={gun_status}, confidence={confidence:.2f}, objects={len(objects)}")
                
                # Mark task as done
                self.gun_detection_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in gun detection thread: {e}")
        
        logger.info("Gun detection worker thread stopped")
    
    def _process_nsfw_detection(self, frame):
        """Internal method to process NSFW detection - called from worker thread"""
        self.detection_count += 1
        logger.info(f"Running NSFW detection in thread (attempt #{self.detection_count})...")
        
        try:
            # Convert OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            logger.debug(f"Frame converted to PIL Image: {pil_image.size}")
            
            # Run NSFW detection with timing
            start_time = time.time()
            results = self.nsfw_pipeline(pil_image)
            inference_time = time.time() - start_time
            
            logger.info(f"NSFW inference completed in {inference_time:.3f} seconds (in thread)")
            logger.debug(f"Raw results: {results}")
            
            # Parse results - look for the highest confidence result
            nsfw_confidence = 0.0
            sfw_confidence = 0.0
            
            for result in results:
                logger.info(f"NSFW detection result (in thread): {result['label']} - Confidence: {result['score']:.3f}")
                if result['label'].lower() == 'nsfw':
                    nsfw_confidence = result['score']
                elif result['label'].lower() in ['normal', 'sfw']:
                    sfw_confidence = result['score']
            
            # Determine if NSFW based on confidence comparison
            if nsfw_confidence > sfw_confidence:
                logger.warning(f"NSFW content detected in thread! Confidence: {nsfw_confidence:.3f}")
                return True, nsfw_confidence
            else:
                logger.info(f"SFW content detected in thread. Confidence: {sfw_confidence:.3f}")
                return False, sfw_confidence
            
        except Exception as e:
            logger.error(f"NSFW detection error in thread: {e}")
            return False, 0.0
            
    def _process_gun_detection(self, frame):
        """Internal method to process gun detection - called from worker thread"""
        self.gun_detection_count += 1
        logger.info(f"Running Gun detection in thread (attempt #{self.gun_detection_count})...")
        
        try:
            # Convert OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert the image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            
            # Run gun detection with timing using base64 encoding
            start_time = time.time()
            
            # Try with URL format - using fake URL with base64 data
            base64_url = "data:image/jpeg;base64," + img_str.decode('utf-8')
            result = self.gun_client.infer(base64_url, model_id="gun-detection-ghlzd/4")
                
            inference_time = time.time() - start_time
            logger.info(f"API response (in thread): {result}")
            logger.info(f"Gun detection completed in {inference_time:.3f} seconds (in thread)")
            
            # Parse results - handle different possible response formats
            try:
                # Try to get predictions directly or from nested structure
                if isinstance(result, dict):
                    predictions = result.get('predictions', [])
                elif isinstance(result, list):
                    predictions = result  # Some APIs return a list directly
                else:
                    predictions = []
                    logger.warning(f"Unexpected response type: {type(result)}")
                
                detected_objects = []
                highest_confidence = 0.0
                
                for prediction in predictions:
                    if not isinstance(prediction, dict):
                        logger.warning(f"Skipping non-dict prediction: {prediction}")
                        continue
                        
                    # Handle different API response formats
                    confidence = prediction.get('confidence', prediction.get('score', prediction.get('probability', 0.0)))
                    class_name = prediction.get('class', prediction.get('label', prediction.get('class_name', 'unknown')))
                    
                    logger.info(f"Gun detection result (in thread): {class_name} - Confidence: {confidence:.3f}")
                    
                    # Extract bounding box coordinates with various possible field names
                    x = prediction.get('x', prediction.get('cx', prediction.get('center_x', 0)))
                    y = prediction.get('y', prediction.get('cy', prediction.get('center_y', 0)))
                    width = prediction.get('width', prediction.get('w', prediction.get('bbox_width', 0)))
                    height = prediction.get('height', prediction.get('h', prediction.get('bbox_height', 0)))
                    
                    # For APIs that return x1,y1,x2,y2 format
                    if 'x1' in prediction and 'x2' in prediction:
                        x1 = prediction.get('x1', 0)
                        y1 = prediction.get('y1', 0)
                        x2 = prediction.get('x2', 0)
                        y2 = prediction.get('y2', 0)
                        width = x2 - x1
                        height = y2 - y1
                        x = x1 + width/2
                        y = y1 + height/2
                    
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    })
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                
                # Determine if guns are detected (confidence threshold of 0.4)
                gun_detected = highest_confidence >= 0.4
                
                if gun_detected:
                    logger.warning(f"Gun/weapon detected in thread! Highest confidence: {highest_confidence:.3f}")
                else:
                    logger.info(f"No guns detected in thread. Highest confidence: {highest_confidence:.3f}")
                
                return gun_detected, highest_confidence, detected_objects
                
            except Exception as e:
                logger.error(f"Error parsing prediction results in thread: {e}")
                return False, 0.0, []
                
        except Exception as e:
            logger.error(f"Gun detection error in thread: {e}")
            return False, 0.0, []
    
    def check_nsfw(self, frame):
        """Queue a frame for NSFW detection in the background thread"""
        if not self.nsfw_detection_available:
            return False, 0.0
            
        # Check if it's time to queue a new detection (3 times per second)
        current_time = time.time()
        time_since_last_check = current_time - self.last_nsfw_check_time
        
        if time_since_last_check >= self.nsfw_time_interval:
            self.last_nsfw_check_time = current_time
            self.frame_count_since_last_nsfw_check = 0
            
            # Don't block on the queue - if it's full, skip this frame
            try:
                # Make a deep copy of the frame to avoid issues with shared memory
                frame_copy = frame.copy()
                
                # Try to put the frame in the queue without blocking
                if self.nsfw_detection_queue.qsize() == 0:
                    self.nsfw_detection_queue.put_nowait(frame_copy)
                    logger.info(f"Queued frame for background NSFW detection")
                else:
                    logger.info(f"Skipped queueing frame - background NSFW detection still in progress")
            except queue.Full:
                logger.info(f"Skipped queueing frame - NSFW queue is full")
                
        else:
            # Count frames since last check (but don't display it)
            self.frame_count_since_last_nsfw_check += 1
        
        # Return the current results (from the thread)
        with self.nsfw_results_lock:
            return self.current_nsfw_status, self.nsfw_confidence
    
    def check_guns(self, frame):
        """Queue a frame for gun detection in the background thread"""
        if not self.gun_detection_available:
            return False, 0.0, []
            
        # Don't block on the queue - if it's full, skip this frame
        try:
            # Make a deep copy of the frame to avoid issues with shared memory
            frame_copy = frame.copy()
            
            # Try to put the frame in the queue without blocking
            if self.gun_detection_queue.qsize() == 0:
                self.gun_detection_queue.put_nowait(frame_copy)
                logger.info(f"Queued frame for background gun detection")
            else:
                logger.info(f"Skipped queueing frame - background detection still in progress")
        except queue.Full:
            logger.info(f"Skipped queueing frame - queue is full")
            
        # Return the current results (from the thread)
        with self.gun_results_lock:
            return self.current_gun_status, self.gun_confidence, self.gun_objects
    
    def blur_frame(self, frame, blur_strength=51):
        """Apply heavy blur to NSFW frame"""
        logger.info(f"Applying blur to NSFW frame (strength: {blur_strength})")
        
        # Make sure blur_strength is an odd number (required by GaussianBlur)
        if blur_strength % 2 == 0:
            blur_strength += 1
            logger.info(f"Adjusted blur strength to odd number: {blur_strength}")
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        
        # Add dark overlay for additional obscuring
        overlay = np.zeros_like(frame)
        overlay[:] = (0, 0, 0)  # Black overlay
        
        # Blend blurred frame with dark overlay
        cv2.addWeighted(blurred, 0.3, overlay, 0.7, 0, blurred)
        
        logger.info("Frame blurred successfully")
        return blurred
        
    def draw_gun_boxes(self, frame, detected_objects):
        """Draw bounding boxes around detected guns/weapons"""
        logger.info(f"Drawing bounding boxes for {len(detected_objects)} detected objects")
        
        try:
            # Create a copy of the frame to avoid modifying the original
            result_frame = frame.copy()
            
            # Draw each bounding box
            for obj in detected_objects:
                try:
                    # Extract coordinates - handle potential type issues
                    x = float(obj.get('x', 0))
                    y = float(obj.get('y', 0))
                    w = float(obj.get('width', 0))
                    h = float(obj.get('height', 0))
                    
                    # Skip if width or height is 0
                    if w <= 0 or h <= 0:
                        logger.warning(f"Invalid box dimensions: w={w}, h={h}")
                        continue
                    
                    # Calculate bounding box coordinates
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    
                    # Ensure coordinates are within frame boundaries
                    height, width = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width-1, x2)
                    y2 = min(height-1, y2)
                    
                    # Draw red rectangle around the object
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add label with confidence
                    confidence = float(obj.get('confidence', 0.0))
                    class_name = str(obj.get('class', 'unknown'))
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(result_frame, (x1, y1-20), (x1+label_size[0], y1), (0, 0, 255), -1)
                    
                    # Draw label text
                    cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                except Exception as e:
                    logger.error(f"Error drawing box for object {obj}: {e}")
                    continue
            
            logger.info("Bounding boxes drawn successfully")
            return result_frame
            
        except Exception as e:
            logger.error(f"Error in draw_gun_boxes: {e}")
            return frame  # Return original frame if there's an error
    
    def receive_video(self):
        """Receive and display video frames"""
        data = b""
        payload_size = struct.calcsize("L")
        
        logger.info("Starting video stream reception...")
        print("Receiving video stream...")
        print("Press 'q' to quit")
        if self.nsfw_detection_available:
            print("NSFW detection: Active (3 checks/sec, model inference every 3 sec)")
            logger.info("NSFW detection is active and ready")
        else:
            print("NSFW detection: Disabled")
            logger.warning("NSFW detection is disabled")
            
        if self.gun_detection_available:
            print(f"Gun detection: Active (every {self.gun_api_time_interval} seconds)")
            logger.info(f"Gun detection is active and ready (API calls every {self.gun_api_time_interval} seconds)")
        else:
            print("Gun detection: Disabled")
            logger.warning("Gun detection is disabled")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        # For frame rate control
        target_fps = 24  # Fixed to match original video's 24 FPS
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        
        try:
            while True:
                # Receive frame size
                while len(data) < payload_size:
                    data += self.socket.recv(4096)
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                # Receive frame data
                while len(data) < msg_size:
                    data += self.socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Decode frame
                try:
                    encoded_frame = pickle.loads(frame_data)
                    frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        frame_count += 1
                        current_time = time.time()
                        
                        # Check for NSFW content (rate limited by check_nsfw)
                        if self.nsfw_detection_available:
                            logger.info(f"Checking frame #{frame_count} for NSFW content...")
                            self.current_nsfw_status, self.nsfw_confidence = self.check_nsfw(frame)
                            
                            if self.current_nsfw_status:
                                print(f"NSFW detected! Confidence: {self.nsfw_confidence:.2f}")
                            else:
                                print(f"SFW - Confidence: {self.nsfw_confidence:.2f}")
                        
                        # Check for guns/weapons based on time interval (not frames)
                        current_time = time.time()
                        time_since_last_call = current_time - self.last_gun_api_call_time
                        
                        if self.gun_detection_available and (time_since_last_call >= self.gun_api_time_interval):
                            logger.info(f"Checking frame #{frame_count} for weapons (time since last check: {time_since_last_call:.1f}s)...")
                            self.current_gun_status, self.gun_confidence, self.gun_objects = self.check_guns(frame)
                            
                            if self.current_gun_status:
                                print(f"WEAPON detected! Confidence: {self.gun_confidence:.2f}")
                            else:
                                print(f"No weapons - Confidence: {self.gun_confidence:.2f}")
                        else:
                            # If not checking this frame, still show status from previous check
                            if self.gun_detection_available:
                                time_remaining = max(0, self.gun_api_time_interval - time_since_last_call)
                                logger.info(f"Skipping weapon detection on frame #{frame_count} (next check in {time_remaining:.1f}s)")
                                if self.current_gun_status:
                                    print(f"WEAPON detected! Confidence: {self.gun_confidence:.2f}")
                                else:
                                    print(f"No weapons - Confidence: {self.gun_confidence:.2f}")
                        
                        # Apply blur if NSFW content detected
                        if self.current_nsfw_status:
                            frame = self.blur_frame(frame)
                            
                        # Draw bounding boxes if guns are detected
                        if self.current_gun_status and self.gun_objects:
                            frame = self.draw_gun_boxes(frame, self.gun_objects)
                        
                        # Calculate and display FPS
                        current_tick = cv2.getTickCount()
                        elapsed_time = (current_tick - start_time) / cv2.getTickFrequency()
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        
                        # Add status text to frame
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Frames: {frame_count}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add NSFW status
                        if self.nsfw_detection_available:
                            if self.current_nsfw_status:
                                cv2.putText(frame, f"NSFW: {self.nsfw_confidence:.2f}", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(frame, "BLURRED", (10, 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, f"SFW: {self.nsfw_confidence:.2f}", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "NSFW Detection: Disabled", (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                                       
                        # Add Gun detection status
                        y_pos = 150  # Starting Y position for gun status text
                        if self.gun_detection_available:
                            if self.current_gun_status:
                                cv2.putText(frame, f"WEAPON: {self.gun_confidence:.2f}", (10, y_pos), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(frame, f"Objects: {len(self.gun_objects)}", (10, y_pos+30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, "No weapons detected", (10, y_pos), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Weapon Detection: Disabled", (10, y_pos), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                        
                        # Display frame
                        cv2.imshow('Video Stream', frame)
                        
                        # Maintain consistent frame rate (similar to original video)
                        # Calculate how long to wait before next frame
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        sleep_time = max(0.001, frame_time - elapsed)
                        
                        # Check for quit key with precise timing to maintain frame rate
                        if cv2.waitKey(int(sleep_time * 1000)) & 0xFF == ord('q'):
                            logger.info("Quit requested by user")
                            print("Quit requested by user")
                            break
                            
                        # Maintain a consistent 24 FPS (no dynamic adjustment)
                        if frame_count == 30:  # Log FPS after first 30 frames
                            actual_fps = 30 / (time.time() - start_time)
                            logger.info(f"Current actual FPS: {actual_fps:.1f}, maintaining target: 24.0")
                            
                        last_frame_time = time.time()
                    else:
                        logger.warning("Error: Received invalid frame")
                        print("Error: Received invalid frame")
                        
                except Exception as e:
                    logger.error(f"Error decoding frame: {e}")
                    print(f"Error decoding frame: {e}")
                    continue
                    
        except (socket.error, ConnectionResetError):
            logger.error("Connection lost to server")
            print("Connection lost to server")
        except KeyboardInterrupt:
            logger.info("Stopping receiver...")
            print("\nStopping receiver...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop the NSFW detection thread if it's running
            if self.nsfw_detection_running:
                logger.info("Stopping NSFW detection thread...")
                self.nsfw_detection_running = False
                
                # Wait for the thread to finish (with timeout)
                if self.nsfw_detection_thread and self.nsfw_detection_thread.is_alive():
                    self.nsfw_detection_thread.join(timeout=2.0)
                    logger.info("NSFW detection thread stopped" if not self.nsfw_detection_thread.is_alive() 
                              else "NSFW detection thread timeout - continuing cleanup")
            
            # Stop the gun detection thread if it's running
            if self.gun_detection_running:
                logger.info("Stopping gun detection thread...")
                self.gun_detection_running = False
                
                # Wait for the thread to finish (with timeout)
                if self.gun_detection_thread and self.gun_detection_thread.is_alive():
                    self.gun_detection_thread.join(timeout=2.0)
                    logger.info("Gun detection thread stopped" if not self.gun_detection_thread.is_alive() 
                              else "Gun detection thread timeout - continuing cleanup")
            
            cv2.destroyAllWindows()
            self.socket.close()
            logger.info("Receiver closed successfully")
            print("Receiver closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            pass

def main():
    logger.info("=== Starting Video Receiver with NSFW Detection ===")
    receiver = VideoReceiver()
    
    if receiver.connect_to_server():
        try:
            receiver.receive_video()
        except KeyboardInterrupt:
            logger.info("Shutting down receiver...")
            print("\nShutting down receiver...")
            receiver.cleanup()

if __name__ == "__main__":
    main() 
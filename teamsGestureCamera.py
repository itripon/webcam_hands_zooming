#!/usr/bin/env python3
"""
Teams Gesture Camera - Virtual Camera with Hand Gesture Zoom Cont        # Interface toggle
        self.show_interface = True  # Interface is ON by default
        
        # Initialize pygame mixer for sound
        try:
            pygame.mixer.init()
            if os.path.exists(self.sound_file):
                self.sound_loaded = True
                logger.info(f"Sound file loaded: {self.sound_file}")
            else:
                self.sound_loaded = False
                logger.warning(f"Sound file not found: {self.sound_file}")
        except Exception as e:
            self.sound_loaded = False
            logger.error(f"Failed to initialize sound: {e}")al camera that can be used with OBS Studio and Microsoft Teams.
 
Features:
- Hand gesture detection for zoom control
- Virtual camera output compatible with OBS Studio
- Natural gesture logic: Close hand = Zoom out, Open hand = Zoom in
- Sensitivity range: 20-150 pixels between thumb and index finger
- Real-time zoom level display
 
Usage:
1. Run this script
2. In OBS Studio, add "Window Capture" source and select "Teams Gesture Camera - Virtual Output"
3. In Teams, select "OBS Virtual Camera" as your camera
 
Controls:
- Close your hand (thumb and index finger together) to zoom out
- Open your hand (spread thumb and index finger apart) to zoom in
- Press 'q' to quit
- Press 'r' to reset zoom to 1.0x
 
Author: Adapted for Teams integration
"""
 
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import logging
import pygame
import os
import threading
 
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
 
class TeamsGestureCamera:
    def __init__(self):
        """Initialize the Teams Gesture Camera with MediaPipe and OpenCV."""
        self.mp_hands = mp.solutions.hands
        # MediaPipe hands will be initialized after we know the camera resolution
        self.hands = None
        self.mediapipe_scale_factor = 1.0  # Scale factor for coordinate conversion
        self.mp_draw = mp.solutions.drawing_utils
       
        # Zoom parameters - INVERTED LOGIC
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.zoom_speed = 0.04
       
        # Gesture detection parameters (will be set based on resolution)
        self.min_distance = 100  # Close hand threshold (zoom out)
        self.max_distance = 400  # Open hand threshold (zoom in)
       
        # Smoothing parameters - separate for zoom and camera movement
        # Zoom speed smoothing (faster response for zoom changes)
        self.distance_history = []
        self.distance_timestamps = []
        self.zoom_history_time_window = 0.3  # 0.3 seconds for zoom responsiveness
        self.history_size = 10  # Fallback max size for zoom
        
        # Camera movement smoothing (slower, more stable for position)
        self.zoom_center_history_x = []
        self.zoom_center_history_y = []
        self.zoom_center_timestamps = []  # Track timestamps for time-based smoothing
        self.center_history_time_window = 5  # 5 seconds of history for moving average
        self.center_history_size = 10  # Fallback max size
       
        # Camera setup
        self.cap = None
        # Remove fixed frame dimensions - will be set based on camera's native resolution
        self.frame_width = None
        self.frame_height = None
       
        # Virtual output window
        self.output_window_name = "Teams Gesture Camera - Virtual Output"
       
        # Status
        self.running = False
        self.last_gesture_time = time.time()
       
        # Sound system for zoom transitions
        self.sound_file = "pam-pam-pammm.mp3"
        self.previous_zoom = 1.0
        self.has_touched_1x = True  # Start as True since we begin at 1x
        self.is_eligible_for_sound = True  # Can play sound when conditions are met
        self.max_zoom_during_transition = 1.0  # Track max zoom during current transition
        self.max_allowed_drop = 0.25  # Maximum allowed drop during 1x->2x transition
       
        # Interface toggle
        self.show_interface = True  # Interface is ON by default
        
        # Functionality toggle
        self.functionality_enabled = True  # Gesture detection is ON by default
        
        # Zoom center tracking (for zooming to finger position)
        self.zoom_center_x = 0.5  # Default to center of frame (normalized coordinates)
        self.zoom_center_y = 0.5  # Default to center of frame (normalized coordinates)
        
        # Visual smoothing for hand landmarks (anti-jitter)
        self.smoothed_landmarks = {}  # Store smoothed positions for each hand
        self.hand_tracking_ids = {}  # Map current hand positions to persistent IDs
        self.next_hand_id = 0  # Counter for assigning new hand IDs
        self.landmark_movement_threshold = 0.005  # Much more sensitive (0.5% instead of 2%)
        self.landmark_smoothing_factor = 0.7  # Faster response (70% new, 30% old instead of 30% new)
       
        # Initialize pygame mixer for sound
        try:
            pygame.mixer.init()
            if os.path.exists(self.sound_file):
                self.sound_loaded = True
                logger.info(f"Sound file loaded: {self.sound_file}")
            else:
                self.sound_loaded = False
                logger.warning(f"Sound file not found: {self.sound_file}")
        except Exception as e:
            self.sound_loaded = False
            logger.error(f"Failed to initialize sound system: {e}")
       
        logger.info("Teams Gesture Camera initialized")
        logger.info("Gesture range will be set based on camera resolution (NATURAL: close=zoom out, open=zoom in)")
        if self.sound_loaded:
            logger.info("[SOUND] Sound system ready - will play sound at half volume to speakers when zooming from 1x to 2.5x smoothly!")
        else:
            logger.info("[WARN] Sound system disabled - check sound file path")
 
    def initialize_camera(self, camera_index=0):
        """Initialize the camera capture and set it to its maximum resolution."""
        try:
            # Initialize camera with DirectShow backend for better Windows support
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                logger.warning("DirectShow failed, trying default backend")
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    raise Exception(f"Could not open camera {camera_index}")
           
            # Get initial camera properties (usually defaults like 640x480)
            initial_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            initial_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Initial camera properties: {initial_width}x{initial_height}")
            
            # Try common high resolutions to find camera's maximum capability
            test_resolutions = [
                (1920, 1080),  # 1080p
                (1680, 1050),  # 16:10 WSXGA+
                (1600, 1200),  # 4:3 UXGA
                (1440, 900),   # 16:10 WXGA+
                (1366, 768),   # 16:9 FWXGA
                (1280, 1024),  # 5:4 SXGA
                (1280, 720),   # 720p
                (1024, 768),   # 4:3 XGA
                (800, 600),    # 4:3 SVGA
                (640, 480)     # 4:3 VGA (fallback)
            ]
            
            best_width, best_height = initial_width, initial_height
            
            logger.info("Testing camera resolutions...")
            for width, height in test_resolutions:
                # Try to set this resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Check what the camera actually set
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Test if we can actually read a frame at this resolution
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    frame_height, frame_width = test_frame.shape[:2]
                    if frame_width == width and frame_height == height:
                        best_width, best_height = width, height
                        logger.info(f"[OK] Confirmed: {width}x{height} works")
                        break  # Found working resolution, use it
                    else:
                        logger.info(f"[NO] {width}x{height} -> got {frame_width}x{frame_height}")
                else:
                    logger.info(f"[NO] {width}x{height} -> failed to read frame")
            
            # Set the best resolution we found
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
            
            # Optimize for maximum frame rate
            # Try to set the highest possible FPS
            target_fps_options = [60, 30, 25, 24, 15]  # Common camera FPS rates, highest first
            for target_fps in target_fps_options:
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if actual_fps >= target_fps * 0.9:  # Accept if within 90% of target
                    logger.info(f"Camera FPS set to: {actual_fps:.1f} fps (target: {target_fps})")
                    break
            else:
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"Camera FPS: {actual_fps:.1f} fps (using camera default)")
            
            # Optimize camera buffer to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to get latest frames
            
            # Read final test frame to confirm dimensions
            ret, final_frame = self.cap.read()
            if ret and final_frame is not None:
                actual_height, actual_width = final_frame.shape[:2]
                self.frame_width = actual_width
                self.frame_height = actual_height
                logger.info(f"Final camera resolution: {self.frame_width}x{self.frame_height}")
            else:
                # Fallback to properties if we can't read a frame
                self.frame_width = best_width
                self.frame_height = best_height
                logger.warning("Could not read final test frame, using set properties")
            
            # Log the final aspect ratio
            aspect_ratio = self.frame_width / self.frame_height
            logger.info(f"Camera aspect ratio: {aspect_ratio:.3f}:1")
            
            if self.frame_width > 640 or self.frame_height > 480:
                logger.info("Successfully set camera to higher resolution than default 640x480")
            else:
                logger.warning("Camera is using 640x480 - may be camera limitation or driver issue")
            
            # Set resolution-relative gesture parameters now that we know the frame dimensions
            self.set_resolution_relative_gesture_params()
            
            # Initialize MediaPipe with resolution-optimized settings
            self.initialize_mediapipe_for_resolution()
            
            return True
           
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
 
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
 
    def smooth_distance(self, distance, num_hands=1):
        """Apply time-based smoothing to distance measurements for zoom responsiveness."""
        current_time = time.time()
        
        # Add current distance and timestamp to history
        self.distance_history.append(distance)
        self.distance_timestamps.append(current_time)
        
        # Remove entries older than the zoom time window
        cutoff_time = current_time - self.zoom_history_time_window
        while (self.distance_timestamps and 
               self.distance_timestamps[0] < cutoff_time):
            self.distance_history.pop(0)
            self.distance_timestamps.pop(0)
        
        # Also maintain maximum size as fallback
        while len(self.distance_history) > self.history_size:
            self.distance_history.pop(0)
            self.distance_timestamps.pop(0)
        
        # Zoom smoothing should be responsive but not jittery
        # Less aggressive smoothing than camera movement for better zoom responsiveness
        hand_factor = min(num_hands / 2.0, 1.5)  # Scale factor based on hand count
        
        if len(self.distance_history) >= 3:
            # Time-weighted average with moderate dampening for zoom
            total_weight = 0
            weighted_sum = 0
            
            for i, timestamp in enumerate(self.distance_timestamps):
                # Age of this sample in seconds
                age = current_time - timestamp
                
                # Base weight decreases exponentially with age (faster decay for zoom)
                base_weight = math.exp(-age * 5.0)  # Faster decay than camera movement
                
                # Apply moderate dampening for multiple hands (less than camera movement)
                stability_factor = 1.0 + (hand_factor - 1.0) * 0.3  # Less dampening for zoom
                adjusted_weight = base_weight / stability_factor
                
                weighted_sum += self.distance_history[i] * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
        
        # Fallback to simple average
        return sum(self.distance_history) / len(self.distance_history)
 
    def update_zoom_from_distance(self, distance, num_hands=1):
        """Update zoom factor based on hand gesture distance with hand count stabilization."""
        if distance < self.min_distance:
            # Very close fingers - zoom out more
            zoom_change = self.zoom_speed * 4
            self.zoom_factor = max(self.min_zoom, self.zoom_factor - zoom_change)
        elif distance > self.max_distance:
            # Far apart fingers - zoom in more
            zoom_change = self.zoom_speed * 4
            self.zoom_factor = min(self.max_zoom, self.zoom_factor + zoom_change)
        else:
            # Proportional zoom based on distance (natural)
            # Closer distance = lower zoom, farther distance = higher zoom
            normalized_distance = (distance - self.min_distance) / (self.max_distance - self.min_distance)
            target_zoom = self.min_zoom + (normalized_distance * (self.max_zoom - self.min_zoom))
           
            # Smooth transition to target zoom with adaptive smoothing based on zoom level and hand count
            zoom_diff = target_zoom - self.zoom_factor
            
            # Base smoothing factor (doubled for 2x faster zoom)
            base_smooth_factor = 0.30
            
            # Apply more aggressive smoothing when zoomed in to reduce jitter
            zoom_based_smoothing = min(self.zoom_factor / 3.0, 0.8)
            
            # Apply additional smoothing based on number of hands (more hands = more smoothing)
            hand_smoothing_factor = min(num_hands / 2.0, 2.0)  # Scale from 1x to 2x smoothing
            
            # Combine all smoothing factors
            total_smoothing = base_smooth_factor * (1 - zoom_based_smoothing * 0.7) / hand_smoothing_factor
            
            self.zoom_factor += zoom_diff * total_smoothing
           
        # Clamp zoom factor
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))
 
    def apply_zoom(self, frame):
        """Apply zoom to the frame by cropping and resizing, centered on finger position."""
        if self.zoom_factor <= 1.0:
            return frame
       
        height, width = frame.shape[:2]
       
        # Calculate crop dimensions
        crop_width = int(width / self.zoom_factor)
        crop_height = int(height / self.zoom_factor)
       
        # Calculate desired center position based on finger position
        center_x = int(self.zoom_center_x * width)
        center_y = int(self.zoom_center_y * height)
        
        # Calculate crop position (centered on finger position)
        start_x = center_x - crop_width // 2
        start_y = center_y - crop_height // 2
        
        # Ensure crop area stays within frame bounds
        start_x = max(0, min(start_x, width - crop_width))
        start_y = max(0, min(start_y, height - crop_height))
       
        # Crop and resize
        cropped = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
       
        return zoomed
 
    def draw_info(self, frame, distance=None, hand_detected=False, num_hands=0, current_fps=None, metric_type="distance"):
        """Draw information overlay on the frame with resolution-relative sizing."""
        height, width = frame.shape[:2]
        
        # Skip drawing interface elements if interface is toggled off
        if not self.show_interface:
            return frame
        
        # Calculate scaling factors based on resolution
        scale_x = width / 1920.0
        scale_y = height / 1080.0
        scale = min(scale_x, scale_y)  # Use smaller scale to maintain aspect ratio
        
        # Function to scale values
        def scale_size(size):
            return max(1, int(size * scale))
        
        # Draw the status indicator only when interface is on
        # Green square = functionality ON, Red square = functionality OFF
        indicator_size = scale_size(30)
        indicator_x = width - indicator_size - scale_size(15)
        indicator_y = scale_size(15)
        indicator_color = (0, 255, 0) if self.functionality_enabled else (0, 0, 255)  # Green or Red
        cv2.rectangle(frame, (indicator_x, indicator_y), 
                     (indicator_x + indicator_size, indicator_y + indicator_size), 
                     indicator_color, -1)
        
        # Add a white border around the indicator
        cv2.rectangle(frame, (indicator_x, indicator_y), 
                     (indicator_x + indicator_size, indicator_y + indicator_size), 
                     (255, 255, 255), scale_size(2))
            
        # Semi-transparent overlay
        overlay = frame.copy()
       
        # Status panel (scaled and enlarged for bigger text)
        panel_x = scale_size(15)
        panel_y = scale_size(15)
        panel_width = scale_size(650)
        panel_height = scale_size(220)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
       
        # Text scaling (increased significantly)
        title_font_scale = 1.44 * scale  # 80% of previous size (was 1.8)
        info_font_scale = 1.4 * scale
        small_font_scale = 1.1 * scale
        instruction_font_scale = 1 * scale  # 60% of previous size (was 1.35)
        fps_font_scale = 1.0 * scale  # New FPS display scale
        
        title_thickness = scale_size(4)
        info_thickness = scale_size(3)
        small_thickness = scale_size(3)
        instruction_thickness = scale_size(2)  # Made bolder (was 2)
        fps_thickness = scale_size(2)  # New FPS display thickness
        
        # Title
        cv2.putText(frame, "Teams Gesture Camera", (scale_size(30), scale_size(65)),
                   cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (0, 255, 255), title_thickness)
       
        # Zoom info
        zoom_text = f"Zoom: {self.zoom_factor:.2f}x"
        cv2.putText(frame, zoom_text, (scale_size(30), scale_size(110)),
                   cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, (0, 255, 0), info_thickness)
       
        # Hand detection status with count
        if hand_detected and num_hands > 0:
            status_color = (0, 255, 0)
            status_text = f"Hands: {num_hands}"
        else:
            status_color = (0, 0, 255)
            status_text = "No Hands"
        cv2.putText(frame, status_text, (scale_size(30), scale_size(155)),
                   cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, status_color, small_thickness)
       
        # Distance/Area info (with black outline for readability)
        if distance is not None and hand_detected:
            if metric_type == "area":
                distance_text = f"Area: {distance:.1f}px (polygon)"
            else:
                distance_text = f"Distance: {distance:.1f}px"
            text_pos = (scale_size(30), scale_size(200))
            # Draw black outline for better readability
            cv2.putText(frame, distance_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (0, 0, 0), small_thickness + 2)
            cv2.putText(frame, distance_text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (255, 255, 255), small_thickness)
       
        # Instructions (bottom of screen, scaled with black outline for readability)
        instructions = [
            "Close hand = Zoom OUT",
            "Open hand = Zoom IN",
            "Press 'q' to quit, 'r' to reset, 'i' to toggle UI, 'x' to toggle functionality"
        ]
       
        for i, instruction in enumerate(instructions):
            y_pos = height - scale_size(120) + (i * scale_size(35))  # Increased spacing
            text_pos = (scale_size(15), y_pos)
            # Draw black outline for better readability
            cv2.putText(frame, instruction, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, (0, 0, 0), instruction_thickness + 3)  # Thicker outline for bolder text
            cv2.putText(frame, instruction, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, (139, 0, 0), instruction_thickness)  # Dark blue color
       
        # FPS display in bottom right corner
        if current_fps is not None:
            fps_text = f"FPS: {current_fps:.1f}"
            # Get text size to position it correctly
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, fps_thickness)[0]
            fps_x = width - text_size[0] - scale_size(15)  # 15px margin from right edge
            fps_y = height - scale_size(15)  # 15px margin from bottom
            fps_pos = (fps_x, fps_y)
            
            # Draw FPS with black outline for better readability
            cv2.putText(frame, fps_text, fps_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (0, 0, 0), fps_thickness + 2)
            cv2.putText(frame, fps_text, fps_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, (0, 255, 0), fps_thickness)  # Green color
       
        return frame
 
    def create_placeholder_frame(self):
        """Create a placeholder frame when camera is not available."""
        # Use default resolution if camera dimensions not set yet
        height = self.frame_height if self.frame_height else 480
        width = self.frame_width if self.frame_width else 640
        
        frame = np.zeros((height, width, 3), dtype=np.uint8)
       
        # Add gradient background
        for y in range(height):
            intensity = int(50 + (y / height) * 50)
            frame[y, :] = [intensity, intensity//2, intensity//3]
       
        # Add text
        text = "Teams Gesture Camera"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height - text_size[1]) // 2
       
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
       
        cv2.putText(frame, "Virtual Camera Ready", (text_x - 50, text_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       
        cv2.putText(frame, "Waiting for camera...", (text_x - 60, text_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
       
        return frame
 
    def check_zoom_transition_and_play_sound(self):
        """Check if zoom transitioned from 1x to 2.5x smoothly and play sound."""
        if not self.sound_loaded:
            return
       
        current_zoom = self.zoom_factor
       
        # Check if we've touched 1x (reset condition)
        if current_zoom <= 1.0:
            if not self.has_touched_1x or not self.is_eligible_for_sound:
                self.has_touched_1x = True
                self.is_eligible_for_sound = True
                self.max_zoom_during_transition = current_zoom
                logger.info(f"[SOUND] Reset: Touched 1x ({current_zoom:.2f}), now eligible for sound trigger")
       
        # If we're eligible and above 1x, start tracking the transition
        if self.is_eligible_for_sound and current_zoom > 1.0:
            # Update max zoom reached during this transition
            self.max_zoom_during_transition = max(self.max_zoom_during_transition, current_zoom)
           
            # Check if zoom dropped too much during the transition
            zoom_drop = self.max_zoom_during_transition - current_zoom
            if zoom_drop > self.max_allowed_drop:
                # Too much drop, disable sound for this transition
                self.is_eligible_for_sound = False
                logger.info(f"[SOUND] Disqualified: Zoom dropped {zoom_drop:.3f} (max allowed: {self.max_allowed_drop:.2f}). Will reset when reaching 1x again.")
           
            # Check if we've reached 2.5x while still eligible
            if self.is_eligible_for_sound and current_zoom >= 2.5 and self.previous_zoom < 2.5:
                # Play sound using simple pygame to speakers only
                self.play_sound_to_speakers()
                logger.info(f"[SOUND] TRIGGERED! Smooth transition from 1x to 2.5x (current: {current_zoom:.2f}x)")
               
                # Disable eligibility until we touch 1x again
                self.is_eligible_for_sound = False
                self.has_touched_1x = False
       
        # Update previous zoom for next iteration
        self.previous_zoom = current_zoom
 
    def play_sound_to_speakers(self):
        """Play sound to speakers only at half volume."""
        if not self.sound_loaded:
            return
       
        def play_audio():
            """Play audio to speakers using pygame."""
            try:
                # Initialize pygame for speakers
                pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
                pygame.mixer.init()
                pygame.mixer.music.load(self.sound_file)
                pygame.mixer.music.set_volume(0.5)  # Half volume
                pygame.mixer.music.play()
               
                # Wait for audio to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                   
                logger.info("[SOUND] Audio played to speakers at half volume")
                   
            except Exception as e:
                logger.error(f"Failed to play sound: {e}")
       
        # Run the playback in a background thread
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()
 
    def run(self):
        """Main loop for the gesture camera."""
        logger.info("Starting Teams Gesture Camera...")
        
        # Initialize camera
        if not self.initialize_camera():
            logger.warning("Camera initialization failed, using placeholder")
       
        self.running = True
       
        # Create output window 
        cv2.namedWindow(self.output_window_name, cv2.WINDOW_NORMAL)
        
        # Size window immediately with camera dimensions if available
        if self.frame_width and self.frame_height:
            display_width = min(800, self.frame_width)
            aspect_ratio = self.frame_width / self.frame_height
            display_height = int(display_width / aspect_ratio)
            cv2.resizeWindow(self.output_window_name, display_width, display_height)
            logger.info(f"Initial window size: {display_width}x{display_height} (camera: {self.frame_width}x{self.frame_height}, ratio: {aspect_ratio:.3f})")
        
        logger.info("Teams Gesture Camera is running!")
        logger.info("GESTURE CONTROLS (NATURAL):")
        logger.info("- Close hand (fingers together) = ZOOM OUT")
        logger.info("- Open hand (fingers apart) = ZOOM IN")
        logger.info("KEYBOARD CONTROLS:")
        logger.info("- Press 'q' to quit")
        logger.info("- Press 'r' to reset zoom to 1.0x")
        logger.info("- Press 'i' to toggle interface ON/OFF")
        logger.info("- Press 'x' to toggle functionality ON/OFF")
        logger.info("For OBS: Add 'Window Capture' and select 'Teams Gesture Camera - Virtual Output'")
       
        try:
            frame_count = 0
            fps_start_time = time.time()
            fps_frame_count = 0
            
            while self.running:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to read frame, using placeholder")
                        frame = self.create_placeholder_frame()
                    else:
                        # Verify frame dimensions on first frame
                        if frame_count == 0:
                            actual_height, actual_width = frame.shape[:2]
                            logger.info(f"First frame verification: {actual_width}x{actual_height}")
                            
                            if actual_width != self.frame_width or actual_height != self.frame_height:
                                logger.warning(f"Frame size changed! Expected: {self.frame_width}x{self.frame_height}, Got: {actual_width}x{actual_height}")
                                # Update our stored dimensions and resize window
                                self.frame_width = actual_width
                                self.frame_height = actual_height
                                
                                # Resize window with corrected dimensions
                                display_width = min(800, self.frame_width)
                                aspect_ratio = self.frame_width / self.frame_height
                                display_height = int(display_width / aspect_ratio)
                                cv2.resizeWindow(self.output_window_name, display_width, display_height)
                                logger.info(f"Window resized to correct ratio: {display_width}x{display_height} (camera: {self.frame_width}x{self.frame_height}, ratio: {aspect_ratio:.3f})")
                            else:
                                logger.info("Frame dimensions confirmed - window sizing is correct")
                            
                            frame_count += 1
                else:
                    frame = self.create_placeholder_frame()
               
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
               
                # Process frame for MediaPipe (may be resized for performance)
                if self.hands:  # Only process if MediaPipe is initialized
                    processing_frame = self.process_frame_for_mediapipe(frame)
                    rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                else:
                    results = None
               
                hand_detected = False
                distance = None
                num_hands_detected = 0  # Initialize hand count
                metric_type = "distance"  # Default metric type
               
                # Process hand landmarks only if functionality is enabled and MediaPipe is initialized
                if results and results.multi_hand_landmarks and self.functionality_enabled:
                    # Filter out duplicate/overlapping hand detections
                    filtered_hands = self.deduplicate_hands(results.multi_hand_landmarks)
                    
                    # Assign persistent IDs to maintain tracking continuity
                    hand_id_mapping = self.assign_persistent_hand_ids(filtered_hands)
                    
                    hand_detected = True
                    num_hands_detected = len(filtered_hands)
                    
                    # Collect all thumb and index positions from all detected hands
                    all_thumbs = []
                    all_indexes = []
                    all_distances = []
                    
                    for hand_index, hand_landmarks in enumerate(filtered_hands):
                        # Get persistent hand ID
                        persistent_hand_id = hand_id_mapping[hand_index]
                        
                        # Apply visual smoothing to reduce jitter
                        smoothed_landmarks = self.smooth_hand_landmarks(hand_landmarks, persistent_hand_id)
                        
                        # Draw hand landmarks only if interface is enabled
                        if self.show_interface:
                            # Create custom drawing specs for consistent dot sizes
                            h, w = frame.shape[:2]
                            scale = min(w / 1920.0, h / 1080.0)
                            landmark_radius = max(3, int(8 * scale))
                            
                            # Custom drawing specs for landmarks
                            landmark_drawing_spec = self.mp_draw.DrawingSpec(
                                color=(0, 255, 255),  # Yellow landmarks
                                thickness=-1,  # Filled circles
                                circle_radius=landmark_radius
                            )
                            connection_drawing_spec = self.mp_draw.DrawingSpec(
                                color=(255, 255, 255),  # White connections
                                thickness=max(1, int(2 * scale))
                            )
                            
                            self.mp_draw.draw_landmarks(
                                frame, 
                                smoothed_landmarks,  # Use smoothed landmarks for drawing
                                self.mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec,
                                connection_drawing_spec
                            )
                       
                        # Get thumb tip and index finger tip from smoothed landmarks
                        thumb_tip = smoothed_landmarks.landmark[4]
                        index_tip = smoothed_landmarks.landmark[8]
                        
                        # Convert to original frame coordinates (MediaPipe uses normalized coordinates)
                        thumb_x, thumb_y = self.convert_mediapipe_coordinates(thumb_tip, frame.shape[1], frame.shape[0])
                        index_x, index_y = self.convert_mediapipe_coordinates(index_tip, frame.shape[1], frame.shape[0])
                        
                        # Store positions for center calculation (normalized coordinates)
                        all_thumbs.append((thumb_tip.x, thumb_tip.y))
                        all_indexes.append((index_tip.x, index_tip.y))
                        
                        # Calculate distance for this hand using original frame dimensions
                        hand_distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                        all_distances.append(hand_distance)
                        
                        # Draw connection line and circles only if interface is enabled
                        if self.show_interface:
                            h, w = frame.shape[:2]
                            
                            # Calculate scaling for hand tracking elements
                            scale = min(w / 1920.0, h / 1080.0)
                            line_thickness = max(2, int(4 * scale))
                            circle_radius = max(5, int(12 * scale))
                            border_thickness = max(1, int(2 * scale))  # Border thickness for circles
                            
                            # Use original frame coordinates for drawing
                            thumb_pos = (int(thumb_x), int(thumb_y))
                            index_pos = (int(index_x), int(index_y))
                            # Use different colors for multiple hands
                            hand_colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), 
                                         (255, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255),
                                         (255, 192, 203), (128, 255, 128)]
                            hand_number = len(all_distances)
                            hand_color = hand_colors[(hand_number - 1) % len(hand_colors)]
                            cv2.line(frame, thumb_pos, index_pos, hand_color, line_thickness)
                            
                            # Draw moderate-sized circles with borders for better visibility
                            # Thumb (green with white border)
                            cv2.circle(frame, thumb_pos, circle_radius, (255, 255, 255), border_thickness)  # White border
                            cv2.circle(frame, thumb_pos, circle_radius - border_thickness, (0, 255, 0), -1)  # Green fill
                            
                            # Index finger (red with white border)
                            cv2.circle(frame, index_pos, circle_radius, (255, 255, 255), border_thickness)  # White border
                            cv2.circle(frame, index_pos, circle_radius - border_thickness, (0, 0, 255), -1)  # Red fill
                    
                    # Calculate zoom metric from all hands
                    if all_distances:
                        num_hands = len(all_distances)
                        
                        if num_hands == 1:
                            # Single hand: use thumb-index distance as before
                            distance = all_distances[0]
                            metric_type = "distance"
                        else:
                            # Multiple hands: use polygon area formed by all thumb and index positions
                            all_points = all_thumbs + all_indexes
                            if len(all_points) >= 3:
                                # Convert to pixel coordinates for area calculation
                                h, w = frame.shape[:2]
                                pixel_points = np.array([(int(point[0] * w), int(point[1] * h)) for point in all_points], np.int32)
                                
                                # Calculate convex hull area
                                hull = cv2.convexHull(pixel_points)
                                polygon_area = cv2.contourArea(hull)
                                
                                # Convert area to a distance-like metric for zoom control
                                # For area, we need to map it to the same range as distances (min_distance to max_distance)
                                # Calculate expected area range: smaller area = zoom out, larger area = zoom in
                                min_area = (self.min_distance ** 2) * 0.5  # Rough estimate for minimum area
                                max_area = (self.max_distance ** 2) * 2.0  # Rough estimate for maximum area
                                
                                # Map polygon area to distance range
                                area_ratio = (polygon_area - min_area) / (max_area - min_area)
                                area_ratio = max(0, min(1, area_ratio))  # Clamp to 0-1
                                distance = self.min_distance + area_ratio * (self.max_distance - self.min_distance)
                                metric_type = "area"
                            else:
                                # Fallback to average distance if not enough points
                                distance = sum(all_distances) / len(all_distances)
                                metric_type = "distance"
                        
                        distance = self.smooth_distance(distance, num_hands)
                        
                        # Calculate center of all thumb and index positions
                        all_points = all_thumbs + all_indexes
                        center_x = sum(point[0] for point in all_points) / len(all_points)
                        center_y = sum(point[1] for point in all_points) / len(all_points)
                        
                        # Apply smoothing to zoom center for stable movement with hand count consideration
                        self.zoom_center_x, self.zoom_center_y = self.smooth_zoom_center(center_x, center_y, num_hands)
                        
                        # Update zoom based on distance (single hand) or polygon area (multiple hands)
                        self.update_zoom_from_distance(distance, num_hands)
                        self.last_gesture_time = time.time()
                        
                        # Draw center point if interface is enabled
                        if self.show_interface:
                            h, w = frame.shape[:2]
                            
                            # Calculate scaling for center elements
                            scale = min(w / 1920.0, h / 1080.0)
                            polygon_thickness = max(1, int(3 * scale))
                            center_radius = max(1, int(8 * scale))
                            text_scale = 1.0 * scale
                            text_thickness = max(1, int(3 * scale))
                            
                            center_pos = (int(center_x * w), int(center_y * h))
                            
                            # Draw semi-transparent shape from all thumb and index positions
                            if len(all_points) >= 3:
                                # Convert points to pixel coordinates
                                pixel_points = np.array([(int(point[0] * w), int(point[1] * h)) for point in all_points], np.int32)
                                
                                # Use convex hull to create a proper polygon shape instead of hourglass
                                hull = cv2.convexHull(pixel_points)
                                
                                # Create overlay for semi-transparent effect
                                overlay = frame.copy()
                                cv2.fillPoly(overlay, [hull], (0, 255, 255))  # Cyan fill
                                cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
                                
                                # Draw polygon outline
                                cv2.polylines(frame, [hull], True, (255, 255, 0), polygon_thickness)  # Yellow outline
                            
                            # Draw smaller center point as a purple filled dot
                            cv2.circle(frame, center_pos, center_radius, (128, 0, 128), -1)
                            
                            # Draw hand count text with black outline for better readability
                            hand_text = f"Hands: {len(all_distances)}"
                            text_pos = (center_pos[0] - int(60 * scale), center_pos[1] - int(30 * scale))
                            # Black outline
                            cv2.putText(frame, hand_text, text_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness + 2)
                            # Yellow text
                            cv2.putText(frame, hand_text, text_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 0), text_thickness)
                        
                elif results.multi_hand_landmarks and not self.functionality_enabled:
                    # Filter out duplicate/overlapping hand detections
                    filtered_hands = self.deduplicate_hands(results.multi_hand_landmarks)
                    
                    # Assign persistent IDs to maintain tracking continuity
                    hand_id_mapping = self.assign_persistent_hand_ids(filtered_hands)
                    
                    # Still detect hands for display purposes, but don't process gestures
                    hand_detected = True
                    
                    # Collect all thumb and index positions from all detected hands (for center tracking)
                    all_thumbs = []
                    all_indexes = []
                    
                    for hand_index, hand_landmarks in enumerate(filtered_hands):
                        # Get persistent hand ID
                        persistent_hand_id = hand_id_mapping[hand_index]
                        
                        # Apply visual smoothing to reduce jitter
                        smoothed_landmarks = self.smooth_hand_landmarks(hand_landmarks, persistent_hand_id)
                        
                        # Get thumb tip and index finger tip from smoothed landmarks
                        thumb_tip = smoothed_landmarks.landmark[4]
                        index_tip = smoothed_landmarks.landmark[8]
                        
                        # Store positions for center calculation
                        all_thumbs.append((thumb_tip.x, thumb_tip.y))
                        all_indexes.append((index_tip.x, index_tip.y))
                        
                        # Draw hand landmarks only if interface is enabled (even when functionality is disabled)
                        if self.show_interface:
                            self.mp_draw.draw_landmarks(frame, smoothed_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            
                            # Still draw tracking points but in gray to show it's disabled
                            h, w = frame.shape[:2]
                            
                            # Calculate scaling for disabled hand tracking elements
                            scale = min(w / 1920.0, h / 1080.0)
                            line_thickness = max(1, int(3 * scale))
                            circle_radius = max(2, int(12 * scale))
                            
                            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                            cv2.line(frame, thumb_pos, index_pos, (128, 128, 128), line_thickness)  # Gray line
                            cv2.circle(frame, thumb_pos, circle_radius, (128, 128, 128), -1)  # Gray circles
                            cv2.circle(frame, index_pos, circle_radius, (128, 128, 128), -1)
                    
                    # Note: When functionality is disabled, zoom center should NOT be updated
                    # to prevent camera movement. Only gesture detection for display is enabled.
               
                # Apply zoom
                frame = self.apply_zoom(frame)
               
                # Clean up old hand tracking data
                self.cleanup_old_hand_data(num_hands_detected)
               
                # Calculate current FPS for display
                fps_frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - fps_start_time
                current_fps = fps_frame_count / elapsed_time if elapsed_time > 0 else 0
               
                # Add information overlay with FPS
                frame = self.draw_info(frame, distance, hand_detected, num_hands_detected, current_fps, metric_type)
               
                # Display frame in virtual output window
                cv2.imshow(self.output_window_name, frame)
               
                # Log FPS performance periodically
                if fps_frame_count % 300 == 0:
                    logger.info(f"Processing FPS: {current_fps:.1f} fps (processed {fps_frame_count} frames in {elapsed_time:.1f}s)")
                    
                # Check zoom transition and play sound if applicable (only when functionality is enabled)
                if self.functionality_enabled:
                    self.check_zoom_transition_and_play_sound()
               
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('r'):
                    self.zoom_factor = 1.0
                    self.previous_zoom = 1.0
                    self.has_touched_1x = True
                    self.is_eligible_for_sound = True
                    self.max_zoom_during_transition = 1.0
                    # Reset both moving averages
                    self.distance_history.clear()
                    self.distance_timestamps.clear()
                    self.zoom_center_history_x.clear()
                    self.zoom_center_history_y.clear()
                    self.zoom_center_timestamps.clear()
                    logger.info("Zoom reset to 1.0x - Sound system and moving averages reset")
                elif key == ord('i'):
                    # Toggle interface display
                    self.show_interface = not self.show_interface
                    status = "ON" if self.show_interface else "OFF"
                    logger.info(f"Interface display toggled {status}")
                elif key == ord('x'):
                    # Toggle functionality (gesture detection and zoom control)
                    self.functionality_enabled = not self.functionality_enabled
                    status = "ENABLED" if self.functionality_enabled else "DISABLED"
                    logger.info(f"Gesture functionality toggled {status}")
                    if not self.functionality_enabled:
                        logger.info("Camera continues recording, but gesture control is disabled")
                    else:
                        logger.info("Gesture control resumed")
               
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
 
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.running = False
       
        if self.cap:
            self.cap.release()
       
        # Clean up pygame mixer
        if self.sound_loaded:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
       
        cv2.destroyAllWindows()
        logger.info("Teams Gesture Camera stopped")
 
    def test_camera_resolutions(self, camera_index=0):
        """Test what resolutions the camera actually supports."""
        logger.info("Testing available camera resolutions...")
        
        test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not test_cap.isOpened():
            logger.error("Could not open camera for resolution testing")
            return
        
        test_resolutions = [
            (1920, 1080), (1680, 1050), (1600, 1200), (1440, 900),
            (1366, 768), (1280, 1024), (1280, 720), (1024, 768),
            (800, 600), (640, 480), (320, 240)
        ]
        
        supported_resolutions = []
        
        for width, height in test_resolutions:
            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                supported_resolutions.append((width, height))
                aspect_ratio = width / height
                logger.info(f"[OK] Supports: {width}x{height} (aspect ratio: {aspect_ratio:.2f})")
        
        test_cap.release()
        
        if supported_resolutions:
            logger.info(f"Camera supports {len(supported_resolutions)} resolutions")
            max_res = max(supported_resolutions, key=lambda x: x[0] * x[1])
            logger.info(f"Maximum resolution: {max_res[0]}x{max_res[1]}")
        else:
            logger.warning("No standard resolutions found, camera may have custom resolution")
 
    def smooth_zoom_center(self, center_x, center_y, num_hands=1):
        """Apply time-based smoothing to camera movement (zoom center) - much slower and more stable than zoom speed."""
        current_time = time.time()
        
        # Add current positions and timestamp to history
        self.zoom_center_history_x.append(center_x)
        self.zoom_center_history_y.append(center_y)
        self.zoom_center_timestamps.append(current_time)
        
        # Remove entries older than the time window (much longer than zoom time window)
        cutoff_time = current_time - self.center_history_time_window
        while (self.zoom_center_timestamps and 
               self.zoom_center_timestamps[0] < cutoff_time):
            self.zoom_center_history_x.pop(0)
            self.zoom_center_history_y.pop(0)
            self.zoom_center_timestamps.pop(0)
        
        # Also maintain maximum size as fallback
        while len(self.zoom_center_history_x) > self.center_history_size:
            self.zoom_center_history_x.pop(0)
            self.zoom_center_history_y.pop(0)
            self.zoom_center_timestamps.pop(0)
        
        # Apply more aggressive smoothing when zoomed in AND when more hands are present
        # Camera movement needs much more stability than zoom changes
        zoom_smoothing_factor = min(self.zoom_factor / 2.0, 1.0)  # More smoothing at higher zoom
        hand_smoothing_factor = min(num_hands / 2.0, 1.5)  # Additional smoothing for more hands
        combined_smoothing = zoom_smoothing_factor * hand_smoothing_factor
        
        if len(self.zoom_center_history_x) >= 3:
            # Time-weighted average with exponential decay (slower decay for camera stability)
            # Camera movement should be much more stable than zoom changes
            total_weight = 0
            weighted_sum_x = 0
            weighted_sum_y = 0
            
            for i, timestamp in enumerate(self.zoom_center_timestamps):
                # Age of this sample in seconds
                age = current_time - timestamp
                
                # Base weight decreases exponentially with age (slower decay than zoom)
                base_weight = math.exp(-age * 0.5)  # Much slower decay than zoom (0.5 vs 5.0)
                
                # Apply heavy dampening when zoomed in with multiple hands
                # This creates maximum stability for camera movement
                stability_factor = 1.0 + combined_smoothing * 4.0  # Much higher stability than zoom
                adjusted_weight = base_weight / stability_factor
                
                weighted_sum_x += self.zoom_center_history_x[i] * adjusted_weight
                weighted_sum_y += self.zoom_center_history_y[i] * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                smooth_x = weighted_sum_x / total_weight
                smooth_y = weighted_sum_y / total_weight
                return smooth_x, smooth_y
        
        # Fallback to simple average
        avg_x = sum(self.zoom_center_history_x) / len(self.zoom_center_history_x)
        avg_y = sum(self.zoom_center_history_y) / len(self.zoom_center_history_y)
        return avg_x, avg_y
        return avg_x, avg_y
 
    def set_resolution_relative_gesture_params(self):
        """Set gesture detection parameters relative to camera resolution for consistent behavior."""
        if not self.frame_width or not self.frame_height:
            logger.warning("Cannot set resolution-relative gesture params - frame dimensions not available")
            return
        
        # Calculate a resolution scale factor based on the diagonal of the frame
        # Using diagonal ensures the gesture range scales consistently for both width and height
        diagonal_pixels = math.sqrt(self.frame_width**2 + self.frame_height**2)
        
        # Base reference: Full HD diagonal
        base_diagonal = math.sqrt(1920**2 + 1080**2)
        scale_factor = diagonal_pixels / base_diagonal
        
        # Target gesture range for Full HD: 100-500 pixels
        base_min_distance = 100
        base_max_distance = 500
        
        # Scale the gesture parameters based on resolution
        self.min_distance = int(base_min_distance * scale_factor)
        self.max_distance = int(base_max_distance * scale_factor)
        
        # Ensure reasonable bounds (don't make gestures too small or too large)
        self.min_distance = max(20, min(self.min_distance, 200))  # Keep between 20-200px
        self.max_distance = max(80, min(self.max_distance, 800))  # Keep between 80-800px
        
        logger.info(f"Resolution-relative gesture params set for {self.frame_width}x{self.frame_height}:")
        logger.info(f"  Diagonal: {diagonal_pixels:.0f}px (scale: {scale_factor:.2f}x vs Full HD)")
        logger.info(f"  Gesture range: {self.min_distance}-{self.max_distance}px (Full HD uses 100-500px)")
        logger.info(f"  NATURAL: close hand ({self.min_distance}px) = zoom out, open hand ({self.max_distance}px) = zoom in")
 
    def initialize_mediapipe_for_resolution(self):
        """Initialize MediaPipe with settings optimized for the current camera resolution."""
        if not self.frame_width or not self.frame_height:
            logger.warning("Cannot initialize MediaPipe - frame dimensions not available")
            return
        
        # Calculate total pixels to determine processing complexity needed
        total_pixels = self.frame_width * self.frame_height
        
        # Determine optimal MediaPipe settings based on resolution
        if total_pixels <= 640 * 480:  # VGA and below
            model_complexity = 1
            max_hands = 6  # Reduced for better FPS
            detection_confidence = 0.6  # Higher to reduce false positives
            tracking_confidence = 0.5   # Balanced for stability
            logger.info("MediaPipe config: Low resolution - balanced detection, optimized for FPS")
        elif total_pixels <= 1280 * 720:  # 720p and below
            model_complexity = 0  # Use faster model for better FPS
            max_hands = 6  # Reduced for better FPS
            detection_confidence = 0.65  # Higher to reduce false positives
            tracking_confidence = 0.5    # Balanced for stability
            logger.info("MediaPipe config: Medium resolution - balanced detection, optimized for FPS")
        elif total_pixels <= 1920 * 1080:  # Full HD
            model_complexity = 0  # Use faster model for better FPS
            max_hands = 6  # Reduce for better FPS
            detection_confidence = 0.7  # Balanced
            tracking_confidence = 0.6   # Consistent with other resolutions
            logger.info("MediaPipe config: Full HD - balanced detection, optimized for FPS")
        else:  # 4K and above
            model_complexity = 0  # Use fastest model for better FPS
            max_hands = 4  # Significantly reduced for 4K performance
            detection_confidence = 0.7  # Higher for 4K stability
            tracking_confidence = 0.5    # Balanced for stability
            logger.info("MediaPipe config: Ultra-high resolution - maximum FPS optimization")
        
        # Initialize MediaPipe with optimized settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=model_complexity,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        logger.info(f"MediaPipe initialized for {self.frame_width}x{self.frame_height} ({total_pixels:,} pixels):")
        logger.info(f"  Model complexity: {model_complexity} (0=fastest, 1=more accurate)")
        logger.info(f"  Max hands: {max_hands}")
        logger.info(f"  Detection confidence: {detection_confidence}")
        logger.info(f"  Tracking confidence: {tracking_confidence}")
    
    def process_frame_for_mediapipe(self, frame):
        """Process frame for MediaPipe - may resize for better performance at high resolutions."""
        # For very high resolutions, resize frame for MediaPipe processing only
        # This dramatically improves performance while maintaining display quality
        height, width = frame.shape[:2]
        total_pixels = width * height
        
        # If frame is larger than Full HD, resize for MediaPipe processing
        if total_pixels > 1920 * 1080:
            # Calculate scale to bring down to approximately Full HD
            scale = math.sqrt((1920 * 1080) / total_pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize for MediaPipe processing
            processing_frame = cv2.resize(frame, (new_width, new_height))
            self.mediapipe_scale_factor = 1.0 / scale  # Store scale factor for coordinate conversion
            
            logger.debug(f"Resized frame for MediaPipe: {width}x{height} -> {new_width}x{new_height} (scale: {scale:.3f})")
            return processing_frame
        else:
            # Use original frame
            self.mediapipe_scale_factor = 1.0
            return frame
    
    def convert_mediapipe_coordinates(self, landmark, original_width, original_height):
        """Convert MediaPipe coordinates back to original frame coordinates."""
        # MediaPipe returns normalized coordinates (0-1), scale to original frame size
        x = landmark.x * original_width
        y = landmark.y * original_height
        
        # Apply scale factor if frame was resized for MediaPipe processing
        if hasattr(self, 'mediapipe_scale_factor') and self.mediapipe_scale_factor != 1.0:
            # Coordinates are already in original frame space since MediaPipe uses normalized coords
            pass
        
        return x, y
    
    def deduplicate_hands(self, hand_landmarks_list):
        """Remove overlapping hand detections that might be duplicates of the same hand."""
        if len(hand_landmarks_list) <= 1:
            return hand_landmarks_list
        
        # Calculate center points for each hand
        hand_centers = []
        for hand_landmarks in hand_landmarks_list:
            # Use the center of the hand (landmark 9 - middle finger MCP)
            center_landmark = hand_landmarks.landmark[9]
            hand_centers.append((center_landmark.x, center_landmark.y))
        
        # Filter out hands that are too close to each other
        filtered_hands = []
        filtered_centers = []
        
        min_distance_threshold = 0.15  # Minimum distance between hand centers (normalized coordinates)
        
        for i, (hand_landmarks, center) in enumerate(zip(hand_landmarks_list, hand_centers)):
            is_duplicate = False
            
            # Check if this hand is too close to any already accepted hand
            for existing_center in filtered_centers:
                distance = math.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
                if distance < min_distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_hands.append(hand_landmarks)
                filtered_centers.append(center)
        
        return filtered_hands
    
    def assign_persistent_hand_ids(self, filtered_hands):
        """Assign persistent IDs to hands based on position continuity."""
        current_hand_centers = []
        
        # Calculate center points for current hands
        for hand_landmarks in filtered_hands:
            center_landmark = hand_landmarks.landmark[9]  # Middle finger MCP
            current_hand_centers.append((center_landmark.x, center_landmark.y))
        
        # Find the best matching between current hands and previous hand IDs
        new_id_mapping = {}
        used_ids = set()
        
        for i, current_center in enumerate(current_hand_centers):
            best_distance = float('inf')
            best_id = None
            
            # Find closest previous hand center
            for prev_id, prev_center in self.hand_tracking_ids.items():
                distance = math.sqrt((current_center[0] - prev_center[0])**2 + 
                                   (current_center[1] - prev_center[1])**2)
                
                # Use this ID if it's the closest and within reasonable distance
                if distance < best_distance and distance < 0.3 and prev_id not in used_ids:
                    best_distance = distance
                    best_id = prev_id
            
            if best_id is not None:
                # Existing hand
                new_id_mapping[i] = best_id
                used_ids.add(best_id)
            else:
                # New hand - assign new ID
                new_id_mapping[i] = self.next_hand_id
                self.next_hand_id += 1
        
        # Update hand tracking centers
        self.hand_tracking_ids = {}
        for i, hand_center in enumerate(current_hand_centers):
            persistent_id = new_id_mapping[i]
            self.hand_tracking_ids[persistent_id] = hand_center
        
        return new_id_mapping
    
    def smooth_hand_landmarks(self, hand_landmarks, hand_id):
        """Smooth hand landmark positions to reduce visual jitter."""
        if hand_id not in self.smoothed_landmarks:
            # First time seeing this hand, initialize with current positions
            self.smoothed_landmarks[hand_id] = {}
            for i, landmark in enumerate(hand_landmarks.landmark):
                self.smoothed_landmarks[hand_id][i] = {'x': landmark.x, 'y': landmark.y}
            return hand_landmarks
        
        # Create a copy of the hand landmarks for smoothing
        smoothed_hand = type(hand_landmarks)()
        smoothed_hand.CopyFrom(hand_landmarks)
        
        # Smooth each landmark
        for i, landmark in enumerate(hand_landmarks.landmark):
            if i in self.smoothed_landmarks[hand_id]:
                prev_x = self.smoothed_landmarks[hand_id][i]['x']
                prev_y = self.smoothed_landmarks[hand_id][i]['y']
                
                # Calculate movement distance
                movement = math.sqrt((landmark.x - prev_x)**2 + (landmark.y - prev_y)**2)
                
                # Only update if movement is significant enough
                if movement > self.landmark_movement_threshold:
                    # Smooth the transition
                    new_x = prev_x + (landmark.x - prev_x) * self.landmark_smoothing_factor
                    new_y = prev_y + (landmark.y - prev_y) * self.landmark_smoothing_factor
                    
                    # Update stored position
                    self.smoothed_landmarks[hand_id][i]['x'] = new_x
                    self.smoothed_landmarks[hand_id][i]['y'] = new_y
                    
                    # Update the landmark
                    smoothed_hand.landmark[i].x = new_x
                    smoothed_hand.landmark[i].y = new_y
                else:
                    # Use previous smooth position (no jitter)
                    smoothed_hand.landmark[i].x = prev_x
                    smoothed_hand.landmark[i].y = prev_y
        
        return smoothed_hand
    
    def cleanup_old_hand_data(self, current_hand_count):
        """Clean up smoothed landmark data for hands that are no longer detected."""
        # Remove smoothing data for hand IDs that are no longer being tracked
        current_hand_ids = set(self.hand_tracking_ids.keys())
        smoothed_hand_ids = set(self.smoothed_landmarks.keys())
        
        # Remove old smoothing data
        hands_to_remove = smoothed_hand_ids - current_hand_ids
        for hand_id in hands_to_remove:
            del self.smoothed_landmarks[hand_id]
 
def main():
    """Main function."""
    try:
        camera = TeamsGestureCamera()
        camera.run()
    except Exception as e:
        logger.error(f"Failed to start Teams Gesture Camera: {e}")
        print(f"Error: {e}")
        input("Press Enter to continue...")
 
if __name__ == "__main__":
    main()

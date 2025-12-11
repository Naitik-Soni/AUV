"""
AUV Computer Vision System for RoboFest Gujarat 5.0
Complete modular implementation for underwater navigation and target detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for all CV parameters"""
    
    # Green Mat Detection (HSV)
    GREEN_LOWER = np.array([35, 40, 40])
    GREEN_UPPER = np.array([85, 255, 255])
    MIN_MAT_AREA = 5000  # minimum pixels for valid mat
    
    # Drum Color Detection (HSV)
    BLUE_LOWER = np.array([100, 50, 50])
    BLUE_UPPER = np.array([130, 255, 255])
    
    RED_LOWER_1 = np.array([0, 50, 50])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([170, 50, 50])
    RED_UPPER_2 = np.array([180, 255, 255])
    
    # Circle Detection
    MIN_DRUM_RADIUS = 20  # pixels
    MAX_DRUM_RADIUS = 150  # pixels
    CIRCLE_PARAM1 = 50
    CIRCLE_PARAM2 = 30
    
    # Tracking
    TARGET_TOLERANCE = 30  # pixels - "centered" threshold
    
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480


class DrumColor(Enum):
    BLUE = "blue"
    RED = "red"
    UNKNOWN = "unknown"


@dataclass
class Position:
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class Drum:
    position: Position
    color: DrumColor
    radius: float


@dataclass
class DetectionResult:
    success: bool
    data: any
    message: str
    timestamp: float


# ============================================================================
# MODULE 1: GREEN MAT DETECTION
# ============================================================================

class GreenMatDetector:
    """Detects the green mat zone on pool floor"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect green mat and return its center position
        
        Args:
            frame: BGR image from camera
            
        Returns:
            DetectionResult with Position data or None
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for green color
            mask = cv2.inRange(hsv, self.config.GREEN_LOWER, self.config.GREEN_UPPER)
            
            # Morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return DetectionResult(False, None, "No green mat detected", time.time())
            
            # Find largest contour (should be the mat)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < self.config.MIN_MAT_AREA:
                return DetectionResult(False, None, f"Mat area too small: {area}", time.time())
            
            # Calculate center
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return DetectionResult(False, None, "Invalid moment calculation", time.time())
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            confidence = min(area / 50000, 1.0)  # normalize by expected area
            
            position = Position(cx, cy, confidence)
            
            return DetectionResult(
                True, 
                {"position": position, "area": area, "contour": largest_contour},
                "Green mat detected successfully",
                time.time()
            )
            
        except Exception as e:
            return DetectionResult(False, None, f"Error: {str(e)}", time.time())
    
    def draw_debug(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detection visualization on frame"""
        debug_frame = frame.copy()
        
        if result.success and result.data:
            pos = result.data["position"]
            contour = result.data["contour"]
            
            # Draw contour
            cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(debug_frame, (int(pos.x), int(pos.y)), 10, (0, 255, 255), -1)
            cv2.putText(debug_frame, "MAT CENTER", (int(pos.x) + 15, int(pos.y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Info text
            cv2.putText(debug_frame, f"Confidence: {pos.confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(debug_frame, result.message, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug_frame


# ============================================================================
# MODULE 2: DRUM DETECTION & COLOR CLASSIFICATION
# ============================================================================

class DrumDetector:
    """Detects and classifies drums by color"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
    
    def detect_drums(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect all drums and classify their colors
        
        Args:
            frame: BGR image from camera
            
        Returns:
            DetectionResult with list of Drum objects
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circles using Hough Transform
            circles = cv2.HoughCircles(
                gray_blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=self.config.CIRCLE_PARAM1,
                param2=self.config.CIRCLE_PARAM2,
                minRadius=self.config.MIN_DRUM_RADIUS,
                maxRadius=self.config.MAX_DRUM_RADIUS
            )
            
            if circles is None:
                return DetectionResult(False, [], "No drums detected", time.time())
            
            circles = np.uint16(np.around(circles))
            drums = []
            
            # Classify each circle
            for circle in circles[0, :]:
                x, y, r = circle
                color = self._classify_drum_color(frame, x, y, r)
                
                drum = Drum(
                    position=Position(float(x), float(y)),
                    color=color,
                    radius=float(r)
                )
                drums.append(drum)
            
            return DetectionResult(
                True,
                drums,
                f"Detected {len(drums)} drums",
                time.time()
            )
            
        except Exception as e:
            return DetectionResult(False, [], f"Error: {str(e)}", time.time())
    
    def _classify_drum_color(self, frame: np.ndarray, x: int, y: int, r: int) -> DrumColor:
        """Classify drum color from its region"""
        # Extract ROI
        y1, y2 = max(0, y - r), min(frame.shape[0], y + r)
        x1, x2 = max(0, x - r), min(frame.shape[1], x + r)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return DrumColor.UNKNOWN
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check blue
        blue_mask = cv2.inRange(hsv_roi, self.config.BLUE_LOWER, self.config.BLUE_UPPER)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        # Check red (two ranges)
        red_mask1 = cv2.inRange(hsv_roi, self.config.RED_LOWER_1, self.config.RED_UPPER_1)
        red_mask2 = cv2.inRange(hsv_roi, self.config.RED_LOWER_2, self.config.RED_UPPER_2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)
        
        # Classify based on which color has more pixels
        if blue_pixels > red_pixels and blue_pixels > 50:
            return DrumColor.BLUE
        elif red_pixels > blue_pixels and red_pixels > 50:
            return DrumColor.RED
        else:
            return DrumColor.UNKNOWN
    
    def draw_debug(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw drum detections on frame"""
        debug_frame = frame.copy()
        
        if result.success and result.data:
            for drum in result.data:
                x, y = int(drum.position.x), int(drum.position.y)
                r = int(drum.radius)
                
                # Color based on classification
                if drum.color == DrumColor.BLUE:
                    color = (255, 0, 0)
                    label = "BLUE DRUM"
                elif drum.color == DrumColor.RED:
                    color = (0, 0, 255)
                    label = "RED DRUM"
                else:
                    color = (128, 128, 128)
                    label = "UNKNOWN"
                
                # Draw circle
                cv2.circle(debug_frame, (x, y), r, color, 3)
                cv2.circle(debug_frame, (x, y), 5, color, -1)
                
                # Draw label
                cv2.putText(debug_frame, label, (x - r, y - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(debug_frame, result.message, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_frame


# ============================================================================
# MODULE 3: TARGET TRACKING
# ============================================================================

class TargetTracker:
    """Tracks specific target drum and provides positioning commands"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.target_drum = None
        
    def track_target(self, frame: np.ndarray, target_color: DrumColor, 
                     drums: List[Drum]) -> DetectionResult:
        """
        Track target drum and calculate positioning offset
        
        Args:
            frame: Current camera frame
            target_color: Color of target drum to track
            drums: List of detected drums
            
        Returns:
            DetectionResult with tracking data
        """
        # Filter drums by target color
        targets = [d for d in drums if d.color == target_color]
        
        if not targets:
            return DetectionResult(False, None, f"No {target_color.value} drum found", time.time())
        
        # If multiple, choose closest to center
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        target = min(targets, key=lambda d: np.hypot(
            d.position.x - frame_center[0],
            d.position.y - frame_center[1]
        ))
        
        # Calculate offset from center
        offset_x = target.position.x - frame_center[0]
        offset_y = target.position.y - frame_center[1]
        distance = np.hypot(offset_x, offset_y)
        
        # Determine if centered
        is_centered = distance < self.config.TARGET_TOLERANCE
        
        # Generate movement command
        if is_centered:
            command = "CENTERED - READY TO DROP"
        else:
            h_cmd = "RIGHT" if offset_x > 0 else "LEFT"
            v_cmd = "FORWARD" if offset_y > 0 else "BACK"
            command = f"MOVE {h_cmd} {abs(int(offset_x))}px, {v_cmd} {abs(int(offset_y))}px"
        
        tracking_data = {
            "target": target,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "distance": distance,
            "is_centered": is_centered,
            "command": command
        }
        
        return DetectionResult(True, tracking_data, command, time.time())
    
    def draw_debug(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw tracking visualization"""
        debug_frame = frame.copy()
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Draw crosshair at center
        cv2.line(debug_frame, (center[0] - 30, center[1]), (center[0] + 30, center[1]), (0, 255, 0), 2)
        cv2.line(debug_frame, (center[0], center[1] - 30), (center[0], center[1] + 30), (0, 255, 0), 2)
        cv2.circle(debug_frame, center, self.config.TARGET_TOLERANCE, (0, 255, 0), 2)
        
        if result.success and result.data:
            target = result.data["target"]
            tx, ty = int(target.position.x), int(target.position.y)
            
            # Draw target
            cv2.circle(debug_frame, (tx, ty), int(target.radius), (0, 255, 255), 3)
            cv2.circle(debug_frame, (tx, ty), 5, (0, 255, 255), -1)
            
            # Draw line from center to target
            cv2.line(debug_frame, center, (tx, ty), (255, 0, 255), 2)
            
            # Status text
            if result.data["is_centered"]:
                color = (0, 255, 0)
                text = "CENTERED!"
            else:
                color = (0, 165, 255)
                text = result.data["command"]
            
            cv2.putText(debug_frame, text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(debug_frame, result.message, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug_frame


# ============================================================================
# MODULE 4: 2D MAPPING
# ============================================================================

class MapGenerator:
    """Generates 2D map of mat and drum positions"""
    
    def __init__(self, map_size: Tuple[int, int] = (800, 600)):
        self.map_size = map_size
        self.detections = []
        
    def add_detection(self, mat_pos: Optional[Position], drums: List[Drum]):
        """Record a detection for mapping"""
        self.detections.append({
            "timestamp": time.time(),
            "mat": mat_pos,
            "drums": drums.copy()
        })
    
    def generate_map(self) -> np.ndarray:
        """Generate final 2D map image"""
        map_img = np.ones((self.map_size[1], self.map_size[0], 3), dtype=np.uint8) * 255
        
        if not self.detections:
            cv2.putText(map_img, "No data collected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return map_img
        
        # Use most recent detection
        latest = self.detections[-1]
        
        # Draw title
        cv2.putText(map_img, "AUV Mission Map", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Draw mat (as rectangle in center)
        if latest["mat"]:
            mat_rect = (150, 150, 500, 300)
            cv2.rectangle(map_img, mat_rect[:2], 
                         (mat_rect[0] + mat_rect[2], mat_rect[1] + mat_rect[3]),
                         (0, 200, 0), -1)
            cv2.putText(map_img, "GREEN MAT", (mat_rect[0] + 180, mat_rect[1] + 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw drums
        for i, drum in enumerate(latest["drums"]):
            # Map drum positions onto the mat area
            drum_x = 200 + (i * 120)
            drum_y = 250
            
            if drum.color == DrumColor.BLUE:
                color = (255, 0, 0)
                label = "BLUE"
            elif drum.color == DrumColor.RED:
                color = (0, 0, 255)
                label = f"RED {i+1}"
            else:
                color = (128, 128, 128)
                label = "UNK"
            
            cv2.circle(map_img, (drum_x, drum_y), 30, color, -1)
            cv2.circle(map_img, (drum_x, drum_y), 30, (0, 0, 0), 2)
            cv2.putText(map_img, label, (drum_x - 25, drum_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add timestamp
        cv2.putText(map_img, f"Generated: {time.strftime('%H:%M:%S')}", 
                   (20, self.map_size[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        return map_img
    
    def save_map(self, filename: str = "auv_mission_map.png"):
        """Save map to file"""
        map_img = self.generate_map()
        cv2.imwrite(filename, map_img)
        return filename


# ============================================================================
# MODULE 5: MAIN VISION SYSTEM
# ============================================================================

class AUVVisionSystem:
    """Main integrated vision system"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.mat_detector = GreenMatDetector(config)
        self.drum_detector = DrumDetector(config)
        self.tracker = TargetTracker(config)
        self.mapper = MapGenerator()
        
    def process_frame(self, frame: np.ndarray, 
                      track_target: Optional[DrumColor] = None) -> Dict:
        """
        Process a single frame through all detection modules
        
        Args:
            frame: BGR camera frame
            track_target: Optional drum color to track
            
        Returns:
            Dictionary with all detection results
        """
        results = {}
        
        # Step 1: Detect green mat
        mat_result = self.mat_detector.detect(frame)
        results["mat"] = mat_result
        
        # Step 2: Detect drums
        drum_result = self.drum_detector.detect_drums(frame)
        results["drums"] = drum_result
        
        # Step 3: Track target if specified
        if track_target and drum_result.success:
            track_result = self.tracker.track_target(frame, track_target, drum_result.data)
            results["tracking"] = track_result
        else:
            results["tracking"] = None
        
        # Step 4: Add to map
        mat_pos = mat_result.data["position"] if mat_result.success else None
        drums = drum_result.data if drum_result.success else []
        self.mapper.add_detection(mat_pos, drums)
        
        return results
    
    def create_debug_display(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Create comprehensive debug visualization"""
        debug_frame = frame.copy()
        
        # Draw mat detection
        if results["mat"].success:
            debug_frame = self.mat_detector.draw_debug(debug_frame, results["mat"])
        
        # Draw drum detection
        if results["drums"].success:
            debug_frame = self.drum_detector.draw_debug(debug_frame, results["drums"])
        
        # Draw tracking
        if results["tracking"]:
            debug_frame = self.tracker.draw_debug(debug_frame, results["tracking"])
        
        return debug_frame


# ============================================================================
# MODULE 6: TESTING & DEMO
# ============================================================================

def demo_live_camera(video_path = None):
    """Demo using live camera feed"""
    print("Starting AUV Vision System Demo...")
    print("Controls:")
    print("  'b' - Track BLUE drum")
    print("  'r' - Track RED drum")
    print("  'm' - Save map")
    print("  'q' - Quit")
    
    # Initialize system
    vision = AUVVisionSystem()
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(Config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    track_target = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        results = vision.process_frame(frame, track_target)
        
        # Create visualization
        display = vision.create_debug_display(frame, results)
        
        # Show results
        cv2.imshow("AUV Vision System", display)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            track_target = DrumColor.BLUE
            print("Tracking BLUE drum")
        elif key == ord('r'):
            track_target = DrumColor.RED
            print("Tracking RED drum")
        elif key == ord('m'):
            filename = vision.mapper.save_map()
            print(f"Map saved to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()


def demo_test_image(image_path: str):
    """Demo using a single test image"""
    vision = AUVVisionSystem()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Process
    results = vision.process_frame(frame, track_target=DrumColor.BLUE)
    
    # Display
    display = vision.create_debug_display(frame, results)
    
    cv2.imshow("Test Image Results", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print results
    print("\n=== DETECTION RESULTS ===")
    print(f"Mat: {results['mat'].message}")
    print(f"Drums: {results['drums'].message}")
    if results['tracking']:
        print(f"Tracking: {results['tracking'].message}")


if __name__ == "__main__":
    # Run live camera demo
    vid_path = r"P:\AUV\AUV\Input\vid_1.mp4"
    demo_live_camera(vid_path)
    
    # Or test with an image:
    # demo_test_image("test_image.jpg")
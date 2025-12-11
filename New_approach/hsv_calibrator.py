"""
HSV Color Calibration Tool
Use this to find the perfect HSV ranges for your specific environment
"""

import cv2
import numpy as np

class HSVCalibrator:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        # Default values
        self.h_min = 0
        self.h_max = 180
        self.s_min = 0
        self.s_max = 255
        self.v_min = 0
        self.v_max = 255
        
    def nothing(self, x):
        """Callback for trackbars"""
        pass
    
    def calibrate(self, color_name="Color"):
        """
        Launch calibration window
        
        Args:
            color_name: Name of color being calibrated (e.g., "Green", "Blue", "Red")
        """
        cap = cv2.VideoCapture(self.camera_id)
        
        # Create window and trackbars
        window_name = f"{color_name} HSV Calibration"
        cv2.namedWindow(window_name)
        
        cv2.createTrackbar("H Min", window_name, 0, 180, self.nothing)
        cv2.createTrackbar("H Max", window_name, 180, 180, self.nothing)
        cv2.createTrackbar("S Min", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("S Max", window_name, 255, 255, self.nothing)
        cv2.createTrackbar("V Min", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("V Max", window_name, 255, 255, self.nothing)
        
        print(f"\n=== {color_name} HSV Calibration ===")
        print("Instructions:")
        print("1. Adjust the sliders to isolate your target color")
        print("2. The mask should show white for target, black for everything else")
        print("3. Press 's' to save values")
        print("4. Press 'q' to quit")
        print("\nAdjusting tips:")
        print("- Start with H (Hue) to get the right color range")
        print("- Adjust S (Saturation) to filter out pale/washed colors")
        print("- Adjust V (Value) to handle lighting variations\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get trackbar positions
            h_min = cv2.getTrackbarPos("H Min", window_name)
            h_max = cv2.getTrackbarPos("H Max", window_name)
            s_min = cv2.getTrackbarPos("S Min", window_name)
            s_max = cv2.getTrackbarPos("S Max", window_name)
            v_min = cv2.getTrackbarPos("V Min", window_name)
            v_max = cv2.getTrackbarPos("V Max", window_name)
            
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply mask to original
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Add text overlay
            cv2.putText(frame, f"{color_name} Calibration", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mask, "Mask (White = Detected)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            
            # Stack images for display
            top_row = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            bottom_row = np.hstack([result, result])
            display = np.vstack([top_row, bottom_row])
            
            # Resize for better viewing
            display = cv2.resize(display, (1280, 720))
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.h_min, self.h_max = h_min, h_max
                self.s_min, self.s_max = s_min, s_max
                self.v_min, self.v_max = v_min, v_max
                
                print(f"\n✓ Saved {color_name} HSV Values:")
                print(f"  LOWER = np.array([{h_min}, {s_min}, {v_min}])")
                print(f"  UPPER = np.array([{h_max}, {s_max}, {v_max}])")
                print("\nCopy these values to Config class in main code!\n")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return (self.h_min, self.h_max, self.s_min, self.s_max, self.v_min, self.v_max)


def calibrate_all_colors():
    """Calibrate all colors needed for AUV challenge"""
    calibrator = HSVCalibrator()
    
    print("\n" + "="*60)
    print("AUV COLOR CALIBRATION WIZARD")
    print("="*60)
    print("\nYou'll calibrate 3 colors: Green (mat), Blue (drum), Red (drum)")
    print("Press ENTER to start each calibration...")
    
    # Green mat
    input("\nPress ENTER to calibrate GREEN MAT...")
    green_values = calibrator.calibrate("Green Mat")
    
    # Blue drum
    input("\nPress ENTER to calibrate BLUE DRUM...")
    blue_values = calibrator.calibrate("Blue Drum")
    
    # Red drum
    input("\nPress ENTER to calibrate RED DRUM...")
    red_values = calibrator.calibrate("Red Drum")
    
    # Print final configuration
    print("\n" + "="*60)
    print("FINAL CONFIGURATION - Copy to Config class:")
    print("="*60)
    print(f"""
class Config:
    # Green Mat Detection
    GREEN_LOWER = np.array([{green_values[0]}, {green_values[2]}, {green_values[4]}])
    GREEN_UPPER = np.array([{green_values[1]}, {green_values[3]}, {green_values[5]}])
    
    # Blue Drum Detection
    BLUE_LOWER = np.array([{blue_values[0]}, {blue_values[2]}, {blue_values[4]}])
    BLUE_UPPER = np.array([{blue_values[1]}, {blue_values[3]}, {blue_values[5]}])
    
    # Red Drum Detection (Note: Red wraps around, may need two ranges)
    RED_LOWER = np.array([{red_values[0]}, {red_values[2]}, {red_values[4]}])
    RED_UPPER = np.array([{red_values[1]}, {red_values[3]}, {red_values[5]}])
    """)


if __name__ == "__main__":
    # Calibrate all colors at once
    calibrate_all_colors()
    
    # Or calibrate one at a time:
    # calibrator = HSVCalibrator()
    # calibrator.calibrate("Green")
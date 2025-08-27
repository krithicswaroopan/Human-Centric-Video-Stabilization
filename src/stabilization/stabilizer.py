"""
Video stabilization module using hip-center based stabilization.
Based on master planning document specifications.
"""
import cv2
import numpy as np
from typing import List, Tuple
from scipy import ndimage


class StabilizationConfig:
    """Configuration for stabilization module."""
    def __init__(self, smoothing_method="moving", smoothing_window=5,
                 border_mode="reflect", border_value=(0, 0, 0), use_cuda=False,
                 target_position=None):
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
        self.border_mode = border_mode
        self.border_value = border_value
        self.use_cuda = use_cuda
        self.target_position = target_position


class Stabilizer:
    """
    Hip-center based video stabilization.
    - Moving average smoothing (5 frame window default)
    - Affine transformation with reflect border mode
    - Target position at frame center
    """
    
    def __init__(self, config=None):
        """
        Initialize stabilizer.
        
        Args:
            config (StabilizationConfig): Configuration object
        """
        self.config = config or StabilizationConfig()
        self.last_valid_offset = (0, 0)
        self.frame_dimensions = None
        
    def compute_offsets(self, hip_centers, target_position=None):
        """
        Compute translation offsets from hip centers to target position.
        Master doc: Δx = x_target - x_hip, Δy = y_target - y_hip
        
        Args:
            hip_centers (list): List of hip center coordinates from pose detection
            target_position (tuple): Target (x, y) position (frame center by default)
            
        Returns:
            List[Tuple]: List of (dx, dy) offsets
        """
        # Use provided target or config default or frame center
        if target_position is None:
            target_position = self.config.target_position
        
        # If still no target, will be set when we know frame dimensions
        target_x, target_y = target_position if target_position else (0.5, 0.5)
        
        offsets = []
        
        for hip_data in hip_centers:
            hip_x = hip_data['x']
            hip_y = hip_data['y']
            confidence = hip_data['confidence']
            
            if confidence > 0.3:  # Valid hip center data
                # Convert normalized coordinates to pixel coordinates if needed
                if hip_x <= 1.0 and hip_y <= 1.0:  # Normalized coordinates
                    if self.frame_dimensions:
                        frame_width, frame_height = self.frame_dimensions
                        hip_x *= frame_width
                        hip_y *= frame_height
                        target_x_px = target_x * frame_width if target_x <= 1.0 else target_x
                        target_y_px = target_y * frame_height if target_y <= 1.0 else target_y
                    else:
                        # Assume target is also normalized
                        target_x_px, target_y_px = target_x, target_y
                else:
                    # Already in pixel coordinates
                    target_x_px, target_y_px = target_x, target_y
                
                # Compute offset: target - hip (as per master doc)
                dx = target_x_px - hip_x
                dy = target_y_px - hip_y
                
                self.last_valid_offset = (dx, dy)
                offsets.append((dx, dy))
            else:
                # Low confidence - carry forward last valid offset (master doc requirement)
                offsets.append(self.last_valid_offset)
        
        return offsets
        
    def smooth_offsets(self, offsets):
        """
        Apply smoothing to offset values.
        Master doc: Moving average (5 frame window default) or Gaussian smoothing
        
        Args:
            offsets (List[Tuple]): Raw offset values
            
        Returns:
            List[Tuple]: Smoothed offset values
        """
        if not offsets:
            return []
        
        # Separate x and y components
        dx_values = np.array([offset[0] for offset in offsets])
        dy_values = np.array([offset[1] for offset in offsets])
        
        if self.config.smoothing_method == "moving":
            # Moving average smoothing (master doc default)
            dx_smooth = self._moving_average(dx_values, self.config.smoothing_window)
            dy_smooth = self._moving_average(dy_values, self.config.smoothing_window)
            
        elif self.config.smoothing_method == "gaussian":
            # Gaussian smoothing (master doc alternative)
            sigma = self.config.smoothing_window / 3.0  # Convert window to sigma
            dx_smooth = ndimage.gaussian_filter1d(dx_values, sigma)
            dy_smooth = ndimage.gaussian_filter1d(dy_values, sigma)
        
        # Recombine into tuples
        smoothed_offsets = [(dx_smooth[i], dy_smooth[i]) for i in range(len(offsets))]
        
        return smoothed_offsets
    
    def _moving_average(self, values, window_size):
        """Apply moving average smoothing."""
        if len(values) < window_size:
            # Use available frames if window is larger than data
            window_size = len(values)
        
        smoothed = np.zeros_like(values)
        
        for i in range(len(values)):
            # Handle edges by using available frames
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            
            smoothed[i] = np.mean(values[start_idx:end_idx])
        
        return smoothed
        
    def apply_transforms(self, frames, smooth_offsets):
        """
        Apply stabilization transforms to frames.
        Master doc: Affine transformation matrix with cv2.warpAffine
        
        Args:
            frames (list): Input frames  
            smooth_offsets (List[Tuple]): Smoothed offsets
            
        Returns:
            list: Stabilized frames
        """
        if not frames or not smooth_offsets:
            return frames
        
        # Store frame dimensions for offset calculation
        if self.frame_dimensions is None:
            self.frame_dimensions = (frames[0].shape[1], frames[0].shape[0])  # (width, height)
        
        stabilized_frames = []
        
        # Set border mode as per master doc
        if self.config.border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT_101
            border_value = None
        elif self.config.border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
            border_value = self.config.border_value
        else:
            border_mode = cv2.BORDER_REFLECT_101
            border_value = None
        
        for i, (frame, offset) in enumerate(zip(frames, smooth_offsets)):
            dx, dy = offset
            
            # Create affine transformation matrix as per master doc
            # M = [[1, 0, dx], [0, 1, dy]]
            transformation_matrix = np.float32([
                [1, 0, dx],
                [0, 1, dy]
            ])
            
            # Apply transformation using cv2.warpAffine
            if border_value is not None:
                stabilized_frame = cv2.warpAffine(
                    frame, 
                    transformation_matrix, 
                    (frame.shape[1], frame.shape[0]),
                    borderMode=border_mode,
                    borderValue=border_value
                )
            else:
                stabilized_frame = cv2.warpAffine(
                    frame, 
                    transformation_matrix, 
                    (frame.shape[1], frame.shape[0]),
                    borderMode=border_mode
                )
            
            stabilized_frames.append(stabilized_frame)
        
        return stabilized_frames
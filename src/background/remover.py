"""
Background removal module using MediaPipe Selfie Segmentation.
Based on master planning document specifications.
"""
import cv2
import numpy as np
import mediapipe as mp


class BackgroundRemovalConfig:
    """Configuration for background removal module."""
    def __init__(self, model_selection=0, confidence_threshold=0.1, 
                 enable_bilateral_filter=True, enable_temporal_smoothing=True,
                 fallback_enabled=True, background_mode="transparent",
                 background_color=(0, 255, 0), blur_strength=15, 
                 enable_edge_smoothing=True):
        # Original working configuration
        self.model_selection = model_selection
        self.confidence_threshold = confidence_threshold
        self.enable_bilateral_filter = enable_bilateral_filter
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.fallback_enabled = fallback_enabled
        
        # Background replacement options
        self.background_mode = background_mode  # "transparent", "color", "blur"
        self.background_color = background_color  # BGR tuple for solid color
        self.blur_strength = blur_strength  # Blur kernel size (must be odd)
        
        # Border smoothing option
        self.enable_edge_smoothing = enable_edge_smoothing


class BackgroundRemover:
    """
    MediaPipe Selfie Segmentation for background removal.
    - Uses General model (0) for higher accuracy
    - Temporal smoothing enabled
    - Returns both person frames and segmentation masks
    """
    
    def __init__(self, config=None):
        """
        Initialize MediaPipe Selfie Segmentation.
        
        Args:
            config (BackgroundRemovalConfig): Configuration object
        """
        self.config = config or BackgroundRemovalConfig()
        self.previous_mask = None
        self.frame_count = 0
        self.quality_scores = []
        self.fallback_triggered = False
        
        # Initialize MediaPipe
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=self.config.model_selection
        )
        
    def process_frame(self, frame):
        """
        Process single frame for background removal.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (person_frame, segmentation_mask)
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.selfie_segmentation.process(rgb_frame)
            
            # Get segmentation mask
            mask = results.segmentation_mask
            
            if mask is not None:
                # Apply bilateral filtering if enabled
                if self.config.enable_bilateral_filter:
                    mask = self._apply_bilateral_filter(mask)
                
                # Apply temporal smoothing if enabled
                if self.config.enable_temporal_smoothing and self.previous_mask is not None:
                    # Enhanced temporal smoothing for walking motion
                    mask = self._apply_temporal_smoothing(mask)
                
                # Store current mask for next frame
                self.previous_mask = mask.copy()
                
                # Calculate quality score for fallback mechanism
                quality_score = np.mean(mask)
                self.quality_scores.append(quality_score)
                self._check_fallback()
                
                # Create binary mask
                binary_mask = (mask > self.config.confidence_threshold).astype(np.uint8)
                
                # Apply mask to create person-only frame
                person_frame = frame.copy()
                person_frame[binary_mask == 0] = 0
                
                # Apply minimal edge smoothing if enabled (conservative approach)
                if self.config.enable_edge_smoothing:
                    person_frame = self._smooth_borders(person_frame, binary_mask)
                
                # Apply background replacement based on selected mode (NEW FEATURE ONLY)
                if hasattr(self.config, 'background_mode') and self.config.background_mode != "transparent":
                    # Convert binary mask back to float for proper blending
                    float_mask = binary_mask.astype(np.float32)
                    final_frame = self._apply_background_replacement(frame, person_frame, float_mask)
                    return final_frame, binary_mask
                else:
                    return person_frame, binary_mask
            else:
                # Person not detected - return original frame
                return frame.copy(), np.ones_like(frame[:,:,0], dtype=np.uint8)
                
        except Exception:
            # Frame processing failure - use previous mask if available
            if self.previous_mask is not None:
                binary_mask = (self.previous_mask > self.config.confidence_threshold).astype(np.uint8)
                person_frame = frame.copy()
                person_frame[binary_mask == 0] = 0
                return person_frame, binary_mask
            else:
                # No previous mask available - return original frame
                return frame.copy(), np.ones_like(frame[:,:,0], dtype=np.uint8)
        
    def process_video(self, frames):
        """
        Process entire video with background removal.
        
        Args:
            frames (list): List of video frames
            
        Yields:
            tuple: (person_frame, segmentation_mask) for each frame
        """
        for frame in frames:
            self.frame_count += 1
            person_frame, mask = self.process_frame(frame)
            yield person_frame, mask
    
    def _apply_bilateral_filter(self, mask):
        """Apply bilateral filtering for smoother mask edges."""
        # Convert to proper data type for bilateral filter
        mask_filtered = cv2.bilateralFilter(
            (mask * 255).astype(np.uint8), 5, 80, 80
        ).astype(np.float32) / 255.0
        return mask_filtered
    
    def _apply_temporal_smoothing(self, mask):
        """Enhanced temporal smoothing for walking motion."""
        # Motion-aware smoothing: stronger smoothing for stable regions
        motion_intensity = np.abs(mask - self.previous_mask).mean()
        
        if motion_intensity > 0.1:  # High motion (walking)
            alpha = 0.6  # Less smoothing to preserve motion
        else:  # Low motion (stable pose)
            alpha = 0.8  # More smoothing for stability
            
        return alpha * mask + (1 - alpha) * self.previous_mask
    
    def _check_fallback(self):
        """Check if model fallback should be triggered."""
        if (not self.config.fallback_enabled or 
            self.fallback_triggered or 
            len(self.quality_scores) < 30):  # Check after 30 frames
            return
            
        # Calculate recent average quality
        recent_quality = np.mean(self.quality_scores[-30:])
        
        if recent_quality < 0.3:  # Low quality threshold
            self._trigger_fallback()
    
    def _trigger_fallback(self):
        """Switch to Landscape model for better performance."""
        self.fallback_triggered = True
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # Landscape model
        )
    
    def _smooth_borders(self, person_frame, binary_mask):
        """
        Smooth pixelated borders by normalizing edge pixels based on closest body pixels.
        This is a targeted approach that only affects border pixels without changing 
        the overall segmentation or stability.
        
        Args:
            person_frame (np.ndarray): Person-only frame with hard edges
            binary_mask (np.ndarray): Binary mask (0 or 1)
            
        Returns:
            np.ndarray: Person frame with smoothed borders
        """
        # Find border pixels (edge detection on mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        border_pixels = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
        
        # Create a slightly dilated version of the person frame for sampling
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        
        # Apply light Gaussian blur only to the person region
        blurred_person = cv2.GaussianBlur(person_frame, (5, 5), 1.0)
        
        # Create result frame starting with original
        result_frame = person_frame.copy()
        
        # For each border pixel, blend with the blurred version
        border_coords = np.where(border_pixels > 0)
        for y, x in zip(border_coords[0], border_coords[1]):
            if dilated_mask[y, x] > 0:  # Only if within dilated body region
                # Blend original and blurred pixel (70% original, 30% blurred)
                result_frame[y, x] = (0.7 * person_frame[y, x] + 0.3 * blurred_person[y, x]).astype(np.uint8)
        
        return result_frame
    
    def _apply_background_replacement(self, original_frame, person_frame, mask):
        """
        Apply different background replacement modes.
        
        Args:
            original_frame (np.ndarray): Original input frame
            person_frame (np.ndarray): Person-only frame (transparent background)
            mask (np.ndarray): Final processed mask (0-1 float)
            
        Returns:
            np.ndarray: Frame with background replacement applied
        """
        if self.config.background_mode == "transparent":
            # Return person-only frame (current default behavior)
            return person_frame
        
        elif self.config.background_mode == "color":
            # Create solid color background
            background = np.full_like(original_frame, self.config.background_color, dtype=np.uint8)
            
            # Blend person with colored background using mask
            mask_3channel = np.stack([mask] * 3, axis=-1)
            result = (person_frame * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)
            
            return result
        
        elif self.config.background_mode == "blur":
            # Create blurred background
            blur_kernel_size = self.config.blur_strength
            if blur_kernel_size % 2 == 0:  # Ensure odd kernel size
                blur_kernel_size += 1
            
            blurred_background = cv2.GaussianBlur(original_frame, (blur_kernel_size, blur_kernel_size), 0)
            
            # Blend person with blurred background using mask
            mask_3channel = np.stack([mask] * 3, axis=-1)
            result = (person_frame * mask_3channel + blurred_background * (1 - mask_3channel)).astype(np.uint8)
            
            return result
        
        else:
            # Default to transparent mode
            return person_frame
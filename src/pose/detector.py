"""
Pose detection module using MediaPipe Pose.
Based on master planning document specifications.
"""
import cv2
import numpy as np
import mediapipe as mp
import json


class PoseDetectionConfig:
    """Configuration for pose detection module."""
    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, hip_calc_method="simple", 
                 interpolate_missing=True):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hip_calc_method = hip_calc_method
        self.interpolate_missing = interpolate_missing


class PoseDetector:
    """
    MediaPipe Pose detection with hip-center calculation.
    - Uses Full model (complexity=1)
    - Single person tracking, 2D keypoints
    - Timeline JSON export format
    """
    
    def __init__(self, config=None):
        """
        Initialize MediaPipe Pose detection.
        
        Args:
            config (PoseDetectionConfig): Configuration object
        """
        self.config = config or PoseDetectionConfig()
        self.previous_pose = None
        self.frame_count = 0
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            enable_segmentation=False
        )
        
    def process_frame(self, frame):
        """
        Process single frame for pose detection.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            dict: Pose data for the frame
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Pose
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                # Calculate smart center (best available axis)
                center_x, center_y, center_confidence, axis_type = self.calculate_smart_center(results.pose_landmarks)
                
                # Also calculate hip center for backward compatibility
                hip_center_x, hip_center_y, hip_confidence = self.calculate_hip_center(results.pose_landmarks)
                
                # Store current pose for error handling
                pose_data = {
                    'frame': self.frame_count,
                    'timestamp': self.frame_count / 30.0,  # Assuming 30 FPS
                    'smart_center': {'x': center_x, 'y': center_y},
                    'smart_confidence': center_confidence,
                    'axis_type': axis_type,
                    'hip_center': {'x': hip_center_x, 'y': hip_center_y},  # Keep for compatibility
                    'hip_confidence': hip_confidence,
                    'keypoints': landmarks,
                    'raw_landmarks': results.pose_landmarks  # Store for overlay visualization
                }
                self.previous_pose = pose_data.copy()
                return pose_data
            else:
                # No detection - use previous frame's pose if available
                if self.previous_pose is not None:
                    return self.previous_pose.copy()
                else:
                    # No previous pose available - return empty
                    return {
                        'frame': self.frame_count,
                        'timestamp': self.frame_count / 30.0,
                        'smart_center': {'x': 0.5, 'y': 0.5},
                        'smart_confidence': 0.0,
                        'axis_type': 'fallback',
                        'hip_center': {'x': 0, 'y': 0},
                        'hip_confidence': 0.0,
                        'keypoints': []
                    }
                    
        except Exception:
            # Frame processing failure - use previous pose if available
            if self.previous_pose is not None:
                return self.previous_pose.copy()
            else:
                return {
                    'frame': self.frame_count,
                    'timestamp': self.frame_count / 30.0,
                    'smart_center': {'x': 0.5, 'y': 0.5},
                    'smart_confidence': 0.0,
                    'axis_type': 'fallback',
                    'hip_center': {'x': 0, 'y': 0},
                    'hip_confidence': 0.0,
                    'keypoints': []
                }
        
    def process_video(self, frames):
        """
        Process entire video for pose detection.
        
        Args:
            frames (list): List of video frames
            
        Returns:
            list: List of pose data dictionaries (one per frame)
        """
        pose_data_list = []
        
        for frame in frames:
            self.frame_count += 1
            pose_data = self.process_frame(frame)
            pose_data_list.append(pose_data)
        
        # Apply interpolation if enabled and needed
        if self.config.interpolate_missing:
            pose_data_list = self._interpolate_missing_pose_data(pose_data_list)
        
        return pose_data_list
        
    
    def calculate_smart_center(self, pose_landmarks):
        """
        Calculate the best available center point from visible keypoints.
        Priority: Shoulders > Hips > Visible keypoints center
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            tuple: (x, y, confidence, axis_type)
        """
        # MediaPipe landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        # Calculate shoulder center
        left_shoulder = pose_landmarks.landmark[LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[RIGHT_SHOULDER]
        shoulder_confidence = (left_shoulder.visibility + right_shoulder.visibility) / 2
        
        # Calculate hip center  
        left_hip = pose_landmarks.landmark[LEFT_HIP]
        right_hip = pose_landmarks.landmark[RIGHT_HIP]
        hip_confidence = (left_hip.visibility + right_hip.visibility) / 2
        
        # Choose best available axis based on confidence and coordinate validity
        axes = []
        
        # Shoulder axis
        if (shoulder_confidence > 0.3 and 
            0 <= left_shoulder.x <= 1 and 0 <= left_shoulder.y <= 1 and
            0 <= right_shoulder.x <= 1 and 0 <= right_shoulder.y <= 1):
            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            axes.append((shoulder_x, shoulder_y, shoulder_confidence, "shoulder"))
        
        # Hip axis
        if (hip_confidence > 0.3 and 
            0 <= left_hip.x <= 1 and 0 <= left_hip.y <= 1 and
            0 <= right_hip.x <= 1 and 0 <= right_hip.y <= 1):
            hip_x = (left_hip.x + right_hip.x) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            axes.append((hip_x, hip_y, hip_confidence, "hip"))
        
        # If both shoulder and hip available, prefer higher confidence
        if axes:
            # Sort by confidence, highest first
            axes.sort(key=lambda x: x[2], reverse=True)
            return axes[0]  # Return best axis
        
        # Fallback: center of all visible keypoints
        visible_points = []
        for idx, landmark in enumerate(pose_landmarks.landmark):
            if (landmark.visibility > 0.3 and 
                0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
                visible_points.append((landmark.x, landmark.y, landmark.visibility))
        
        if visible_points:
            # Calculate weighted center of visible points
            total_weight = sum(point[2] for point in visible_points)
            if total_weight > 0:
                center_x = sum(point[0] * point[2] for point in visible_points) / total_weight
                center_y = sum(point[1] * point[2] for point in visible_points) / total_weight
                avg_confidence = total_weight / len(visible_points)
                return center_x, center_y, avg_confidence, "keypoints_center"
        
        # Last resort: return frame center with zero confidence
        return 0.5, 0.5, 0.0, "fallback"

    def calculate_hip_center(self, pose_landmarks):
        """
        Calculate hip center from MediaPipe pose landmarks.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            tuple: (x, y, confidence)
        """
        # MediaPipe landmark indices for hips
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        left_hip = pose_landmarks.landmark[LEFT_HIP]
        right_hip = pose_landmarks.landmark[RIGHT_HIP]
        
        # Check if both hips are visible and valid
        if (left_hip.visibility > 0.3 and right_hip.visibility > 0.3 and
            0 <= left_hip.x <= 1 and 0 <= left_hip.y <= 1 and
            0 <= right_hip.x <= 1 and 0 <= right_hip.y <= 1):
            
            if self.config.hip_calc_method == "weighted":
                # Weighted average based on visibility
                total_weight = left_hip.visibility + right_hip.visibility
                hip_x = (left_hip.x * left_hip.visibility + right_hip.x * right_hip.visibility) / total_weight
                hip_y = (left_hip.y * left_hip.visibility + right_hip.y * right_hip.visibility) / total_weight
            else:
                # Simple average
                hip_x = (left_hip.x + right_hip.x) / 2
                hip_y = (left_hip.y + right_hip.y) / 2
            
            confidence = (left_hip.visibility + right_hip.visibility) / 2
            return hip_x, hip_y, confidence
        else:
            # Hip not detected - return center with zero confidence
            return 0.5, 0.5, 0.0
        
    def export_to_json(self, pose_data, output_path):
        """
        Export pose data to JSON file.
        
        Args:
            pose_data (dict): Pose timeline data
            output_path (str): Output JSON file path
        """
        with open(output_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
    
    def _interpolate_missing_pose_data(self, pose_data_list):
        """
        Interpolate missing pose data.
        
        Args:
            pose_data_list (list): Pose data list with potential missing values
            
        Returns:
            list: Pose data with interpolated values
        """
        for i in range(len(pose_data_list)):
            if pose_data_list[i]['smart_confidence'] < self.config.min_detection_confidence:
                # Find previous and next valid points
                prev_valid = None
                next_valid = None
                
                # Look backward
                for j in range(i-1, -1, -1):
                    if pose_data_list[j]['smart_confidence'] >= self.config.min_detection_confidence:
                        prev_valid = j
                        break
                
                # Look forward
                for j in range(i+1, len(pose_data_list)):
                    if pose_data_list[j]['smart_confidence'] >= self.config.min_detection_confidence:
                        next_valid = j
                        break
                
                # Interpolate if we have valid points
                if prev_valid is not None and next_valid is not None:
                    # Linear interpolation
                    weight = (i - prev_valid) / (next_valid - prev_valid)
                    
                    pose_data_list[i]['smart_center']['x'] = (
                        pose_data_list[prev_valid]['smart_center']['x'] * (1 - weight) +
                        pose_data_list[next_valid]['smart_center']['x'] * weight
                    )
                    pose_data_list[i]['smart_center']['y'] = (
                        pose_data_list[prev_valid]['smart_center']['y'] * (1 - weight) +
                        pose_data_list[next_valid]['smart_center']['y'] * weight
                    )
                    pose_data_list[i]['smart_confidence'] = max(
                        pose_data_list[prev_valid]['smart_confidence'],
                        pose_data_list[next_valid]['smart_confidence']
                    ) * 0.7  # Reduced confidence for interpolated data
                elif prev_valid is not None:
                    # Use previous valid point
                    pose_data_list[i]['smart_center'] = pose_data_list[prev_valid]['smart_center'].copy()
                    pose_data_list[i]['smart_confidence'] = pose_data_list[prev_valid]['smart_confidence'] * 0.5
                elif next_valid is not None:
                    # Use next valid point
                    pose_data_list[i]['smart_center'] = pose_data_list[next_valid]['smart_center'].copy()
                    pose_data_list[i]['smart_confidence'] = pose_data_list[next_valid]['smart_confidence'] * 0.5
        
        return pose_data_list
    
    
    def draw_full_pose_skeleton(self, frame, pose_landmarks, smart_center_pixel, axis_type, frame_num):
        """
        Draw comprehensive pose skeleton with larger keypoints for stabilized output.
        
        Args:
            frame: Input frame (person-only segmented frame)
            pose_landmarks: MediaPipe pose landmarks
            smart_center_pixel: Smart center in pixel coordinates (x, y)
            axis_type: Type of axis used ("shoulder", "hip", "keypoints_center", "fallback")
            frame_num: Frame number for labeling
            
        Returns:
            Frame with full pose skeleton overlay
        """
        if pose_landmarks is None:
            return frame
        
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # MediaPipe pose connections for skeleton
        pose_connections = [
            # Head and torso
            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),  # Head
            (4, 5), (5, 6), (6, 8),  # More head
            (9, 10),  # Mouth
            (11, 12),  # Shoulders  
            (11, 23), (12, 24),  # Torso to hips
            (23, 24),  # Hip line
            
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            
            # Right arm  
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            
            # Left leg
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
            
            # Right leg
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
        ]
        
        # Draw pose connections (skeleton lines)
        for connection in pose_connections:
            start_idx, end_idx = connection
            start_point = pose_landmarks.landmark[start_idx]
            end_point = pose_landmarks.landmark[end_idx]
            
            # Only draw if both points are visible and valid
            if (start_point.visibility > 0.3 and end_point.visibility > 0.3 and
                0 <= start_point.x <= 1 and 0 <= start_point.y <= 1 and
                0 <= end_point.x <= 1 and 0 <= end_point.y <= 1):
                
                start_pixel = (int(start_point.x * width), int(start_point.y * height))
                end_pixel = (int(end_point.x * width), int(end_point.y * height))
                
                # Draw skeleton line
                cv2.line(overlay_frame, start_pixel, end_pixel, (0, 255, 255), 3)  # Yellow lines
        
        # Draw all pose landmarks with larger, color-coded keypoints
        for idx, landmark in enumerate(pose_landmarks.landmark):
            if landmark.visibility > 0.3 and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Color coding for different body parts
                if idx in [11, 12]:  # Shoulders
                    color = (0, 255, 0)  # Green
                    radius = 10
                elif idx in [23, 24]:  # Hips
                    color = (255, 0, 0)  # Blue
                    radius = 10
                elif idx in [13, 14, 15, 16]:  # Arms
                    color = (255, 255, 0)  # Cyan
                    radius = 8
                elif idx in [25, 26, 27, 28]:  # Legs
                    color = (0, 0, 255)  # Red
                    radius = 8
                elif idx <= 10:  # Head/face
                    color = (255, 0, 255)  # Magenta
                    radius = 6
                else:  # Other points
                    color = (128, 128, 128)  # Gray
                    radius = 5
                
                # Draw keypoint
                cv2.circle(overlay_frame, (x, y), radius, color, -1)
                
                # Add labels for key landmarks
                if idx in [11, 12, 23, 24]:
                    labels = {11: "L_SHOULDER", 12: "R_SHOULDER", 23: "L_HIP", 24: "R_HIP"}
                    cv2.putText(overlay_frame, labels[idx], (x + 12, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw smart center point with distinctive marking
        if smart_center_pixel:
            center_x, center_y = int(smart_center_pixel[0]), int(smart_center_pixel[1])
            
            # Draw center point based on axis type
            center_colors = {
                "shoulder": (0, 255, 0),      # Green
                "hip": (255, 0, 0),           # Blue  
                "keypoints_center": (255, 255, 0),  # Cyan
                "fallback": (128, 128, 128)   # Gray
            }
            center_color = center_colors.get(axis_type, (255, 255, 255))
            
            # Draw large center point
            cv2.circle(overlay_frame, (center_x, center_y), 15, center_color, 4)
            cv2.circle(overlay_frame, (center_x, center_y), 8, (255, 255, 255), -1)  # White center
            
            # Add center point label
            cv2.putText(overlay_frame, f"CENTER ({axis_type.upper()})", (center_x + 20, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, center_color, 2)
        
        # Add frame information
        info_color = (255, 255, 255)  # White text
        cv2.putText(overlay_frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2)
        cv2.putText(overlay_frame, f"Axis: {axis_type.upper()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
        if smart_center_pixel:
            cv2.putText(overlay_frame, f"Center: ({smart_center_pixel[0]:.0f}, {smart_center_pixel[1]:.0f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
        
        return overlay_frame
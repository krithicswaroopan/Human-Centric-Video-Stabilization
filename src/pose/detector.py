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
                
                # Calculate hip center
                hip_center_x, hip_center_y, hip_confidence = self.calculate_hip_center(results.pose_landmarks)
                
                # Store current pose for error handling
                pose_data = {
                    'frame': self.frame_count,
                    'timestamp': self.frame_count / 30.0,  # Assuming 30 FPS
                    'hip_center': {'x': hip_center_x, 'y': hip_center_y},
                    'hip_confidence': hip_confidence,
                    'keypoints': landmarks
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
            dict: Complete pose timeline data in Timeline JSON format
        """
        hip_centers = []
        all_keypoints = []
        
        for frame in frames:
            self.frame_count += 1
            pose_data = self.process_frame(frame)
            
            # Add to hip centers timeline
            hip_centers.append({
                'frame': pose_data['frame'],
                'timestamp': pose_data['timestamp'],
                'x': pose_data['hip_center']['x'],
                'y': pose_data['hip_center']['y'],
                'confidence': pose_data['hip_confidence']
            })
            
            # Add to all keypoints
            all_keypoints.append({
                'frame': pose_data['frame'],
                'keypoints': pose_data['keypoints']
            })
        
        # Apply interpolation if enabled and needed
        if self.config.interpolate_missing:
            hip_centers = self._interpolate_missing_data(hip_centers)
        
        # Return Timeline JSON format as per master doc
        return {
            'video_info': {
                'fps': 30,  # Assumed FPS
                'total_frames': len(frames)
            },
            'hip_centers': hip_centers,
            'all_keypoints': all_keypoints
        }
        
    def calculate_hip_center(self, pose_landmarks):
        """
        Calculate hip center from pose landmarks.
        Left Hip: Landmark #23, Right Hip: Landmark #24
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            tuple: (x, y, confidence)
        """
        left_hip = pose_landmarks.landmark[23]  # LEFT_HIP
        right_hip = pose_landmarks.landmark[24]  # RIGHT_HIP
        
        if self.config.hip_calc_method == "simple":
            # Simple average as per master doc
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            hip_confidence = (left_hip.visibility + right_hip.visibility) / 2
            
        elif self.config.hip_calc_method == "weighted":
            # Confidence-weighted calculation
            left_weight = left_hip.visibility
            right_weight = right_hip.visibility
            total_weight = left_weight + right_weight
            
            if total_weight > 0:
                hip_center_x = (left_hip.x * left_weight + right_hip.x * right_weight) / total_weight
                hip_center_y = (left_hip.y * left_weight + right_hip.y * right_weight) / total_weight
                hip_confidence = total_weight / 2
            else:
                # Fallback to simple average if no visibility
                hip_center_x = (left_hip.x + right_hip.x) / 2
                hip_center_y = (left_hip.y + right_hip.y) / 2
                hip_confidence = 0.0
        
        return hip_center_x, hip_center_y, hip_confidence
        
    def export_to_json(self, pose_data, output_path):
        """
        Export pose data to JSON file.
        
        Args:
            pose_data (dict): Pose timeline data
            output_path (str): Output JSON file path
        """
        with open(output_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
    
    def _interpolate_missing_data(self, hip_centers):
        """
        Interpolate missing hip center data.
        
        Args:
            hip_centers (list): Hip center data with potential missing values
            
        Returns:
            list: Hip centers with interpolated data
        """
        for i in range(len(hip_centers)):
            if hip_centers[i]['confidence'] < self.config.min_detection_confidence:
                # Find previous and next valid points
                prev_valid = None
                next_valid = None
                
                # Look backward
                for j in range(i-1, -1, -1):
                    if hip_centers[j]['confidence'] >= self.config.min_detection_confidence:
                        prev_valid = j
                        break
                
                # Look forward
                for j in range(i+1, len(hip_centers)):
                    if hip_centers[j]['confidence'] >= self.config.min_detection_confidence:
                        next_valid = j
                        break
                
                # Interpolate if we have valid points
                if prev_valid is not None and next_valid is not None:
                    # Linear interpolation
                    weight = (i - prev_valid) / (next_valid - prev_valid)
                    
                    hip_centers[i]['x'] = (
                        hip_centers[prev_valid]['x'] * (1 - weight) +
                        hip_centers[next_valid]['x'] * weight
                    )
                    hip_centers[i]['y'] = (
                        hip_centers[prev_valid]['y'] * (1 - weight) +
                        hip_centers[next_valid]['y'] * weight
                    )
                    hip_centers[i]['confidence'] = max(
                        hip_centers[prev_valid]['confidence'],
                        hip_centers[next_valid]['confidence']
                    ) * 0.7  # Reduced confidence for interpolated data
                elif prev_valid is not None:
                    # Use previous valid point
                    hip_centers[i]['x'] = hip_centers[prev_valid]['x']
                    hip_centers[i]['y'] = hip_centers[prev_valid]['y']
                    hip_centers[i]['confidence'] = hip_centers[prev_valid]['confidence'] * 0.5
                elif next_valid is not None:
                    # Use next valid point
                    hip_centers[i]['x'] = hip_centers[next_valid]['x']
                    hip_centers[i]['y'] = hip_centers[next_valid]['y']
                    hip_centers[i]['confidence'] = hip_centers[next_valid]['confidence'] * 0.5
        
        return hip_centers
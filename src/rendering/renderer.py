"""
Video rendering module for output generation.
Based on master planning document specifications.
"""
import cv2
import numpy as np
import os


class RenderingConfig:
    """Configuration for rendering module."""
    def __init__(self, video_codec='mp4v', video_quality=1.0, comparison_layout='horizontal',
                 max_total_pixels=2073600, scale_large_videos=True):  # 1920x1080 = 2.07MP
        self.video_codec = video_codec
        self.video_quality = video_quality
        self.comparison_layout = comparison_layout
        # Universal scaling based on total pixel count (resolution-agnostic)
        self.max_total_pixels = max_total_pixels    # Max pixels for comparison video
        self.scale_large_videos = scale_large_videos # Whether to scale down large videos


class Renderer:
    """
    Video rendering for stabilized output and side-by-side comparison.
    - Stabilized video output
    - Side-by-side comparison (original | stabilized)
    """
    
    def __init__(self, output_dir, config=None):
        """
        Initialize renderer.
        
        Args:
            output_dir (str): Output directory path
            config (RenderingConfig): Configuration object
        """
        self.output_dir = output_dir
        self.config = config or RenderingConfig()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def render_video(self, frames, filename, fps=30):
        """
        Render frames to video file.
        
        Args:
            frames (list): List of frames to render
            filename (str): Output filename
            fps (int): Frame rate
        """
        if not frames:
            print(f"Warning: No frames to render for {filename}")
            return
        
        # Get video properties from first frame
        height, width = frames[0].shape[:2]
        output_path = os.path.join(self.output_dir, filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return
        
        try:
            # Write frames to video
            for frame in frames:
                # Ensure frame is in correct format
                if len(frame.shape) == 3:
                    video_writer.write(frame)
                else:
                    # Convert grayscale to BGR if needed
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    video_writer.write(frame_bgr)
            
            print(f"Successfully rendered video: {output_path}")
            
        except Exception as e:
            print(f"Error writing video {output_path}: {e}")
        finally:
            video_writer.release()
        
    def render_comparison(self, orig_frames, stab_frames, filename, fps=30):
        """
        Render side-by-side comparison video (original | stabilized).
        Master doc: Side-by-side comparison using OpenCV hconcat
        
        Args:
            orig_frames (list): Original frames
            stab_frames (list): Stabilized frames  
            filename (str): Output filename
            fps (int): Frame rate
        """
        
        if not orig_frames or not stab_frames:
            print(f"Warning: Missing frames for comparison video {filename}")
            return
        
        if len(orig_frames) != len(stab_frames):
            print(f"Warning: Frame count mismatch for {filename}")
            min_frames = min(len(orig_frames), len(stab_frames))
            orig_frames = orig_frames[:min_frames]
            stab_frames = stab_frames[:min_frames]
        
        # Get frame properties
        height, width = orig_frames[0].shape[:2]
        output_path = os.path.join(self.output_dir, filename)
        
        # Universal resolution scaling based on total pixel count (works for all aspect ratios)
        scale_factor = 1.0
        if self.config.scale_large_videos:
            # Calculate projected comparison video dimensions
            if self.config.comparison_layout == 'horizontal':
                comp_width = width * 2  # Side-by-side doubles width
                comp_height = height
            else:
                comp_width = width
                comp_height = height * 2  # Vertical doubles height
            
            # Calculate total pixels in comparison video
            total_pixels = comp_width * comp_height
            
            # Scale down if exceeds maximum
            if total_pixels > self.config.max_total_pixels:
                scale_factor = np.sqrt(self.config.max_total_pixels / total_pixels)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                if self.config.comparison_layout == 'horizontal':
                    final_comp_width = new_width * 2
                    final_comp_height = new_height
                else:
                    final_comp_width = new_width  
                    final_comp_height = new_height * 2
                
        
        # Create comparison frames
        comparison_frames = []
        
        for i, (orig_frame, stab_frame) in enumerate(zip(orig_frames, stab_frames)):
            # Ensure both frames have same dimensions
            if orig_frame.shape != stab_frame.shape:
                stab_frame = cv2.resize(stab_frame, (width, height))
            
            # Scale down frames if needed for large videos
            if scale_factor < 1.0:
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
                orig_frame = cv2.resize(orig_frame, (scaled_width, scaled_height))
                stab_frame = cv2.resize(stab_frame, (scaled_width, scaled_height))
            
            # Create side-by-side layout as per master doc
            if self.config.comparison_layout == 'horizontal':
                # Horizontal concatenation: original | stabilized
                comparison_frame = cv2.hconcat([orig_frame, stab_frame])
            else:
                # Vertical concatenation: original above, stabilized below
                comparison_frame = cv2.vconcat([orig_frame, stab_frame])
            
            comparison_frames.append(comparison_frame)
            
        
        # Get comparison video dimensions
        comp_height, comp_width = comparison_frames[0].shape[:2]
        
        # Initialize video writer for comparison
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (comp_width, comp_height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            
            # Try alternative codec
            fourcc_alt = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc_alt, fps, (comp_width, comp_height))
            
            if not video_writer.isOpened():
                print("Error: Alternative codec also failed")
                return
        
        try:
            # Write comparison frames
            for frame in comparison_frames:
                video_writer.write(frame)
            
            print(f"Successfully rendered comparison video: {output_path}")
            
        except Exception as e:
            print(f"Error writing comparison video {output_path}: {e}")
        finally:
            video_writer.release()
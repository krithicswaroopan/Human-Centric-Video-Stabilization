"""
CLI entry point for Human-Centric Video Stabilization.
Based on master planning document specifications.
"""
import argparse
import sys
import os

# Suppress TensorFlow oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from background.remover import BackgroundRemover, BackgroundRemovalConfig
from pose.detector import PoseDetector, PoseDetectionConfig
from stabilization.stabilizer import Stabilizer, StabilizationConfig
from rendering.renderer import Renderer, RenderingConfig
import cv2
import numpy as np


def parse_arguments():
    """
    Parse command line arguments based on master document specifications.
    """
    parser = argparse.ArgumentParser(
        description="Human-Centric Video Stabilization Pipeline"
    )
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    # Background removal options
    parser.add_argument("--bg_model", choices=["general", "landscape"], 
                       default="general", help="Background model variant")
    parser.add_argument("--bg_mode", choices=["transparent", "color", "blur"],
                       default="color", help="Background replacement mode")
    parser.add_argument("--bg_color", type=str, default="0,255,0",
                       help="Background color in BGR format (e.g. '0,255,0' for green)")
    parser.add_argument("--blur_strength", type=int, default=15,
                       help="Background blur strength (kernel size, must be odd)")
    parser.add_argument("--enable_edge_smoothing", choices=["true", "false"],
                       default="true", help="Enable border smoothing for cleaner edges")
    
    # Pose detection options
    parser.add_argument("--pose_model", choices=["lite", "full", "heavy"],
                       default="full", help="Pose model complexity")
    parser.add_argument("--hip_calc_method", choices=["simple", "weighted"],
                       default="weighted", help="Hip center calculation method")
    parser.add_argument("--min_confidence", type=float, default=0.7,
                       help="Minimum pose detection confidence")
    parser.add_argument("--interpolate_missing", choices=["true", "false"],
                       default="true", help="Interpolate missing pose data")
    
    # Stabilization options
    parser.add_argument("--smoothing_method", choices=["moving", "gaussian"],
                       default="gaussian", help="Smoothing method")
    parser.add_argument("--smoothing_window", type=int, default=7,
                       help="Smoothing window size")
    parser.add_argument("--border_mode", choices=["reflect", "constant"],
                       default="constant", help="Border handling mode")
    parser.add_argument("--use_cuda", choices=["true", "false"],
                       default="false", help="Enable CUDA acceleration")
    
    return parser.parse_args()


def load_video(video_path):
    """Load video frames into memory (master doc: load entire video to RAM)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"Loaded {len(frames)} frames at {fps} FPS")
    return frames, fps


def main():
    """
    Main pipeline execution based on master document workflow.
    Sequential processing: Background → Pose → Stabilization → Rendering
    """
    args = parse_arguments()
    
    print("Human-Centric Video Stabilization Pipeline")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    try:
        # Step 1: Load video (master doc: load entire video to RAM)
        print("Step 1: Loading video...")
        frames, fps = load_video(args.input)
        
        # Step 2: Initialize modules with configurations
        print("Step 2: Initializing modules...")
        
        # Parse background color from string
        try:
            bg_color_values = [int(x) for x in args.bg_color.split(',')]
            if len(bg_color_values) != 3:
                raise ValueError("Background color must have exactly 3 values")
            bg_color = tuple(bg_color_values)
        except (ValueError, AttributeError) as e:
            print(f"Invalid background color format '{args.bg_color}'. Using default green (0,255,0)")
            bg_color = (0, 255, 0)
        
        # Background removal config with enhancements
        bg_config = BackgroundRemovalConfig(
            model_selection=0 if args.bg_model == "general" else 1,
            confidence_threshold=0.1,
            enable_bilateral_filter=True,
            enable_temporal_smoothing=True,
            fallback_enabled=True,
            # Background options
            background_mode=args.bg_mode,
            background_color=bg_color,
            blur_strength=args.blur_strength,
            # Border smoothing
            enable_edge_smoothing=args.enable_edge_smoothing == "true"
        )
        background_remover = BackgroundRemover(bg_config)
        
        # Pose detection config  
        pose_complexity_map = {"lite": 0, "full": 1, "heavy": 2}
        pose_config = PoseDetectionConfig(
            model_complexity=pose_complexity_map[args.pose_model],
            min_detection_confidence=args.min_confidence,
            min_tracking_confidence=args.min_confidence,
            hip_calc_method=args.hip_calc_method,
            interpolate_missing=args.interpolate_missing == "true"
        )
        pose_detector = PoseDetector(pose_config)
        
        # Stabilization config
        stabilization_config = StabilizationConfig(
            smoothing_method=args.smoothing_method,
            smoothing_window=args.smoothing_window,
            border_mode=args.border_mode,
            use_cuda=args.use_cuda == "true"
        )
        stabilizer = Stabilizer(stabilization_config)
        
        # Rendering config with universal video compatibility settings
        rendering_config = RenderingConfig(
            max_total_pixels=2073600,  # Full HD total pixels (1920x1080) - universal limit
            scale_large_videos=True    # Enable scaling for large videos
        )
        renderer = Renderer(args.output_dir, rendering_config)
        
        # Step 3: Background removal processing
        print("Step 3: Processing background removal...")
        bg_results = list(background_remover.process_video(frames))
        person_frames = [result[0] for result in bg_results]
        
        # Step 4: Pose detection processing
        print("Step 4: Processing pose detection...")
        pose_data_list = pose_detector.process_video(frames)
        
        detected_count = sum(1 for p in pose_data_list if p['smart_confidence'] > 0.3)
        print(f"Detected poses in {detected_count}/{len(pose_data_list)} frames")
        
        # Export pose data as JSON
        pose_json_path = os.path.join(args.output_dir, "pose_data.json")
        import json
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            json_data = convert_numpy_types(pose_data_list)
            with open(pose_json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Pose data exported to: {pose_json_path}")
        except Exception as e:
            print(f"Warning: Failed to export pose data: {e}")
        
        # Step 5: Stabilization processing
        print("Step 5: Processing stabilization...")
        
        # Set target position to frame center
        if frames:
            frame_height, frame_width = frames[0].shape[:2] 
            stabilizer.config.target_position = (frame_width // 2, frame_height // 2)
        
        # Compute and smooth offsets using smart centers
        raw_offsets = stabilizer.compute_offsets(pose_data_list, normalized=True)
        smooth_offsets = stabilizer.smooth_offsets(raw_offsets)
        
        # Apply transforms to person frames  
        stabilized_frames = stabilizer.apply_transforms(person_frames, smooth_offsets)
        
        # Add pose skeleton overlay to stabilized frames
        print("Adding pose skeleton overlay to stabilized frames...")
        stabilized_frames_with_pose = []
        
        # Get frame dimensions for coordinate conversion
        frame_height, frame_width = frames[0].shape[:2]
        
        for i, (stabilized_frame, frame_pose_data) in enumerate(zip(stabilized_frames, pose_data_list)):
            if 'raw_landmarks' in frame_pose_data and frame_pose_data['raw_landmarks']:
                # Convert smart center to pixel coordinates
                smart_center = frame_pose_data['smart_center']
                smart_center_pixel = (
                    smart_center['x'] * frame_width if smart_center['x'] <= 1.0 else smart_center['x'],
                    smart_center['y'] * frame_height if smart_center['y'] <= 1.0 else smart_center['y']
                )
                
                # Add full pose skeleton to stabilized frame
                frame_with_pose = pose_detector.draw_full_pose_skeleton(
                    stabilized_frame,
                    frame_pose_data['raw_landmarks'],
                    smart_center_pixel,
                    frame_pose_data.get('axis_type', 'unknown'),
                    i
                )
                stabilized_frames_with_pose.append(frame_with_pose)
            else:
                # No pose detected, use stabilized frame as-is
                stabilized_frames_with_pose.append(stabilized_frame)
        
        # Step 6: Rendering outputs
        print("Step 6: Rendering output videos...")
        
        # Render stabilized video with pose skeleton overlay
        renderer.render_video(stabilized_frames_with_pose, "stabilized_output.mp4", fps)
        
        # Render side-by-side comparison (original | stabilized with pose overlay)
        renderer.render_comparison(frames, stabilized_frames_with_pose, "comparison_output.mp4", fps)
        
        print("\n" + "=" * 50)
        print("Pipeline execution completed successfully!")
        print(f"Outputs saved to: {args.output_dir}")
        print("- stabilized_output.mp4: Stabilized video")  
        print("- comparison_output.mp4: Side-by-side comparison")
        print("- pose_data.json: Pose detection data")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
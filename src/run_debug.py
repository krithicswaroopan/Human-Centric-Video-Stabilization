"""
DEBUG VERSION: CLI entry point for Human-Centric Video Stabilization.
Same logic as run.py but with enhanced debugging output and additional files.
"""
import argparse
import sys
import os
import json
import logging
from datetime import datetime

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


def setup_debug_logging(output_dir):
    """Setup comprehensive logging for debugging."""
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Setup logging
    log_filename = os.path.join(debug_dir, f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return debug_dir


def parse_arguments():
    """Parse command line arguments (same as run.py)."""
    parser = argparse.ArgumentParser(
        description="Human-Centric Video Stabilization Pipeline - DEBUG VERSION"
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
    
    # Debug-specific options
    parser.add_argument("--debug_frames", type=int, default=25,
                       help="Number of sample frames to save for debugging")
    
    return parser.parse_args()


def load_video(video_path):
    """Load video frames into memory (same as run.py)."""
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


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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


def main():
    """DEBUG VERSION: Main pipeline execution with enhanced debugging."""
    try:
        args = parse_arguments()
        
        # Setup debug logging
        debug_dir = setup_debug_logging(args.output_dir)
        logging.info("=" * 65)
        logging.info("Human-Centric Video Stabilization - DEBUG VERSION")
        logging.info("=" * 65)
        
        # Step 1: Load video
        print("Step 1: Loading video...")
        logging.info("STEP 1: Loading video...")
        frames, fps = load_video(args.input)
        frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
        logging.info(f"Video loaded: {len(frames)} frames, {fps} FPS, {frame_width}x{frame_height}")
        
        # Step 2: Initialize modules (EXACT SAME AS RUN.PY)
        print("Step 2: Initializing modules...")
        logging.info("STEP 2: Initializing modules...")
        
        # Parse background color from string
        try:
            bg_color_values = [int(x) for x in args.bg_color.split(',')]
            if len(bg_color_values) != 3:
                raise ValueError("Background color must have exactly 3 values")
            bg_color = tuple(bg_color_values)
        except (ValueError, AttributeError) as e:
            print(f"Invalid background color format '{args.bg_color}'. Using default green (0,255,0)")
            bg_color = (0, 255, 0)
        
        # Background removal config (SAME AS RUN.PY)
        bg_config = BackgroundRemovalConfig(
            model_selection=0 if args.bg_model == "general" else 1,
            confidence_threshold=0.1,
            enable_bilateral_filter=True,
            enable_temporal_smoothing=True,
            fallback_enabled=True,
            background_mode=args.bg_mode,
            background_color=bg_color,
            blur_strength=args.blur_strength,
            enable_edge_smoothing=args.enable_edge_smoothing == "true"
        )
        background_remover = BackgroundRemover(bg_config)
        logging.info(f"Background config: model={args.bg_model}, mode={args.bg_mode}, color={bg_color}")
        
        # Pose detection config (SAME AS RUN.PY)
        pose_complexity_map = {"lite": 0, "full": 1, "heavy": 2}
        pose_config = PoseDetectionConfig(
            model_complexity=pose_complexity_map[args.pose_model],
            min_detection_confidence=args.min_confidence,
            min_tracking_confidence=args.min_confidence,
            hip_calc_method=args.hip_calc_method,
            interpolate_missing=args.interpolate_missing == "true"
        )
        pose_detector = PoseDetector(pose_config)
        logging.info(f"Pose config: model={args.pose_model}, confidence={args.min_confidence}")
        
        # Stabilization config (SAME AS RUN.PY)
        stabilization_config = StabilizationConfig(
            smoothing_method=args.smoothing_method,
            smoothing_window=args.smoothing_window,
            border_mode=args.border_mode,
            use_cuda=args.use_cuda == "true"
        )
        stabilizer = Stabilizer(stabilization_config)
        logging.info(f"Stabilization config: smoothing={args.smoothing_method}, window={args.smoothing_window}")
        
        # Rendering config (SAME AS RUN.PY)
        rendering_config = RenderingConfig(
            max_total_pixels=2073600,  # Full HD total pixels (1920x1080) - universal limit
            scale_large_videos=True    # Enable scaling for large videos
        )
        renderer = Renderer(args.output_dir, rendering_config)
        logging.info("Modules initialized successfully")
        
        # Step 3: Background removal processing (SAME AS RUN.PY)
        print("Step 3: Processing background removal...")
        logging.info("STEP 3: Processing background removal...")
        bg_results = list(background_remover.process_video(frames))
        person_frames = [result[0] for result in bg_results]
        
        # DEBUG: Save sample background frames
        frames_debug_dir = os.path.join(debug_dir, "frames")
        os.makedirs(frames_debug_dir, exist_ok=True)
        for i, (person_frame, original_frame) in enumerate(zip(person_frames[:args.debug_frames], frames)):
            if i % 5 == 0:  # Every 5th frame
                # Save original frame
                cv2.imwrite(os.path.join(frames_debug_dir, f"original_{i:04d}.jpg"), original_frame)
                # Save person frame
                cv2.imwrite(os.path.join(frames_debug_dir, f"person_{i:04d}.jpg"), person_frame)
                logging.info(f"Saved debug frame {i}")
        
        logging.info(f"Background removal completed: {len(person_frames)} frames processed")
        
        # Step 4: Pose detection (SAME AS RUN.PY)
        print("Step 4: Detecting poses...")
        logging.info("STEP 4: Detecting poses...")
        pose_data_list = pose_detector.process_video(frames)
        
        # DEBUG: Log pose detection statistics
        detected_count = sum(1 for p in pose_data_list if p['smart_confidence'] > 0.3)
        logging.info(f"Pose detection stats: {detected_count}/{len(pose_data_list)} frames with valid poses")
        
        # DEBUG: Log first few pose detections
        for i, pose_data in enumerate(pose_data_list[:5]):
            logging.info(f"Frame {i}: confidence={pose_data['smart_confidence']:.3f}, "
                        f"center=({pose_data['smart_center']['x']:.3f}, {pose_data['smart_center']['y']:.3f}), "
                        f"axis={pose_data.get('axis_type', 'unknown')}")
        
        print(f"Detected poses in {detected_count}/{len(pose_data_list)} frames")
        
        # Step 5: Stabilization (SAME AS RUN.PY)
        print("Step 5: Computing stabilization...")
        logging.info("STEP 5: Computing stabilization...")
        
        # Compute offsets
        raw_offsets = stabilizer.compute_offsets(pose_data_list, normalized=True)
        logging.info(f"Computed {len(raw_offsets)} raw offsets")
        
        # Apply smoothing
        smooth_offsets = stabilizer.smooth_offsets(raw_offsets)
        smoothing_sigma = stabilizer.get_smoothing_sigma()
        print(f"Gaussian smoothing applied with sigma {smoothing_sigma}.")
        logging.info(f"Applied smoothing with sigma {smoothing_sigma}")
        
        # DEBUG: Log offset statistics
        if smooth_offsets:
            avg_offset_x = np.mean([offset[0] for offset in smooth_offsets])
            avg_offset_y = np.mean([offset[1] for offset in smooth_offsets])
            max_offset_x = max([abs(offset[0]) for offset in smooth_offsets])
            max_offset_y = max([abs(offset[1]) for offset in smooth_offsets])
            logging.info(f"Offset stats: avg=({avg_offset_x:.2f}, {avg_offset_y:.2f}), "
                        f"max=({max_offset_x:.2f}, {max_offset_y:.2f})")
        
        # Apply stabilization transforms
        stabilized_frames = stabilizer.apply_transforms(person_frames, smooth_offsets)
        logging.info(f"Applied stabilization transforms to {len(stabilized_frames)} frames")
        
        # Step 6: Add pose overlay (SAME AS RUN.PY)
        print("Adding pose skeleton overlay to stabilized frames...")
        logging.info("STEP 6: Adding pose overlays...")
        stabilized_frames_with_pose = []
        
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
                stabilized_frames_with_pose.append(stabilized_frame)
        
        # DEBUG: Save sample stabilized frames
        stabilized_debug_dir = os.path.join(debug_dir, "stabilized_frames")
        os.makedirs(stabilized_debug_dir, exist_ok=True)
        for i, stabilized_frame in enumerate(stabilized_frames_with_pose[:args.debug_frames]):
            if i % 5 == 0:  # Every 5th frame
                cv2.imwrite(os.path.join(stabilized_debug_dir, f"stabilized_{i:04d}.jpg"), stabilized_frame)
                logging.info(f"Saved debug stabilized frame {i}")
        
        logging.info("Pose overlays added successfully")
        
        # Step 7: Rendering output videos (SAME AS RUN.PY)
        print("Step 6: Rendering output videos...")
        logging.info("STEP 7: Rendering output videos...")
        
        # Render stabilized video
        renderer.render_video(stabilized_frames_with_pose, "stabilized_output.mp4", fps)
        
        # Render comparison video
        renderer.render_comparison(frames, stabilized_frames_with_pose, "comparison_output.mp4", fps)
        
        # DEBUG: Export pose data to JSON
        pose_data_json = os.path.join(args.output_dir, "pose_data.json")
        try:
            json_data = convert_numpy_types(pose_data_list)
            with open(pose_data_json, 'w') as f:
                json.dump(json_data, f, indent=2)
            logging.info(f"Pose data exported to: {pose_data_json}")
        except Exception as e:
            logging.error(f"Failed to export pose data: {e}")
        
        # DEBUG: Export stabilization data
        stabilization_data = {
            "raw_offsets": convert_numpy_types(raw_offsets),
            "smooth_offsets": convert_numpy_types(smooth_offsets),
            "smoothing_sigma": float(smoothing_sigma),
            "frame_count": len(frames),
            "detection_rate": detected_count / len(pose_data_list)
        }
        
        stabilization_json = os.path.join(args.output_dir, "stabilization_data.json")
        try:
            with open(stabilization_json, 'w') as f:
                json.dump(stabilization_data, f, indent=2)
            logging.info(f"Stabilization data exported to: {stabilization_json}")
        except Exception as e:
            logging.error(f"Failed to export stabilization data: {e}")
        
        print("\n" + "=" * 65)
        print("DEBUG Pipeline execution completed successfully!")
        print(f"Outputs saved to: {args.output_dir}")
        print(f"- stabilized_output.mp4: Stabilized video")
        print(f"- comparison_output.mp4: Side-by-side comparison")
        print(f"- pose_data.json: Pose detection data")
        print(f"- stabilization_data.json: Stabilization analysis")
        print(f"Debug outputs in: {debug_dir}")
        print(f"- debug_log_*.txt: Comprehensive logging")
        print(f"- frames/: Sample original and person frames")
        print(f"- stabilized_frames/: Sample stabilized frames")
        print("=" * 65)
        
        logging.info("DEBUG pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
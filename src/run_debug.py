"""
DEBUG VERSION: CLI entry point for Human-Centric Video Stabilization.
Enhanced with comprehensive logging and diagnostic outputs.
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
    """Parse command line arguments for debug version."""
    parser = argparse.ArgumentParser(
        description="Human-Centric Video Stabilization Pipeline - DEBUG VERSION"
    )
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    # Background removal options
    parser.add_argument("--bg_model", choices=["general", "landscape"], 
                       default="general", help="Background model variant")
    
    # Pose detection options
    parser.add_argument("--pose_model", choices=["lite", "full", "heavy"],
                       default="full", help="Pose model complexity")
    parser.add_argument("--hip_calc_method", choices=["simple", "weighted"],
                       default="simple", help="Hip center calculation method")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                       help="Minimum pose detection confidence")
    parser.add_argument("--interpolate_missing", choices=["true", "false"],
                       default="true", help="Interpolate missing pose data")
    
    # Stabilization options
    parser.add_argument("--smoothing_method", choices=["moving", "gaussian"],
                       default="moving", help="Smoothing method")
    parser.add_argument("--smoothing_window", type=int, default=5,
                       help="Smoothing window size")
    parser.add_argument("--border_mode", choices=["reflect", "constant"],
                       default="reflect", help="Border handling mode")
    parser.add_argument("--use_cuda", choices=["true", "false"],
                       default="false", help="Enable CUDA acceleration")
    
    # Debug options
    parser.add_argument("--debug_frames", type=int, default=30,
                       help="Number of frames to process for debugging (default: 30)")
    
    return parser.parse_args()


def load_video(video_path, max_frames=None):
    """Load video frames with debugging info."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logging.info(f"VIDEO INFO:")
    logging.info(f"  - FPS: {fps}")
    logging.info(f"  - Total frames: {total_frame_count}")
    logging.info(f"  - Dimensions: {width}x{height}")
    logging.info(f"  - Duration: {total_frame_count/fps:.2f} seconds")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        
        # Limit frames for debugging
        if max_frames and frame_count >= max_frames:
            logging.info(f"Limiting to {max_frames} frames for debugging")
            break
    
    cap.release()
    logging.info(f"Loaded {len(frames)} frames")
    return frames, fps


def main():
    """Main debug pipeline execution."""
    args = parse_arguments()
    
    print("Human-Centric Video Stabilization Pipeline - DEBUG VERSION")
    print("=" * 65)
    print(f"Input: {args.input}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Debug Frames: {args.debug_frames}")
    print()
    
    try:
        # Setup debug environment
        debug_dir = setup_debug_logging(args.output_dir)
        
        # Step 1: Load video
        logging.info("STEP 1: Loading video...")
        frames, fps = load_video(args.input, args.debug_frames)
        
        # Log video analysis
        frame_height, frame_width = frames[0].shape[:2]
        logging.info(f"FRAME ANALYSIS:")
        logging.info(f"  - Frame dimensions: {frame_width}x{frame_height}")
        logging.info(f"  - Aspect ratio: {frame_width/frame_height:.3f}")
        logging.info(f"  - Orientation: {'Horizontal' if frame_width > frame_height else 'Vertical'}")
        
        # Step 2: Initialize modules
        logging.info("STEP 2: Initializing modules...")
        
        # Background removal config
        bg_config = BackgroundRemovalConfig(
            model_selection=0 if args.bg_model == "general" else 1,
            confidence_threshold=0.1,
            enable_bilateral_filter=True,
            enable_temporal_smoothing=True,
            fallback_enabled=True
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
        
        # Rendering config
        rendering_config = RenderingConfig()
        renderer = Renderer(args.output_dir, rendering_config)
        
        # Step 3: Process with debugging
        logging.info("STEP 3: Processing with full debugging...")
        
        # Create debug subdirectories
        frames_debug_dir = os.path.join(debug_dir, "frames")
        os.makedirs(frames_debug_dir, exist_ok=True)
        
        debug_data = {
            "video_info": {
                "fps": fps,
                "width": frame_width,
                "height": frame_height,
                "total_frames": len(frames),
                "aspect_ratio": frame_width/frame_height,
                "orientation": "horizontal" if frame_width > frame_height else "vertical"
            },
            "coordinate_conversions": [],
            "stabilization_debug": {},
            "frame_analysis": []
        }
        
        # Process frames with detailed logging
        bg_results = []
        pose_results = []
        
        for i, frame in enumerate(frames[:args.debug_frames]):
            logging.info(f"Processing frame {i+1}/{args.debug_frames}")
            
            # Background processing
            person_frame, mask = background_remover.process_frame(frame)
            bg_results.append((person_frame, mask))
            
            # Pose processing with coordinate logging
            pose_detector.frame_count = i
            pose_data = pose_detector.process_frame(frame)
            
            # Convert smart center to pixel coordinates for logging
            if pose_data['smart_confidence'] > 0:
                center_pixel_x = pose_data['smart_center']['x'] * frame_width if pose_data['smart_center']['x'] <= 1.0 else pose_data['smart_center']['x']
                center_pixel_y = pose_data['smart_center']['y'] * frame_height if pose_data['smart_center']['y'] <= 1.0 else pose_data['smart_center']['y']
                
                coord_debug = {
                    "frame": i,
                    "smart_center_normalized": {"x": pose_data['smart_center']['x'], "y": pose_data['smart_center']['y']},
                    "smart_center_pixel": {"x": center_pixel_x, "y": center_pixel_y},
                    "axis_type": pose_data.get('axis_type', 'unknown'),
                    "frame_center_pixel": {"x": frame_width//2, "y": frame_height//2},
                    "confidence": pose_data['smart_confidence'],
                    # Keep hip data for comparison
                    "hip_normalized": {"x": pose_data['hip_center']['x'], "y": pose_data['hip_center']['y']},
                    "hip_confidence": pose_data['hip_confidence']
                }
                debug_data["coordinate_conversions"].append(coord_debug)
                
                logging.info(f"  Frame {i}: Smart center ({pose_data.get('axis_type', 'unknown')}) normalized=({pose_data['smart_center']['x']:.3f}, {pose_data['smart_center']['y']:.3f}), pixel=({center_pixel_x:.1f}, {center_pixel_y:.1f})")
            
            pose_results.append(pose_data)
            
            # Save debug frames every 5th frame
            if i % 5 == 0:
                # Original with pose overlay
                if pose_data['smart_confidence'] > 0 and 'raw_landmarks' in pose_data:
                    overlay_frame = pose_detector.draw_pose_overlay(
                        frame, 
                        pose_data['raw_landmarks'],
                        (center_pixel_x, center_pixel_y),
                        i
                    )
                    cv2.imwrite(os.path.join(frames_debug_dir, f"frame_{i:04d}_original_overlay.jpg"), overlay_frame)
                    
                    # Also save with full skeleton overlay
                    skeleton_frame = pose_detector.draw_full_pose_skeleton(
                        person_frame,
                        pose_data['raw_landmarks'], 
                        (center_pixel_x, center_pixel_y),
                        pose_data.get('axis_type', 'unknown'),
                        i
                    )
                    cv2.imwrite(os.path.join(frames_debug_dir, f"frame_{i:04d}_skeleton.jpg"), skeleton_frame)
                else:
                    # Save original frame without overlay if no pose detected
                    cv2.imwrite(os.path.join(frames_debug_dir, f"frame_{i:04d}_original_no_pose.jpg"), frame)
                
                # Segmented frame
                cv2.imwrite(os.path.join(frames_debug_dir, f"frame_{i:04d}_segmented.jpg"), person_frame)
        
        # Step 4: Test stabilization calculations (but don't apply transforms yet)
        logging.info("STEP 4: Testing stabilization calculations...")
        
        # Use pose results directly for stabilization testing (smart centers)
        # Set target position and test offset calculations
        stabilizer.config.target_position = (frame_width // 2, frame_height // 2)
        logging.info(f"Target position set to: ({frame_width // 2}, {frame_height // 2}) - frame center")
        
        # Compute offsets with debugging using smart centers
        raw_offsets = stabilizer.compute_offsets(pose_results)
        smoothed_offsets = stabilizer.smooth_offsets(raw_offsets)
        
        # Get debug data from stabilizer
        stabilization_debug_data = stabilizer.get_debug_data()
        smoothing_debug_data = stabilizer.log_smoothing_debug(raw_offsets, smoothed_offsets)
        
        debug_data["stabilization_debug"] = {
            "target_position": {"x": frame_width // 2, "y": frame_height // 2},
            "offset_calculations": stabilization_debug_data,
            "smoothing_analysis": smoothing_debug_data,
            "raw_offsets_summary": {
                "count": len(raw_offsets),
                "first_5": raw_offsets[:5],
                "last_5": raw_offsets[-5:] if len(raw_offsets) > 5 else raw_offsets
            },
            "smoothed_offsets_summary": {
                "count": len(smoothed_offsets),
                "first_5": smoothed_offsets[:5],
                "last_5": smoothed_offsets[-5:] if len(smoothed_offsets) > 5 else smoothed_offsets
            }
        }
        
        # Log critical stabilization findings
        logging.info("STABILIZATION ANALYSIS:")
        logging.info(f"  - Total valid smart centers: {len([d for d in stabilization_debug_data if d.get('confidence', 0) > 0.3])}")
        
        # Count axis types used
        axis_types = {}
        for d in stabilization_debug_data:
            axis_type = d.get('axis_type', 'unknown')
            axis_types[axis_type] = axis_types.get(axis_type, 0) + 1
        logging.info(f"  - Axis types used: {axis_types}")
        
        if raw_offsets and all(isinstance(o, (list, tuple)) and len(o) == 2 for o in raw_offsets):
            logging.info(f"  - Raw offsets range: dx=[{min(o[0] for o in raw_offsets):.1f}, {max(o[0] for o in raw_offsets):.1f}], dy=[{min(o[1] for o in raw_offsets):.1f}, {max(o[1] for o in raw_offsets):.1f}]")
            logging.info(f"  - Smoothed offsets range: dx=[{min(o[0] for o in smoothed_offsets):.1f}, {max(o[0] for o in smoothed_offsets):.1f}], dy=[{min(o[1] for o in smoothed_offsets):.1f}, {max(o[1] for o in smoothed_offsets):.1f}]")
        else:
            logging.info(f"  - No valid offsets calculated")
        
        # Save debug data
        debug_json_path = os.path.join(debug_dir, "debug_data.json")
        
        # Convert all types to JSON-serializable format
        import numpy as np
        
        def fix_json_types(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                               np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: fix_json_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [fix_json_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # any remaining numpy scalar
                return obj.item()
            return obj
        
        try:
            with open(debug_json_path, 'w') as f:
                json.dump(fix_json_types(debug_data), f, indent=2)
        except Exception as e:
            logging.error(f"JSON serialization failed, saving as text: {e}")
            with open(debug_json_path.replace('.json', '.txt'), 'w') as f:
                f.write(str(debug_data))
        
        logging.info(f"Debug data saved to: {debug_json_path}")
        logging.info(f"Debug frames saved to: {frames_debug_dir}")
        
        # PHASE 2: Apply transforms and test video rendering
        logging.info("PHASE 2: Applying stabilization transforms and testing rendering...")
        
        # Create stabilized debug directory
        stabilized_debug_dir = os.path.join(debug_dir, "stabilized_frames")
        os.makedirs(stabilized_debug_dir, exist_ok=True)
        
        # Extract person frames from background results
        person_frames = [result[0] for result in bg_results]
        
        # Apply stabilization transforms
        logging.info("Applying stabilization transforms...")
        stabilized_frames = stabilizer.apply_transforms(person_frames, smoothed_offsets)
        
        # Save stabilized frames for comparison
        for i, stabilized_frame in enumerate(stabilized_frames[:10]):  # Save first 10
            if i % 2 == 0:  # Every other frame
                cv2.imwrite(os.path.join(stabilized_debug_dir, f"stabilized_{i:04d}.jpg"), stabilized_frame)
                logging.info(f"Saved stabilized frame {i}")
                
                # Log frame analysis
                non_zero_pixels = cv2.countNonZero(cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2GRAY))
                logging.info(f"  Frame {i}: Non-zero pixels = {non_zero_pixels}")
                
                # Check if person is visible in expected center region
                h, w = stabilized_frame.shape[:2]
                center_region = stabilized_frame[h//4:3*h//4, w//4:3*w//4]
                center_pixels = cv2.countNonZero(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY))
                logging.info(f"  Frame {i}: Center region pixels = {center_pixels}")
        
        # Add pose overlay to stabilized frames
        logging.info("Adding pose overlays to stabilized frames...")
        stabilized_frames_with_pose = []
        
        for i, (stabilized_frame, pose_data) in enumerate(zip(stabilized_frames, pose_results)):
            if pose_data['smart_confidence'] > 0 and 'raw_landmarks' in pose_data:
                # Convert smart center to pixel coordinates
                center_pixel_x = pose_data['smart_center']['x'] * frame_width if pose_data['smart_center']['x'] <= 1.0 else pose_data['smart_center']['x']
                center_pixel_y = pose_data['smart_center']['y'] * frame_height if pose_data['smart_center']['y'] <= 1.0 else pose_data['smart_center']['y']
                
                # Log coordinates before overlay
                logging.info(f"Frame {i}: Adding overlay at original smart center pixel=({center_pixel_x:.1f}, {center_pixel_y:.1f})")
                
                # Add pose overlay using ORIGINAL coordinates (not transformed)
                frame_with_pose = pose_detector.draw_full_pose_skeleton(
                    stabilized_frame,
                    pose_data['raw_landmarks'],
                    (center_pixel_x, center_pixel_y),  # Original coordinates
                    pose_data.get('axis_type', 'unknown'),
                    i
                )
                stabilized_frames_with_pose.append(frame_with_pose)
                
                # Save overlay frame for debugging
                if i % 5 == 0:
                    cv2.imwrite(os.path.join(stabilized_debug_dir, f"overlay_{i:04d}.jpg"), frame_with_pose)
                    logging.info(f"Saved overlay frame {i}")
            else:
                stabilized_frames_with_pose.append(stabilized_frame)
        
        # Test video rendering
        logging.info("Testing video rendering...")
        try:
            renderer.render_video(stabilized_frames_with_pose, "debug_stabilized_output.mp4", fps)
            logging.info("Video rendering completed successfully")
        except Exception as e:
            logging.error(f"Video rendering failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        print("\n" + "=" * 65)
        print("DEBUG PHASE 2 COMPLETED!")
        print(f"Check debug output in: {debug_dir}")
        print(f"- debug_log_*.txt: Comprehensive logging")
        print(f"- debug_data.json: Coordinate and calculation data")
        print(f"- frames/: Sample original frames with overlays")
        print(f"- stabilized_frames/: Stabilized frames analysis")
        print(f"- debug_stabilized_output.mp4: Test video output")
        print("\nCompare debug frames with video to identify the issue.")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
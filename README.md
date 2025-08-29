# Human-Centric Video Stabilization

A comprehensive Python pipeline that stabilizes videos of walking humans by keeping the subject anchored at a defined position while allowing natural background movement. The system combines MediaPipe pose detection, background segmentation, and advanced stabilization algorithms to create smooth, professional-quality stabilized videos.

## Overview

This system provides an end-to-end solution for human-centric video stabilization, utilizing state-of-the-art computer vision models and robust processing algorithms. The pipeline processes videos offline to achieve optimal quality results with extensive customization options for different use cases.

**Key Innovation**: Smart center calculation prioritizing shoulders > hips > visible keypoints for more robust stabilization compared to traditional hip-only approaches.

## Features

### Core Functionality
- **Smart-center based stabilization** - Intelligent axis selection (shoulders > hips > keypoint centroid) for robust stabilization
- **MediaPipe integration** - Advanced pose detection and background segmentation
- **Multiple background modes** - Transparent, solid color replacement, and background blur options
- **Edge smoothing** - Anti-aliased borders using morphological operations and Gaussian blur
- **Universal video compatibility** - Automatic scaling for different resolutions and aspect ratios
- **Comprehensive pose overlays** - Full skeleton visualization with color-coded body parts
- **Motion smoothing** - Advanced Gaussian and moving average filters with configurable parameters
- **Side-by-side comparison** - Before/after videos with synchronized playback
- **Pose data export** - Complete pose timeline data in JSON format

### Enhanced Processing Features
- **Temporal smoothing** - Frame-to-frame consistency for background removal
- **Bilateral filtering** - Smoother segmentation edges
- **Automatic fallback** - Model switching for challenging scenarios
- **Linear interpolation** - Missing pose data recovery
- **Debug mode** - Enhanced logging and frame-by-frame analysis
- **Memory optimization** - Efficient processing for large videos

### Output Options
- **Multiple video formats** - MP4 with codec fallback support
- **Flexible layouts** - Horizontal/vertical comparison arrangements  
- **Resolution scaling** - Automatic downscaling for large videos
- **Data export** - JSON pose data and stabilization metrics

## Requirements

- **Python 3.10+** (tested up to 3.12)
- **CPU-based processing** (GPU acceleration optional)
- **Memory**: 8GB+ RAM recommended (16GB+ for 4K videos)
- **Storage**: 2-3x input video size for output files

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd Human-Centric-Video-Stabilization
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `mediapipe>=0.10.0` - Pose detection and background segmentation
- `opencv-python>=4.8.0` - Video processing and transformations  
- `numpy>=1.24.0` - Array operations and mathematical computations
- `scipy>=1.10.0` - Signal processing for advanced smoothing algorithms

## Usage

### Basic Usage
```bash
python src/run.py --input input_video.mp4 --output_dir ./output
```

### Debug Mode (Enhanced Output)
```bash
python src/run_debug.py --input input_video.mp4 --output_dir ./output
```

### Advanced Configuration
```bash
python src/run.py \
  --input walking_video.mp4 \
  --output_dir ./results \
  --bg_mode blur \
  --bg_color "0,255,0" \
  --pose_model heavy \
  --hip_calc_method weighted \
  --smoothing_method gaussian \
  --smoothing_window 7 \
  --border_mode constant \
  --min_confidence 0.8 \
  --enable_edge_smoothing true
```

## Configuration Options

### Background Removal Options
- **`--bg_model {general,landscape}`** - Model variant selection (default: `general`)
  - `general`: Higher accuracy, better for complex backgrounds
  - `landscape`: Faster processing, optimized for landscape videos

- **`--bg_mode {transparent,color,blur}`** - Background replacement mode (default: `color`)
  - `transparent`: Remove background completely (black)
  - `color`: Replace with solid color
  - `blur`: Replace with blurred version of original background

- **`--bg_color "R,G,B"`** - Background color in BGR format (default: `"0,255,0"` = green)
  - Examples: `"255,0,0"` (blue), `"0,0,255"` (red), `"128,128,128"` (gray)

- **`--blur_strength INT`** - Background blur kernel size (default: `15`, must be odd)
  - Range: 3-51, higher values = more blur

- **`--enable_edge_smoothing {true,false}`** - Border smoothing for cleaner edges (default: `true`)

### Pose Detection Options
- **`--pose_model {lite,full,heavy}`** - Model complexity trade-off (default: `full`)
  - `lite`: Fastest processing, lower accuracy
  - `full`: Balanced speed and accuracy  
  - `heavy`: Highest accuracy, slower processing

- **`--hip_calc_method {simple,weighted}`** - Hip center calculation method (default: `weighted`)
  - `simple`: Basic average of left/right hip landmarks
  - `weighted`: Confidence-weighted average for better accuracy

- **`--min_confidence FLOAT`** - Minimum pose detection confidence (default: `0.7`)
  - Range: 0.0-1.0, higher values = stricter detection

- **`--interpolate_missing {true,false}`** - Interpolate missing pose data (default: `true`)

### Stabilization Options
- **`--smoothing_method {moving,gaussian}`** - Smoothing algorithm (default: `gaussian`)
  - `moving`: Moving average filter - simple and fast
  - `gaussian`: Gaussian filter - smoother results, better for walking motion

- **`--smoothing_window INT`** - Smoothing window size (default: `7`)
  - Range: 3-21, larger values = smoother but potentially over-smoothed results

- **`--border_mode {reflect,constant}`** - Border handling method (default: `constant`)
  - `reflect`: Mirror pixels at frame edges
  - `constant`: Fill with solid background color

- **`--use_cuda {true,false}`** - Enable CUDA GPU acceleration (default: `false`)
  - Requires CUDA-compatible GPU and drivers

### Debug Options (run_debug.py only)
- **`--debug_frames INT`** - Number of sample frames to save for analysis (default: `25`)

## Output Files

### Standard Output (`run.py`)
1. **`stabilized_output.mp4`** - Stabilized video with pose skeleton overlays
2. **`comparison_output.mp4`** - Side-by-side comparison (original | stabilized)
3. **`pose_data.json`** - Complete pose detection data

### Debug Output (`run_debug.py`)
4. **`debug/debug_log_*.txt`** - Comprehensive processing logs
5. **`debug/frames/`** - Sample original and processed frames
6. **`debug/stabilized_frames/`** - Sample stabilized frames with overlays
7. **`stabilization_data.json`** - Detailed stabilization metrics

### Pose Data JSON Format
```json
[
  {
    "frame": 0,
    "timestamp": 0.0,
    "smart_center": {"x": 0.52, "y": 0.68},
    "smart_confidence": 0.85,
    "axis_type": "shoulder",
    "hip_center": {"x": 0.51, "y": 0.72},
    "hip_confidence": 0.78,
    "keypoints": [
      {"x": 0.5, "y": 0.6, "z": 0.1, "visibility": 0.9}
    ]
  }
]
```

## Key Functions and Architecture

### 1. Smart Center Calculation
```python
# Priority-based center calculation
def calculate_smart_center(pose_landmarks):
    # 1st priority: Shoulder center (most stable)
    # 2nd priority: Hip center (traditional approach)  
    # 3rd priority: Visible keypoints centroid (fallback)
```

**Advantages over hip-only approaches:**
- More stable for varying walking styles
- Better handling of partial occlusion
- Reduced sensitivity to clothing/body shape variations

### 2. Background Replacement System
```python
# Multiple background modes with edge smoothing
def _apply_background_replacement(original, person, mask):
    if mode == "color": # Solid color replacement
    if mode == "blur":  # Blurred background  
    if mode == "transparent": # Remove completely
```

### 3. Universal Video Compatibility
```python
# Automatic scaling based on total pixel count
def render_comparison(orig_frames, stab_frames):
    total_pixels = comp_width * comp_height
    if total_pixels > max_pixels:
        scale_factor = sqrt(max_pixels / total_pixels)
```

**Supports all aspect ratios:** 16:9, 4:3, 1:1, portrait modes

### 4. Advanced Stabilization Algorithm
```python
# Motion-aware smoothing with configurable methods
def smooth_offsets(offsets):
    if method == "gaussian":
        sigma = window_size / 3.0
        return gaussian_filter1d(values, sigma)
    elif method == "moving":
        return moving_average(values, window_size)
```

### 5. Comprehensive Pose Visualization
```python
# Full skeleton with color-coded body parts
def draw_full_pose_skeleton(frame, landmarks):
    # Green: Shoulders, Blue: Hips, Cyan: Arms
    # Red: Legs, Magenta: Head, Gray: Other points
```

## Pipeline Architecture

The system follows a modular, sequential processing architecture:

```
Input Video → Background Removal → Pose Detection → Stabilization → Rendering → Output
     ↓              ↓                    ↓              ↓             ↓
MediaPipe      Smart Center       Motion Smoothing  Video Export  JSON Export
Segmentation   Calculation        & Transformation   & Comparison  & Debug Data
```

### Module Details

**Module 1: Background Removal (`background/remover.py`)**
- MediaPipe Selfie Segmentation with temporal smoothing
- Multiple background replacement modes
- Edge smoothing using morphological operations
- Automatic model fallback for challenging scenarios

**Module 2: Pose Detection (`pose/detector.py`)**
- MediaPipe Pose with 33-keypoint detection
- Smart center calculation with priority-based selection
- Linear interpolation for missing data
- Comprehensive pose visualization system

**Module 3: Stabilization (`stabilization/stabilizer.py`)**
- Smart center-based offset calculation
- Advanced smoothing algorithms (Gaussian/Moving Average)
- Affine transformation with configurable border handling
- Motion-aware parameter adjustment

**Module 4: Rendering (`rendering/renderer.py`)**
- Universal resolution compatibility
- Side-by-side comparison generation
- Multiple output formats with codec fallback
- Automatic scaling for large videos

## Performance Characteristics

### Processing Speed (approximate, varies by hardware)
- **Lite model**: 0.5-1x real-time
- **Full model**: 0.2-0.5x real-time  
- **Heavy model**: 0.1-0.2x real-time

### Memory Usage
- **1080p video**: ~2-4GB RAM
- **4K video**: ~8-16GB RAM
- **Background removal**: +30% memory overhead
- **Debug mode**: +20% additional overhead

### Quality vs Speed Trade-offs
- **Fast processing**: `lite` + `landscape` + `moving` + `window=3`
- **Balanced**: `full` + `general` + `gaussian` + `window=7` (default)
- **High quality**: `heavy` + `general` + `gaussian` + `window=9`

## Examples

### Basic Stabilization
```bash
# Default settings - good balance of quality and speed
python src/run.py --input walk.mp4 --output_dir ./results
```

### Green Screen Effect
```bash
# Replace background with green color for chroma key
python src/run.py \
  --input walk.mp4 \
  --output_dir ./results \
  --bg_mode color \
  --bg_color "0,255,0"
```

### Blurred Background Effect
```bash
# Artistic blurred background effect
python src/run.py \
  --input walk.mp4 \
  --output_dir ./results \
  --bg_mode blur \
  --blur_strength 25
```

### High-Quality Processing
```bash
# Maximum quality settings (slower processing)
python src/run.py \
  --input walk.mp4 \
  --output_dir ./results \
  --pose_model heavy \
  --smoothing_method gaussian \
  --smoothing_window 9 \
  --min_confidence 0.8 \
  --enable_edge_smoothing true
```

### Fast Processing
```bash
# Speed-optimized settings (lower quality)
python src/run.py \
  --input walk.mp4 \
  --output_dir ./results \
  --pose_model lite \
  --bg_model landscape \
  --smoothing_method moving \
  --smoothing_window 3
```

### Debug Analysis
```bash
# Full debug output with frame samples
python src/run_debug.py \
  --input walk.mp4 \
  --output_dir ./results \
  --debug_frames 50
```

### Performance Optimization

**For faster processing:**
```bash
--pose_model lite --bg_model landscape --smoothing_method moving --smoothing_window 3
```

**For better quality:**
```bash
--pose_model heavy --smoothing_method gaussian --smoothing_window 9 --min_confidence 0.8
```

**For memory-constrained systems:**
- Process shorter video segments
- Use lower resolution input videos
- Close other applications during processing

## License

This project is released under the MIT License. See LICENSE file for details.
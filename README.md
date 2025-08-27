# Human-Centric Video Stabilization

A modular Python pipeline that stabilizes videos of walking humans by keeping the subject fixed at a defined screen location while allowing the background to drift naturally.

## Overview

This system uses **MediaPipe** for pose detection and background segmentation to create smooth, stabilized videos focused on human subjects. The pipeline processes videos offline to achieve optimal quality results.

## Features

- **Hip-center based stabilization** - Keeps the human subject centered using pose keypoint tracking
- **Background removal** - Isolates the human subject from the background
- **Motion smoothing** - Reduces jitter while preserving natural walking motion
- **Side-by-side comparison** - Generates before/after comparison videos
- **Pose data export** - Saves pose keypoints and hip center data as JSON
- **Configurable pipeline** - Extensive customization options for all modules

## Requirements

- **Python 3.10+**
- **CPU-based processing** (GPU optional)
- **Memory**: 8GB+ RAM recommended for video processing

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

Required packages:
- `mediapipe>=0.10.0` - Pose detection and background segmentation
- `opencv-python>=4.8.0` - Video processing and transformations
- `numpy>=1.24.0` - Array operations
- `scipy>=1.10.0` - Signal processing for smoothing

## Usage

### Basic Usage
```bash
python src/run.py --input input_video.mp4 --output_dir ./output
```

### Advanced Usage with Options
```bash
python src/run.py \
  --input walking_video.mp4 \
  --output_dir ./results \
  --pose_model full \
  --hip_calc_method weighted \
  --smoothing_method moving \
  --smoothing_window 7 \
  --border_mode reflect
```

## Configuration Options

### Background Removal
- `--bg_model {general,landscape}` - Model variant (default: general)
  - **general**: Higher accuracy, slower processing
  - **landscape**: Faster processing, optimized for landscape videos

### Pose Detection  
- `--pose_model {lite,full,heavy}` - Model complexity (default: full)
  - **lite**: Fastest, least accurate
  - **full**: Balanced speed/accuracy
  - **heavy**: Most accurate, slowest
- `--hip_calc_method {simple,weighted}` - Hip center calculation (default: simple)
  - **simple**: Average of left/right hip landmarks
  - **weighted**: Confidence-weighted average
- `--min_confidence FLOAT` - Minimum pose detection confidence (default: 0.5)
- `--interpolate_missing {true,false}` - Interpolate missing pose data (default: true)

### Stabilization
- `--smoothing_method {moving,gaussian}` - Smoothing algorithm (default: moving)
  - **moving**: Moving average filter
  - **gaussian**: Gaussian filter (smoother results)
- `--smoothing_window INT` - Smoothing window size (default: 5)
- `--border_mode {reflect,constant}` - Border handling (default: reflect)
  - **reflect**: Mirror pixels at edges
  - **constant**: Fill with solid color
- `--use_cuda {true,false}` - Enable GPU acceleration (default: false)

## Output Files

The pipeline generates three output files in the specified output directory:

1. **`stabilized_output.mp4`** - Hip-center stabilized video
2. **`comparison_output.mp4`** - Side-by-side comparison (original | stabilized)
3. **`pose_data.json`** - Pose detection data in Timeline JSON format

### Pose Data Format
```json
{
  "video_info": {
    "fps": 30,
    "total_frames": 900
  },
  "hip_centers": [
    {
      "frame": 0,
      "timestamp": 0.033,
      "x": 320,
      "y": 240,
      "confidence": 0.85
    }
  ],
  "all_keypoints": [
    {
      "frame": 0,
      "keypoints": [
        {"x": 0.5, "y": 0.6, "z": 0.1, "visibility": 0.9}
      ]
    }
  ]
}
```

## Methods Used

### 1. Background Removal
- **MediaPipe Selfie Segmentation** with General model for person/background separation
- **Temporal smoothing** to reduce mask flickering across frames
- **Bilateral filtering** for smoother segmentation edges
- **Automatic fallback** to Landscape model if segmentation quality drops

### 2. Pose Detection
- **MediaPipe Pose** with Full model for 33-keypoint detection
- **Hip center calculation** using landmarks #23 (left hip) and #24 (right hip)
- **Linear interpolation** for missing or low-confidence pose data
- **Single person tracking** optimized for walking scenarios

### 3. Stabilization Algorithm
- **Hip-center anchoring** - Uses calculated hip center as stabilization point
- **Target positioning** - Centers subject at frame center (configurable)
- **Motion smoothing** - Moving average filter to reduce jitter
- **Affine transformation** - Pure translation using cv2.warpAffine
- **Border reflection** - Maintains visual continuity at frame edges

### 4. Video Rendering
- **Sequential processing** - Offline processing for optimal quality
- **Side-by-side layout** - Horizontal concatenation for comparison
- **MP4 output** - Standard video format with configurable codec

## Pipeline Architecture

The system follows a modular architecture with four main components:

```
Input Video → Background Removal → Pose Detection → Stabilization → Rendering → Output
```

1. **Module 1**: Background removal isolates the human subject
2. **Module 2**: Pose detection tracks hip center position over time  
3. **Module 3**: Stabilization computes and applies translation transforms
4. **Module 4**: Rendering generates final videos and exports data

## Known Limitations

### Processing Constraints
- **Memory intensive** - Loads entire video into RAM for processing
- **CPU-bound processing** - Limited by single-threaded MediaPipe operations
- **Sequential only** - Cannot process multiple videos simultaneously

### Video Requirements
- **Single person focus** - Designed for videos with one primary walking subject
- **Clear subject visibility** - Poor lighting or occlusion may affect pose detection
- **Walking motion** - Optimized for walking gait, may not work well for other activities

### Technical Limitations
- **No real-time processing** - Offline processing only, not suitable for live video
- **Fixed stabilization point** - Hip-center only, cannot stabilize on other body parts
- **2D stabilization only** - No rotation or 3D perspective correction
- **MediaPipe dependencies** - Requires specific MediaPipe model versions

### Output Limitations
- **No cropping/zooming** - Maintains original frame dimensions
- **Border artifacts** - May show reflected or filled borders after stabilization
- **Pose tracking failures** - Stabilization quality depends on consistent pose detection
- **Format constraints** - Outputs MP4 only, limited codec options

### Performance Considerations
- **Video length impact** - Longer videos require proportionally more memory and processing time
- **Resolution dependency** - Higher resolution videos significantly increase processing time
- **Model complexity trade-offs** - More accurate models are substantially slower

## Examples

### Basic Stabilization
```bash
# Process a walking video with default settings
python src/run.py --input walk.mp4 --output_dir ./results
```

### High-Quality Processing  
```bash
# Use heavy pose model with Gaussian smoothing for best quality
python src/run.py \
  --input walk.mp4 \
  --output_dir ./results \
  --pose_model heavy \
  --smoothing_method gaussian \
  --smoothing_window 7
```

### Fast Processing
```bash
# Use lite pose model with landscape background model for speed
python src/run.py \
  --input walk.mp4 \
  --output_dir ./results \
  --pose_model lite \
  --bg_model landscape \
  --smoothing_window 3
```

## Troubleshooting

### Common Issues

**"Could not open video file"**
- Ensure video file path is correct and file is readable
- Verify video format is supported by OpenCV (MP4, AVI, MOV, etc.)

**"No pose detected"**  
- Check video has clear human subject visibility
- Try lowering `--min_confidence` value
- Ensure subject is facing camera during walking

**"Out of memory errors"**
- Reduce video resolution or length
- Process shorter video segments
- Increase system RAM if possible

**Poor stabilization quality**
- Try `--smoothing_method gaussian` for smoother results
- Increase `--smoothing_window` for more stable output
- Use `--pose_model heavy` for better pose accuracy

**TensorFlow warnings**
- oneDNN optimization warnings are automatically suppressed
- MediaPipe feedback manager warnings are normal and can be ignored
- XNNPACK delegate messages indicate CPU optimizations are working

## License

This project is released under the MIT License.
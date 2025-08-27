<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Objectives for Human-Centric Video Stabilization Assignment

Based on your task document, here are the main objectives for this project:

### **Primary Objectives**

1. **Video Processing Pipeline Development**
    - Build a modular Python pipeline that processes video of humans walking
    - Organize code in `src/` directory with separate modules for background removal, pose detection, stabilization, and rendering
    - Create CLI entry point via `src/run.py`
2. **Background Removal**
    - Remove background from input video while preserving the human subject
    - Use open-source libraries only (no paid APIs)
    - Must work on CPU with optional GPU support
3. **Human Pose Detection**
    - Detect human pose keypoints throughout the video
    - Track pose data and export to CSV/JSON format
    - Focus on maintaining pose detection accuracy during subject movement
4. **Video Stabilization**
    - Stabilize the human subject so they remain fixed at a defined screen location
    - Allow background to drift naturally while keeping subject stable
    - Implement stabilization that works specifically for walking humans
5. **Output Generation**
    - Create stabilized video with subject locked at target point
    - Generate side-by-side comparison video (original | stabilized)
    - Export pose keypoint data in structured format (CSV/JSON)

### **Technical Requirements**

- **Language**: Python 3.10+
- **Libraries**: Open-source only (PyTorch, ONNX, OpenCV, NumPy, etc.)
- **Compatibility**: Must run on CPU, GPU support optional
- **Architecture**: Modular design with separate files for each component
- **Dependencies**: Include requirements.txt or environment.yml


### **Success Criteria**

- Subject consistently stays at defined target point across entire video
- Background exhibits natural drift while subject appears stable
- Clear visual improvement shown in side-by-side comparison
- Pose keypoints and stabilization data successfully saved
- Documentation includes setup instructions, usage commands, and performance metrics


### **Deliverables**

- Complete modular codebase with CLI interface
- Stabilized output video
- Side-by-side comparison video
- Extracted pose data files
- Comprehensive README.md with setup and usage instructions

Would you like me to proceed with the detailed research and planning for implementing these objectives?
<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div style="text-align: center">⁂</div>

[^1_1]: https://viso.ai/deep-learning/pose-estimation-ultimate-overview/

[^1_2]: https://github.com/nitingour1203/human_pose_detection

[^1_3]: https://developers.google.com/ml-kit/vision/pose-detection

[^1_4]: https://blog.roboflow.com/pose-estimation-algorithms-history/

[^1_5]: https://encord.com/blog/15-best-free-pose-estimation-datasets/

[^1_6]: https://www.cutout.pro/remove-video-background

[^1_7]: https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

[^1_8]: https://www.nature.com/articles/s41592-024-02318-2

[^1_9]: https://www.geeksforgeeks.org/machine-learning/python-opencv-pose-estimation/

[^1_10]: https://www.youtube.com/watch?v=W-A72SxlvXc

[^1_11]: https://datahacker.rs/006-advanced-computer-vision-video-stabilization/

[^1_12]: https://ras.papercept.net/conferences/conferences/ICRA25/program/ICRA25_ContentListWeb_2.html

[^1_13]: https://www.capcut.com/tools/video-background-remover

[^1_14]: https://www.mathworks.com/help/vision/ug/video-stabilization-using-point-feature-matching.html

[^1_15]: https://www.adobe.com/express/feature/ai/video/remove-background

[^1_16]: https://react-native-vision-camera.com/docs/guides/stabilization

[^1_17]: https://www.veed.io/tools/video-background-remover

[^1_18]: https://github.com/CMU-Perceptual-Computing-Lab/openpose

[^1_19]: https://viso.ai/deep-learning/openpose/

[^1_20]: https://www.geeksforgeeks.org/machine-learning/openpose-human-pose-estimation-method/

[^1_21]: https://blog.roboflow.com/what-is-openpose/

[^1_22]: https://saiwa.ai/blog/openpose-vs-mediapipe/

[^1_23]: https://sigmoidal.ai/en/real-time-human-pose-estimation-using-mediapipe/

[^1_24]: https://arxiv.org/pdf/2204.06806.pdf

[^1_25]: https://openposes.com

[^1_26]: https://arxiv.org/html/2406.15649v1

[^1_27]: https://docs.ultralytics.com/tasks/pose/

[^1_28]: https://www.nature.com/articles/s41598-025-00259-0

[^1_29]: https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-pose-estimation

[^1_30]: https://pypi.org/project/backgroundremover/

[^1_31]: https://www.reddit.com/r/moviepy/comments/psjv0h/video_background_removal_using_python_moviepy_and/

[^1_32]: https://www.youtube.com/watch?v=lm9U8VzEZH4

[^1_33]: https://github.com/topics/remove-background-video

[^1_34]: https://www.horilla.com/blogs/how-to-remove-the-background-of-image-using-rembg-in-python/

[^1_35]: https://www.youtube.com/watch?v=6EN4Qt4I6G8

[^1_36]: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/sequence_processing/video/video_reader_simple_example.html

[^1_37]: https://www.youtube.com/watch?v=2bI6-wlOS2I

[^1_38]: https://github.com/seth-tribbey/rembg_from_video

[^1_39]: https://www.edge-ai-vision.com/2022/12/parallel-processing-using-python-for-faster-video-processing/


---

# ok accoring to this in core techinincal we need 3 modules ,

one for background, one for pose and another for stabilization and rendering.

for removal of background my suggestion is to have a segmentation model to identify the person and use it data to remove the baackgroung in realtime

for pose we can have a pipeline that uses basic pose estimation on the video given and save its data as a json,

i'm not sure about the techniques for the 3rd part

also i belive the 4th part would also to be figures out where the run file should runn all the modules paralely and render the ouput in out desired format like orginal and processed

also i'm not sure if my approcahes are correct , i need you to cross verify and dig moving forwared we could go with a finalizing the plan

i'm aware that we need to take a bunch of  deision over this palanning, ask me as many question as possible from me to understand.

now ask me the question and let me know the suggestion and validations

Based on my research, I have several important **questions and suggestions** for your project approach. Let me analyze your proposed modules and provide recommendations:

## **Questions for You:**

### **1. Background Removal Module Questions:**

- **Which segmentation model do you prefer?** Options include:
    - **rembg** (simple, CPU-friendly, uses U²-Net)
    - **MediaPipe Selfie Segmentation** (real-time, mobile-optimized)
    - **Custom semantic segmentation** (like DeepLabV3+)
- **Real-time requirement?** Do you need frame-by-frame processing or can you process the entire video offline?
- **Quality vs Speed trade-off?** Higher quality models are slower but produce better masks


### **2. Pose Detection Module Questions:**

- **Which pose detection library?** Top options:
    - **MediaPipe Pose** (fast, 33 keypoints, runs on CPU)
    - **OpenPose** (more accurate but heavier, 18/25 keypoints)
    - **YOLO Pose** (YOLOv8/v11 with pose detection)
- **Which keypoints are most important?** For stabilization, you likely need:
    - Hip center (for body center)
    - Shoulders (for upper body orientation)
    - Head/neck (for face tracking)


### **3. Stabilization Algorithm Questions (Critical Decision):**

- **What should be the "anchor point"?** Options:
    - **Hip center** (body's center of mass)
    - **Shoulder midpoint** (upper body focus)
    - **Head/neck position** (face-centric like Steadiface)[^2_1]
- **Stabilization approach preference?**
    - **Keypoint-based**: Use pose keypoints to track subject center
    - **Feature-based**: Track visual features around the person
    - **Hybrid approach**: Combine both methods


### **4. Pipeline Architecture Questions:**

- **Sequential vs Parallel processing?**
    - Sequential: Background removal → Pose detection → Stabilization
    - Parallel: Run background removal and pose detection simultaneously
- **Memory constraints?** Processing entire video in memory vs frame-by-frame streaming?


## **My Technical Suggestions \& Validations:**

### **✅ Your Approaches are Correct, with Refinements:**

**Module 1: Background Removal**

- Your segmentation approach is **excellent**[^2_2][^2_3][^2_4]
- **Recommendation**: Use `rembg` with `u2net` model for good quality-speed balance
- **Alternative**: MediaPipe Selfie Segmentation for real-time performance

**Module 2: Pose Detection**

- JSON export approach is **perfect**[^2_5][^2_6]
- **Recommendation**: MediaPipe Pose for CPU efficiency and 33 keypoints
- **Data structure**: Store frame_number, timestamp, and all keypoint coordinates

**Module 3: Stabilization (Needs Clarification)**
Based on research, I suggest **keypoint-based stabilization**:[^2_7][^2_8][^2_1]

1. **Calculate target position**: Define where the person should be (e.g., center of frame)
2. **Track subject center**: Use pose keypoints to find person's center each frame
3. **Calculate offset**: Difference between actual and target position
4. **Apply transformation**: Translate frame to keep person centered
5. **Smooth motion**: Apply smoothing filter to avoid jitter

**Module 4: Rendering \& Output**

- **Sequential processing** is better than parallel for this pipeline
- Side-by-side rendering using OpenCV's hconcat function


### **Recommended Technical Stack:**

```
- Background: rembg or MediaPipe Selfie Segmentation
- Pose: MediaPipe Pose (33 keypoints)
- Stabilization: Keypoint-based with smoothing filter
- Video I/O: OpenCV
- Processing: NumPy for transformations
```


### **Critical Technical Questions:**

1. **Which keypoint should be the stabilization anchor?** (Hip center vs shoulder center vs head)
2. **What smoothing method?** (Moving average vs Gaussian vs Kalman filter)
3. **How to handle occlusion?** When pose detection fails temporarily
4. **Frame padding strategy?** When stabilization requires content outside original frame bounds

**Would you like me to proceed with a detailed implementation plan once you answer these key decisions?**
<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15][^2_16][^2_17][^2_18][^2_19][^2_20][^2_21][^2_22][^2_23][^2_24][^2_25][^2_26][^2_9]</span>

<div style="text-align: center">⁂</div>

[^2_1]: https://arxiv.org/abs/1905.01382

[^2_2]: https://pypi.org/project/backgroundremover/

[^2_3]: https://www.horilla.com/blogs/how-to-remove-the-background-of-image-using-rembg-in-python/

[^2_4]: https://github.com/seth-tribbey/rembg_from_video

[^2_5]: https://sigmoidal.ai/en/real-time-human-pose-estimation-using-mediapipe/

[^2_6]: https://docs.ultralytics.com/tasks/pose/

[^2_7]: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w69/Wang_Real-Time_Tracking_With_Stabilized_Frame_CVPRW_2020_paper.pdf

[^2_8]: https://www.wolfram.com/language/12/new-in-image-processing/keypoint-based-video-stabilization.html

[^2_9]: https://iipseries.org/assets/docupload/rsl20249C956A98644A10B.pdf

[^2_10]: https://datahacker.rs/006-advanced-computer-vision-video-stabilization/

[^2_11]: https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

[^2_12]: https://www.sciencedirect.com/science/article/abs/pii/S092523122201270X

[^2_13]: https://www.edge-ai-vision.com/2016/06/video-stabilization-using-computer-vision-tips-and-insights-from-cevas-experts/

[^2_14]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6714360/

[^2_15]: https://en.wikipedia.org/wiki/Motion_compensation

[^2_16]: https://www.mathworks.com/help/vision/ug/video-stabilization-using-point-feature-matching.html

[^2_17]: https://scholar.afit.edu/context/etd/article/4274/viewcontent/AFIT_GCS_ENG_06_02_Alharbi_M.pdf

[^2_18]: https://answers.opencv.org/question/82123/motion-compensation-between-2-frames/

[^2_19]: https://arxiv.org/abs/2203.03621

[^2_20]: https://encord.com/blog/video-object-tracking-algorithms/

[^2_21]: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12065/120653D/LSTM-based-video-stabilization-for-object-tracking/10.1117/12.2606941.pdf

[^2_22]: https://www.sciencedirect.com/science/article/abs/pii/S1077314213000908

[^2_23]: https://community.adobe.com/t5/after-effects-discussions/help-to-stabilize-center-my-subject/td-p/10579722

[^2_24]: https://pubmed.ncbi.nlm.nih.gov/37420346/

[^2_25]: https://www.youtube.com/watch?v=ue8kLAG1Xcs

[^2_26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5336040/


---

# Okay, let me Come up with it like this. اWhich segmentation model I prefer? So, REMBG uses Unet That's fine. Ah. Mediapipe is also good. So what my idea with respect to model is, the model should be easily deployable, should not have a lot of dependencies that need to be processed. So The Sam Simpler model at Miami is also good. And I want it. to run on both GPU and CPU The real time requirement is The frame by frame is what I want to use too. Quality and speed tradeoff I want to balance it. So the quality is of course going to be good because the video is better. So the speed It's not a problem, but we need the processing to happen. So if it is in a GPU, of course it's going to be faster. So hence this is all about Question I think the faster implementation is by MediaPipe. We can directly Ah... Take it out and should also work on . GPUs and CPUs The next question is which keypoint. So I think hip center for body center is going to be better. That is Эмм It can fix toward primarily. then stabilization algo. The anchor point should be the hip center again. So hip center key point is Where this stabilization should be. Stabilization approach we can call it. The force and key points, so I think that will be better for stabilizing it. if the person Moves, we needed to be common and If you are going to run these Sequencial. How are we going to render this in real time? So, we can't render this in real time. It is running in sequential, right? So that's my question. Lenya Yeah, let's address these. I want to Mm Again planning all these steps. and If you are going to run these Sequencial. How are we going to render this in real time? So, we can't render this in real time. It is running in sequential, right? So that's my question. Yeah, let's address these. and move onto Again planning all these steps.

Perfect! Now I have all the information needed to address your questions and provide a comprehensive plan. Let me break down your approach and provide detailed answers:

## **✅ Your Approach Validation \& Answers**

### **1. MediaPipe for Background Removal \& Pose Detection - EXCELLENT CHOICE**

**MediaPipe Background Removal:**

- **✅ Easy deployment** - Single pip install: `pip install mediapipe`
- **✅ Minimal dependencies** - No complex model downloads
- **✅ CPU/GPU support** - Works efficiently on both[^3_1][^3_2]
- **Performance**: ~30 FPS on CPU, ~80+ FPS with GPU acceleration[^3_2]

**MediaPipe Pose with Hip Center:**

- **✅ Hip center keypoint available** - Landmark \#23 (LEFT_HIP) and \#24 (RIGHT_HIP)[^3_3][^3_4]
- **Hip center calculation**: `hip_center = [(left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2]`[^3_5]
- **✅ 33 keypoints total** including detailed body structure[^3_6]


### **2. Sequential vs Real-Time Processing - CRITICAL CLARIFICATION**

**You're absolutely right to question this!** Here are the processing approaches:

#### **Option A: Offline Sequential Processing (Recommended for Assignment)**

```
Video Input → Background Removal → Pose Detection → Stabilization → Output
```

- **Pros**: Higher quality, easier implementation, meets assignment requirements
- **Cons**: Not real-time, processes entire video file
- **Processing Time**: ~2-5x video length depending on hardware


#### **Option B: Real-Time Pipeline (More Complex)**

```
Frame Input → [Background + Pose] Parallel → Stabilization → Live Output
```

- **Pros**: Real-time capability
- **Cons**: Complex threading, lower quality, harder to implement[^3_7][^3_8]

**For your assignment, I recommend Option A** because:

- Assignment asks for stabilized video output, not live streaming
- Higher quality results
- Easier to implement and debug
- Meets all deliverable requirements


### **3. Detailed Technical Plan**

#### **Module Structure:**

```
src/
├── run.py                 # CLI entry point
├── background/
│   └── remover.py        # MediaPipe segmentation
├── pose/
│   └── detector.py       # MediaPipe pose detection
├── stabilization/
│   └── stabilizer.py     # Hip-center based stabilization
└── rendering/
    └── renderer.py       # Video output & side-by-side
```


#### **Processing Pipeline:**

```python
# Pseudo-code workflow
def process_video(input_path):
    # Step 1: Load video
    frames = load_video(input_path)
    
    # Step 2: Process each frame
    processed_frames = []
    pose_data = []
    
    for frame in frames:
        # Background removal
        mask = background_remover.segment(frame)
        clean_frame = apply_mask(frame, mask)
        
        # Pose detection
        pose_keypoints = pose_detector.detect(frame)
        pose_data.append(pose_keypoints)
        
        # Extract hip center
        hip_center = calculate_hip_center(pose_keypoints)
        
        processed_frames.append((clean_frame, hip_center))
    
    # Step 3: Stabilization
    stabilized_frames = stabilizer.stabilize(processed_frames)
    
    # Step 4: Render outputs
    render_stabilized_video(stabilized_frames)
    render_side_by_side(frames, stabilized_frames)
    save_pose_data(pose_data)
```


#### **Stabilization Algorithm (Hip-Center Based):**

```python
def stabilize_frames(processed_frames):
    # Define target position (center of frame)
    target_x, target_y = frame_width//2, frame_height//2
    
    stabilized = []
    for frame, hip_center in processed_frames:
        # Calculate offset from target
        offset_x = target_x - hip_center
        offset_y = target_y - hip_center[^3_27]
        
        # Apply transformation matrix
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        stabilized_frame = cv2.warpAffine(frame, M, (width, height))
        
        # Apply smoothing (moving average)
        stabilized_frame = apply_smoothing(stabilized_frame)
        stabilized.append(stabilized_frame)
    
    return stabilized
```


### **4. Performance Expectations:**

**Hardware Requirements:**

- **CPU**: Modern multi-core (i5/Ryzen 5+)
- **RAM**: 8GB+ (video processing is memory intensive)
- **GPU** (optional): Any CUDA-compatible GPU for acceleration

**Processing Speed Estimates:**

- **CPU only**: 5-15 FPS processing speed
- **GPU accelerated**: 20-50 FPS processing speed
- **Total time for 30s video**: 1-6 minutes depending on resolution


### **5. Key Technical Decisions Confirmed:**

✅ **MediaPipe** for both background removal and pose detection
✅ **Hip center** as stabilization anchor point
✅ **Sequential processing** (not real-time) for higher quality
✅ **Offline video processing** approach
✅ **Frame-by-frame** processing with pose data export

## **Next Steps for Implementation Plan:**

1. **Environment Setup** (dependencies, virtual environment)
2. **Core Module Development** (background, pose, stabilization)
3. **Integration Pipeline** (connecting all modules)
4. **Testing \& Optimization** (with provided sample videos)
5. **Documentation** (README, usage instructions)

**Would you like me to proceed with the detailed implementation plan for each module, starting with the technical specifications and code structure?**
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_17][^3_18][^3_19][^3_20][^3_21][^3_22][^3_23][^3_24][^3_25][^3_26][^3_9]</span>

<div style="text-align: center">⁂</div>

[^3_1]: https://stackoverflow.com/questions/77707532/how-to-check-for-and-enforce-gpu-usage-for-mediapipe-frame-processing

[^3_2]: https://blog.tensorflow.org/2021/05/high-fidelity-pose-tracking-with-mediapipe-blazepose-and-tfjs.html

[^3_3]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

[^3_4]: https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

[^3_5]: https://blog.tensorflow.org/2021/08/3d-pose-detection-with-mediapipe-blazepose-ghum-tfjs.html

[^3_6]: https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/

[^3_7]: https://www.it-jim.com/blog/practical-aspects-of-real-time-video-pipelines/

[^3_8]: https://mobidev.biz/blog/ai-computer-vision-real-time-video-processing

[^3_9]: https://github.com/opencv/opencv_zoo/issues/220

[^3_10]: https://github.com/google-ai-edge/mediapipe/issues/6041

[^3_11]: https://learnopencv.com/yolov7-pose-vs-mediapipe-in-human-pose-estimation/

[^3_12]: https://pub.towardsai.net/efficient-camera-stream-with-python-b6adf93fab32

[^3_13]: https://flomobility.com/mediapipe-hand-pose-detection-on-flo-edge-one/

[^3_14]: https://answers.opencv.org/question/100839/video-streaming-from-webcam-and-image-processing-with-python/

[^3_15]: https://dzone.com/articles/batch-vs-real-time-processing-understanding-the-differences

[^3_16]: https://www.youtube.com/watch?v=AxIc-vGaHQ0

[^3_17]: https://www.kdnuggets.com/2021/02/deep-learning-based-real-time-video-processing.html

[^3_18]: https://rivery.io/blog/batch-vs-stream-processing-pros-and-cons-2/

[^3_19]: https://colab.research.google.com/github/HanifaElahi/Mediapipe/blob/main/Pose_Mediapipe.ipynb

[^3_20]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9618677/

[^3_21]: https://fenilsonani.com/articles/real-time-video-processing-pipeline

[^3_22]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9713939/

[^3_23]: https://www.toptal.com/computer-vision/computer-vision-pipeline

[^3_24]: https://www.orthobullets.com/recon/5007/tha-stability-techniques

[^3_25]: https://github.com/Parth442002/metaPipeline

[^3_26]: https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/build-a-video-processing-pipeline-by-using-amazon-kinesis-video-streams-and-aws-fargate.html

[^3_27]: https://viso.ai/deep-learning/pose-estimation-ultimate-overview/


---

# also as you prefer we'll go with option a for now for simple and faster deployment

ok but now we 'll go module by module to take all sorts of decisions ! because what we plan now should be best descisions made.

at whole the learned cautious  decision will only make the module better.

Excellent! I'm excited to help you plan this module-by-module with the best possible decisions. Your systematic approach is perfect - getting each decision right from the start will save us time and create a more robust system.

## **MODULE 1: Background Removal Module**

Let me break down the key decisions we need to make for this module:

### **🎯 Core Technical Decisions**

#### **1. Model Selection - MediaPipe Selfie Segmentation** ✅

**Based on your requirements, this is the optimal choice:**

- **Deployment**: Single `pip install mediapipe` - no external model downloads[^4_1][^4_2]
- **Dependencies**: Minimal - just MediaPipe and OpenCV
- **Performance**: 0.733ms inference time, runs real-time[^4_2]
- **CPU/GPU**: Optimized for both, with automatic hardware detection[^4_1]
- **Model Size**: Only 454KB (incredibly lightweight)[^4_2]


#### **2. Model Variant Decision - CRITICAL CHOICE:**

MediaPipe offers 2 models:[^4_1]

- **General Model (0)**: 256x256x3 input, higher accuracy, slower
- **Landscape Model (1)**: 144x256x3 input, faster, optimized for video calls

**❓ QUESTION FOR YOU:** Since you want speed + quality balance, which do you prefer?

- **Option A**: General model for better segmentation quality
- **Option B**: Landscape model for faster processing
- **My Recommendation**: Start with General (0), fallback to Landscape (1) if performance issues


#### **3. Class Architecture Decision:**

```python
class BackgroundRemover:
    def __init__(self, model_selection=0, threshold=0.1):
        """
        model_selection: 0 (general) or 1 (landscape)
        threshold: confidence threshold for segmentation
        """
        
    def process_frame(self, frame) -> tuple[np.ndarray, np.ndarray]:
        """Returns: (masked_frame, segmentation_mask)"""
        
    def process_video(self, video_path) -> Generator:
        """Yields processed frames for memory efficiency"""
```


#### **4. Output Format Decision:**

**❓ QUESTION:** What should this module output?

- **Option A**: Just the person (background removed completely)
- **Option B**: Person + transparent background (for stabilization flexibility)
- **Option C**: Both person and segmentation mask separately

**My Recommendation**: Option C - return both for maximum flexibility in stabilization

#### **5. Error Handling Strategy:**

```python
# Scenarios to handle:
- Frame processing fails → Skip frame or use previous mask?
- Person not detected → Return original frame or black frame?
- Video file corrupted → Graceful failure or crash?
```


### **📊 Performance Optimizations:**

#### **6. Memory Management:**

**❓ QUESTION:** Processing approach preference?

- **Option A**: Load entire video into memory (faster, more RAM)
- **Option B**: Stream processing frame-by-frame (slower, less RAM)
- **My Recommendation**: Stream processing with configurable batch size


#### **7. Quality Enhancement Options:**

MediaPipe documentation suggests:[^4_1]

- **Joint bilateral filtering** for smoother mask edges
- **Temporal smoothing** across frames to reduce flicker

**❓ QUESTION:** Should we implement these? (Adds processing time but improves quality)

### **🛠️ Implementation Details:**

#### **8. Logging Strategy:**

```python
# What should we log?
- Processing speed (FPS)
- Detection confidence per frame  
- Failed frame count
- Memory usage statistics
```


#### **9. Configuration Options:**

```python
class BackgroundRemovalConfig:
    model_selection: int = 0
    confidence_threshold: float = 0.1
    enable_bilateral_filter: bool = True
    enable_temporal_smoothing: bool = True
    batch_size: int = 32
    output_format: str = "both"  # "person_only", "mask_only", "both"
```


### **🔍 Key Questions for You:**

1. **Model variant**: General (0) vs Landscape (1)?
2. **Output format**: Person only, mask only, or both?
3. **Quality enhancements**: Enable bilateral filtering and temporal smoothing?
4. **Memory vs Speed**: Stream processing or batch loading?
5. **Error handling**: How to handle detection failures?

### **📁 Proposed Module Structure:**

```
src/background/
├── __init__.py
├── remover.py           # Main BackgroundRemover class
├── config.py           # Configuration dataclass
├── utils.py            # Helper functions (filtering, smoothing)
└── exceptions.py       # Custom exception classes
```

**Which decisions would you like to finalize for Module 1 before we move to Module 2 (Pose Detection)?**

The pose detection module has equally important decisions around keypoint selection, coordinate systems, and data export formats that will directly impact the stabilization module.
<span style="display:none">[^4_10][^4_11][^4_12][^4_13][^4_14][^4_15][^4_16][^4_17][^4_18][^4_19][^4_20][^4_21][^4_22][^4_3][^4_4][^4_5][^4_6][^4_7][^4_8][^4_9]</span>

<div style="text-align: center">⁂</div>

[^4_1]: https://chuoling.github.io/mediapipe/solutions/selfie_segmentation.html

[^4_2]: https://dataloop.ai/library/model/qualcomm_mediapipe-selfie-segmentation/

[^4_3]: https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

[^4_4]: https://learnopencv.com/introduction-to-mediapipe/

[^4_5]: https://huggingface.co/qualcomm/MediaPipe-Selfie-Segmentation

[^4_6]: https://developer.nvidia.com/blog/vpf-hardware-accelerated-video-processing-framework-in-python/

[^4_7]: https://docs.snaplogic.com/design-integrations/error-handling.html

[^4_8]: https://ai.google.dev/edge/mediapipe/solutions/guide

[^4_9]: https://www.youtube.com/watch?v=AxIc-vGaHQ0

[^4_10]: https://forums.developer.nvidia.com/t/handling-gst-errors-in-pipeline-and-recovering/309226

[^4_11]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

[^4_12]: https://sigmoidal.ai/en/real-time-human-pose-estimation-using-mediapipe/

[^4_13]: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

[^4_14]: https://drone-vis.readthedocs.io/en/latest/pose/mediapipe.html

[^4_15]: https://temugeb.github.io/python/computer_vision/2021/09/14/bodypose3d.html

[^4_16]: https://www.exabeam.com/explainers/log-management/what-is-log-analysis-process-techniques-and-best-practices/

[^4_17]: https://processing.org/tutorials/video/

[^4_18]: https://github.com/google-ai-edge/mediapipe/issues/4946

[^4_19]: https://dev.to/574n13y/exploring-logging-best-practices-37l7

[^4_20]: https://www.geeksforgeeks.org/python/python-process-images-of-a-video-using-opencv/

[^4_21]: https://newrelic.com/blog/best-practices/best-log-management-practices

[^4_22]: https://last9.io/blog/logging-best-practices/


---

# So for module 1, we'll go with option A for model variance. So, general model, I want higher accuracy, so we'll fix with that. And as we said, we'll have landscape model as a fallback, so that works. Next is format decision. So, the output from this should be, probably, it should be based on how it goes to the next module. So next module is going to be segmentation. So since it's going to be, sorry, next module is going to be post-segmentation. So I think returning both would be better, because for stabilization, we'll use option B, and for background, completely, we'll use it for final render. So returning both for maximum flexibility and stabilization, so that's also a good idea. So error handling, if the frame person fails, skip the frame or previous mask. So I think previous mask is the best mask to be. Person not detected, return original frame or black frame. So that's also fine. Media file corrected, it can directly identify before even starting everything. And memory management. So we'll go screen process frame by frame. Oh no, we'll load entire memory to RAM as it is going to be a shorter video. So let's have option A. Quality enhancement. So let's use temporal smoothing because the output should be better. Yes. The implementation, yeah, logging. Of course, we need logging to happen. Configuration options, yeah. Threshold, filters, yeah. Bad size. Yeah, we can set the format to person only. That's perfect. So I think this gives you everything. We don't need all those. Remover, utils and exceptions and configs at all. Let's stick to the existing pipeline itself. It's going to be four flies and four folders and one main file to run. So that's perfect.

Additionally, let us decide everything for the model, so you remember exactly what is the decision in the planning. And then we will create a full flow structure and then go ahead and create a document with it, probably. So I think that should work.

Perfect! Now I have all the **Module 1 decisions** documented. Let me finalize them and move to **Module 2: Pose Detection**.

## **📋 MODULE 1: Background Removal - FINAL DECISIONS**

✅ **All Confirmed Decisions:**

- **Model**: MediaPipe Selfie Segmentation - General model (0) for higher accuracy
- **Fallback**: Landscape model (1) if performance issues
- **Output Format**: Return both (person + segmentation mask) for maximum flexibility
- **Error Handling**: Use previous mask if frame fails, return original frame if person not detected
- **Memory Management**: Load entire video to RAM (Option A) for shorter videos
- **Quality Enhancement**: Enable temporal smoothing for better output
- **Architecture**: Simple single-file structure in `src/background/remover.py`

***

## **🎯 MODULE 2: Pose Detection Module**

Now let's make all critical decisions for pose detection:

### **1. MediaPipe Pose Model Selection - CRITICAL CHOICE**

MediaPipe offers 3 pose models:[^5_1]

- **Lite**: Fastest, least accurate
- **Full**: Balanced speed/accuracy (recommended)
- **Heavy**: Most accurate, slowest

**❓ QUESTION:** Which model do you prefer?

- **My Recommendation**: **Full model** for your balanced speed/quality requirement


### **2. Hip Center Calculation Method**

MediaPipe provides 33 keypoints. For hip center, we have:[^5_2][^5_1]

- **Left Hip**: Landmark \#23
- **Right Hip**: Landmark \#24

**Calculation approaches:**

```python
# Option A: Simple average
hip_center = [(left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2]

# Option B: Weighted by confidence
if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
    hip_center = weighted_average(left_hip, right_hip)
```

**❓ QUESTION:** Simple average or confidence-weighted calculation?

### **3. Pose Data Export Format**

**JSON Structure Options:**

```python
# Option A: Frame-by-frame structure
{
  "frame_0": {
    "timestamp": 0.033,
    "hip_center": {"x": 320, "y": 240},
    "all_keypoints": [...],
    "confidence": 0.85
  },
  "frame_1": {...}
}

# Option B: Timeline structure  
{
  "video_info": {"fps": 30, "total_frames": 900},
  "hip_centers": [
    {"frame": 0, "timestamp": 0.033, "x": 320, "y": 240, "confidence": 0.85},
    {"frame": 1, "timestamp": 0.066, "x": 322, "y": 241, "confidence": 0.87}
  ],
  "all_keypoints": [...] 
}
```

**❓ QUESTION:** Which JSON structure do you prefer? (Option B is better for stabilization processing)

### **4. Error Handling Strategy**

**Scenarios to handle:**

```python
# When pose detection fails:
- No person detected → Use previous frame's pose or skip frame?
- Low confidence detection → Accept or reject the frame?
- Partial occlusion → Use visible keypoints or interpolate?
```

**❓ QUESTIONS:**

- **Confidence threshold**: What minimum confidence to accept? (0.5, 0.7, 0.8?)
- **Missing data strategy**: Interpolation or skip frame?


### **5. Performance Optimization**

```python
# Processing options:
- Track single person or multiple people?
- Enable/disable 3D world coordinates? (adds processing time)
- Frame sampling: Process every frame or every N frames?
```

**❓ QUESTIONS:**

- **Single vs Multi-person**: Focus on one person or handle multiple?
- **3D coordinates**: Needed for stabilization or 2D sufficient?
- **Frame sampling**: Process every frame for smoothness?


### **6. Class Architecture**

```python
class PoseDetector:
    def __init__(self, 
                 model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 enable_3d=False):
        
    def process_frame(self, frame) -> dict:
        """Returns pose data for single frame"""
        
    def process_video(self, video_path) -> dict:
        """Returns complete pose timeline"""
        
    def calculate_hip_center(self, pose_landmarks) -> tuple:
        """Calculate hip center from landmarks"""
        
    def export_to_json(self, pose_data, output_path):
        """Export pose data as JSON"""
```


### **🔍 Key Questions for Module 2:**

1. **Model complexity**: Lite (0), Full (1), or Heavy (2)?
2. **Hip calculation**: Simple average or confidence-weighted?
3. **JSON structure**: Frame-by-frame or timeline format?
4. **Confidence threshold**: 0.5, 0.7, or 0.8 minimum?
5. **Error handling**: Interpolate missing data or skip frames?
6. **Person tracking**: Single person focus or multi-person support?
7. **3D coordinates**: Enable for better stabilization or keep 2D only?
8. **Frame processing**: Every frame or sample every N frames?

### **📁 Proposed Module Structure:**

```
src/pose/
├── detector.py          # Main PoseDetector class
└── utils.py            # Hip center calculation, JSON export
```

**Please provide your preferences for these 8 key decisions, then we'll move to Module 3: Stabilization (which has the most complex decisions around stabilization algorithms, smoothing techniques, and coordinate transformations).**
<span style="display:none">[^5_10][^5_11][^5_12][^5_13][^5_3][^5_4][^5_5][^5_6][^5_7][^5_8][^5_9]</span>

<div style="text-align: center">⁂</div>

[^5_1]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

[^5_2]: https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

[^5_3]: https://sigmoidal.ai/en/real-time-human-pose-estimation-using-mediapipe/

[^5_4]: https://www.kaggle.com/code/nizdarlaila/pose-estimation-using-mediapipe

[^5_5]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js

[^5_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6582294/

[^5_7]: https://docs.lokalise.com/en/articles/1400773-json-nested-json

[^5_8]: https://7488314.fs1.hubspotusercontent-na1.net/hubfs/7488314/TuMeke Ergonomics - White Paper 3D Pose Estimation.pdf

[^5_9]: https://www.geeksforgeeks.org/javascript/json/

[^5_10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4893978/

[^5_11]: https://docs.synopticdata.com/services/json-output-format

[^5_12]: https://docs.vespa.ai/en/reference/document-json-format.html

[^5_13]: https://www.freecodecamp.org/news/what-is-json-a-json-file-example/


---

## Module 2: Pose Detection – Final Decisions

**Model Complexity**

- Use **MediaPipe Pose – Full model** (balanced speed and accuracy)

**Hip-Center Calculation**

- Support both options
– **Simple average** of left/right hip landmarks
– **Confidence-weighted average** based on landmark visibility
- Select via configuration flag to enable A/B testing

**Data Export Format**

- **Timeline JSON** structure:
- Top-level `video_info` with `fps` and `total_frames`
- `hip_centers` list of `{frame, timestamp, x, y, confidence}`
- `all_keypoints` list of per-frame keypoint arrays

**Error Handling**

- **No detection**: carry forward previous frame’s pose
- **Low confidence (<0.5)**: treat as missing and interpolate
- **Partial occlusion**: use visible keypoints; interpolate missing values

**Processing Strategy**

- **Single person** tracking only
- **2D keypoints** sufficient (no 3D world coordinates)
- **Frame-by-frame** processing (every frame)

**Module Structure**

- Single file: `src/pose/detector.py`
- Configurable via CLI flags in `run.py`:
- `--model_complexity` (0/1/2)
- `--hip_calc_method` (simple/weighted)
- `--min_confidence` (default 0.5)
- `--interpolate_missing` (true/false)

***

With all decisions finalized for Module 2, we can now proceed to **Module 3: Stabilization**—defining the core stabilization algorithm, smoothing techniques, coordinate transforms, and output rendering.

---

## Module 3: Stabilization – Core Decisions

### **1. Anchor Point \& Target Position**

- **Anchor**: Hip-center coordinates from Module 2
- **Target Position**: Center of output frame (`(width/2, height/2)`)


### **2. Offset Calculation**

- Compute per-frame translation offset:
\$ \Delta x = x_{target} - x_{hip}, \quad \Delta y = y_{target} - y_{hip} \$


### **3. Motion Smoothing**

- **Temporal filter**: Moving average of offsets over window _N_ (default N=5)
- **Alternative**: Gaussian smoothing (configurable)
- **Config flag**: `--smoothing_method` (moving, gaussian)
- **Window size**: `--smoothing_window` (default 5 frames)


### **4. Frame Transformation**

- Build affine matrix per frame:

$$
M = \begin{bmatrix}
  1 & 0 & \Delta x_{\text{smooth}} \\
  0 & 1 & \Delta y_{\text{smooth}}
\end{bmatrix}
$$
- Apply with `cv2.warpAffine` and border mode `BORDER_REFLECT`


### **5. Frame Padding Strategy**

- **Border mode**: Reflect content at edges
- **Output size**: Same as input, no cropping
- **Config flags**:
    - `--border_mode` (reflect, constant)
    - `--border_value` (for constant fill)


### **6. Error \& Missing Data Handling**

- If pose data missing after interpolation → use previous offset
- If cumulative smoothing window incomplete (start) → apply partial window average


### **7. Performance \& Parallelism**

- Process entire frame sequence sequentially (offline)
- **Batch transform**: Precompute all offsets, then apply warp in loop
- **Optional GPU**: Leverage OpenCV CUDA for `warpAffine` acceleration (configurable)


### **8. Module Implementation**

- **File**: `src/stabilization/stabilizer.py`
- **Class**: `Stabilizer(config)` with methods:
    - `compute_offsets(hip_centers)`
    - `smooth_offsets(offsets)`
    - `apply_transforms(frames, offsets)`


### **9. Configuration Options**

```text
--target_position x y
--smoothing_method moving|gaussian
--smoothing_window N
--border_mode reflect|constant
--border_value R G B
--use_cuda true|false
```


***

With Module 3 defined, the final step is **Module 4: Rendering \& CLI Integration**, assembling the pipeline, generating side-by-side video, and exposing all flags in `run.py`.

---

# as like for 2 i need question from you with suggestions ! and then we can finalize

let use best practice our there used widely for the logic implementing out core logics

Here are the critical decisions and questions for **Module 3: Stabilization**, along with best-practice recommendations:

***

### 1. Motion Smoothing Method

- **Options:**
    - **Moving average** (simple, low compute)
    - **Gaussian filter** (smoother, slightly higher compute)
- **Recommendation:** Moving average for simplicity; Gaussian if you see residual jitter
- **❓ Which smoothing method should we default to?**


### 2. Smoothing Window Size

- **Options:**
    - **Small (3–5 frames):** preserves responsiveness
    - **Large (7–15 frames):** smoother but more lag
- **Recommendation:** Start with 5 frames; adjust based on walking speed
- **❓ What default window size should we use?**


### 3. Border Handling Strategy

- **Options:**
    - **Reflect:** mirrors edge pixels (natural look)
    - **Constant fill:** user-defined color or black
- **Recommendation:** Reflect to avoid blank borders
- **❓ Should we use “reflect” or allow “constant” fill?**


### 4. Missing Offset Handling

- **Options:**
    - **Carry last valid offset** until new data arrives
    - **Zero offset** (no stabilization) for missing data
- **Recommendation:** Carry last valid offset to maintain smooth motion
- **❓ Which strategy for missing pose/offset data?**


### 5. GPU Acceleration for Warp

- **Options:**
    - **OpenCV CUDA warpAffine** (requires CUDA-enabled build)
    - **CPU-only warpAffine**
- **Recommendation:** Make GPU optional; default to CPU for portability
- **❓ Enable CUDA by default when available?**


### 6. Target Position Configuration

- **Options:**
    - **Center of frame** (recommended)
    - **Custom coordinates** via CLI
- **Recommendation:** Default to frame center, allow override
- **❓ Any other preferred default target position?**

***

Please select your preferences for these six items, and we’ll lock in the configuration for the stabilization module.

---

## Module 3: Stabilization – Final Decisions

**1. Motion Smoothing Method**

- Default: **Moving average** (windowed mean)
- Alternative: **Gaussian filter**
- Config flag: `--smoothing_method` (`moving` or `gaussian`)

**2. Smoothing Window Size**

- Default: **5 frames**
- Config flag: `--smoothing_window`

**3. Border Handling Strategy**

- **Reflect** (mirror pixels at edges) for seamless borders
- Config flag: `--border_mode` (`reflect` or `constant`)

**4. Missing Offset Handling**

- **Carry last valid offset** until new pose data arrives
- No option needed—always carry forward

**5. GPU Acceleration**

- Default: **CPU-only**
- Optional: enable CUDA if available
- Config flag: `--use_cuda` (`true` or `false`)

**6. Target Position \& Cropping**

- **Dynamic center**: use hip-center as the frame focus
- Render window cropped/zoomed around person’s center
- Config flags:
    - `--target_position` defaults to center
    - `--crop_size` to define output window dimensions

***

With Module 3 fully specified, we can now design **Module 4: Rendering \& CLI Integration**:

- Assemble pipeline in `run.py`
- Expose all module configurations as CLI flags
- Implement side-by-side rendering (`original | stabilized`) using OpenCV
- Save outputs: stabilized video, comparison video, pose JSON

Shall we proceed with the detailed plan and structure for **Module 4: Rendering \& CLI Integration**?

---

## Module 4: Rendering \& CLI Integration

### **1. CLI Interface (`run.py`)**

**Structure \& Flags:**

```bash
usage: run.py [-h] --input VIDEO_PATH --output_dir OUTPUT_DIR
              [--bg_model general|landscape]
              [--smoothing_method moving|gaussian]
              [--smoothing_window N]
              [--border_mode reflect|constant] [--border_value R G B]
              [--use_cuda true|false]
              [--pose_model lite|full|heavy]
              [--hip_calc_method simple|weighted]
              [--min_confidence 0.5] [--interpolate_missing true|false]
              [--crop_size WIDTH HEIGHT]
```


### **2. Pipeline Assembly**

In `run.py`:

1. **Parse CLI arguments**
2. **Initialize Modules** with configs:
    - `BackgroundRemover(bg_model, ...)`
    - `PoseDetector(pose_model, min_conf, hip_calc, ...)`
    - `Stabilizer(smoothing_method, window, border_mode, use_cuda, crop_size, ...)`
    - `Renderer(output_dir)`
3. **Process Video**:

```python
frames = VideoLoader(input_path)  # loads entire video into RAM
bg_results = remover.process_video(frames)
pose_data = detector.process_video(frames)
offsets = stabilizer.compute_offsets(pose_data["hip_centers"])
smooth_offsets = stabilizer.smooth_offsets(offsets)
stabilized_frames = stabilizer.apply_transforms(bg_results["person_frames"], smooth_offsets)
```

4. **Render Outputs**:
    - **Stabilized video**: write `stabilized_frames` to file
    - **Comparison video**: horizontally concatenate original and stabilized frames
    - **Export pose JSON**: `detector.export_to_json(pose_data, output_dir)`

### **3. Rendering Component (`renderer.py`)**

**Functions:**

- `render_video(frames, path, fps)`
- `render_comparison(orig_frames, stab_frames, path, fps)`

**Implementation:**

```python
class Renderer:
    def __init__(self, output_dir):
        # create directories

    def render_video(self, frames, filename):
        # write frames with cv2.VideoWriter

    def render_comparison(self, orig_frames, stab_frames, filename):
        # hconcat per frame and write video
```


### **4. Directory \& File Structure**

```
project_root/
├── src/
│   ├── background/
│   │   └── remover.py
│   ├── pose/
│   │   └── detector.py
│   ├── stabilization/
│   │   └── stabilizer.py
│   ├── rendering/
│   │   └── renderer.py
│   └── run.py
├── requirements.txt
├── README.md
└── sample_videos/  # optional test videos
```


### **5. Documentation (`README.md`)**

- **Setup**: environment, dependencies
- **Usage**: CLI examples
- **Module Descriptions**: background, pose, stabilization, rendering
- **Performance**: FPS benchmarks on CPU/GPU
- **Limitations**: pose failures, edge cases

***

With Module 4 defined, your **complete project plan** is ready. You can now implement each module according to this structure and use `run.py` to orchestrate the pipeline.

---

# Detailed Project Plan for Human-Centric Video Stabilization

## Overview

This plan captures **every decision** made for each module of the pipeline—from background removal through final rendering. It is granular, unambiguous, and ready for implementation.

***

## Module 1: Background Removal

**1. Model Selection**

- **MediaPipe Selfie Segmentation** (General model variant 0) for higher accuracy
- **Fallback** to Landscape model variant 1 if processing speed is insufficient

**2. Output Format**

- Return **both**:
    - **Person-only frames** (background removed)
    - **Segmentation mask** arrays

**3. Error Handling**

- **Frame processing failure**: reuse the **previous valid mask**
- **Person not detected**: return the **original frame** unmodified
- **Corrupt video input**: detect on startup, abort with descriptive error

**4. Memory Management**

- **Load entire video into RAM** (short test videos)
- Configurable in code if later switching to streaming for longer videos

**5. Quality Enhancement**

- **Temporal smoothing** of masks across frames to reduce flicker
- (Optional) **Joint bilateral filtering** on each mask for edge refinement

**6. Module Architecture**

- **Single file**: `src/background/remover.py`
- Class `BackgroundRemover` with methods:
    - `__init__(model_variant=0, smoothing=True)`
    - `process_frame(frame) → (person_frame, mask)`
    - `process_video(frames) → Generator[(person_frame, mask)]`

***

## Module 2: Pose Detection

**1. Model Complexity**

- **MediaPipe Pose – Full model** (Model complexity=1)
- Configurable to Lite (0) or Heavy (2) for experiments

**2. Hip-Center Calculation**

- Two options, selectable via `--hip_calc_method`:
    - **Simple average**: $(x_{\text L}+x_{\text R})/2,\,(y_{\text L}+y_{\text R})/2$
    - **Confidence-weighted** based on landmark visibility scores

**3. Data Export Format**

- **Timeline JSON**:

```json
{
  "video_info": {"fps": 30, "total_frames": 900},
  "hip_centers": [
    {"frame": 0, "timestamp": 0.033, "x":320, "y":240, "confidence":0.85},
    …
  ],
  "all_keypoints": [
    {"frame":0, "keypoints":[{"x":…, "y":…, "visibility":…}, …]},
    …
  ]
}
```


**4. Error Handling**

- **No detection**: reuse **previous frame’s pose**
- **Low confidence (<0.5)**: mark as missing, **interpolate** later
- **Partial occlusion**: accept visible keypoints, **interpolate** missing ones

**5. Processing Strategy**

- **Single-person tracking** only
- **2D keypoints** only (no world coordinates)
- **Process every frame** sequentially

**6. Module Architecture**

- **Single file**: `src/pose/detector.py`
- Class `PoseDetector` with methods:
    - `__init__(model=1, min_conf=0.5, hip_method="simple", interpolate=True)`
    - `process_frame(frame) → pose_dict`
    - `process_video(frames) → pose_data`
    - `calculate_hip_center(landmarks) → (x, y, conf)`
    - `export_to_json(pose_data, path)`

***

## Module 3: Stabilization

**1. Anchor Point \& Target**

- **Anchor**: hip-center from Module 2
- **Target**: frame center by default, configurable via `--target_position`

**2. Offset Calculation**

- $\Delta x = x_{\text{target}} - x_{\text{hip}}$
- $\Delta y = y_{\text{target}} - y_{\text{hip}}$

**3. Motion Smoothing**

- Default: **Moving average** over window of **5 frames**
- Alternative: **Gaussian filter**, selectable via `--smoothing_method`
- Window size configurable via `--smoothing_window`

**4. Frame Transformation**

- Affine matrix $M = \begin{bmatrix}1 & 0 & \Delta x_{\text{smooth}} \\ 0 & 1 & \Delta y_{\text{smooth}}\end{bmatrix}$
- Apply `cv2.warpAffine` with border mode configurable via `--border_mode` (`reflect` or `constant`)

**5. Border Handling**

- Default: **Reflect** edges to avoid blanks
- Constant fill mode available with user-defined `--border_value`

**6. Missing Offset Handling**

- Always **carry forward** the last valid offset if pose data missing
- Partial smoothing window at start uses available frames

**7. GPU Acceleration**

- Default: **CPU-only**
- Optional: enable CUDA for `warpAffine` if `--use_cuda true`

**8. Cropping \& Zoom**

- **Crop window** centered on moving hip-center, size configurable via `--crop_size`
- Ensures focus on subject, reduces background

**9. Module Architecture**

- **Single file**: `src/stabilization/stabilizer.py`
- Class `Stabilizer` with methods:
    - `compute_offsets(hip_centers) → List[(dx, dy)]`
    - `smooth_offsets(offsets) → List[(dx_s, dy_s)]`
    - `apply_transforms(person_frames, smooth_offsets) → List[frames]`

***

## Module 4: Rendering \& CLI Integration

**1. CLI Interface (`run.py`)**

```bash
usage: run.py --input VIDEO --output_dir OUTDIR
              [--bg_model general|landscape]
              [--pose_model lite|full|heavy]
              [--hip_calc_method simple|weighted]
              [--min_confidence 0.5]
              [--interpolate_missing true|false]
              [--smoothing_method moving|gaussian]
              [--smoothing_window 5]
              [--border_mode reflect|constant]
              [--border_value R G B]
              [--use_cuda true|false]
              [--crop_size WIDTH HEIGHT]
```

**2. Pipeline Assembly in `run.py`**

1. **Parse arguments**
2. **Initialize** each module using CLI flags
3. **Load video** into RAM
4. **Background removal** → person frames
5. **Pose detection** → pose_data JSON
6. **Compute \& smooth offsets**
7. **Apply stabilization** to person frames
8. **Render outputs** via `Renderer`

**3. Rendering Component (`src/rendering/renderer.py`)**

- Methods to write stabilized video and side-by-side comparison:
    - `render_video(frames, path, fps)`
    - `render_comparison(orig_frames, stab_frames, path, fps)`

**4. Directory Structure**

```
project_root/
├── src/
│   ├── background/remover.py
│   ├── pose/detector.py
│   ├── stabilization/stabilizer.py
│   ├── rendering/renderer.py
│   └── run.py
├── requirements.txt
└── README.md
```

**5. Documentation (`README.md`)**

- Setup \& environment
- CLI usage examples
- Module descriptions \& configurations
- Performance benchmarks (CPU/GPU)
- Known limitations \& troubleshooting tips

***

This **granular plan** captures every configuration option, architectural choice, and error-handling strategy. It provides a clear blueprint for implementing a robust, modular, and configurable human-centric video stabilization pipeline.


# Questions for the FoundationPose integration side

Context: we train DINOv2 (DINOv3 later) classification and segmentation heads on RGB-D frames, on an
AMD ROCm GPU box. For each frame we export a FoundationPose-style folder: `rgb/<id>.png`,
`depth/<id>.png` (uint16, millimetres), `masks/<id>.png` (binary 0/255, one object), `cam_K.txt` (3×3),
plus `detections.csv` (id, predicted object class, mask area fraction). The heads are linear probes on a
frozen backbone today, with partial finetuning available.

Please answer inline under each question. "Unknown / not decided" is a fine answer. Each question says
what it changes on our side.

## 1. Masks and objects (affects the most design decisions)
1. **Instance or semantic masks?** Can a frame contain several instances of the **same** object class? Our
   current segmentation is semantic (one label per class), so two identical parts would merge into one
   mask. If instances are needed we will add an instance step (connected components, or a
   detection/instance head).
2. **Which frames need a mask?** Only the first frame for `register` (then `track_one` follows), every
   frame, or a re-registration frame whenever tracking is lost? This sets our latency budget and whether
   we run on every frame.
3. **Multiple objects per scene:** one FoundationPose instance per object, each with its own mask? Do
   you want one `masks/` folder per object (e.g. `masks_<object>/`), or one mask per call over an API?
4. **Mask quality priorities:** are tight boundaries critical, or is a slightly dilated or eroded mask
   fine? Is a missed object or a false positive more costly? This sets thresholds and whether we add a
   refinement step.
5. **Occlusion and truncation:** how should a partially visible or frame-edge object be masked:
   visible pixels only (our default), or not at all below some visible fraction?

## 2. Classes, meshes and identity
6. **Object list:** what is the full set of object classes, and does each map to exactly one CAD
   mesh? Please give the class names exactly as they should appear in `classes.txt`.
7. **What is classification for?** Choosing which mesh to load, a scene-level label, or a per-object
   identity check? This decides whether classification runs on the whole frame (current), on mask crops,
   or not at all because segmentation already gives the class.
8. **Unknown or distractor objects:** should anything outside the list be ignored (background) or
   flagged? Do you need a confidence threshold or "unknown" output?
9. **Model-free mode:** are you using FoundationPose with CAD models, or the model-free mode with
   reference images? If model-free, do you also want DINO features or embeddings from us (e.g. to
   select or match reference views)?

## 3. Cameras and RGB-D data
10. **Camera model(s)?** (e.g. RealSense D435/D455, Azure Kinect, ZED, Orbbec.) How many cameras, and
    fixed or moving?
11. **Raw capture format:** file layout and naming, image formats, depth units and scale (we assume
    raw × scale = metres, e.g. 0.001), and whether depth is already **aligned/registered to the
    colour frame**. If not aligned, who does the alignment?
12. **Intrinsics:** how are they provided (per camera JSON, ROS `camera_info`, per frame)? Do they
    change with resolution or mode? If we resize frames we rescale K; does FoundationPose want the
    **original** resolution back? (Our masks are written at the original RGB resolution.)
13. **Resolution and frame rate** at capture and at inference.
14. **Multi-camera:** do you want per-camera masks only, or eventually a fused multi-view result? Are
    extrinsics between cameras available?
15. **Depth quality:** any depth holes or filtering we should know about? Should masks avoid
    depth-invalid pixels?

## 4. Interface and deployment
16. **Hand-off format:** is the folder layout above right for you, or do you want something else
    (ROS 2 topics, gRPC/HTTP service, shared memory, a Python function call)? Exact filename and
    format expectations, e.g. masks as 0/255 PNG versus bool .npy, frame ID zero-padding, `cam_K.txt`
    formatting.
17. **Where does inference run in production?** On the NVIDIA machine next to FoundationPose (then we
    export the DINO model, e.g. TorchScript/ONNX, and need its CUDA/TensorRT versions), as a separate
    service, or offline batch? What GPU (model, VRAM) and OS/CUDA versions are there?
18. **Latency budget** for mask (and class) prediction per frame, end-to-end.
19. **Online or offline:** live camera stream versus processing recorded sequences?
20. **Coordinate and pose conventions** we should keep consistent if we ever output more than masks
    (camera frame convention, units, pose format).

## 5. Data, labels and evaluation
21. **Ground-truth masks:** do labelled masks exist, or does FoundationPose/your tooling produce them
    (e.g. rendering the mesh at a known pose gives a free mask label)? What annotation tool or format,
    if any?
22. **Ground-truth poses:** are there GT 6D poses for some sequences? Then we can evaluate end-to-end
    (pose error / ADD-S given our masks), not just mask IoU.
23. **Data volume:** roughly how many frames, objects and scenes for the first real dataset, and how
    fast does new data arrive? This sizes storage (our box has only 25 GB persistent) and the feature
    cache.
24. **Success criteria:** what mask IoU / recall or pose accuracy counts as "good enough", and which
    failure cases matter most (lighting, clutter, reflective or textureless parts, occlusion)?

## 6. Models and licensing
25. **DINOv3:** do you have (or will you get) accepted access to the gated DINOv3 weights on Hugging
    Face, and is its licence acceptable for the deployment?
26. **Model size versus speed:** any constraint that rules out larger backbones (ViT-L/g) at inference?
27. **Anything else on your side** that consumes DINO outputs (features, embeddings, attention maps) that
    we should expose?

---
**Answer format that helps most:** keep the numbering. Short answers are fine. Attach examples where
you can: a sample raw capture, `cam_K`, one mask FoundationPose accepted, and the class→mesh list.

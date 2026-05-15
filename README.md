# VisDrone Human & Car Detection System
 
Fine-tuned YOLOv8s on VisDrone2019 for real-time human and car detection from drone/aerial imagery, with person counting and bounding box visualization.
 
---
 
## Demo Video
[Watch Demo Video](YOUR_GOOGLE_DRIVE_LINK_HERE)
 
---
 
## Results
 
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| person | 0.690 | 0.454 | 0.411 | 0.169 |
| car | 0.843 | 0.772 | 0.756 | 0.521 |
| **all** | **0.767** | **0.613** | **0.652** | **0.369** |
 
- **Inference speed**: 199 FPS on Tesla T4 GPU
- **Training duration**: 1.74 hours / 50 epochs
- **Model size**: 22.5 MB
---
 
## Sample Outputs
 
![Inference Results](outputs/inference_results.png)
![Training Curves](outputs/training_curves.png)
 
---
 
## Repository Structure
 
```
visdrone-detection/
├── notebooks/
│   └── visdrone_detection.ipynb    # Full pipeline — run top to bottom
├── outputs/
│   ├── inference_results.png       # 6-image inference grid
│   ├── training_curves.png         # Loss and mAP curves across 50 epochs
│   ├── sanity_check.png            # Preprocessing verification
│   ├── confusion_matrix.png        # Per-class confusion matrix
│   └── detection_summary.csv       # Per-image person/car counts
├── assets/
│   └── PR_curve.png                # Precision-recall curve
└── README.md
```
 
> **Model weights**: `best.pt` is available on [Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE) (22.5 MB)
 
---
 
## Dataset
 
**VisDrone2019-DET** — [Kaggle link](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
 
| Split | Images | Objects (after remap) |
|-------|--------|-----------------------|
| Train | 6,471 | 276,219 |
| Val | 548 | 30,008 |
| Test | 1,610 | 61,227 |
 
### Label Remapping
 
VisDrone has 10 classes. Remapped to 2:
 
| Original ID | Original Name | Remapped To |
|-------------|---------------|-------------|
| 0 | pedestrian | person (0) |
| 1 | people | person (0) |
| 3 | car | car (1) |
| 4 | van | car (1) |
| 2, 5, 6, 7, 8, 9 | bicycle, truck, tricycle, awning-tricycle, bus, motor | discarded |
 
Labels were already in YOLO normalized format — no coordinate conversion required.
 
---
 
## Setup & Installation
 
```bash
git clone https://github.com/YOUR_USERNAME/visdrone-detection.git
cd visdrone-detection
pip install ultralytics opencv-python matplotlib pandas pyyaml
```
 
Download `best.pt` from [Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE) and place it in the `weights/` folder.
 
---
 
## Usage
 
### Run inference on a single image
 
```python
from ultralytics import YOLO
 
model = YOLO("weights/best.pt")
results = model.predict(source="your_image.jpg", conf=0.25, iou=0.45)
 
# Count persons
person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)
car_count    = sum(1 for c in results[0].boxes.cls if int(c) == 1)
 
print(f"Persons: {person_count} | Cars: {car_count}")
```
 
### Run full pipeline
Open `notebooks/visdrone_detection.ipynb` on [Kaggle](https://www.kaggle.com) with GPU enabled and run all cells top to bottom. Skip the training cell (Cell 6) if using pretrained weights.
 
---
 
## Training Configuration
 
| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8s (COCO pretrained) |
| Optimizer | AdamW |
| Learning rate | 0.001 → 0.00001 (cosine) |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 × 640 |
| GPU | Tesla T4 (16 GB) |
| Augmentations | Mosaic, MixUp, Copy-Paste, HSV jitter, rotation ±10°, scale ±50%, horizontal flip |
 
---
 
## Evaluation & Discussion
 
### Strengths
- Car detection is strong — mAP50 of 0.756 with precision 0.843. Cars are larger, less occluded objects at drone altitude
- 199 FPS inference makes real-time deployment feasible
- Mosaic augmentation significantly improves small object generalization
- Fine-tuning from COCO pretrained weights converges faster and outperforms training from scratch
### Limitations
- Person recall is 0.454 — the model misses approximately 55% of humans
- Root cause: many persons in VisDrone are 8–15 pixels tall at 640×640 resolution, near the physical detection limit of any convolutional detector
- Heavily occluded crowd scenes cause merged or missed bounding boxes
- Higher input resolution (1280px) would improve small object recall at the cost of speed
### Challenges Faced
- VisDrone class IDs differ from COCO — required careful remapping before training
- Dense annotation files with 80+ objects per image slow down preprocessing scans
- Tuning confidence threshold: too high misses tiny persons, too low floods output with false positives. conf=0.25 was the best balance
---
 
## Acknowledgements
 
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [VisDrone Kaggle](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
 

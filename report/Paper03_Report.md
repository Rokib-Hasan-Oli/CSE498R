# Comprehensive Training and Evaluation Report: Paper #3

**Paper Title:** A Cloud-Based Hybrid Model for Real-Time Detection of BRTA-Approved Licence Plates Using YOLO Tiny and Haar Cascade  
**Implementer:** Md. Yousuf  
**Date:** October 2026  
**Environment:** Kaggle Notebook (GPU T4 x2)  
**Dataset:** 498R BanglaLPR All-in-One v1.0

---

## 1. Executive Summary

This document provides a comprehensive technical report on the re-implementation and evaluation of the methodology proposed in Paper #3. The original paper proposed a two-stage hybrid pipeline: utilizing a lightweight YOLOv4-tiny model to detect vehicles, followed by a custom-trained Haar Cascade classifier to locate the license plate within the vehicle's Region of Interest (ROI).

Due to the unavailability of the paper's proprietary dataset (IAT-BUET gate video) and their specific custom-trained Bangla Haar cascade, this implementation was adapted. We utilized the team's `498R BanglaLPR All-in-One` dataset and modern YOLOv8n (nano) models as a substitute for the older YOLOv4-tiny. To test the paper's hypothesis regarding Haar cascades, we employed OpenCV's generic Russian license plate Haar cascade as a baseline.

### 1.1 Key Findings

1. **Single-Stage YOLO Superiority:** The single-stage YOLOv8n model (without ROI cropping) achieved an F1 score of **95.25%**, significantly outperforming the paper's reported YOLO-tiny F1 of 92%.
2. **Hybrid Pipeline Failure:** The paper's proposed vehicle-ROI + Haar cascade pipeline achieved only a **4.19%** F1 score in our environment. This catastrophic drop is attributed to domain shift (a Russian Haar model evaluating Bangla plates) and cascading errors.
3. **Two-Stage YOLO vs. Single-Stage:** Replacing the Haar cascade with our trained YOLOv8n inside the vehicle ROI resulted in an F1 score of **83.93%**. This confirms that restricting the field of view to a vehicle ROI actually degrades detection performance compared to running YOLO on the full image.

---

## 2. Methodology and System Architecture

### 2.1 Original Paper vs. Our Implementation

| Component | Original Paper | Our Implementation | Reason for Deviation |
| :--- | :--- | :--- | :--- |
| **Vehicle Detector** | YOLOv4-tiny (COCO) | YOLOv8n (COCO) | YOLOv4-tiny is deprecated and not natively supported by Ultralytics; YOLOv8n is the modern equivalent. |
| **Plate Detector** | Custom Bangla Haar Cascade | 1. YOLOv8n (Trained on Bangla) <br> 2. Generic Russian Haar | The paper's custom Haar is not public. We used a generic OpenCV Haar to test the paper's logic. |
| **Input Resolution** | 464x464 (Fixed) | 416x416 and 640x640 | 416 is closest to the paper's resolution; 640 is modern standard. |
| **Deployment** | Flask Server / RTSP | Offline Evaluation | No live IP camera available; testing done on static images. |

### 2.2 Dataset Details

The evaluation utilized the team's custom dataset, split as follows:

- **Training Set:** 8,574 images (4,067 human-annotated + 4,507 auto-annotated).
- **Validation Set:** 896 images (human-annotated).
- **Test Set:** 1,153 images (human-annotated).
- **Conditions:** The dataset includes specific subsets for Clear, Night, Blurry, Small Plates, Synthetic Rain, Synthetic Fog, and No-Plate (for false alarm testing).

### 2.3 Hardware and Software Specifications

- **GPU:** Kaggle NVIDIA Tesla T4 x2 (16GB VRAM each).
- **CPU:** Kaggle CPU environment (simulating the 4-thread constraint of the paper's Ryzen 3).
- **Software Stack:** Python 3.13.15, PyTorch 2.11.0, Ultralytics 8.4.170, OpenCV 4.14.0.

---

## 3. Detector Training Phase

Two separate YOLOv8n models were trained concurrently to evaluate the impact of input resolution on detection accuracy and speed.

- **`v8n_640`:** Trained at 640x640 resolution, batch size 16.
- **`v8n_416`:** Trained at 416x416 resolution, batch size 16.
- **Optimizer:** Stochastic Gradient Descent (SGD).
- **Epochs:** Maximum 60 epochs (safeguarded by a 3.5-hour time cap).

### 3.1 Training Results

| Metric | v8n_640 | v8n_416 |
| :--- | :--- | :--- |
| **mAP50 (%)** | 98.7 | 98.8 |
| **mAP50-95 (%)** | 45.2 | 45.8 |
| **Training Time (min)** | 108.20 | 102.50 |
| **GPU FPS (FP16)** | - | 76.40 |
| **Hardware Used** | Tesla T4 | Tesla T4 |

---

## 4. Pipeline Evaluation (Ours vs. Paper)

The evaluation compared four distinct pipelines against the paper's reported metrics. Inference speed was measured on a CPU (4 threads) to simulate the paper's Ryzen 3 environment.

| Metric | Paper (p.6, Table II) | YOLO Alone | Haar Alone | Hybrid (Veh + Haar) | Hybrid (Veh + YOLO) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Precision (%)** | 94 | 93.35 | 11.07 | 13.81 | 84.18 |
| **Recall (%)** | 93 | 97.23 | 2.77 | 2.47 | 83.68 |
| **F1 Score (%)** | 93 | 95.25 | 4.43 | 4.19 | 83.93 |
| **CPU FPS** | 30 | 22.3 | 21.7 | 13.3 | 6.5 |
| **GPU FPS** | - | 94.1 | 12.0 | 21.0 | 39.0 |

### 4.1 Analysis of the Hybrid Failure

The proposed Hybrid model (Vehicle YOLO + Haar) achieved an F1 score of just 4.19%. The failure is rooted in two primary issues:

1. **Domain Shift:** The Russian Haar cascade (`haarcascade_russian_plate_number.xml`) expects high-contrast, single-line alphanumeric plates with a specific aspect ratio. Bangla plates are often two-line, contain complex conjuncts, and have different lighting profiles. The Haar features fail to find matching patterns.
2. **Cascading Errors:** When the vehicle detector (YOLOv8n COCO) provides a tight bounding box, it might clip the edges of the license plate. Even if the Haar cascade was perfect, it cannot detect a partially cropped plate.

### 4.2 Analysis of the YOLO Degradation in ROI

The `veh+yolo` pipeline (83.93% F1) performed worse than `yolo` alone (95.25%). This indicates that global context is crucial for YOLO. When the image is cropped to the vehicle ROI, the model loses contextual background information and the plate might be upscaled or downscaled poorly during the resize operation, affecting accuracy.

---

## 5. Condition-Based Performance Analysis

Detector performance was evaluated across various environmental conditions to test robustness.

| Run | All Test | Clear | Night | Blurry | Small Plates | Rain (Synth) | Fog (Synth) | False Alarms (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v8n_416** | 98.8 | 99.0 | 97.4 | 98.1 | 88.6 | 86.1 | 97.7 | 5.00 |
| **v8n_640** | 98.7 | 99.0 | 94.5 | 97.8 | 92.8 | 83.8 | 96.3 | 6.88 |

### 5.1 Key Observations

- **Resolution vs. Weather:** The 416px model outperformed the 640px model on Night (97.4% vs 94.5%) and Synthetic Rain (86.1% vs 83.8%). This suggests that larger input sizes might sometimes overfit to high-resolution noise in poor visibility conditions.
- **Small Plates:** The 640px model performed notably better on small plates (92.8% vs 88.6%). Higher resolution preserves the pixels required to detect distant or small license plates.
- **False Alarms:** The false alarm rate (detecting a plate where there is none) was 5.00% for the 416px model and 6.88% for the 640px model. This is acceptable but indicates room for improvement in negative mining.

---

## 6. Hardware Performance Analysis (CPU vs. GPU)

The paper emphasized real-time CPU performance (30 FPS on Ryzen 3).

- **Our GPU Performance:** YOLOv8n alone reached 94.1 FPS on the Tesla T4 (FP32), and 76.40 FPS with FP16 quantization. This proves the model is highly capable of real-time performance.
- **Our CPU Performance:** YOLOv8n achieved 22.3 FPS on the Kaggle CPU (4 threads). This is lower than the paper's 40 FPS for YOLOv4-tiny.
- **Reason for the Gap:** PyTorch (used by YOLOv8) has higher CPU overhead compared to Darknet (used by YOLOv4). Additionally, Kaggle's standard CPU (often Intel Xeon) may have lower single-core clock speeds compared to a desktop Ryzen 3 2200G. To achieve 40+ FPS on CPU, deploying YOLOv8n via ONNX Runtime or OpenVINO is highly recommended.

---

## 7. Conclusion and Recommendations

Based on the empirical evidence from this implementation:

1. **Drop the Haar Cascade:** The hybrid approach using a generic Haar cascade is entirely ineffective for Bangla license plates due to severe domain shift. The team should not pursue this two-stage approach.
2. **Adopt Single-Stage YOLOv8n:** YOLOv8n running on the full image achieved an F1 score of 95.25%, proving it is the most robust and accurate method among the tested pipelines.
3. **Keep the 640px Model for Accuracy:** For static or high-accuracy scenarios (like parking lot check-ins), `v8n_640` is superior for small plates.
4. **Keep the 416px Model for Speed:** For real-time edge deployment where speed is critical, `v8n_416` offers a better trade-off without significant loss in mAP.

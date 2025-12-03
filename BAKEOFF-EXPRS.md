# 🌾 Fields of The World (FTW) — Bake-off Experiments

This document lists planned experiments for evaluating and improving different components of the FTW model training and design pipeline.

---

## **1. Input Ordering**

**Setting:**  
For the baseline **FTW-3-class** configuration (`efficientnet-b3`, trained for 100 epochs, default hyperparameters):

- `random_shuffle: false` — *Default in v1*. The model showed noticeable sensitivity to the ordering of `window_A` and `window_B`.  
- `random_shuffle: true` — Makes the model invariant to the ordering of paired images.

Note: during evaluation we keep `random_shuffle: false`  for reproducibility.

### **Metrics Table**

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| random_shuffle: false | – | – | – | – | – | – |
| random_shuffle: true  | – | – | – | – | – | – |

👉 All experiments below will use **`random_shuffle: true`** during training with all other settings kept at default.

---

## **2. Loss Functions**

**Setting:** Ablation of different loss formulations using baseline class weights.

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| Cross-Entropy | – | – | – | – | – | – |
| Dice | 0.822 | 0.903 | 0.901 | 0.562 | 0.342| 0.425 |
| Log-Cosh Dice |0.828 | 0.912 | 0.900 | 0.559 | 0.380 | 0.452 |
| Fractal Tanimoto | – | – | – | – | – | – |
| Tversky | – | – | – | – | – | – |
| 0.5 × (CE + Dice) | 0.794 | 0.922 | 0.851 | 0.452 |0.376 | 0.411 |
| 0.5 × (CE + Log-Cosh Dice) | 0.779 | 0.922 | 0.834 | 0.411 |0.359 | 0.383|
| 0.5 × (CE + Fractal Tanimoto) | – | – | – | – | – | – |
| 0.5 × (CE + Tversky) | – | – | – | – | – | – |

---

## **3. Backbones**

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| EfficientNet-B3 | – | – | – | – | – | – |
| EfficientNet-B4 | – | – | – | – | – | – |
| EfficientNet-B5 | – | – | – | – | – | – |
| EfficientNet-B6 | – | – | – | – | – | – |
| EfficientNet-B7 | – | – | – | – | – | – |

---

## **4. Training Settings**

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| linear – 100 epochs | – | – | – | – | – | – |
| linear – 300 epochs | – | – | – | – | – | – |
| cosine – 100 epochs | – | – | – | – | – | – |
| cosine – 300 epochs | – | – | – | – | – | – |

---

## **5. Best Setting Combinations**

Train a model using the **best combination** of results from Sections **2**, **3**, and **4**.

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| Best Combo | – | – | – | – | – | – |

---

# 🧠 Foundation Model Evaluations

Using the best settings obtained from Sections **2**, **3**, and **4**, evaluate **pretrained transformer-based geospatial foundation models** that provide token embeddings of shape **(N, d)** —  
where *d* is the embedding dimension and *N* is the number of image patches.

**Goal:**  
Select one state-of-the-art foundation model and ablate the following components.

---

## **6. Fusion Strategies for (window_A, window_B)**

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| Channel-wise concat (`N × 2d`) | – | – | – | – | – | – |
| Channel-wise mean (`N × d`) | – | – | – | – | – | – |
| Token-wise concat (`2N × d`) | – | – | – | – | – | – |
| Bidirectional token fuser (`N × d`) | – | – | – | – | – | – |
| Other | – | – | – | – | – | – |

---

## **7. Decoder Design**

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| Decoder 1 | – | – | – | – | – | – |
| Decoder 2 | – | – | – | – | – | – |
| Decoder 3 | – | – | – | – | – | – |
| Decoder 4 | – | – | – | – | – | – |
| Decoder 5 | – | – | – | – | – | – |

---

## **8. Foundation Model Choices**

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| Model A (Galileo) | – | – | – | – | – | – |
| Model B (Galileo) | – | – | – | – | – | – |
| Model C (Galileo) | – | – | – | – | – | – |
| Model D (Galileo) | – | – | – | – | – | – |
| Model E (Galileo) | – | – | – | – | – | – |
| Model F (Galileo) | – | – | – | – | – | – |
| Model G (Galileo) | – | – | – | – | – | – |
| Clay (Subash) | – | – | – | – | – | – |
| TerraFM (Subash) | – | – | – | – | – | – |
| AlphaEarth Embedding (Subash) | – | – | – | – | – | – |
| DINOv3 (Subash) | – | – | – | – | – | – |

---

## **9. Instance Segmentation**

Using the best training configuration from Sections **2–4**, train and evaluate the following segmentation architectures:

| Component | pixel_iou | pixel_precision | pixel_recall | object_precision | object_recall | object_f1 |
|------------|------------|----------------|---------------|------------------|----------------|------------|
| SAM (Segment Anything Model) | – | – | – | – | – | – |
| RF-DETR | – | – | – | – | – | – |
| Mask2Former | – | – | – | – | – | – |
| Other | – | – | – | – | – | – |

---
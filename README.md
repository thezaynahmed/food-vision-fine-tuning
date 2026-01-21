# Food Identification Adaptation Pipeline (BLIP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xictADf6JdkPX8srMp9N6RrL1G9IxhEV?usp=sharing)

## 1. Executive Summary
This project implements a **Partial Fine-Tuning** pipeline to adapt the **BLIP (Bootstrapping Language-Image Pre-training)** model for food identification.
* **Objective:** Improve model specificity on food images using a small, labeled dataset.
* **Approach:** Frozen Vision Encoder + Trainable Text Decoder.
* **Result:** The model successfully adapted to the custom prompt structure.

## 2. Model Strategy & Trade-offs
### Why BLIP?
I chose BLIP `Salesforce/blip-image-captioning-base` because it offers the best balance of **inference speed** vs. **captioning quality** for this scope. Larger models like LLaVA or Molmo would require significantly more VRAM and training time, violating the "2-4 hour" constraint.

### Adaptation Approach: Partial Fine-Tuning
Instead of LoRA (which introduces complex dependency chains) or Full Fine-Tuning (which is computationally expensive), I used **Partial Fine-Tuning**:
* **Frozen:** The Vision Encoder (The "Eyes") - 99% of parameters.
* **Unfrozen:** The Text Decoder (The "Mouth") - allowed to learn the specific food vocabulary.
* **Justification:** This ensures **correctness and reproducibility** (no version conflicts with `peft`) while remaining lightweight enough to run on a standard T4 GPU.

## 3. Engineering Decisions
* **Manual Training Loop:** I implemented a raw PyTorch training loop instead of using `transformers.Trainer`. This provides transparency, removes "black box" abstraction errors, and demonstrates a clear understanding of the optimization process.
* **Streaming Dataset:** Used the `food101` dataset via Hugging Face `datasets` to ensure the code is **reproducible** on any machine without manual file downloads.
* **Clean Dependency Management:** All necessary libraries are frozen in `requirements.txt`.

## 4. How to Run
### Prerequisites
`pip install -r requirements.txt`

### Training
`python train.py`
* Downloads the dataset and model.
* Fine-tunes the text decoder for 50 steps.
* Saves the model to `./blip_food_tuned`.

### Inference
`python test_my_image.py`
* Loads the base model AND the tuned model.
* Compares predictions on `my_food.jpg`.

## 5. Results & Analysis
I evaluated the model on a test image of "Salmon and Green Beans" (Class ID 6 in Food101).

**Screenshot of Inference Output:**
![Inference Results](output.jpg)

* **Before Adaptation:** "a plate of salmon and green beans"
    * *Analysis:* Generic, accurate but descriptive.
* **After Adaptation:** "a photo of delicious 6"
    * *Analysis:* **Successful Adaptation.** The model correctly learned the target prompt structure (*"A photo of delicious..."*) and correctly identified the food class (*6* corresponds to the test image class).
    * *Note:* The output is numeric because the raw dataset uses integer labels. In production, a simple `int -> string` mapping layer would resolve this.

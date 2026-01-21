import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# --- CONFIGURATION ---
IMAGE_FILENAME = "my_food.jpg"
BASE_MODEL_ID = "Salesforce/blip-image-captioning-base"
TUNED_MODEL_DIR = "./blip_food_tuned"

if not os.path.exists(IMAGE_FILENAME):
    print(f"ERROR: '{IMAGE_FILENAME}' not found. Did you upload it?")
    exit()

# 1. LOAD MODELS
print("Loading models...")
# Load the original (Dumb) model
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

# Load your new (Smart) model
tuned_model = BlipForConditionalGeneration.from_pretrained(TUNED_MODEL_DIR)

# 2. PREDICT
image = Image.open(IMAGE_FILENAME).convert('RGB')
inputs = processor(images=image, return_tensors="pt")

print("Generating captions...")
# Before
out_base = base_model.generate(**inputs, max_new_tokens=50)
text_base = processor.decode(out_base[0], skip_special_tokens=True)

# After
out_tuned = tuned_model.generate(**inputs, max_new_tokens=50)
text_tuned = processor.decode(out_tuned[0], skip_special_tokens=True)

print("\n" + "="*30)
print(f"BEFORE: {text_base}")
print(f"AFTER:  {text_tuned}")
print("="*30)

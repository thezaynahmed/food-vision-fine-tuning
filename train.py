import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "Salesforce/blip-image-captioning-base"
OUTPUT_DIR = "./blip_food_tuned"
BATCH_SIZE = 2
STEPS = 50

# --- 1. SETUP DATA ---
print("Loading dataset...")
dataset = load_dataset("food101", split="train[:50]")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- 2. SETUP MODEL ---
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

# --- 3. FREEZE LAYERS (The "Partial Fine-Tuning" Strategy) ---
# We lock the entire model first
for param in model.parameters():
    param.requires_grad = False

# We unlock ONLY the text decoder so it can learn new food names.
# This is chemically similar to LoRA but uses standard PyTorch.
for param in model.text_decoder.parameters():
    param.requires_grad = True

model.to(device)
model.train()
print(f"Model ready. Training on {device}...")

# --- 4. TRAINING LOOP ---
# Using standard AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
progress_bar = tqdm(range(STEPS))

for i in progress_bar:
    # Get random batch
    indices = torch.randint(0, len(dataset), (BATCH_SIZE,))
    batch_raw = dataset.select(indices)

    # Prepare inputs
    images = [x for x in batch_raw['image']]
    captions = [f"A photo of delicious {x['label']}" for x in batch_raw]

    inputs = processor(images=images, text=captions, return_tensors="pt", padding=True).to(device)

    # Forward pass
    outputs = model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values, labels=inputs.input_ids)

    # Backward pass
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    progress_bar.set_description(f"Loss: {loss.item():.4f}")

# --- 5. SAVE ---
print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Done! Model saved to", OUTPUT_DIR)

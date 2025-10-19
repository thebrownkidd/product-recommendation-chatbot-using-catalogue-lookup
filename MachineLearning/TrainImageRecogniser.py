"""
TrainImageCat.py (Multi-Label)
Fine-tunes a pretrained ResNet50 for multi-label classification.
Each image can belong to multiple categories.
"""

import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# =========================================================
# Config
# =========================================================

DATA_JSON = "img_cat.json"
IMAGE_DIR = "."
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


# =========================================================
# Dataset
# =========================================================

class MultiLabelImageDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        all_cats = set()
        for uid, entry in data.items():
            cats = entry.get("categories", [])
            imgs = entry.get("images", [])
            if len(imgs) == 0 or len(cats) == 0:
                continue
            for c in cats:
                all_cats.add(c)
            self.samples.append({"uid": uid, "images": imgs, "categories": cats})

        self.all_categories = sorted(list(all_cats))
        self.cat2idx = {c: i for i, c in enumerate(self.all_categories)}
        self.idx2cat = {i: c for c, i in self.cat2idx.items()}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path = os.path.join(IMAGE_DIR, entry["images"][0])
        image = Image.open(img_path).convert("RGB")

        # Multi-hot encode categories
        label = torch.zeros(len(self.all_categories), dtype=torch.float32)
        for c in entry["categories"]:
            if c in self.cat2idx:
                label[self.cat2idx[c]] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================
# Model
# =========================================================

def create_model(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V2")
    for param in model.parameters():
        param.requires_grad = False  # freeze backbone
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Sigmoid()  # multi-label activation
    )
    return model


# =========================================================
# Training
# =========================================================
import torchmetrics

def train_model(json_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiLabelImageDataset(json_path, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(dataset.all_categories)
    print(f"✅ Found {num_classes} unique categories")

    model = create_model(num_classes).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    # metrics
    f1_metric = torchmetrics.classification.MultilabelF1Score(num_labels=num_classes, threshold=0.5).to(DEVICE)
    acc_metric = torchmetrics.classification.MultilabelAccuracy(num_labels=num_classes, threshold=0.5).to(DEVICE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        f1_metric.reset()
        acc_metric.reset()

        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            f1_metric.update(outputs, labels.int())
            acc_metric.update(outputs, labels.int())

        avg_loss = total_loss / len(loader)
        epoch_f1 = f1_metric.compute().item()
        epoch_acc = acc_metric.compute().item()

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, F1 = {epoch_f1:.4f}, Accuracy = {epoch_acc:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "class_map": dataset.cat2idx
    }, "image_multilabel_classifier.pt")
    print("✅ Training complete. Model saved as image_multilabel_classifier.pt")

# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    train_model(DATA_JSON)

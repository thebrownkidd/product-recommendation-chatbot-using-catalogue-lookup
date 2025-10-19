"""
TrainDiscCat.py
---------------
Trains dual encoders (CategoryEncoder, TextEncoder) so that
category vectors and title+description vectors align in embedding space.

Input file: disc_cat.json
Structure: {
  "uniq_id_1": {
      "title": "...",
      "description": "...",
      "categories": ["furniture", "chair", "wooden"]
  },
  ...
}
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================
# Configs
# =============================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# Dataset
# =============================

class DiscCatDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for uid, entry in data.items():
            title = entry.get("title", "")
            desc = entry.get("description", "")
            cats = entry.get("categories", [])
            if len(cats) > 0:
                self.samples.append({
                    "uid": uid,
                    "title": title,
                    "description": desc,
                    "categories": cats
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):

    batch_out = {
        "uid": [item["uid"] for item in batch],
        "title": [item["title"] for item in batch],
        "description": [item["description"] for item in batch],
        "categories": [item["categories"] for item in batch],
    }
    return batch_out

# =============================
# Dual Encoder Model
# =============================

class DualEncoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.cat_encoder = AutoModel.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_categories(self, categories):
        """Encode list of category lists"""
        cat_vecs = []
        for cat_list in categories:
            embs = []
            for c in cat_list:
                tokens = self.tokenizer(
                    c, return_tensors='pt', truncation=True, padding=True
                ).to(DEVICE)
                with torch.no_grad():
                    out = self.cat_encoder(**tokens)
                    emb = out.last_hidden_state.mean(dim=1)
                embs.append(emb)
            cat_vecs.append(torch.mean(torch.stack(embs), dim=0))
        return torch.cat(cat_vecs, dim=0)

    def encode_text(self, titles, descs):
        texts = [str(t if isinstance(t, str) else "") + " " + str(d if isinstance(d, str) else "")
                for t, d in zip(titles, descs)]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        outputs = self.text_encoder(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb


    def forward(self, categories, titles, descs):
        cat_emb = self.encode_categories(categories)
        text_emb = self.encode_text(titles, descs)
        return cat_emb, text_emb


# =============================
# Contrastive Loss
# =============================

def contrastive_loss(cat_emb, text_emb, temperature=0.07):
    cat_emb = F.normalize(cat_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = cat_emb @ text_emb.T / temperature
    labels = torch.arange(cat_emb.size(0)).to(cat_emb.device)
    return F.cross_entropy(logits, labels)


# =============================
# Training Loop
# =============================

def train_model(json_path):
    dataset = DiscCatDataset(json_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = DualEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            cats = batch["categories"]
            titles = batch["title"]
            descs = batch["description"]

            cat_emb, text_emb = model(cats, titles, descs)
            loss = contrastive_loss(cat_emb, text_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "disc_cat_dualencoder.pt")
    print("✅ Training complete. Model saved as disc_cat_dualencoder.pt")


# =============================
# Run
# =============================

if __name__ == "__main__":
    train_model("disc_cat.json")

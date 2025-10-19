"""
Generate Product Vectors Script

This script processes a catalogue CSV file and generates vector embeddings
for each product using the trained dual encoder model.

Input: catelogue.csv
Output: products_with_vectors.csv
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import os
import json
import sys
from pathlib import Path

# Model paths
DUAL_ENCODER_PATH = "models/disc_cat_dualencoder.pt"
IMG_CLASSIFIER_PATH = "models/image_multilabel_classifier.pt"
CATALOGUE_PATH = "catalogue.csv"
OUTPUT_PATH = "data/products_with_vectors.csv"
IMAGE_DIR = "."  # Base directory for images, adjust if needed

# Define transforms for image processing - matching the training code
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DualEncoder(torch.nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.cat_encoder = AutoModel.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
    def encode_categories(self, categories, tokenizer, device="cpu"):
        """Encode list of category lists"""
        cat_vecs = []
        for cat_list in categories:
            embs = []
            for c in cat_list:
                tokens = tokenizer(
                    c, return_tensors='pt', truncation=True, padding=True
                ).to(device)
                with torch.no_grad():
                    out = self.cat_encoder(**tokens)
                    emb = out.last_hidden_state.mean(dim=1)
                embs.append(emb)
            cat_vecs.append(torch.mean(torch.stack(embs), dim=0) if embs else torch.zeros(384).to(device).unsqueeze(0))
        return torch.cat(cat_vecs, dim=0)

    def encode_text(self, titles, descs, tokenizer, device="cpu"):
        """Encode title and description pairs"""
        texts = [str(t if isinstance(t, str) else "") + " " + str(d if isinstance(d, str) else "")
                for t, d in zip(titles, descs)]
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
        return emb

def create_image_model(num_classes):
    """Create image model architecture matching the training code"""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Sigmoid()  # multi-label activation
    )
    return model

def load_models():
    """Load the trained models"""
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Load dual encoder model
    dual_encoder = DualEncoder()
    try:
        state_dict = torch.load(DUAL_ENCODER_PATH, map_location=device)
        dual_encoder.load_state_dict(state_dict)
        dual_encoder = dual_encoder.to(device)
        dual_encoder.eval()
    except Exception as e:
        print(f"Error loading dual encoder model: {e}")
        sys.exit(1)
    
    # Load image classifier model
    try:
        img_data = torch.load(IMG_CLASSIFIER_PATH, map_location=device)
        class_map = img_data["class_map"]  # Dictionary mapping category names to indices
        
        # Create model with correct number of classes
        img_model = create_image_model(len(class_map))
        img_model.load_state_dict(img_data["model_state"])
        img_model.to(device)
        img_model.eval()
    except Exception as e:
        print(f"Error loading image model: {e}")
        class_map = {}
        img_model = None
    
    print("✓ Models loaded successfully")
    return dual_encoder, tokenizer, img_model, class_map, device

def extract_categories(row):
    """Extract categories from the product row"""
    # Try to get categories from various possible fields
    categories = []
    
    # Check if there's a 'categories' column
    if 'categories' in row and pd.notna(row['categories']):
        try:
            # Try parsing as JSON list
            if isinstance(row['categories'], str):
                if row['categories'].startswith('['):
                    categories = json.loads(row['categories'])
                else:
                    # Comma-separated string
                    categories = [c.strip() for c in row['categories'].split(',')]
        except:
            # Fallback: treat as a single category
            categories = [row['categories']]
    
    # Check for category, category_1, category_2, etc.
    for col in row.index:
        if col.startswith('category') and pd.notna(row[col]):
            categories.append(row[col])
    
    # Check for tags
    if 'tags' in row and pd.notna(row['tags']):
        try:
            if isinstance(row['tags'], str):
                if row['tags'].startswith('['):
                    tags = json.loads(row['tags'])
                else:
                    tags = [t.strip() for t in row['tags'].split(',')]
                categories.extend(tags)
        except:
            pass
    
    # Deduplicate
    return list(set(categories))

def get_image_categories(image_path, img_model, class_map, device, threshold=0.5):
    """Get categories from an image using the trained multi-label classifier"""
    if not img_model or not os.path.exists(image_path):
        return []
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = image_transform(img).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = img_model(img_tensor)
        
        # Convert to probabilities with sigmoid
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Get categories where probability > threshold
        idx2cat = {v: k for k, v in class_map.items()}
        predicted_categories = [idx2cat[i] for i, prob in enumerate(probs) if prob > threshold]
        
        return predicted_categories
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def generate_vectors():
    """Generate vectors for products in the catalogue"""
    # Load models
    dual_encoder, tokenizer, img_model, class_map, device = load_models()
    
    # Load catalogue
    print(f"Loading catalogue from {CATALOGUE_PATH}...")
    try:
        df = pd.read_csv(CATALOGUE_PATH)
        print(f"✓ Loaded {len(df)} products")
    except Exception as e:
        print(f"Error loading catalogue: {e}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_vectors = []
    
    # Extract categories for all products
    print("Extracting categories...")
    df['extracted_categories'] = df.apply(extract_categories, axis=1)
    
    # Extract categories from images if available
    if img_model is not None and 'image_path' in df.columns:
        print("Extracting categories from images...")
        img_categories = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_path = row.get('image_path', '')
            if pd.notna(image_path) and image_path:
                full_path = os.path.join(IMAGE_DIR, image_path)
                img_cats = get_image_categories(full_path, img_model, class_map, device)
                img_categories.append(img_cats)
            else:
                img_categories.append([])
        
        df['img_categories'] = img_categories
        
        # Merge extracted categories with image categories
        for i, row in df.iterrows():
            all_cats = list(set(row['extracted_categories'] + row['img_categories']))
            df.at[i, 'extracted_categories'] = all_cats
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Generating vectors for {len(df)} products in {total_batches} batches...")
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size].copy()
        
        # Get titles and descriptions
        titles = batch_df['title'].fillna('').tolist()
        descriptions = batch_df['description'].fillna('').tolist()
        
        # Get categories
        categories = batch_df['extracted_categories'].tolist()
        
        # Generate vectors - we'll use text embeddings only if available
        # otherwise fall back to category embeddings
        try:
            # First try with text encoder
            text_vectors = dual_encoder.encode_text(titles, descriptions, tokenizer, device).cpu().numpy()
            batch_vectors = text_vectors
            
            # If categories exist and text is empty, use category vectors
            for j, (title, desc, cats) in enumerate(zip(titles, descriptions, categories)):
                if (not title and not desc) and cats:
                    cat_vector = dual_encoder.encode_categories([cats], tokenizer, device).cpu().numpy()[0]
                    batch_vectors[j] = cat_vector
        except Exception as e:
            print(f"Error encoding batch {i//batch_size + 1}/{total_batches}: {e}")
            # Fallback to zeros
            batch_vectors = np.zeros((len(batch_df), 384))
            
        all_vectors.extend(batch_vectors.tolist())
    
    # Add vectors to dataframe
    df['vector'] = all_vectors
    
    # Save to CSV
    print(f"Saving {len(df)} products with vectors to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("✓ Done!")
    
if __name__ == "__main__":
    generate_vectors()
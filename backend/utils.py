import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from PIL import Image
import io
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from torchvision import transforms, models
import torch.nn as nn

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define transforms for image processing - matching the training code
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DualEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.cat_encoder = AutoModel.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
    def encode_categories(self, categories):
        """Encode list of category lists"""
        # This function is implemented in load_models for actual use
        pass
        
    def encode_text(self, titles, descs):
        """Encode title and description pairs"""
        # This function is implemented in load_models for actual use
        pass
        
    def forward(self, categories, titles, descs):
        cat_emb = self.encode_categories(categories)
        text_emb = self.encode_text(titles, descs)
        return cat_emb, text_emb

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
    """Load the trained models."""
    # Load tokenizer for text processing
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Load dual encoder model
    model_state_dict = torch.load("/etc/secrets/disc_cat_dualencoder.pt", map_location=torch.device('cpu'))
    dual_encoder = DualEncoder()
    dual_encoder.load_state_dict(model_state_dict)
    dual_encoder.eval()
    
    # Load image classification model
    img_data = torch.load("/etc/secrets/image_multilabel_classifier.pt", map_location=torch.device('cpu'))
    class_map = img_data["class_map"]  # Dictionary mapping category names to indices
    
    # Create model with correct number of classes
    img_model = create_image_model(len(class_map))
    img_model.load_state_dict(img_data["model_state"])
    img_model.eval()
    
    return dual_encoder, tokenizer, img_model, class_map

def load_product_data():
    """Load product catalog with precomputed vectors."""
    products_df = pd.read_csv("data/products_with_vectors.csv")
    # Convert string representation of vectors back to numpy arrays
    products_df['vector'] = products_df['vector'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=',')
    )
    return products_df

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def find_similar_products(query_vector: np.ndarray, products_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Find most similar products by vector similarity."""
    # Calculate similarities
    similarities = products_df['vector'].apply(lambda x: cosine_similarity(query_vector, x))
    
    # Add similarities to the dataframe
    temp_df = products_df.copy()
    temp_df['similarity'] = similarities
    
    # Sort by similarity and return top_n
    return temp_df.sort_values('similarity', ascending=False).head(top_n)

def categories_to_vector(categories: List[str], dual_encoder, tokenizer) -> np.ndarray:
    """
    Convert a list of categories to a vector using the trained dual encoder.
    This matches the encode_categories method in the training code.
    """
    cat_embs = []
    for cat in categories:
        tokens = tokenizer(
            cat, return_tensors='pt', truncation=True, padding=True
        )
        with torch.no_grad():
            out = dual_encoder.cat_encoder(**tokens)
            emb = out.last_hidden_state.mean(dim=1)
        cat_embs.append(emb)
    
    if cat_embs:
        # Average the embeddings as done in training
        cat_vec = torch.mean(torch.stack(cat_embs), dim=0).cpu().numpy()
        return cat_vec
    else:
        # Return zero vector if no categories
        return np.zeros(384)  # MiniLM-L6-v2 embedding size

def image_to_categories(image, img_model, class_map, threshold=0.5) -> List[str]:
    """
    Process image through the trained multi-label classifier and return predicted categories.
    Implements the inference flow from the training code.
    """
    # Preprocess image
    img_tensor = image_transform(image).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        outputs = img_model(img_tensor)
    
    # Convert to probabilities with sigmoid
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Get categories where probability > threshold
    idx2cat = {v: k for k, v in class_map.items()}
    predicted_categories = [idx2cat[i] for i, prob in enumerate(probs) if prob > threshold]
    
    return predicted_categories

def format_products_for_response(products_df):
    """Format products dataframe to response format."""
    products = []
    for _, row in products_df.iterrows():
        product = {
            "id": str(row.get("id", "")),
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "price": float(row.get("price", 0)) if not pd.isna(row.get("price", 0)) else 0,
            "image_url": row.get("image_url", ""),
            "similarity": float(row.get("similarity", 0)),
            "attributes": {}
        }
        
        # Add any other columns as attributes
        for col in row.index:
            if col not in ["id", "title", "description", "price", "image_url", "similarity", "vector"]:
                product["attributes"][col] = row[col]
                
        products.append(product)
    
    return products

# Direct Google Generative AI functions - no LangChain dependency
def text_to_categories(query: str) -> List[str]:
    """Use Gemini to determine relevant categories from text."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    Given the following user query for product search: "{query}"
    
    Identify the 3 most relevant product categories that match this query.
    Return only the category names, separated by commas.
    
    Categories should be general product types, attributes, or use cases.
    """
    
    response = model.generate_content(prompt)
    categories = [cat.strip() for cat in response.text.split(',')]
    return categories

def generate_product_description(product_info):
    """Generate product description using Gemini if one doesn't exist."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Prepare attributes string
    attributes = ', '.join([f"{k}: {v}" for k, v in product_info.get("attributes", {}).items()])
    
    prompt = f"""
    Generate a compelling product description for:
    
    Product Title: {product_info.get("title", "")}
    Product Image URL: {product_info.get("image_url", "")}
    Additional Features: {attributes}
    
    The description should be informative, highlight key features, and be approximately 2-3 sentences long.
    """
    
    response = model.generate_content(prompt)
    return response.text

def rag_enhance_results(query: str, products: List[Dict]) -> str:
    """Use Gemini to provide enhanced recommendations."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Prepare product data for the prompt
    products_context = ""
    for i, product in enumerate(products, 1):
        products_context += f"""
        Product {i}:
        - Title: {product['title']}
        - Description: {product.get('description', 'No description available')}
        - Price: {product.get('price', 'N/A')}
        - Features: {', '.join([f"{k}: {v}" for k, v in product.get('attributes', {}).items()])}
        """
    
    prompt = f"""
    User query: "{query}"
    
    Based on the following product results:
    {products_context}
    
    Provide a helpful response that:
    1. Acknowledges the user's query
    2. Summarizes the top matches and why they might be relevant
    3. Highlights any specific features that align with the user's needs
    4. Offers any additional recommendations or suggestions based on the results
    
    Keep your response concise and focused on helping the user find the right product.
    """
    
    response = model.generate_content(prompt)
    return response.text
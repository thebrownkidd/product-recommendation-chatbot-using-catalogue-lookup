from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
from PIL import Image
import io
from typing import List, Dict

from models import TextQueryRequest, SearchResponse, ProductResponse
from utils import (
    load_models, load_product_data, text_to_categories,
    image_to_categories, find_similar_products, categories_to_vector,
    rag_enhance_results, format_products_for_response,
    generate_product_description
)

app = FastAPI(title="Product Recommendation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data on startup
@app.on_event("startup")
async def startup_event():
    # Load the models and data
    app.state.dual_encoder, app.state.tokenizer, app.state.img_model, app.state.class_map = load_models()
    app.state.products_df = load_product_data()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(request: TextQueryRequest):
    try:
        # Get relevant categories from text
        categories = text_to_categories(request.query)
        
        # Convert categories to vectors using the dual encoder
        query_vector = categories_to_vector(
            categories, app.state.dual_encoder, app.state.tokenizer
        )
        
        if len(query_vector) > 0:
            # Find similar products
            similar_products_df = find_similar_products(query_vector, app.state.products_df)
            
            # Format products for response
            products = format_products_for_response(similar_products_df)
            
            # Generate descriptions for products without one
            for product in products:
                if not product.get("description"):
                    product["description"] = generate_product_description(product)
            
            # Enhance results with RAG
            enhanced_response = rag_enhance_results(request.query, products)
            
            return SearchResponse(
                products=products,
                enhanced_response=enhanced_response
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to generate query vector")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(file: UploadFile = File(...)):
    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get categories from image
        predicted_categories = image_to_categories(
            image, app.state.img_model, app.state.class_map
        )
        
        # Convert categories to vectors using the dual encoder
        query_vector = categories_to_vector(
            predicted_categories, app.state.dual_encoder, app.state.tokenizer
        )
        
        if len(query_vector) > 0:
            # Find similar products
            similar_products_df = find_similar_products(query_vector, app.state.products_df)
            
            # Format products for response
            products = format_products_for_response(similar_products_df)
            
            # Generate descriptions for products without one
            for product in products:
                if not product.get("description"):
                    product["description"] = generate_product_description(product)
            
            # Enhance results with RAG
            enhanced_response = rag_enhance_results(
                f"Products similar to the uploaded image (detected categories: {', '.join(predicted_categories)})", 
                products
            )
            
            return SearchResponse(
                products=products,
                enhanced_response=enhanced_response
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to generate query vector from image")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
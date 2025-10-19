# Product Recommendation Chatbot

A full-stack application that provides product recommendations based on text queries or image uploads. The system uses dual encoders to match categories with products and a multi-label image classifier to extract categories from images.

## Features

- Text-based product search
- Image-based product search
- AI-powered recommendations using Gemini
- Vector similarity search for accurate product matching

## Architecture

- **Frontend**: React application with tabbed interface
- **Backend**: FastAPI + LangChain + PyTorch
- **ML Models**: 
  - Dual encoder for category-to-vector conversion
  - ResNet50-based image classifier for category detection

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- Google Gemini API key

### Backend Setup

1. Create a virtual environment and install dependencies:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
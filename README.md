# NeuroAd: Multimodal RAG System for Ad Analysis

## Overview

NeuroAd is a system that analyzes ad images and generates improvement recommendations using machine learning and artificial intelligence. The system extracts text from images, detects objects, creates embeddings, finds similar high-performing ads, and uses an AI model to suggest improvements.

## Requirements

- Python 3.9 or higher
- 8GB RAM minimum
- GPU is optional but recommended for faster processing

## Installation

### Step 1: Install Python Dependencies

Run this command to install all required packages:

```bash
pip install -r requirements.txt
```

### Step 2: Install and Setup Ollama

Ollama is needed to run the Llama3 language model locally.

1. Download Ollama from https://ollama.ai
2. Install it on your system
3. Open a terminal and run:

```bash
ollama serve
```

4. In a separate terminal, pull the Llama3 model:

```bash
ollama pull llama3
```

Keep the `ollama serve` terminal running while using the app.

## Running the Application

In a terminal, navigate to the project folder and run:

```bash
streamlit run test1.py
```

The app will open in your browser at http://localhost:8501

## How to Use

1. Create a folder called `ad_images` in your project directory
2. Add ad images (JPG or PNG) to this folder
3. In the app, click "Re-Index Database" in the sidebar to process all images
4. Upload an ad image you want to analyze
5. Enter the goal or intent of your ad
6. Click Submit to see:
   - Detected objects and extracted text
   - Top 3 similar high-performing ads
   - AI-generated recommendations

## How It Works

The system follows this process:

1. OCR Text Extraction: Uses easyOCR to extract visible text from images
2. Object Detection: Uses YOLOv8 to identify objects in the image
3. Embedding Creation: Creates a combined image and text embedding using CLIP
4. Similarity Search: Uses FAISS to find the most similar ads from the database
5. Ranking: Sorts results by CTR (click-through rate) performance
6. Recommendation: Sends the results to Llama3 to generate improvement suggestions

## System Components

Object Detection (YOLOv8): Identifies objects in images from 80 common categories
Text Extraction (easyOCR): Extracts readable text from images
Embeddings (CLIP): Creates vector representations combining image and text information
Retrieval (FAISS): Performs fast similarity search across indexed ads
Language Model (Llama3): Generates human-readable recommendations

## Limitations

Object detection works well for common objects like people, cars, chairs, and cups. For specialized products like headphones, watches, or cosmetics, detection may be less accurate. The system accounts for this by focusing recommendations on text content and visual composition rather than object labels.

The dataset currently uses simulated captions and metrics. In production, this would use real ad data with actual performance metrics.

## Tips

Make sure you have at least 10-20 images in the ad_images folder for good retrieval results.

If the app runs slowly, close other applications to free up RAM.

If you see connection errors to Ollama, make sure the `ollama serve` command is running in another terminal.

You can edit the marketing_captions list in test1.py to use different ad captions.

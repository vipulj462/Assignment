import streamlit as st
import os
import uuid
import json
import numpy as np
import faiss
import easyocr
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from langchain_community.llms import Ollama

# ==========================================
# 1. CONFIG
# ==========================================
DATA_DIR = "ad_images"
INDEX_FILE = "ad_index_v2.bin"
METADATA_FILE = "ad_metadata_v2.json"
MODEL_NAME = "clip-ViT-B-32" 
LLM_MODEL = "llama3"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

st.set_page_config(layout="wide", page_title="NeuroAd: Multimodal RAG System")

# ==========================================
# 2. LOAD MODELS (Cached)
# ==========================================
@st.cache_resource
def load_models():
    """Load all ML models"""
    embed_model = SentenceTransformer(MODEL_NAME)
    yolo_model = YOLO("yolov8n.pt")
    ocr_reader = easyocr.Reader(['en'])
    return embed_model, yolo_model, ocr_reader

embed_model, yolo_model, ocr_reader = load_models()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def process_image(image_path):
    """Extract OCR text and detected objects from image"""
    
    # 1. OCR - Extract text from image
    try:
        ocr_result = ocr_reader.readtext(image_path, detail=0)
        text_content = " ".join(ocr_result)
    except:
        text_content = ""

    # 2. Object Detection - What objects are in the image
    yolo_results = yolo_model(image_path, verbose=False)
    detected_objects = []
    detection_details = []
    
    for result in yolo_results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            detected_objects.append(class_name)
            detection_details.append({
                'class': class_name,
                'confidence': round(confidence, 3)
            })
    
    return text_content, list(set(detected_objects)), detection_details

def create_multimodal_embedding(image_path, caption, ocr_text, embed_model):
    """
    FIXED: Create multimodal embedding from RELIABLE signals only
    
    KEY FIX: Do NOT include YOLO objects in the embedding!
    
    Why:
    - YOLO objects are unreliable (non-COCO classes misclassified)
    - Confidence score ≠ semantic correctness
    - Including wrong objects corrupts the semantic vector
    - This cascades through retrieval and LLM reasoning
    
    What we embed instead:
    - Image embedding (visual features - reliable)
    - Caption (known good - we set it)
    - OCR text (extracted directly - reliable)
    - NOT: YOLO objects (potentially wrong)
    """
    
    # 1. Image Embedding
    try:
        img = Image.open(image_path)
        img_emb = embed_model.encode(img)
    except:
        img_emb = np.zeros(embed_model.get_sentence_embedding_dimension())
    
    # 2. Text Embedding - ONLY reliable signals
    # REMOVED: {' '.join(objects)} ← This was the problem!
    reliable_text = f"{caption} {ocr_text}"
    text_emb = embed_model.encode(reliable_text)
    
    # 3. Fusion: Weighted average
    fusion_emb = (0.6 * img_emb) + (0.4 * text_emb)
    
    # 4. Normalize for cosine similarity
    norm = np.linalg.norm(fusion_emb)
    if norm > 0:
        fusion_emb = fusion_emb / norm
        
    return fusion_emb.astype('float32')

# ==========================================
# 4. INDEXING PIPELINE
# ==========================================
def build_index():
    """Build FAISS index from all images in ad_images folder"""
    
    image_files = [f for f in os.listdir(DATA_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.error("No images found in ad_images/ folder!")
        return None, None

    embeddings = []
    metadata = []
    
    marketing_captions = [
        "Experience timeless elegance with our luxury collection.",
        "Boost your energy with organic natural ingredients.",
        "The ultimate performance gear for athletes.",
        "Refresh your senses with our new summer scent.",
        "Precision engineering meets modern design.",
        "Transform your lifestyle with innovation.",
        "Quality that speaks for itself.",
        "Discover what excellence feels like.",
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(image_files):
        path = os.path.join(DATA_DIR, file)
        
        # Extract features from image
        ocr_text, objects, detection_details = process_image(path)
        
        # Simulate marketing caption
        import random
        sim_caption = random.choice(marketing_captions)
        
        # Simulate CTR - based on reliable signals only
        base_ctr = random.uniform(0.5, 2.0)
        if len(ocr_text) > 5: 
            base_ctr += 0.5
        final_ctr = round(base_ctr, 2)

        # FIXED: Create embedding WITHOUT objects
        # Only use caption + OCR (reliable signals)
        emb = create_multimodal_embedding(path, sim_caption, ocr_text, embed_model)
        
        embeddings.append(emb)
        metadata.append({
            "id": str(uuid.uuid4()),
            "filename": file,
            "filepath": path,
            "caption": sim_caption,
            "ocr_text": ocr_text,
            # Store objects separately - for display/reference only
            # NOT used in embedding or semantic reasoning
            "objects_detected": objects,
            "detection_details": detection_details,
            "ctr": final_ctr
        })
        
        progress = (i + 1) / len(image_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{len(image_files)} images...")

    # Build FAISS index
    embeddings_np = np.array(embeddings)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    status_text.empty()
    progress_bar.empty()
    
    return index, metadata

def load_index():
    """Load existing FAISS index from disk"""
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return index, metadata
    return None, None

# ==========================================
# 5. RAG & LLM GENERATION
# ==========================================
def generate_ad_advice(input_data, retrieved_ads):
    """
    FIXED: Generate recommendations using ONLY reliable context
    
    KEY FIX: Do NOT pass YOLO objects to LLM
    
    We pass:
    - User's caption intent (reliable)
    - User's OCR text (reliable)
    - Reference ads' captions and OCR text (reliable)
    
    We DON'T pass:
    - YOLO objects (unreliable for non-COCO classes)
    """
    
    try:
        llm = Ollama(model=LLM_MODEL)
    except Exception as e:
        return f"⚠️ Error: Cannot connect to Ollama. Make sure to run 'ollama serve' in another terminal. Error: {str(e)}"

    # Sort by performance (CTR)
    retrieved_ads.sort(key=lambda x: x['ctr'], reverse=True)
    top_performers = retrieved_ads[:3]

    # Build context from top-performing ads
    # ONLY using reliable signals (caption + OCR)
    context_str = ""
    for idx, ad in enumerate(top_performers, 1):
        context_str += f"""
[Reference Ad #{idx}]
Performance Metric: {ad['ctr']}% CTR
Marketing Caption: "{ad['caption']}"
Text in Image: "{ad['ocr_text']}"
---
"""

    # Create prompt with ONLY reliable context
    prompt = f"""You are a Senior Creative Strategist analyzing ad performance.

USER'S AD CONCEPT:
- Ad Goal/Intent: "{input_data['caption_intent']}"
- Text in image: "{input_data['ocr_text']}"

[Note: We're focusing on text messaging and visual composition, which are reliable signals for ad effectiveness]

HIGH-PERFORMING REFERENCE ADS:
{context_str}

TASK:
Provide exactly 3 concrete, actionable recommendations to improve the user's ad.
Reference the appropriate ad when relevant.

FORMAT YOUR RESPONSE AS:
1. Copywriting Tweak: [specific text/messaging advice]
2. Visual Composition: [specific visual/design advice]  
3. Strategic Insight: [overall approach advice]
"""
    
    response = llm.invoke(prompt)
    return response

# ==========================================
# 6. STREAMLIT UI
# ==========================================
st.title("🚀 NeuroAd: AI-Powered Ad Analysis System")
st.markdown("""
**Multimodal RAG System** combining:
- 🎨 Computer Vision (OCR + Object Detection)
- 📊 Embeddings (CLIP multimodal fusion)
- 🔍 Retrieval (FAISS vector search)
- 🤖 Generative AI (LLM recommendations)

**Note**: Recommendations based on reliable signals (text + visual composition). 
Object detection shown for reference only.
""")

# Sidebar controls
with st.sidebar:
    st.header("🔧 Database Management")
    
    if st.button("🔄 Re-Index Database", use_container_width=True):
        with st.spinner("⏳ Processing images and building index..."):
            idx, meta = build_index()
            if idx:
                st.success(f"✅ Successfully indexed {len(meta)} ads")
            else:
                st.error("❌ No images found. Add images to 'ad_images/' folder")

    # Load current index
    index, metadata = load_index()
    if index:
        st.info(f"📚 Knowledge Base: {len(metadata)} ads indexed")
    else:
        st.warning("⚠️ No index found. Click 'Re-Index Database' to create one.")

# Main content
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

# LEFT COLUMN: User Input
with col1:
    st.subheader("📤 Upload Your Ad")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
    
    user_intent = st.text_input(
        "What is the goal of this ad?",
        value="Increase sales and brand awareness"
    )

# RIGHT COLUMN: Results
with col2:
    st.subheader("🏆 Reference Ads")
    st.info("Top 3 similar high-performing ads will appear here")

# MAIN ANALYSIS
st.divider()

if uploaded_file and index and user_intent:
    
    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display user's ad
    with col1:
        st.image(uploaded_file, caption="📸 Your Ad Concept", use_column_width=True)
    
    # Process the uploaded image
    with st.spinner("⚙️ Analyzing your ad..."):
        
        # Step 1: Extract features
        u_text, u_objs, u_details = process_image(temp_filename)
        
        # Step 2: Create embedding (FIXED - without objects)
        u_emb = create_multimodal_embedding(temp_filename, user_intent, u_text, embed_model)
        
        # Step 3: Retrieve similar ads
        k = 10
        D, I = index.search(np.array([u_emb]), k)
        
        # Step 4: Rerank by CTR (performance)
        candidates = [metadata[i] for i in I[0] if i < len(metadata)]
        candidates.sort(key=lambda x: x['ctr'], reverse=True)
        top_winners = candidates[:3]
    
    # Display analysis results
    st.subheader("🔍 Analysis Results")
    
    # Show detected objects with disclaimer
    if u_details:
        st.write("**Objects Detected by YOLO (for reference only):**")
        st.info("⚠️ Note: Object detection has limited accuracy on non-standard products. "
                "These are shown for information only. Recommendations are based on text and visual composition.")
        
        obj_text = ", ".join([f"{d['class']} ({d['confidence']*100:.1f}%)" 
                              for d in u_details])
        st.caption(obj_text)
    else:
        st.info("ℹ️ No objects detected")
    
    # Show OCR text if any
    if u_text:
        st.write("**Text in image:**")
        st.info(f'"{u_text}"')
    
    # Display retrieved reference ads
    st.divider()
    st.subheader("📊 Top 3 Similar High-Performers")
    
    ref_tabs = st.tabs([f"Reference Ad #{i+1}" for i in range(3)])
    
    for tab_idx, tab in enumerate(ref_tabs):
        with tab:
            if tab_idx < len(top_winners):
                ad = top_winners[tab_idx]
                
                col_img, col_info = st.columns([1, 1])
                
                with col_img:
                    st.image(ad['filepath'], width=250)
                
                with col_info:
                    st.metric("CTR Performance", f"{ad['ctr']}%")
                    st.write(f"**Caption:** {ad['caption']}")
                    
                    if ad['ocr_text']:
                        st.write(f"**Text:** {ad['ocr_text']}")
                    
                    # FIXED: Safely check for objects_detected (backward compatible)
                    objects_list = ad.get('objects_detected', [])  # ← FIX: use .get() with default
                    if objects_list:
                        objects_str = ", ".join(objects_list)
                        st.caption(f"Detected objects: {objects_str}")
            else:
                st.write("No reference ad available")
    
    # Generate recommendations
    st.divider()
    st.subheader("💡 AI-Powered Recommendations")
    
    # FIXED: Don't pass objects to recommendation engine
    input_data = {
        "ocr_text": u_text,
        "caption_intent": user_intent
        # ✅ Removed: "objects": u_objs
    }
    
    with st.spinner("🤖 Generating insights from Llama3..."):
        advice = generate_ad_advice(input_data, top_winners)
        st.markdown(advice)
    
    # Cleanup
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

else:
    if not index:
        st.warning("⚠️ No index loaded. Please click 'Re-Index Database' in the sidebar.")
    elif not uploaded_file:
        st.info("👈 Upload an image to get started")
    elif not user_intent:
        st.info("👈 Enter the ad intent to proceed")

st.divider()
st.caption("NeuroAd v2.0 | Multimodal RAG with Semantic Integrity")
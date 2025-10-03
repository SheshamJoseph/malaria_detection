import streamlit as st
from PIL import Image
import json
import os
from typing import List
import random
from typing import Tuple

def load_sample_cells(data_dir: str = "dataset") -> Tuple[List[Image.Image], List[str]]:
    """
    Load sample cells from your dataset directory with proper format conversion.
    """
    sample_cells = []
    sample_labels = []
    
    data_dir = "./cell_images" 
    try:
        parasitized_dir = os.path.join(data_dir, "Parasitized")
        uninfected_dir = os.path.join(data_dir, "Uninfected")
        
        # Load parasitized cells
        if os.path.exists(parasitized_dir):
            parasitized_files = [f for f in os.listdir(parasitized_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            selected_parasitized = random.sample(parasitized_files, 
                                               min(20, len(parasitized_files)))
            
            for filename in selected_parasitized:
                img_path = os.path.join(parasitized_dir, filename)
                try:
                    img = Image.open(img_path)
                    # Convert to RGB and ensure consistent size
                    img = img.convert('RGB')
                    # Resize to a standard size if needed
                    img = img.resize((128, 128), Image.Resampling.LANCZOS)
                    sample_cells.append(img)
                    sample_labels.append("parasitized")
                except Exception as e:
                    print(f"Could not load {img_path}: {e}")
        
        # Load uninfected cells
        if os.path.exists(uninfected_dir):
            uninfected_files = [f for f in os.listdir(uninfected_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            selected_uninfected = random.sample(uninfected_files, 
                                             min(20, len(uninfected_files)))
            
            for filename in selected_uninfected:
                img_path = os.path.join(uninfected_dir, filename)
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = img.resize((128, 128), Image.Resampling.LANCZOS)
                    sample_cells.append(img)
                    sample_labels.append("uninfected")
                except Exception as e:
                    print(f"Could not load {img_path}: {e}")
        
        print(f"✅ Loaded {len(sample_cells)} sample cells")
        
    except Exception as e:
        print(f"Error loading sample cells: {e}")
    
    return sample_cells, sample_labels

def display_results_single(confidence: float, inference_time: float, threshold: float = 0.5):
    """Display results for single image classification."""
    st.subheader("🔎 Prediction Results")
    
    if confidence >= threshold:
        st.error(f"**Parasitized** (Confidence: {confidence:.2%})")
    else:
        st.success(f"**Uninfected** (Confidence: {confidence:.2%})")
    
    st.write(f"**⏱ Inference Time:** {inference_time:.2f} ms")
    
    # Confidence meter
    st.progress(float(confidence))
    st.caption(f"Confidence score: {confidence:.3f}")

def display_results_batch(results: List[dict]):
    """Display results for batch processing."""
    st.subheader("📊 Batch Analysis Results")
    
    parasitized_count = sum(1 for r in results if r['classification'] == 'Parasitized')
    total_count = len(results)
    
    if total_count > 0:
        parasitemia = (parasitized_count / total_count) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cells", total_count)
        with col2:
            st.metric("Parasitized", parasitized_count)
        with col3:
            st.metric("Parasitemia", f"{parasitemia:.1f}%")
        
        # Display individual results in an expandable section
        with st.expander("View Individual Cell Results"):
            for i, result in enumerate(results):
                status_color = "🔴" if result['classification'] == 'Parasitized' else "🟢"
                st.write(f"{status_color} Cell {i+1}: {result['classification']} "
                        f"(Confidence: {result['confidence']:.2%})")
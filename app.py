import streamlit as st
import numpy as np
from PIL import Image
from detector import MalariaDetector, SlideAnalyzer
import utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🧪",
    layout="wide"
)

# --- Initialize Models ---
@st.cache_resource
def init_detector():
    return MalariaDetector("assets/model.tflite")

@st.cache_resource
def init_analyzer():
    return SlideAnalyzer()

# --- Main App ---
def main():
    st.title("🦠 Malaria Detection System")
    st.markdown("""
    This system detects malaria parasites in blood smear images using a SqueezeNet model 
    optimized for edge devices with TensorFlow Lite.
    """)
    
    # Initialize models
    detector = init_detector()
    analyzer = init_analyzer()
    
    # Sidebar for mode selection
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode:",
        ["Single Cell Analysis", "Synthetic Slide Analysis", "About"]
    )
    
    if app_mode == "Single Cell Analysis":
        single_cell_analysis(detector)
    elif app_mode == "Synthetic Slide Analysis":
        synthetic_slide_analysis(detector, analyzer)
    else:
        about_page()

def single_cell_analysis(detector: MalariaDetector):
    """Single cell classification interface."""
    st.header("🔬 Single Cell Analysis")
    st.write("Upload a single blood cell image for malaria detection.")
    
    uploaded_file = st.file_uploader(
        "Choose a cell image...", 
        type=["jpg", "png", "jpeg"],
        key="single_cell"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Cell Image", use_column_width=True)
        
        with col2:
            with st.spinner("Analyzing cell..."):
                confidence, inference_time = detector.predict_single_image(image)
                utils.display_results_single(confidence, inference_time)

def synthetic_slide_analysis(detector: MalariaDetector, analyzer: SlideAnalyzer):
    """Synthetic slide analysis interface."""
    st.header("🧫 Synthetic Slide Analysis")
    st.write("""
    This mode demonstrates the complete pipeline:
    1. Creates a synthetic blood smear slide
    2. Detects individual cells
    3. Classifies each cell
    4. Calculates overall parasitemia
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        num_cells = st.slider("Number of cells in synthetic slide", 10, 100, 30)
    with col2:
        analysis_threshold = st.slider("Classification threshold", 0.1, 0.9, 0.5, 0.1)
    
    if st.button("🔄 Generate and Analyze Synthetic Slide", type="primary"):
        with st.spinner("Creating synthetic slide and analyzing cells..."):
            # Note: You'll need to provide sample cells from your dataset
            # For now, this is a placeholder structure
            sample_cells, sample_labels = utils.load_sample_cells()
            
            if not sample_cells:
                st.warning("""
                **Demo Note:** Sample cells not loaded. 
                In a full implementation, you would load actual cell images from your test dataset here.
                """)
                # Create a placeholder synthetic slide for demo purposes
                placeholder_slide = create_placeholder_slide()
                st.image(placeholder_slide, caption="Synthetic Blood Smear (Placeholder)", use_column_width=True)
                return
            
            # Create synthetic slide
            synthetic_slide = analyzer.create_synthetic_slide(
                sample_cells, sample_labels, num_cells
            )
            
            # Display the slide
            st.subheader("Generated Synthetic Slide")
            st.image(synthetic_slide, caption="Synthetic Blood Smear", use_column_width=True)
            
            # Detect and analyze cells
            detected_cells = analyzer.detect_cells_in_slide(synthetic_slide)
            
            if detected_cells:
                st.subheader(f"Detected {len(detected_cells)} Cells")
                
                # Classify each cell
                results = []
                for i, cell in enumerate(detected_cells):
                    confidence, inference_time = detector.predict_single_image(cell)
                    classification, _ = detector.classify_cell(confidence, analysis_threshold)
                    
                    results.append({
                        'cell_id': i,
                        'classification': classification,
                        'confidence': confidence,
                        'inference_time': inference_time
                    })
                
                # Display batch results
                utils.display_results_batch(results)
                
            else:
                st.warning("No cells detected in the slide. Try adjusting the detection parameters.")

def create_placeholder_slide():
    """Create a placeholder slide for demo purposes."""
    from PIL import ImageDraw
    slide = Image.new('RGB', (800, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(slide)
    
    # Draw some circles to represent cells
    for _ in range(20):
        x = np.random.randint(50, 750)
        y = np.random.randint(50, 550)
        r = np.random.randint(20, 40)
        color = 'red' if np.random.random() > 0.7 else 'green'
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline='black')
    
    return slide

def about_page():
    """About page with project information."""
    st.header("About This Project")
    st.markdown("""
    ## Malaria Detection System
    
    **Final Year Project** 
    
    This system uses a SqueezeNet architecture converted to TensorFlow Lite for efficient 
    edge deployment to classify malaria-infected blood cells.
    
    ### Features:
    - 🧪 **Single Cell Analysis**: Upload individual cell images for immediate classification
    - 🧫 **Synthetic Slide Analysis**: Complete pipeline demonstration with synthetic slides
    - ⚡ **Edge Optimized**: Uses TFLite for fast inference on resource-constrained devices
    - 📊 **Comprehensive Results**: Detailed analysis with confidence scores and metrics
    
    ### Technical Stack:
    - **Model**: SqueezeNet (TFLite)
    - **Interface**: Streamlit
    - **Image Processing**: OpenCV, PIL
    - **Deployment**: Edge-device compatible
    """)

if __name__ == "__main__":
    main()
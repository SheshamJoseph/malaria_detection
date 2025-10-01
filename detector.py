import numpy as np
import time
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2
import random
from typing import Tuple, List, Dict, Optional


class MalariaDetector:
    """Malaria detection model handler for both single cell and whole slide analysis."""
    
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        # Assuming model expects (1, height, width, 3)
        self.target_size = (self.input_shape[2], self.input_shape[1])  # (width, height)
    
    def preprocess_single_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess a single cell image for classification."""
        image = image.convert("RGB").resize((self.input_shape[2], self.input_shape[1]))
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict_single_image(self, image: Image.Image) -> Tuple[float, float]:
        """Predict a single image and return confidence and inference time."""
        processed_image = self.preprocess_single_image(image)
        
        # Run inference
        start_time = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        self.interpreter.invoke()
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # ms
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Assuming binary classification: [prob_uninfected, prob_parasitized]
        confidence = float(output_data[0][1])  # Probability of being parasitized
        return confidence, inference_time
    
    def classify_cell(self, confidence: float, threshold: float = 0.5) -> Tuple[str, str]:
        """Classify cell based on confidence score."""
        if confidence > threshold:
            return "Parasitized", "red"
        else:
            return "Uninfected", "green"


class SlideAnalyzer:
    """Handles synthetic slide creation and cell detection."""
    
    def __init__(self, slide_size: Tuple[int, int] = (1200, 800)):
        self.slide_size = slide_size
        self.background_color = (240, 240, 240)  # Light gray background
    
    def create_synthetic_slide(self, cell_images: List[Image.Image], 
                             labels: List[str], num_cells: int = 50) -> Image.Image:
        """Create a synthetic blood smear slide with random cell placement."""
        # Create blank slide
        slide = np.ones((self.slide_size[1], self.slide_size[0], 3), dtype=np.uint8)
        slide = slide * self.background_color
        
        # Convert PIL Images to OpenCV format
        cv_slide = slide.copy()
        
        for _ in range(num_cells):
            # Randomly select a cell
            idx = random.randint(0, len(cell_images) - 1)
            cell_img = cell_images[idx]
            label = labels[idx]
            
            # Convert PIL to OpenCV
            cell_cv = cv2.cvtColor(np.array(cell_img), cv2.COLOR_RGB2BGR)
            
            # Random position (with border padding)
            pad = 100
            x = random.randint(pad, self.slide_size[0] - cell_cv.shape[1] - pad)
            y = random.randint(pad, self.slide_size[1] - cell_cv.shape[0] - pad)
            
            # Simple overlay (in real implementation, you'd do proper blending)
            try:
                cv_slide[y:y+cell_cv.shape[0], x:x+cell_cv.shape[1]] = cell_cv
            except ValueError:
                continue  # Skip if placement doesn't work
        
        return Image.fromarray(cv2.cvtColor(cv_slide, cv2.COLOR_BGR2RGB))
    
    def detect_cells_in_slide(self, slide_image: Image.Image) -> List[Image.Image]:
        """Detect and extract individual cells from a slide image."""
        # Convert to OpenCV format
        slide_cv = cv2.cvtColor(np.array(slide_image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(slide_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold using Otsu's method
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        extracted_cells = []
        min_area = 1000  # Minimum area to be considered a cell
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add padding and extract ROI
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(slide_cv.shape[1] - x, w + 2 * padding)
                h = min(slide_cv.shape[0] - y, h + 2 * padding)
                
                cell_roi = slide_cv[y:y+h, x:x+w]
                cell_pil = Image.fromarray(cv2.cvtColor(cell_roi, cv2.COLOR_BGR2RGB))
                extracted_cells.append(cell_pil)
        
        return extracted_cells
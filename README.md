
# Malaria Detection System

**Final Year Project**

## Overview
This project provides an automated system for detecting malaria parasites in blood smear images. It leverages a SqueezeNet architecture converted to TensorFlow Lite for efficient edge deployment, enabling fast and accurate classification of malaria-infected blood cells.

## Features
- 🧪 **Single Cell Analysis**: Upload individual cell images for immediate classification
- 🧫 **Synthetic Slide Analysis**: Complete pipeline demonstration with synthetic slides
- ⚡ **Edge Optimized**: Uses TFLite for fast inference on resource-constrained devices
- 📊 **Comprehensive Results**: Detailed analysis with confidence scores and metrics

## Technical Stack
- **Model**: SqueezeNet (TensorFlow Lite)
- **Interface**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Deployment**: Edge-device compatible

## Installation
1. **Clone the repository:**
	```bash
	git clone https://github.com/SheshamJoseph/malaria_detection.git
	cd malaria_detection
	```
2. **Install Python 3.10 (recommended):**
	Use [pyenv](https://github.com/pyenv/pyenv) or your system package manager.
3. **Create and activate a virtual environment:**
	```bash
	python3.10 -m venv .venv
	source .venv/bin/activate
	```
4. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
Follow the instructions in the web interface to analyze cell images or synthetic slides.

## Dataset
The project uses a dataset of blood smear cell images, organized as follows:

- `cell_images/Parasitized/` — Images of malaria-infected cells
- `cell_images/Uninfected/` — Images of healthy cells

Link to the dataset [here](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip)

## Model
The SqueezeNet model is trained and converted to TensorFlow Lite format for edge deployment. Model files are located in the `assets/` directory:
- `model.tflite` — Main TFLite model
- `best_model.keras`, `fine_tuned_model.keras` — Keras model checkpoints

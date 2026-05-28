import os
import numpy as np
import torch
import torch.nn as nn
import timm
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tempfile
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Deepfake Detection API")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the same Xception model class as in training
class XceptionDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionDeepfakeModel, self).__init__()
        
        # Load Xception from timm - use legacy_xception to match training
        self.xception = timm.create_model('legacy_xception', pretrained=True, num_classes=0)
        
        # Freeze all layers first (as done in training)
        for param in self.xception.parameters():
            param.requires_grad = False
        
        # Unfreeze last few blocks (as done in training)
        children = list(self.xception.children())
        for i, child in enumerate(children[-3:]):  # Last 3 blocks
            for param in child.parameters():
                param.requires_grad = True
        
        # Get feature dimension
        self.feature_dim = self.xception.num_features
        
        # Custom classifier (same as in training)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.xception(x)
        return self.classifier(features)

# Initialize model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = XceptionDeepfakeModel(num_classes=2).to(device)

# Load the trained model
MODEL_PATH = "xception_deepfake_model.pth"
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Extract model_state_dict from checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from checkpoint")
        
        # Print test accuracy from training if available
        if 'test_accuracy' in checkpoint:
            print(f"Model test accuracy during training: {checkpoint['test_accuracy']:.4f}")
    else:
        # If it's directly the state_dict
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully (direct state_dict)")
    
    model.eval()
    print(f"Model evaluation mode set")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# SRM filter functions (same as in training)
def srm_grayscale(image):
    """Apply SRM filter to extract noise residuals in grayscale."""
    # Ensure image is numpy array
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose(1, 2, 0)
    
    # Convert to float32
    img = np.array(image, dtype=np.float32)
    
    # Convert to grayscale if RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        img_gray = img
    
    # Add batch dimension
    img_gray = np.expand_dims(img_gray, axis=(0, -1))  # Shape: (1, H, W, 1)
    
    # SRM 5x5 filters (same as in training)
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    
    # Normalization factors
    q = [4.0, 12.0, 2.0]
    
    # Create normalized filters
    filter1 = np.asarray(filter1, dtype=np.float32) / q[0]
    filter2 = np.asarray(filter2, dtype=np.float32) / q[1]
    filter3 = np.asarray(filter3, dtype=np.float32) / q[2]
    
    # Create filters for single-channel input
    filters_list = []
    for f in [filter1, filter2, filter3]:
        filter_single = np.expand_dims(np.expand_dims(f, axis=-1), axis=-1)
        filters_list.append(filter_single)
    
    # Stack filters to get shape (5, 5, 1, 3)
    filters = np.concatenate(filters_list, axis=-1)
    
    # Apply convolution using TensorFlow
    input_tensor = tf.constant(img_gray, dtype=tf.float32)
    
    # Convolution with 3 filters
    output = tf.nn.conv2d(
        input=input_tensor,
        filters=filters,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    
    # Take absolute value and combine
    output = tf.abs(output)
    output_combined = tf.reduce_max(output, axis=-1, keepdims=True)
    
    # Convert to numpy
    res = output_combined.numpy()[0, :, :, 0]
    
    return res

def normalize_srm(srm_output):
    """Normalize SRM output to [0, 255] for saving"""
    srm_norm = srm_output.astype(np.float32)
    min_val = np.min(srm_norm)
    max_val = np.max(srm_norm)
    
    if max_val - min_val == 0:
        return np.zeros_like(srm_norm)
    
    normalized = (srm_norm - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def preprocess_for_xception(srm_image):
    """Convert SRM image to Xception input format"""
    IMG_SIZE = 299  # Xception input size
    
    # Resize
    srm_resized = cv2.resize(srm_image, (IMG_SIZE, IMG_SIZE))
    
    # Convert single channel to 3 channels
    if len(srm_resized.shape) == 2:
        srm_resized = np.stack([srm_resized, srm_resized, srm_resized], axis=-1)
    
    # Normalize to [-1, 1] (same as in training)
    srm_normalized = srm_resized.astype(np.float32) / 255.0
    srm_normalized = (srm_normalized * 2.0) - 1.0
    
    return srm_normalized

def process_image(image_bytes):
    """Process uploaded image for prediction"""
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to numpy array (RGB)
        image = np.array(pil_image.convert('RGB'))
        
        # Apply SRM filter
        srm_raw = srm_grayscale(image)
        srm_normalized = normalize_srm(srm_raw)
        
        # Preprocess for Xception
        processed_image = preprocess_for_xception(srm_normalized)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection API is running",
        "device": str(device),
        "model": "Xception with SRM filter"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Check file size
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            return {
                "status": "error",
                "message": "File size too large. Maximum size is 10MB."
            }
        
        # Process image
        image_tensor = process_image(contents)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Get confidence scores
        real_prob = probabilities[0][0].item() * 100
        fake_prob = probabilities[0][1].item() * 100
        
        # Determine result (0: real, 1: fake as per training)
        result = "REAL" if predicted.item() == 0 else "FAKE"
        confidence = real_prob if result == "REAL" else fake_prob
        
        return {
            "status": "success",
            "prediction": result,
            "confidence": round(confidence, 2),
            "real_probability": round(real_prob, 2),
            "fake_probability": round(fake_prob, 2),
            "model_confidence": "high" if confidence > 85 else "medium" if confidence > 70 else "low"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
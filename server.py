from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import numpy as np
import torch
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import os
import json


# Define the models for the request data
class Point(BaseModel):
    x: int
    y: int

class PointSegmentationRequest(BaseModel):
    points: List[Point]
    labels: List[int]
    alpha_only: bool = False  # Add this field with a default value

class HoverPreviewRequest(BaseModel):
    point: Point

class AlphaMaskRequest(BaseModel):
    point: Point


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the SAM model
MODEL_TYPE = "vit_h"  # Options: vit_h, vit_l, vit_b
CHECKPOINT_PATH = "backend/sam_vit_h.pth"  # Path to your downloaded checkpoint

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=device)
predictor = SamPredictor(sam)

# Create a mask generator for automatic masks (for hover feature) - Modified for fine-grained segmentation
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,  # Increase from 32 to 64 for finer granularity
    pred_iou_thresh=0.86,  # Slightly lower threshold to capture more details
    stability_score_thresh=0.92,  # Lower stability threshold for smaller objects
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  # Reduce to capture smaller objects
    box_nms_thresh=0.7,  # Lower NMS threshold for more mask variety
    crop_nms_thresh=0.7,
    point_grids=None  # Use default grid
)

# Keep a reference to the current image and pre-computed masks
current_image = None
all_masks = None

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    global current_image, all_masks
    
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Convert to RGB (in case it's RGBA or another format)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array for SAM
    image_np = np.array(image)
    
    # Store the image for later use
    current_image = image_np
    
    # Set the image in the predictor
    predictor.set_image(image_np)
    
    # Pre-compute all possible masks for hover-based interaction
    # Note: This can be slow for large images, so we do it at upload time
    print("Generating automatic masks... (this may take a while)")
    all_masks = mask_generator.generate(image_np)
    print(f"Generated {len(all_masks)} masks")
    
    return {"message": "Image uploaded and processed successfully"}

@app.post("/hover_preview")
async def hover_preview(request: HoverPreviewRequest):
    global current_image, all_masks
    
    if current_image is None:
        raise HTTPException(status_code=400, detail="Please upload an image first")
    
    if all_masks is None or len(all_masks) == 0:
        raise HTTPException(status_code=400, detail="No masks available for hover preview")
    
    # Get the hover point
    hover_x = request.point.x
    hover_y = request.point.y
    
    # Find suitable masks at the hovered position
    suitable_masks = []
    
    for i, mask in enumerate(all_masks):
        # Check if the point is inside this mask
        if mask["segmentation"][hover_y, hover_x]:
            # Calculate a score that prioritizes smaller masks
            area = mask["area"]
            iou_score = mask["predicted_iou"]
            # Prioritize smaller areas with decent IoU scores
            # This formula prioritizes smaller masks but still considers quality
            combined_score = iou_score * (1.0 - (area / (current_image.shape[0] * current_image.shape[1])))
            suitable_masks.append((i, combined_score, mask["segmentation"]))
    
    # Sort by combined score (descending)
    suitable_masks.sort(key=lambda x: x[1], reverse=True)
    
    # If no mask was found, return empty mask
    if not suitable_masks:
        # Create a transparent mask
        mask_rgba = np.zeros((current_image.shape[0], current_image.shape[1], 4), dtype=np.uint8)
    else:
        # Take the top-scoring mask (prioritizing smaller objects)
        best_mask = suitable_masks[0][2]
        
        # Create the mask overlay
        mask_rgba = np.zeros((current_image.shape[0], current_image.shape[1], 4), dtype=np.uint8)
        
        # Set the mask with a highlight color
        highlight_color = np.array([255, 255, 0, 180])  # Yellow with semi-transparency
        
        # Set the color for the mask area
        for i in range(4):  # for RGBA channels
            mask_rgba[:, :, i] = np.where(best_mask, highlight_color[i], 0)
    
    # Convert to PIL Image
    result_pil = Image.fromarray(mask_rgba, 'RGBA')
    
    # Create a BytesIO object to save the image
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.post("/alpha_mask")
async def get_alpha_mask(request: AlphaMaskRequest):
    global current_image, all_masks
    
    if current_image is None:
        raise HTTPException(status_code=400, detail="Please upload an image first")
    
    if all_masks is None or len(all_masks) == 0:
        raise HTTPException(status_code=400, detail="No masks available")
    
    # Get the hover point
    point_x = request.point.x
    point_y = request.point.y
    
    # Find suitable masks at the clicked position
    suitable_masks = []
    
    for i, mask in enumerate(all_masks):
        # Check if the point is inside this mask
        if mask["segmentation"][point_y, point_x]:
            # Calculate a score that prioritizes smaller masks
            area = mask["area"]
            iou_score = mask["predicted_iou"]
            # Prioritize smaller areas with decent IoU scores
            combined_score = iou_score * (1.0 - (area / (current_image.shape[0] * current_image.shape[1])))
            suitable_masks.append((i, combined_score, mask["segmentation"]))
    
    # Sort by combined score (descending)
    suitable_masks.sort(key=lambda x: x[1], reverse=True)
    
    # If no mask was found, return an empty mask
    if not suitable_masks:
        # Create a transparent mask
        mask_image = np.zeros((current_image.shape[0], current_image.shape[1], 4), dtype=np.uint8)
    else:
        # Take the top-scoring mask (prioritizing smaller objects)
        best_mask = suitable_masks[0][2]
        
        # Create an RGBA image with ONLY alpha channel (pure alpha mask)
        mask_image = np.zeros((current_image.shape[0], current_image.shape[1], 4), dtype=np.uint8)
        
        # Set alpha channel to 255 where mask is True
        mask_image[:, :, 3] = np.where(best_mask, 255, 0)
        
        # Set RGB to white for better visibility
        for i in range(3):  # for RGB channels
            mask_image[:, :, i] = 255
    
    # Convert to PIL Image
    result_pil = Image.fromarray(mask_image, 'RGBA')
    
    # Create a BytesIO object to save the image
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.post("/segment_points")
async def segment_with_points(request: PointSegmentationRequest):
    global current_image
    
    if current_image is None:
        raise HTTPException(status_code=400, detail="Please upload an image first")
    
    if not request.points:
        raise HTTPException(status_code=400, detail="No points provided")
    
  # Convert points and labels to NumPy arrays
    input_points = np.array([[p.x, p.y] for p in request.points])
    input_labels = np.array(request.labels)
    
    # Check if we have valid points and labels
    if len(input_points) != len(input_labels):
        raise HTTPException(status_code=400, detail="Number of points and labels must match")
    
    # Generate masks using SAM's point prompts
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,  # Return multiple masks
    )
    
    # Get the highest-scoring mask
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    
    # Create an RGBA image
    mask_rgba = np.zeros((current_image.shape[0], current_image.shape[1], 4), dtype=np.uint8)
    
    if request.alpha_only:
        # Set alpha channel to 255 where mask is True
        mask_rgba[:, :, 3] = np.where(best_mask, 255, 0)
        
        # Set RGB to white for better visibility
        for i in range(3):  # for RGB channels
            mask_rgba[:, :, i] = 255
    else:
        # Set the mask with color (adjust color as needed)
        mask_color = np.array([30, 144, 255, 180])  # RGBA: Dodger Blue with semi-transparency
        
        # Set the color for the mask area
        for i in range(4):  # for RGBA channels
            mask_rgba[:, :, i] = np.where(best_mask, mask_color[i], 0)
    
    # Convert to PIL Image
    result_pil = Image.fromarray(mask_rgba, 'RGBA')
    
    # Create a BytesIO object to save the image
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.post("/segment")
async def segment_image(brushed_mask: UploadFile = File(...)):
    global current_image
    
    if current_image is None:
        raise HTTPException(status_code=400, detail="Please upload an image first")
    
    # Read the brushed mask from the request
    contents = await brushed_mask.read()
    mask_img = Image.open(io.BytesIO(contents))
    mask_np = np.array(mask_img)
    
    # Extract the red channel since you're drawing with red in your frontend
    mask_red = mask_np[:,:,0]
    
    # Create a binary mask from the brushed areas
    binary_mask = mask_red > 100  # Threshold to avoid background noise
    
    # Use this mask as an additional input to guide SAM
    # First get points from the brushed area
    points = np.argwhere(binary_mask)
    
    if len(points) == 0:
        raise HTTPException(status_code=400, detail="No brushed areas found in mask")
    
    # Convert from (y, x) to (x, y) as required by SAM
    input_points = np.flip(points, axis=1)
    
    # Calculate the bounding box of the brush strokes
    y_indices, x_indices = np.where(binary_mask)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    
    # Calculate the center of the brush strokes
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2
    
    # Add a center point and some points around the boundary
    selected_points = []
    
    # Add center point
    selected_points.append([center_x, center_y])
    
    # Add boundary points
    boundary_points = [
        [min_x, min_y],  # Top-left
        [max_x, min_y],  # Top-right
        [min_x, max_y],  # Bottom-left
        [max_x, max_y],  # Bottom-right
        [center_x, min_y],  # Top-center
        [center_x, max_y],  # Bottom-center
        [min_x, center_y],  # Left-center
        [max_x, center_y],  # Right-center
    ]
    
    for point in boundary_points:
        if binary_mask[point[1], point[0]]:
            selected_points.append(point)
    
    # Add some random points from the brush strokes for better coverage
    indices = np.random.choice(len(input_points), min(40, len(input_points)), replace=False)
    for idx in indices:
        selected_points.append([input_points[idx][0], input_points[idx][1]])
    
    # Convert to numpy array
    input_points = np.array(selected_points)
    
    # Get point labels (all foreground for brushed areas)
    input_labels = np.ones(len(input_points))
    
    # Generate masks
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,  # Return multiple masks
    )
    
    # Get the highest-scoring mask
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    
    # Convert the boolean mask to an image with alpha channel
    mask_image = np.zeros((best_mask.shape[0], best_mask.shape[1], 4), dtype=np.uint8)
    
    # Create a pure alpha mask (white with alpha)
    # Set RGB to white
    for i in range(3):
        mask_image[:, :, i] = 255
    
    # Set alpha channel - fully opaque where mask is True, transparent elsewhere
    mask_image[:, :, 3] = np.where(best_mask, 255, 0)
    
    # Convert result to PIL Image
    result_pil = Image.fromarray(mask_image, 'RGBA')
    
    # Create a BytesIO object to save the image
    img_byte_arr = io.BytesIO()
    result_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
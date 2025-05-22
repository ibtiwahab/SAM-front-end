from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

@app.post("/segment")
async def segment_area(brushed_mask: UploadFile, image_url: str = Form(...)):
    # Load user image and brushed mask
    user_image = np.array(Image.open(image_url))
    mask_img = Image.open(io.BytesIO(await brushed_mask.read())).convert("L")
    binary_mask = np.array(mask_img) > 128

    # Apply SAM segmentation on the masked area
    masks = mask_generator.generate(user_image, input_mask=binary_mask)

    # Compose alpha mask output
    alpha = np.zeros_like(binary_mask, dtype=np.uint8)
    for m in masks:
        alpha[m['segmentation']] = 255

    result = Image.fromarray(alpha).convert("L")
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")

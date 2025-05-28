# api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import uvicorn
import mimetypes

#from predictor import load_model, predict_quality, MultiTaskModel
from src.fastapi.predictor import load_model, predict_quality, MultiTaskModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Quality Prediction Service",
    description="A service to predict image quality via classification probability and regression score",
    version="1.0.0"
)

# Load model at startup
@app.on_event("startup")
async def load_model_event():
    global model
    try:
        logger.info("Loading model...")
        model = load_model(model_path="model.pth", device="cpu")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise

# Response models
class HealthResponse(BaseModel):
    status: str

class QualityResponse(BaseModel):
    classification_prob: float
    regression_score: float

@app.get("/ping", response_model=HealthResponse)
async def ping():
    """
    Health check endpoint.
    Returns a simple JSON payload indicating the service is alive.
    """
    return HealthResponse(status="alive")

@app.post("/predict", response_model=QualityResponse)
async def predict_image_quality(file: UploadFile = File(...)):
    """
    Predict image quality from an uploaded image file.

    - **file**: UploadFile, the image to analyze.
    """
    # Determine file name and extension
    filename = file.filename or ""
    ext = filename.lower().split('.')[-1]
    if ext not in {"png", "jpg", "jpeg"}:
        # Fallback to content-type detection if no extension
        content_type = file.content_type or mimetypes.guess_type(filename)[0] or ""
        if not content_type.startswith("image/"):
            logger.error("Unsupported file type: filename=%s, content_type=%s", filename, content_type)
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PNG or JPEG image.")

    try:
        image_bytes = await file.read()
        cls_prob, reg_score = predict_quality(model, image_bytes, device="cpu")
        response = QualityResponse(
            classification_prob=round(cls_prob, 4),
            regression_score=round(reg_score, 4)
        )
        logger.info("Prediction successful: %s", response)
        return response

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
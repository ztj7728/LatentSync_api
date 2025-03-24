import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from latentsync_service import LatentSyncService

app = FastAPI()
service = LatentSyncService()

class InferenceRequest(BaseModel):
    video_path: str
    audio_path: str
    output_path: str
    guidance_scale: float = 1.5
    inference_steps: int = 20
    seed: int = None

@app.on_event("startup")
async def startup_event():
    """Startup hook to load models"""
    service.load_model()
    print("Service initialized and ready!")

@app.post("/inference")
async def inference(request: InferenceRequest):
    try:
        service.inference(
            video_path=request.video_path,
            audio_path=request.audio_path,
            output_path=request.output_path,
            guidance_scale=request.guidance_scale,
            inference_steps=request.inference_steps,
            seed=request.seed
        )
        return JSONResponse(content={
            "status": "success",
            "output_path": request.output_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    """Check if service is ready"""
    return {"status": "ready" if service.pipeline is not None else "not_ready"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port) 
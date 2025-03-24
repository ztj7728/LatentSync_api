from fastapi import FastAPI
import torch
from omegaconf import OmegaConf
from scripts.inference import LipsyncPipeline, Audio2Feature, UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed  # 修正这里的导入路径
from pydantic import BaseModel
import os

app = FastAPI()

class InferenceRequest(BaseModel):
    video_path: str
    audio_path: str
    video_out_path: str
    inference_steps: int = 20
    guidance_scale: float = 1.0
    seed: int = 1247
    device: str = "auto"

class ModelService:
    def __init__(self, unet_config_path, inference_ckpt_path):
        self.device = "cpu"  # 默认设置
        self.dtype = torch.float32  # 默认精度设置
        self.model = self.load_model(unet_config_path, inference_ckpt_path)

    def load_model(self, unet_config_path, inference_ckpt_path):
        print(f"Loading model from {inference_ckpt_path}")
        
        # 加载配置文件
        config = OmegaConf.load(unet_config_path)
        
        # 加载其他组件
        scheduler = DDIMScheduler.from_pretrained("configs")

        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device=self.device,
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        denoising_unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            inference_ckpt_path,
            device="cpu",  # Load to CPU first, then move to target device
        )

        denoising_unet = denoising_unet.to(device=self.device, dtype=self.dtype)

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        ).to(self.device)

        return pipeline

    def inference(self, video_path, audio_path, video_out_path, inference_steps=20, guidance_scale=1.0, seed=1247):
        # 根据提供的参数执行推理
        set_seed(seed)
        self.model(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=video_out_path,
            video_mask_path=video_out_path.replace(".mp4", "_mask.mp4"),
            num_frames=20,  # config 中的 frames，可以根据需要调整
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=self.dtype,
            width=512,  # 假设分辨率
            height=512,
        )

# 初始化模型服务
model_service = ModelService(unet_config_path="configs/unet.yaml", inference_ckpt_path="checkpoints/latentsync_unet.pt")

@app.post("/inference/")
async def inference(request: InferenceRequest):
    # 调用已加载的模型进行推理
    model_service.inference(
        video_path=request.video_path,
        audio_path=request.audio_path,
        video_out_path=request.video_out_path,
        inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
    )
    return {"message": "Inference completed successfully."}

# 启动 FastAPI 服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

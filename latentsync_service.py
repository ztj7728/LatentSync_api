import os
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

class LatentSyncService:
    def __init__(self):
        self.pipeline = None
        
    def load_model(self, unet_config_path="configs/unet/stage2.yaml", 
                  inference_ckpt_path="checkpoints/latentsync_unet.pt"):
        """Initialize and load all models"""
        if self.pipeline is not None:
            return
            
        # Check if GPU supports float16
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32
        
        print("Loading models...")
        
        # Load config
        self.config = OmegaConf.load(unet_config_path)
        
        # Initialize scheduler
        scheduler = DDIMScheduler.from_pretrained("configs")
        
        # Load audio encoder
        if self.config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif self.config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
            
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=self.config.data.num_frames,
            audio_feat_length=self.config.data.audio_feat_length,
        )
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        # Load UNet
        denoising_unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            inference_ckpt_path,
            device="cpu",
        )
        denoising_unet = denoising_unet.to(dtype=dtype)
        
        # Create pipeline
        self.pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        ).to("cuda")
        
        print("Models loaded and ready for inference!")

    def inference(self, video_path, audio_path, output_path,
                 guidance_scale=1.5, inference_steps=20, seed=None):
        """Run inference with loaded models"""
        if self.pipeline is None:
            raise RuntimeError("Models not loaded. Call load_model() first!")
            
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video path '{video_path}' not found")
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio path '{audio_path}' not found")
            
        print(f"Running inference...")
        print(f"Input video: {video_path}")
        print(f"Input audio: {audio_path}")
        print(f"Output path: {output_path}")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        # Run inference
        self.pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_path,
            video_mask_path=output_path.replace(".mp4", "_mask.mp4"),
            num_frames=self.config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=torch.float16,
            width=self.config.data.resolution,
            height=self.config.data.resolution,
            mask_image_path=self.config.data.mask_image_path,
        )
        
        print(f"Inference complete! Output saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, default="checkpoints/latentsync_unet.pt")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True) 
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Create service and load models
    service = LatentSyncService()
    service.load_model(args.unet_config_path, args.inference_ckpt_path)
    
    # Run inference
    service.inference(
        video_path=args.video_path,
        audio_path=args.audio_path, 
        output_path=args.output_path,
        guidance_scale=args.guidance_scale,
        inference_steps=args.inference_steps,
        seed=args.seed
    ) 

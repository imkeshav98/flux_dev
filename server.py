from io import BytesIO
import torch
import litserve as ls
from diffusers import FluxPipeline
from PIL import Image
from fastapi import Response

class FluxLitAPI(ls.LitAPI):
    def setup(self, device):
        # Load the Flux model with torch_dtype=torch.bfloat16
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )

    def decode_request(self, request):
        # Extract and validate parameters from the request
        try:
            prompt = request.get("prompt", "")
            height = int(request.get("height", 1024))
            width = int(request.get("width", 1024))
            guidance_scale = float(request.get("guidance_scale", 3.5))
            num_inference_steps = int(request.get("num_inference_steps", 50))
            max_sequence_length = int(request.get("max_sequence_length", 512))
            seed = int(request.get("seed", 0))
        except ValueError as e:
            raise ValueError(f"Invalid parameter value: {e}")
        return (
            prompt, height, width, guidance_scale,
            num_inference_steps, max_sequence_length, seed
        )

    def predict(self, prompt, height, width, guidance_scale, num_inference_steps, max_sequence_length, seed):
        # Generate image using the Flux pipeline
        generator = torch.Generator("cpu").manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]
        return image

    def encode_response(self, image):
        try:
            # Convert the image to bytes and return as a Response
            buffered = BytesIO()
            image.save(buffered, format="PNG")

            # save output image to disk
            image.save("output.png")
            
            return Response(content=buffered.getvalue(), headers={"Content-Type": "image/png"})
        except Exception as e:
            # Handle exceptions
            return Response(content=str(e), status_code=500)
        
# Start the LitServer
if __name__ == "__main__":
    api = FluxLitAPI()
    server = ls.LitServer(api, accelerator="gpu")
    server.run(port=8000)
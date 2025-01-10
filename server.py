from io import BytesIO
import torch
import time
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
        # Extract prompt from request
        prompt = request["prompt"]
        return prompt

    def predict(self, prompt):
        # Generate image using the Flux pipeline
        generator = torch.Generator().manual_seed(int(time.time()))
        image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=30,
            num_inference_steps=28,
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
from io import BytesIO
import torch
import time
from flask import Flask, request, send_file, jsonify
from diffusers import FluxPipeline
from PIL import Image

app = Flask(__name__)

# Initialize the Flux pipeline globally
pipe = None

def initialize_model():
    global pipe
    torch.cuda.empty_cache()
    
    if pipe is None:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")  # Move model to GPU

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        # Extract prompt from request
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "No prompt provided"}), 400
            
        prompt = data["prompt"]
        
        # Generate image using the Flux pipeline
        generator = torch.Generator().manual_seed(int(time.time()))
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=30,
            num_inference_steps=28,
            generator=generator
        ).images[0]
        
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        # Save output image to disk (optional)
        image.save("output.png")
        
        # Return the image
        return send_file(
            buffered,
            mimetype="image/png"
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Initialize the model before starting the server
    initialize_model()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=8000, debug=False)
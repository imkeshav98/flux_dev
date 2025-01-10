import requests

url = "http://localhost:8000/predict/"
headers = {"Content-Type": "application/json"}
data = {
    "prompt": "Create a vibrant summer-themed modern social media advertisement for an Indian Clothing Brand. A small brand logo is visible in the corner of the image. The brand name 'Suta' can be seen with a simple professional font. A brand tagline in simple font reads 'Feel the Breeze, Wear the Comfort'. A call-to-action button with the text 'Explore Collection' is placed at the bottom of the image. The primary focus is a breezy, lightweight summer dress, elegantly draped and exuding comfort and style. The background is a minimalistic depiction of a sunlit beach with gentle waves and a clear blue sky, enhancing the summer vibe. The playful theme is accentuated by the brand's warm golden-yellow color, creating a visually appealing final image.",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 7.5,
    "num_inference_steps": 28,
    "max_sequence_length": 256,
    "seed": 42
}
response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
else:
    print(f"Error: {response.status_code} - {response.text}")
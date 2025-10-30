import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse  # Import the argument parser library

def generate_image(prompt, negative_prompt, model_id, file_name="generated_image.png"):
    print("Checking for CUDA (GPU) availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available. Running on CPU. This will be very slow.")
    else:
        print(f"CUDA is available! Using {torch.cuda.get_device_name(0)}")

    print(f"Loading model: {model_id}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        # torch_dtype=torch.float16,  <- I am removing this line to force float32
        use_safensors=True
    )
    
    pipe = pipe.to(device)
    
    # This line disables the NSFW safety checker
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    print(f"Generating image for prompt: '{prompt}'...")
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=50, 
            guidance_scale=7.5
        ).images[0]

    print(f"Saving image to {file_name}...")
    
    image.save(file_name)
    print("Done!")

if __name__ == "__main__":
    
    # --- Set up the command-line argument parser ---
    parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion.")
    
    # Required argument for the prompt
    parser.add_argument(
        "-p", 
        "--prompt", 
        type=str, 
        required=True, 
        help="The text prompt to generate an image from."
    )
    
    # Optional argument for the negative prompt
    parser.add_argument(
        "-np",
        "--negative_prompt",
        type=str,
        default="blurry, low quality, bad art, ugly, deformed, watermark, text, signature",
        help="The negative prompt (what you don't want to see)."
    )
    
    # Optional argument for the output filename
    parser.add_argument(
        "-o",
        "--output_filename",
        type=str,
        default="generated_image.png",
        help="The name of the file to save the image as."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # --- Call the function with the arguments from the terminal ---
    
    model_to_use = "runwayml/stable-diffusion-v1-5"
    
    generate_image(
        prompt=args.prompt, 
        negative_prompt=args.negative_prompt, 
        model_id=model_to_use, 
        file_name=args.output_filename  # <-- This line is now fixed
    )



# ComfyUI-LTXVideo
ComfyUI-LTXVideo is a collection of custom nodes for ComfyUI designed to integrate the LTXVideo diffusion model. These nodes enable workflows for text-to-video, image-to-video, and video-to-video generation. The main LTXVideo repository can be found [here](https://github.com/Lightricks/LTX-Video).

### Single ltx 13b
```python
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    #model, vae, _ = CheckpointLoaderSimple('ltxv-13b-0.9.7-distilled-fp8.safetensors')
    model, clip, vae = CheckpointLoaderSimple('ltxv-13b-0.9.7-distilled-fp8.safetensors')
    clip = CLIPLoader('t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors', 'ltxv', 'default')
    clip_text_encode_positive_prompt_conditioning = CLIPTextEncode('', clip)
    clip_text_encode_negative_prompt_conditioning = CLIPTextEncode('low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly', clip)
    positive, negative = LTXVConditioning(clip_text_encode_positive_prompt_conditioning, clip_text_encode_negative_prompt_conditioning, 24.000000000000004)
    guider = STGGuiderAdvanced(model, positive, negative, 0.9970000000000002, True, '1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180', '1,1,1,1,1,1', '0,0,0,0,0,0', '1, 1, 1, 1, 1, 1', '[25], [35], [35], [42], [42], [42]', None)
    sampler = KSamplerSelect('euler_ancestral')
    float = StringToFloatList('1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219, 0.0')
    sigmas = FloatToSigmas(float)
    noise = RandomNoise(118)
    image, _ = LoadImage('037.png')
    denoised_output = LTXVBaseSampler(model, vae, 768, 512, 97, guider, sampler, sigmas, noise, image, '0', 0.8, 'center', 30, 1)
    vae2 = SetVAEDecoderNoise(vae, 0.05, 0.025, 42)
    image2 = VAEDecode(denoised_output, vae2)
    _ = VHSVideoCombine(image2, 24, 0, 'ltxv-base', 'video/h264-mp4', False, False, None, None, None)
```

### batch ltx 13b
```python
import os
import time
import subprocess
import shutil
from pathlib import Path

# Configuration
SEED = 661695664686456
SOLAR_TERMS_IMAGES_DIR = 'solar_terms_images'
INPUT_DIR = 'ComfyUI/input'
OUTPUT_DIR = 'ComfyUI/temp'
PYTHON_PATH = '/environment/miniconda3/bin/python'

def copy_images_to_input():
    """Copy all images from solar_terms_images to ComfyUI/input"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # Clear input directory first
    for file in Path(INPUT_DIR).glob('*'):
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # Copy new images
    for img_file in Path(SOLAR_TERMS_IMAGES_DIR).glob('*'):
        if img_file.is_file():
            try:
                shutil.copy(img_file, INPUT_DIR)
            except Exception as e:
                print(f"Error copying {img_file}: {e}")

def get_latest_video_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_video(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
    timeout = 3000  # seconds (increased for video generation)
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_count = get_latest_video_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(1)  # check less frequently for videos
    return False

def generate_script(input_image, SEED):
    """Generate the ComfyUI script for the given input image"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model, clip, vae = CheckpointLoaderSimple('ltxv-13b-0.9.7-distilled-fp8.safetensors')
    clip = CLIPLoader('t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors', 'ltxv', 'default')
    clip_text_encode_positive_prompt_conditioning = CLIPTextEncode('', clip)
    clip_text_encode_negative_prompt_conditioning = CLIPTextEncode('low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly', clip)
    positive, negative = LTXVConditioning(clip_text_encode_positive_prompt_conditioning, clip_text_encode_negative_prompt_conditioning, 24.000000000000004)
    guider = STGGuiderAdvanced(model, positive, negative, 0.9970000000000002, True, '1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180', '1,1,1,1,1,1', '0,0,0,0,0,0', '1, 1, 1, 1, 1, 1', '[25], [35], [35], [42], [42], [42]', None)
    sampler = KSamplerSelect('euler_ancestral')
    float = StringToFloatList('1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219, 0.0')
    sigmas = FloatToSigmas(float)
    noise = RandomNoise({SEED})
    image, _ = LoadImage('{input_image}')
    denoised_output = LTXVBaseSampler(model, vae, 768, 512, 97, guider, sampler, sigmas, noise, image, '0', 0.8, 'center', 30, 1)
    vae2 = SetVAEDecoderNoise(vae, 0.05, 0.025, 42)
    image2 = VAEDecode(denoised_output, vae2)
    _ = VHSVideoCombine(image2, 24, 0, 'ltxv-base', 'video/h264-mp4', False, False, None, None, None)
"""
    return script_content

def main():
    SEED = 661695664686456
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Copy all images to input directory
    copy_images_to_input()
    
    # Get list of input images
    input_images = list(Path(INPUT_DIR).glob('*.png'))
    if not input_images:
        print(f"No images found in {INPUT_DIR}")
        return
    
    print(f"Found {len(input_images)} images to process")
    
    # Process each image
    for img_path in input_images:
        # Get current video count before running
        initial_count = get_latest_video_count()
        
        # Generate script for this image
        script = generate_script(
            str(img_path).split("/")[-1], SEED
        )
        
        # Write script to file
        with open('run_ltxv_generation.py', 'w') as f:
            f.write(script)
        
        print(f"Processing image: {img_path.name} with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_ltxv_generation.py'])
        
        # Wait for new video
        if not wait_for_new_video(initial_count):
            print(f"Timeout waiting for video generation for {img_path.name}. Continuing to next image.")
        
        # Increment seed for next generation
        SEED -= 1

if __name__ == "__main__":
    main()

```

## Installation

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-LTXVideo` in the list of nodes and follow installation instructions.

### Manual installation

1. Install ComfyUI
2. Clone this repository to `custom-nodes` folder in your ComfyUI installation directory.
3. Install the required packages:
```bash
cd custom_nodes/ComfyUI-LTXVideo && pip install -r requirements.txt
```
For portable ComfyUI installations, run
```
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-LTXVideo\requirements.txt
```

### Models

1. Download [ltx-video-2b-v0.9.safetensors](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors) from Hugging Face and place it under `models/checkpoints`.
2. [Install git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) and clone the text encoder model to `models/text_encoders`:
```bash
cd models/text_encoders && git clone https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS
```

## Example workflows

Note that to run the example workflows, you need to have [ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite) installed.

### Text-to-video

[Download workflow](assets/ltxvideo-t2v.json)
![workflow](assets/ltxvideo-t2v.png)

### Image-to-video

[Download workflow](assets/ltxvideo-i2v.json)
![workflow](assets/ltxvideo-i2v.png)

### Video-to-video

[Download workflow](assets/ltxvideo-v2v.json)
![workflow](assets/ltxvideo-v2v.png)

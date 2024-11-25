
from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

with Workflow(): 
	noise = RandomNoise(42)
	model, vae = LTXVLoader('ltx-video-2b-v0.9.safetensors', None)
	image, _ = LoadImage('baloons.png')
	model, latent, sigma_shift = LTXVModelConfigurator(model, vae, 'Custom', 768, 512, 65, 25, 1, True, image)
	clip = LTXVCLIPModelLoader('PixArt-XL-2-1024-MS/text_encoder/model-00001-of-00002.safetensors')
	conditioning = CLIPTextEncode('''A group of colorful hot air balloons take off at dawn in Cappadocia, Turkey.
	Dozens of balloons in various bright colors and patterns slowly rise into the pink and orange sky. Below them, the unique landscape of Cappadocia unfolds, with its distinctive "fairy chimneys" - tall, cone-shaped rock formations scattered across the valley. The rising sun casts long shadows across the terrain, highlighting the otherworldly topography.''', clip)
	conditioning2 = CLIPTextEncode('worst quality, inconsistent motion, blurry, jittery, distorted, watermarks', clip)
	guider = CFGGuider(model, conditioning, conditioning2, 4)
	sampler = KSamplerSelect('euler')
	sigmas = BasicScheduler(model, 'normal', 20, 1)
	sigmas = LTXVShiftSigmas(sigmas, sigma_shift, True, 0.1)
	latent, _ = SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent)
	image2 = VAEDecode(latent, vae)
	# _ = VHSVideoCombine(image2, 25, 0, 'LTXVideo', 'video/h264-mp4', False, True, None, None, None)


sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg

python -m pip install comfy-cli
comfy --here install


cd ComfyUI/custom_nodes
git clone https://github.com/Chaoses-Ib/ComfyScript.git
cd ComfyScript
python -m pip install -e ".[cli,default]"


pip uninstall aiohttp
pip install -U aiohttp

wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.safetensors -O ltx-video-2b-v0.9.safetensors

models/checkpoints

cd models/text_encoders && git clone https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS

custom_nodes 

git clone https://github.com/Lightricks/ComfyUI-LTXVideo
pip install -r requirements.txt
#pip uninstall torch
#pip install torch torchaudio torchvision

git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
pip install -r requirements.txt

python -m comfy_script.transpile lora.json
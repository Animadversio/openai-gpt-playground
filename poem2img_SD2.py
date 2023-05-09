import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
# get StableDiffusion 2.1


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
#%%
from pathlib import Path
import json
from tqdm import tqdm, trange
rootdir = r"/home/binxu/DL_Projects/poem2diffusion"
text_prompt_pairs = json.load(open(Path(rootdir)/"text_prompt_pairs.json"))
#%%
for linei, (text_cn, text_en, prompt) in tqdm(enumerate(text_prompt_pairs)):
    # 5mins for 19 lines of poems
    images = pipe(prompt, num_images_per_prompt=4).images
    for i, image in enumerate(images):
        uuid = i
        while (Path(rootdir)/f"poem_L{linei:02d}_img{uuid:02d}.png").exists():
            uuid += 1
        image.save(Path(rootdir)/f"poem_L{linei:02d}_img{uuid:02d}.png")

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
#%%
# generate the diffusion latent trajectory and visualization gif
image_reservoir = []
latents_reservoir = []
@torch.no_grad()
def plot_show_callback(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
    image = pipe.vae.decode(1 / 0.18215 * latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    # plt_show_image(image)
    # plt.imsave(f"diffprocess/sample_{i:02d}.png", image)
    image_reservoir.append(image)

@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())

#%%
# save the image_reservoir as a gif
import imageio
imageio.mimsave("diffprocess/sample.gif", image_reservoir, fps=10)
#%%
for i, image in enumerate(images):
    uuid = i
    while (Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}.png").exists():
        uuid += 1
    image.save(Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}.png")
    imgseq = [imgs[i] for imgs in image_reservoir]
    imageio.mimsave(Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}.gif", imgseq, fps=10)
    imageio.mimsave(Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}.mp4", imgseq, fps=10)
    latent_traj = torch.stack(latents_reservoir)
#%%
for linei, (text_cn, text_en, prompt) in tqdm(enumerate(text_prompt_pairs)):
    # 40 mins for 19 lines of poems, with image saving
    image_reservoir = []
    latents_reservoir = []
    images = pipe(prompt, num_images_per_prompt=4, callback=plot_show_callback).images
    latents_seq_all = torch.stack(latents_reservoir)
    for i, image in enumerate(images):
        uuid = i
        while (Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}.png").exists():
            uuid += 1
        image.save(Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}.png")
        imgseq = [imgs[i] for imgs in image_reservoir]
        # make montage of the images and save as jpg
        imageio.mimsave(Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}_rev.gif", imgseq[::-1], fps=10)
        imageio.mimsave(Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}_rev.mp4", imgseq[::-1], fps=10)
        torch.save(latents_seq_all[:, i], Path(rootdir) / f"poem_L{linei:02d}_img{uuid:02d}_latents.pt")
#%%
# extract frames from the mp4 files


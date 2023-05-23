import random

import torch
from pathlib import Path
import json
from tqdm import tqdm, trange
from stable_diffusion_videos import StableDiffusionWalkPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

rootdir = r"/home/binxu/DL_Projects/poem2diffusion"
#%%
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
#%%
# pipeline = StableDiffusionWalkPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
# ).to("cuda")
#%%
pipeline = StableDiffusionWalkPipeline(
    vae = pipe.vae,
    text_encoder = pipe.text_encoder,
    tokenizer = pipe.tokenizer,
    unet = pipe.unet,
    scheduler = pipe.scheduler,
    safety_checker = None,
    feature_extractor = pipe.feature_extractor,
    requires_safety_checker = False,
)
#%%
# rootdir = r"/home/binxu/DL_Projects/poem2diffusion"
# text_prompt_pairs = json.load(open(Path(rootdir)/"text_prompt_pairs.json"))
text_prompt_pairs = json.load(open(Path(rootdir)/"text_prompt_pairs_singleline.json"))
#%%
prompt_seq = []
for linei, (text_cn, text_en, prompt) in tqdm(enumerate(text_prompt_pairs)):
    prompt_seq.append(prompt.replace("Prompt: ", "").replace('"', ""))
#%%
RNDseed_seq = [42+i for i in range(len(prompt_seq))]
video_path = pipeline.walk(
    prompts=prompt_seq,
    seeds=RNDseed_seq,
    num_interpolation_steps=32,
    height=768,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=768,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='/home/binxu/DL_Projects/poem2diffusion',        # Where images/videos will be saved
    name='Singer_video',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default

)

#%%
sameRNDseed_seq = [42] * len(prompt_seq)
video_path = pipeline.walk(
    prompts=prompt_seq,
    seeds=sameRNDseed_seq,
    num_interpolation_steps=32,
    height=768,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=768,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='/home/binxu/DL_Projects/poem2diffusion',        # Where images/videos will be saved
    name='Singer_video_42',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
)

#%%

#%%
timing_pair = json.load(open(Path(rootdir)/"lyric_timing.json"))
timing_pair.insert(0, [0,0])



#%%
timing_prompt_pair = json.load(open(Path(rootdir)/"lyric_timing_prompt_pairs_singleline.json"))
# start 9, 27, 33
# end 258,
#%%
audio_offsets = [offset for offset, text_cn, text_en, prompt in timing_prompt_pair]
prompt_seq_music = [prompt for offset, text_cn, text_en, prompt in timing_prompt_pair[:-1]]
prompt_seq_music = [prompt.replace("Prompt: ", "").replace('"', "") for prompt in prompt_seq_music]
RNDseed_seq_music = [random.randint(0, 1000) for offset, text_cn, text_en, prompt in timing_prompt_pair[:-1]]
fps = 10
num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]
#%%
video_path = pipeline.walk(
    prompts=prompt_seq_music,
    seeds=RNDseed_seq_music,
    num_interpolation_steps=num_interpolation_steps,
    audio_filepath='/home/binxu/DL_Projects/poem2diffusion/Singer_Song_TanWeiwei.mp3',
    audio_start_sec=9,
    fps=fps,
    height=768,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=768,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='/home/binxu/DL_Projects/poem2diffusion',  # Where images/videos will be saved
    name='Singer_video_music_Refined',  # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
    batch_size=2,
)






#%%
audio_offsets = [offset for idx, offset in timing_pair]
prompt_seq_music = [prompt_seq[idx] for idx, offset in timing_pair]
RNDseed_seq_music = [random.randint(0, 1000) for idx in timing_pair]
fps = 5
num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]
#%
# audio_offsets = [146, 148]  # [Start, end]
# fps = 30  # Use lower values for testing (5 or 10), higher values for better quality (30 or 60)
#
# # Convert seconds to frames
# num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]

video_path = pipeline.walk(
    prompts=prompt_seq_music,
    seeds=RNDseed_seq_music,
    num_interpolation_steps=num_interpolation_steps,
    audio_filepath='/home/binxu/DL_Projects/poem2diffusion/Singer_Song_TanWeiwei.mp3',
    audio_start_sec=0,
    fps=fps,
    height=768,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=768,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='/home/binxu/DL_Projects/poem2diffusion',  # Where images/videos will be saved
    name='Singer_video_music',  # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default

)
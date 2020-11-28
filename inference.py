import torch
from torch import nn, optim
from torchsummary import summary

import os
import numpy as np
from PIL import Image

from helper import *
from model.generator import SkipEncoderDecoder, input_noise

DTYPE = torch.cuda.FloatTensor
INPUT_DEPTH = 32
LR = 0.01 
TRAINING_STEPS = 6001
SHOW_STEP = 50
REG_NOISE = 0.03

image_path  = os.path.join('data', 'me.jpg')
watermark_path = os.path.join('data', 'watermark2.png')

image_pil = read_image(image_path)
image_pil = image_pil.convert('RGB')

image_mask_pil = read_image(watermark_path)
image_mask_pil = image_mask_pil.convert('RGB')
image_mask_pil = image_mask_pil.resize((image_pil.size[0], image_pil.size[1]))

image_np = pil_to_np_array(image_pil)
image_mask_np = pil_to_np_array(image_mask_pil)
image_mask_np[image_mask_np == 0.0] = 1.0

image_mask_var = np_to_torch_array(image_mask_np).type(DTYPE)

visualize_sample(image_np, image_mask_np, image_mask_np * image_np, nrow = 3, size_factor = 12)


generator = SkipEncoderDecoder(
    INPUT_DEPTH,
    num_channels_down = [128] * 5,
    num_channels_up = [128] * 5,
    num_channels_skip = [128] * 5
).type(DTYPE)
generator_input = input_noise(INPUT_DEPTH, image_np.shape[1:]).type(DTYPE)
summary(generator, generator_input.shape[1:])

objective = torch.nn.MSELoss().type(DTYPE)
optimizer = optim.Adam(generator.parameters(), LR)

image_var = np_to_torch_array(image_np).type(DTYPE)
mask_var = np_to_torch_array(image_mask_np).type(DTYPE)

generator_input_saved = generator_input.detach().clone()
noise = generator_input.detach().clone()

for step in range(TRAINING_STEPS):
    optimizer.zero_grad()
    generator_input = generator_input_saved

    if REG_NOISE > 0:
        generator_input = generator_input_saved + (noise.normal_() * REG_NOISE)
        
    out = generator(generator_input)
   
    loss = objective(out * mask_var, image_var * mask_var)
    loss.backward()
        
    if step % SHOW_STEP == 0:
        out_np = torch_to_np_array(out)
        visualize_sample(np.clip(out_np, 0, 1), nrow = 1, size_factor = 5)
        
    optimizer.step()

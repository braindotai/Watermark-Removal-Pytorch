import torch
from torch import nn, optim
from torchsummary import summary

import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from helper import *
from model.generator import SkipEncoderDecoder, input_noise

import argparse

parser = argparse.ArgumentParser(description = 'Removing Watermark')
parser.add_argument('--image-path', type = str, default = './data/watermark-available/me.jpg', help = 'Path to the "watermarked" image.')
parser.add_argument('--watermark-path', type = str, default = './data/watermark-available/watermark.png', help = 'Path to the "watermark" image.')
parser.add_argument('--input-depth', type = int, default = 32, help = 'Max channel dimension of the noise input. Set it based on gpu/device memory you have available.')
parser.add_argument('--lr', type = float, default = 0.01, help = 'Learning rate.')
parser.add_argument('--training-steps', type = int, default = 3000, help = 'Number of training iterations.')
parser.add_argument('--show-steps', type = int, default = 200, help = 'Interval for visualizing results.')
parser.add_argument('--reg-noise', type = float, default = 0.03, help = 'Hyper-parameter for regularized noise input.')
parser.add_argument('--device', type = str, default = 'cuda', help = 'Device for pytorch, either "cpu" or "cuda".')
parser.add_argument('--max-dim', type = float, default = 512, help = 'Max dimension of the final output image')

args = parser.parse_args()
if args.device == 'cuda' and not torch.cuda.is_available():
    args.device = 'cpu'
    print('\nSetting device to "cpu", since torch is not built with "cuda" support...')
    print('It is recommended to use GPU if possible...')

DTYPE = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor

image_pil = read_image(args.image_path)
image_pil = image_pil.convert('RGB')
image_pil = image_pil.resize((128, 128))

image_mask_pil = read_image(args.watermark_path)
image_mask_pil = image_mask_pil.convert('RGB')
image_mask_pil = image_mask_pil.resize((image_pil.size[0], image_pil.size[1]))

image_np = pil_to_np_array(image_pil)
image_mask_np = pil_to_np_array(image_mask_pil)
image_mask_np[image_mask_np == 0.0] = 1.0

image_var = np_to_torch_array(image_np).type(DTYPE)
mask_var = np_to_torch_array(image_mask_np).type(DTYPE)

visualize_sample(image_np, image_mask_np, image_mask_np * image_np, nrow = 3, size_factor = 12)

print('Building model...\n')

generator = SkipEncoderDecoder(
    args.input_depth,
    num_channels_down = [128] * 5,
    num_channels_up = [128] * 5,
    num_channels_skip = [128] * 5
).type(DTYPE)
generator_input = input_noise(args.input_depth, image_np.shape[1:]).type(DTYPE)
summary(generator, generator_input.shape[1:])

objective = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), args.lr)

generator_input_saved = generator_input.clone()
noise = generator_input.clone()
generator_input = generator_input_saved

print('\nStarting training...\n')

progress_bar = tqdm(range(TRAINING_STEPS), desc = 'Completed', ncols = 800)

for step in progress_bar:
    optimizer.zero_grad()
    generator_input = generator_input_saved

    if args.reg_noise > 0:
        generator_input = generator_input_saved + (noise.normal_() * REG_NOISE)
        
    output = generator(generator_input)
   
    loss = objective(output * mask_var, watermarked_var * mask_var)
    loss.backward()

    if step % args.show_steps == 0:
        output_image = torch_to_np_array(output)
        visualize_sample(watermarked_np, output_image, nrow = 2, size_factor = 10)
    
    progress_bar.set_postfix(Step = step, Loss = loss.item())
        
    optimizer.step()

# for step in tqdm(range(args.training_steps), desc = 'Completed', ncols = 100):
#     optimizer.zero_grad()

#     if args.reg_noise > 0:
#         generator_input = generator_input_saved + (noise.normal_() * args.reg_noise)
        
#     out = generator(generator_input)
   
#     loss = objective(out * mask_var, image_var * mask_var)
#     loss.backward()
        
#     if step % args.show_steps == 0:
#         out_np = torch_to_np_array(out)
#         visualize_sample(np.clip(out_np, 0, 1), nrow = 1, size_factor = 5)
        
#     optimizer.step()
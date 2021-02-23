# import argparse
# from api import remove_watermark

# parser = argparse.ArgumentParser(description = 'Removing Watermark')
# parser.add_argument('--image-path', type = str, default = './data/watermark-unavailable/watermarked/watermarked0.png', help = 'Path to the "watermarked" image.')
# parser.add_argument('--mask-path', type = str, default = './data/watermark-unavailable/masks/mask0.png', help = 'Path to the "watermark" image.')
# parser.add_argument('--input-depth', type = int, default = 32, help = 'Max channel dimension of the noise input. Set it based on gpu/device memory you have available.')
# parser.add_argument('--lr', type = float, default = 0.01, help = 'Learning rate.')
# parser.add_argument('--training-steps', type = int, default = 3000, help = 'Number of training iterations.')
# parser.add_argument('--show-step', type = int, default = 200, help = 'Interval for visualizing results.')
# parser.add_argument('--reg-noise', type = float, default = 0.03, help = 'Hyper-parameter for regularized noise input.')
# parser.add_argument('--max-dim', type = float, default = 512, help = 'Max dimension of the final output image')

# args = parser.parse_args()

# remove_watermark(
#     image_path = args.image_path,
#     mask_path = args.mask_path,
#     max_dim = args.max_dim,
#     show_step = args.show_step,
#     reg_noise = args.reg_noise,
#     input_depth = args.input_depth,
#     lr = args.lr,
#     training_steps = args.training_steps,
# )

import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('./data/watermark-unavailable/watermarked/watermarked0.png')
mask = cv2.imread('./data/watermark-unavailable/masks/mask0.png', 0)
zeros = mask == 0
ones = mask == 1
mask[zeros] = 0
mask[ones] = 1
plt.imshow(mask, cmap = 'gray')
plt.show()
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
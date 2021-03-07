import imageio
import numpy as np
import os

data_path = './logs/fern_exp1/fern_exp1_spiral_080000_rgb.mp4'
output_dir = './simulate/fern_exp1'

video = imageio.mimread(data_path)
video_np = np.array(video)

os.makedirs(output_dir, exist_ok=True)
new_video_list = ['r', 'g', 'b']


for i in range(3):
    gray = np.tile(video_np[..., i][..., None], [1, 1, 1, 3])
    imageio.mimwrite(os.path.join(output_dir, f'{new_video_list[i]}.mp4'), gray, fps=30, macro_block_size=1)

print('Video Process Ended')
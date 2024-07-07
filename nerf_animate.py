import nerf
import torch
import sys
import numpy as np
import imageio

nerf_model = nerf.NerfModel('cuda')
nerf_model.load_state_dict(torch.load(sys.argv[1]))
dataset = torch.from_numpy(np.load(sys.argv[2]))
output_filepath = sys.argv[3]
img_width = int(sys.argv[4])
img_height = int(sys.argv[5])
num_frames = int(sys.argv[6])

frames = []
for i in range(num_frames):
    rays = dataset[i * img_height * img_width : (i + 1) * img_height * img_width, :6].cuda()
    
    # Render in chuncks so that we don't run out of memory
    chunk_size = 10
    img_pixels = []
    for j in range(int(np.ceil(img_height / chunk_size))):
        ray_origins = rays[j * img_width * chunk_size : (j + 1) * img_width * chunk_size, :3]
        ray_directions = rays[j * img_width * chunk_size : (j + 1) * img_width * chunk_size, 3:6]
        with torch.no_grad():
            rendered_pixels = nerf_model.render(ray_origins, ray_directions, 192, 1.0, 10.0)
        img_pixels.append(rendered_pixels)
    img_pixels = torch.cat(img_pixels).data.cpu().numpy().reshape(img_height, img_width, 3) * 255
    img_pixels = np.round(img_pixels).astype(np.uint8) 
    frames.append(img_pixels)
    print("Processed frame", i)

imageio.mimsave(output_filepath, frames, format='GIF', fps=20)
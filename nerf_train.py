import nerf
import sys

nerf_model = nerf.NerfModel('cuda')
nerf_model.train(sys.argv[1], num_epochs=16, ray_sample_count=192, ray_near=1.0, ray_far=10.0, test_img_width=int(sys.argv[2]), test_img_height=int(sys.argv[3]), test_img_index=0)
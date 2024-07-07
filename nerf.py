import torch
import numpy as np
import torch.utils.data
import os
import cv2

class NerfModel(torch.nn.Module):
    def __init__(self, device_type):
        super(NerfModel, self).__init__()
        
        self.device = device_type
        self.position_encoding_dim = 10
        self.direction_encoding_dim = 4
        self.hidden_layer_feature_count = 256
        
        self.sub_netork_1 = torch.nn.Sequential(
            torch.nn.Linear(self.position_encoding_dim * 6, self.hidden_layer_feature_count),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_feature_count, self.hidden_layer_feature_count),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_feature_count, self.hidden_layer_feature_count),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_feature_count, self.hidden_layer_feature_count),
            torch.nn.ReLU(),
        )
        
        self.sub_netork_2 = torch.nn.Sequential(
            torch.nn.Linear(self.position_encoding_dim * 6 + self.hidden_layer_feature_count, self.hidden_layer_feature_count),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_feature_count, self.hidden_layer_feature_count),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_feature_count, self.hidden_layer_feature_count),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_feature_count, self.hidden_layer_feature_count + 1)
        )
        
        self.sub_netork_3 = torch.nn.Sequential(
            torch.nn.Linear(self.direction_encoding_dim * 6 + self.hidden_layer_feature_count, self.hidden_layer_feature_count // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_feature_count // 2, 3),
            torch.nn.Sigmoid()
        )
        
        self.to(self.device)
        
    def train(self, dataset_path: str, num_epochs: int, ray_sample_count: int, ray_near: float, ray_far: float, test_img_width: int, test_img_height: int, test_img_index: int):
        dataset_root_path = os.path.dirname(dataset_path)
        model_save_path = os.path.join(dataset_root_path, 'trained_nerf.pt')
        test_results_path = os.path.join(dataset_root_path, 'test_results')
        
        # Load the dataset. We use batches of 4096 as described in the paper
        dataset = torch.from_numpy(np.load(dataset_path))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True)
        
        # In the paper it is explained that Adam optimizer is used and initially the learning rate starts at 5 * 10^-4 and decays exponentially to 5 * 10^-4
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        
        # Calculate how much the learning rate has to decay over the course of the training
        lr_decay_milestones = []
        milestone = 2
        
        while milestone <= (num_epochs / 2):
            lr_decay_milestones.append(milestone)
            milestone *= 2
        
        lr_modifier = 0.1 ** (1.0 / len(lr_decay_milestones))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_milestones, gamma=lr_modifier)
        
        # Start training
        for _ in range(1, num_epochs):
            counter = 0
            for ray_batch in data_loader:
                ray_origins = ray_batch[:, :3].to(self.device)
                ray_directions = ray_batch[:, 3:6].to(self.device)
                original_pixels = ray_batch[:, 6:].to(self.device)

                rendered_pixels = self.render(ray_origins, ray_directions, ray_sample_count, ray_near, ray_far)
                loss = torch.pow(original_pixels - rendered_pixels, 2.0).sum()
                print(counter, " ", loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                counter = counter + 1
                if counter % 500 == 0:
                    # Every 500 batches save and test the model
                    print("Saving and testing model...")
                    torch.save(self.state_dict(), model_save_path)
                    self.test(test_results_path, dataset[:test_img_width * test_img_height, :6], ray_sample_count, ray_near, ray_far, test_img_index, test_img_width, test_img_height)
                    test_img_index = test_img_index + 1
                    
            lr_scheduler.step()
            
            # Save the model and test it after each epoch
            print("Saving and testing model...")
            torch.save(self.state_dict(), model_save_path)
            self.test(test_results_path, dataset[:test_img_width * test_img_height, :6], ray_sample_count, ray_near, ray_far, test_img_index, test_img_width, test_img_height)
            test_img_index = test_img_index + 1
    
    def render(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, ray_sample_step_count: int, ray_near: float, ray_far: float) -> torch.Tensor:
        sample_steps = torch.linspace(ray_near, ray_far, ray_sample_step_count, device=self.device).expand([ray_origins.shape[0], ray_sample_step_count]) # [num_rays, num_samples]
        deltas = torch.cat([sample_steps[:, 1:] - sample_steps[:, :-1], torch.tensor([1000000], device=self.device).expand([ray_origins.shape[0], 1])], dim=1) # [num_rays, num_samples]
        positions_at_steps = ray_origins.unsqueeze(1) + sample_steps.unsqueeze(2) * ray_directions.unsqueeze(1); # [num_rays, num_samples, 3 (xyz)]
        ray_directions = ray_directions.expand([ray_sample_step_count, ray_directions.shape[0], 3]).transpose(0, 1); # [num_rays, num_samples, 3 (xyz)]
        
        # Reshape the positions and directions so that we have a tensor of size [num_rays * num_samples, 3 (xyz)] and pass them to the network
        colors, sigmas = self.forward(positions_at_steps.reshape([-1, 3]), ray_directions.reshape([-1, 3]))
        
        # Reshape the colors and sigmas back to the [num_rays, num_samples, 3 (xyz)] and [num_rays, num_samples] formats respectively
        colors, sigmas = colors.reshape(positions_at_steps.shape), sigmas.reshape([positions_at_steps.shape[0], positions_at_steps.shape[1]])
        
        alphas = 1.0 - torch.exp(-sigmas * deltas)
        transmittance = torch.cumprod(torch.exp(-sigmas * deltas), dim=1) # T = exp(-sum(sigma_i * delta_i))
        
        # Make sure the transmittance is always 1 in the first iteration since we haven't hit anything yet
        transmittance = torch.cat([torch.ones([transmittance.shape[0], 1], device=self.device), transmittance[:, :-1]], dim=1)
        
        return (transmittance.unsqueeze(2) * alphas.unsqueeze(2) * colors).sum(dim=1) # Sum the result from all samples
        
    def forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_positions = NerfModel.apply_positional_encoding(ray_origins, self.position_encoding_dim)
        encoded_directions = NerfModel.apply_positional_encoding(ray_directions, self.direction_encoding_dim)
        
        network_1_output = self.sub_netork_1(encoded_positions)
        network_2_output = self.sub_netork_2(torch.cat([network_1_output, encoded_positions], dim=1))
        
        sigmas = torch.relu(network_2_output[:, -1]) # Make sure sigma is non-negative by using ReLU
        remaining = network_2_output[:, :-1]
        
        colors = self.sub_netork_3(torch.cat([remaining, encoded_directions], dim=1))
        return (colors, sigmas)
    
    @torch.no_grad()
    def test(self, test_results_path: str, test_data: torch.Tensor, ray_sample_count: float, ray_near: float, ray_far: float, img_index: int, img_width: int, img_height: int):
        # Generate ray origins and directions for the test
        ray_origins = test_data[:, :3]
        ray_directions = test_data[:, 3:6]
        chunk_size = 10

        img_pixels = []
        for i in range(int(np.ceil(img_height / chunk_size))):
            ray_o = ray_origins[i * img_width * chunk_size: (i + 1) * img_width * chunk_size].to(self.device)
            ray_d = ray_directions[i * img_width * chunk_size: (i + 1) * img_width * chunk_size].to(self.device)        
            rendered_pixels = self.render(ray_o, ray_d, ray_sample_count, ray_near, ray_far)
            img_pixels.append(rendered_pixels)
        
        img_pixels = torch.cat(img_pixels).data.cpu().numpy().reshape(img_height, img_width, 3) * 255
        img_pixels = np.round(img_pixels).astype(np.uint8)
        cv2.imwrite(os.path.join(test_results_path, f'test_{img_index}.png'), img_pixels)
    
    def apply_positional_encoding(input: torch.Tensor, dimension: int) -> torch.Tensor:
        result = []
        for i in range(dimension):
            result.append(torch.sin(2 ** i * input))
            result.append(torch.cos(2 ** i * input))
        return torch.cat(result, dim=1)
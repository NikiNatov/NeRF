import subprocess
import os.path
import struct
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import sys

class Camera:
    def __init__(self, width: int, height: int, focal_length: float):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        
class Image:
    def __init__(self, name: str, rotation: np.ndarray, translation: np.ndarray):
        self.name = name
        world_to_camera = np.concatenate([Rotation.from_quat(rotation).as_matrix(), translation.reshape([3, 1])], axis=1) # add a translation column
        world_to_camera = np.concatenate([world_to_camera, np.array([0, 0, 0, 1]).reshape([1, 4])], axis=0) # add a bottom row of 0,0,0,1
        self.camera_to_world = np.linalg.inv(world_to_camera)
        
def run_colmap_feature_extractor(database_path: str, input_images_path: str):
    args = [
        'colmap', 'feature_extractor', 
        '--database_path', database_path, 
        '--image_path', input_images_path,
        '--SiftExtraction.use_gpu', '1',
        '--ImageReader.single_camera', '1'
    ]
    output = subprocess.check_output(args)
    print('Colmap feature_extractor: ', output)
    
def run_colmap_matcher(database_path: str):
    args = [
        'colmap', 'exhaustive_matcher', 
        '--database_path', database_path
    ]
    output = subprocess.check_output(args)
    print('Colmap matcher: ', output)
    
def run_colmap_mapper(database_path: str, input_images_path: str, output_path: str):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    args = [
        'colmap', 'mapper', 
        '--database_path', database_path,
        '--image_path', input_images_path,
        '--output_path', output_path,
        '--Mapper.num_threads', '4',
        '--Mapper.extract_colors', '0',
        '--Mapper.multiple_models', '0',
    ]
    output = subprocess.check_output(args)
    print('Colmap mapper: ', output)
    
def read_colmap_camera(cameras_path: str) -> Camera:
    with open(cameras_path, "rb") as file:
        _ = struct.unpack("<Q", file.read(8))[0] # num_cameras
        _ = struct.unpack("<i", file.read(4))[0] # camera_id
        _ = struct.unpack("<i", file.read(4))[0] # camera_model
        camera_width = struct.unpack("<Q", file.read(8))[0]
        camera_height = struct.unpack("<Q", file.read(8))[0]
        focal_length = struct.unpack("<d", file.read(8))[0]
        return Camera(camera_width, camera_height, focal_length)
    
def read_colmap_images(images_path: str) -> list:
    images = []
    with open(images_path, "rb") as file:
        num_images = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_images):
            _ = struct.unpack("<i", file.read(4))[0] # image_id
            transform_properties = struct.unpack("<ddddddd", file.read(56))
            rotation = np.array([transform_properties[1], transform_properties[2], transform_properties[3], transform_properties[0]])
            translation = np.array(transform_properties[4:7])
            _ = struct.unpack("<i", file.read(4))[0] # camera_id
            name = ""
            c = struct.unpack("<c", file.read(1))[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = struct.unpack("<c", file.read(1))[0]
            num_points2D = struct.unpack("<Q", file.read(8))[0]
            _ = struct.unpack("<" + "ddQ" * num_points2D, file.read(24 * num_points2D)) # x y id
            images.append(Image(name, rotation, translation))
    return images

def generate_ray(x: int, y: int, camera: Camera, camera_to_world_matrix: np.ndarray) -> np.ndarray:
    # The origin is the translation component of the transformation matrix
    origin = camera_to_world_matrix[:3, 3].astype(np.float32)
    # Get the direction in camera space
    direction = np.array([x - camera.width / 2.0, y - camera.height / 2.0, camera.focal_length]).reshape([3, 1])
    # Transform the direction to world space
    direction = camera_to_world_matrix[:3, :3] @ direction
    # Normalize
    direction = (direction.reshape([3]) / np.linalg.norm(direction)).astype(np.float32)
    return np.array([origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]])

def generate_nerf_dataset(input_images_path: str, output_path: str, dataset_name: str):
    colmap_directory = os.path.join(output_path, "colmap")
    database_path = os.path.join(colmap_directory, "database.db")
    
    if not os.path.exists(colmap_directory):
        os.makedirs(colmap_directory)
    
    run_colmap_feature_extractor(database_path, input_images_path)
    run_colmap_matcher(database_path)
    run_colmap_mapper(database_path, input_images_path, colmap_directory)
    
    camera = read_colmap_camera(os.path.join(colmap_directory, "0", "cameras.bin"))
    images = read_colmap_images(os.path.join(colmap_directory, "0", "images.bin"))
    
    rays = []
    for image in images:
        image_data = cv2.imread(os.path.join(input_images_path, image.name))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        for y in range(camera.height):
            for x in range(camera.width):
                ray = generate_ray(x, y, camera, image.camera_to_world)
                rgb = (image_data[y, x] / 255.0).astype(np.float32)
                rays.append(np.concatenate([ray, rgb], axis=0))
        print("Serialized image", image.name)
        
    np.save(os.path.join(output_path, dataset_name + '.npy'), rays)
    
generate_nerf_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
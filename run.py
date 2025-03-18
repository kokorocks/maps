import cv2
import numpy as np
import torch
import open3d as o3d
from torchvision import transforms
from PIL import Image
from midas.model_loader import default_models, load_model

# Load MiDaS depth estimation model
def load_depth_model():
    model_type = "DPT_Large"  # Use a smaller model for speed: "MiDaS_small"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_model(model_type, device)
    return model, transform, device

# Estimate depth from a panorama image
def estimate_depth(image_path, model, transform, device):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = model(img)
    depth = depth.squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return depth

# Convert depth map to a 3D point cloud for a panorama
def depth_to_point_cloud(depth_map):
    h, w = depth_map.shape
    fx, fy = w / (2 * np.pi), h / np.pi  # Approximate focal length for equirectangular projection
    cx, cy = w / 2, h / 2  # Optical center

    points = []
    for y in range(h):
        theta = (y / h) * np.pi  # Vertical angle
        for x in range(w):
            phi = (x / w) * 2 * np.pi  # Horizontal angle
            z = depth_map[y, x] / 255.0  # Scale depth values
            X = z * np.sin(theta) * np.cos(phi)
            Y = z * np.cos(theta)
            Z = z * np.sin(theta) * np.sin(phi)
            points.append([X, Y, Z])

    return np.array(points)

# Create a mesh from the point cloud
def create_3d_mesh(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]
    return mesh

if __name__ == "__main__":
    image_path = "panorama.jpg"  # Change this to your panorama image
    model, transform, device = load_depth_model()
    depth_map = estimate_depth(image_path, model, transform, device)
    points = depth_to_point_cloud(depth_map)
    mesh = create_3d_mesh(points)
    o3d.visualization.draw_geometries([mesh])

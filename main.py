import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
import numpy as np
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import open3d as o3d

# 2. load the model first
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

#3. Load and resize the image
image = Image.open("input/space.jpg")    #give the img path
image = image.convert('RGB')      #convert to rgb
new_height = 512 if image.height > 480 else image.height   # choose the req height
new_height -= (new_height % 32)       # it should be multiple of 32 
new_width = int(new_height * image.width / image.height)    # cal width from height
diff = new_width % 32         # it should be multiple of 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)      # resize wihthe new dimensions 

#4. convert image into format that model understands (like converting to tensor and all)
inputs = feature_extractor(images=image, return_tensors="pt")  # exracthe features from the image

#5. predict depth (we're not training here)
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# 6. Post-processing - convert the output tensor to numpy array
out = predicted_depth.squeeze().cpu().numpy()
# resize the depth map to match image size
out_resize = Image.fromarray(out).resize((new_width, new_height), resample=Image.BILINEAR)
out = np.array(out_resize)
out_norm = out / np.max(out) # normalize depth values between 0 to 1

# 7. # prepare for Open3D (Open3D needs depth and color images)
width, height = image.size
image_np = np.array(image)

# create rgbd image
depth_scaled = out_norm * 1.7     # multiply depth a bit to bring some contrast, and convert to float32 as required
depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.float32))
image_o3d = o3d.geometry.Image(image_np)

# image with both color and depth info
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    image_o3d, depth_o3d,
    convert_rgb_to_intensity=False
)

# 8. Creating a Camera - saying how the camera sees the image
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 350, 350, width/2, height/2)

# 10. Create 3D point cloud from the rgbd image and camera info
pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
# o3d.visualization.draw_geometries([pcd_raw])

# 11. Post-processing the 3D Point Cloud
#outliers removal
cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd_raw.select_by_index(ind)

#estimate normals
pcd.estimate_normals()

pcd.orient_normals_to_align_with_direction()
# o3d.visualization.draw_geometries([pcd])

# 12. reconstruct surface from point cloud using Poisson reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=15)[0]
mesh= mesh.filter_smooth_simple(number_of_iterations=1)

#rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0,0,0))

#display the mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

o3d.io.write_triangle_mesh('output/space.obj', mesh)

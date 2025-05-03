# -Photo-to-3D-Model

# Photo-to-3D Mesh Generator using AI

## 🚀 What We’re Doing

In this project we’re going to take a 2D image and turn it into a 3D mesh! We’ll be using some AI ttools to figure out the depth from the image, and then we’ll use that info to create a 3D model.

## 🧠 How It Works

1. **Depth Estimation:**  
   We use the GLPN model (which you can find on HuggingFace) to predict the depth of the image, so that we know how far things are in 3D space.
   The GLPN model - AI used for monocular depth estimation, meaning it can determine the depth of objects within a single image.

3. **Point Cloud:**  
   We then combine the depth info with the original image to create an RGB-D image (it’s an image with both color and depth data).
   A 3D point cloud - digital representation of an object or environment composed of numerous individual data points in a 3D coordinate system.

5. **Making the Mesh:**  
   From this RGB-D image, we create a 3D point cloud, and then we turn that into a mesh using a technique called Poisson surface reconstruction.
   Poisson surface reconstruction - technique used to generate a smooth, triangular mesh from a set of 3D point data

7. **Export & View:**  
   The final 3D mesh gets saved as a .ply file, and we can display it on the screen using Open3D!

## 📸 What You Need

- Just place the 2D image (like `input.jpg`) in the `input/` folder. That's it! 

## 📦 What You’ll Get

- A 3D model saved as `output.ply` in the `output/` folder.
- You’ll also see a preview of the 3D mesh as the program runs.

## 🛠️ Getting Started

1. **Clone the repo:**

   Open your terminal/command prompt and type:
   ```bash
   git clone https://github.com/your-repo/photo-to-3D-Model.git
   cd photo-to-3D-Model

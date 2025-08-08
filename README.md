## Project - Mesh based Image Vectorization

This folder contains code for:
1. Generating 2D triangular meshes from images using two different approaches
2. Reconstruction of Images from Meshes
3. Clustering of canny edge image meshes using k means clustering
4. Reconstruction of Radial Gradients in Mesh space
5. Reconstruction of Freeform Gradients in Mesh space
6. Fitting gradient parameters and choosing the suitable gradient reconstruction method (Radial/Linear/Solid fill) based on gradient reconstruction error for segment wise meshes
   
## Approaches

1. Mesh Generation for edge result (`edge_mesh.py`)
2. Mesh Generation for colored segmentation result (`mesh.py`)

## Setup
Clone the current repository

Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage
1. Creating 2D meshes
   
### Option 1: Use the main dispatcher
```bash
# For edge images
python3 main.py -e "path/to/edge_image.png"

# For color segmentation images  
python3 main.py -c "path/to/colored_image.png"
```

### Option 2: Run files directly
```bash
# Edge-based mesh generation
python3 edge_mesh.py "path/to/edge_image.png"

# Color-based mesh generation
python3 mesh.py "path/to/colored_image.png"
```
### The meshes have variable resolution in terms of volume and minimum angles of the triangle. It can be modified by changing max_volume and min_angle paramters


2. Reconstruction of Radial Gradients in Mesh space
```bash
python3 radial_gradient.py "path/to/canny_edge_image.png" "path/to/original_image.png"
```
3. Reconstruction of Freeform Gradients in Mesh space
```bash
python3 freeform.py "path/to/canny_edge_image.png" "path/to/original_image.png"
```
4. Fitting gradient parameters and choosing the suitable gradient reconstruction method (Radial/Linear/Solid fill) for segment wise meshes
```bash
python3 segment-gradient.py "path/to/canny_edge_image.png" "path/to/original_image.png"
```

## Files

- `main.py` - Simple dispatcher script
- `edge_mesh.py` - Edge-based mesh generation
- `mesh.py` - Color-based mesh generation  
- `reconstruct.py` - Image reconstruction from mesh
- `kmeans.py` - k means clustering on meshes
- `radial_gradient.py` - Reconstruction of radial gradients from meshes
- `freeform.py` - Reconstruction of freeform gradients from meshes
- `freeform2.py` - Reconstruction of freeform gradients with visualisation of optimisation process
- `segment-gradient.py` - Creation of 2D meshes and fitting suitable gradients for each segment (Radial/Linear/Solid fill) based on gradient reconstruction error
- `requirements.txt` - Python dependencies

For any information feel free to reach out to shalinionev@gmail.com

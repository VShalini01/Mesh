# reconstruction of mesh 
import numpy as np
from PIL import Image
import vedo
import vtk
from vtk.util import numpy_support
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def kmeans_cluster_and_visualize_auto_k(vertices, triangles, vertex_colors, 
                                        k_min=2, k_max=20, window_size=(1000, 1400)):
    """
    choose best k using elbow method (inertia) on vertices for KMeans clustering 
    on vertex colors (Lab space), then visualize mesh colored by cluster labels.
    """
    print(f"Selecting k from range [{k_min}, {k_max}] using elbow method (inertia)")

    colors_rgb = np.array(vertex_colors, dtype=np.uint8)
    colors_rgb_norm = colors_rgb / 255.0
    colors_lab = rgb2lab(colors_rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3)

    inertias = []
    kmeans_models = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(colors_lab)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        kmeans_models.append(kmeans)
        print(f"k={k}, inertia={inertia:.2f}")

    # Use KneeLocator to find elbow point
    knee = KneeLocator(range(k_min, k_max + 1), inertias, curve="convex", direction="decreasing")
    best_k = knee.knee or k_min  # Fallback to k_min if no clear knee found

    print(f"Chosen k={best_k} using elbow method")

    best_kmeans = kmeans_models[best_k - k_min]
    best_labels = best_kmeans.predict(colors_lab)

    # Convert cluster centers back to RGB for visualization
    centroids_lab = best_kmeans.cluster_centers_.reshape(-1, 1, 3)
    centroids_rgb = lab2rgb(centroids_lab).reshape(-1, 3)
    centroids_rgb_clipped = np.clip(centroids_rgb, 0, 1)
    cluster_colors = (centroids_rgb_clipped * 255).astype(np.uint8)

    clustered_vertex_colors = cluster_colors[best_labels]

    # Visualization
    window_width, window_height = window_size
    plotter = vedo.Plotter(N=1, axes=0, bg='white', size=(window_width, window_height))

    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    width = max_x - min_x
    height = max_y - min_y

    padding = 0.05
    scale_x = (1 - 2 * padding) * window_width / width
    scale_y = (1 - 2 * padding) * window_height / height
    scale = min(scale_x, scale_y)

    transformed_vertices = np.copy(vertices)
    transformed_vertices[:, 0] = scale * (vertices[:, 0] - min_x) + window_width * padding
    transformed_vertices[:, 1] = scale * (vertices[:, 1] - min_y) + window_height * padding
    transformed_vertices[:, 1] = window_height - transformed_vertices[:, 1]

    vedo_vertices_for_display = np.c_[transformed_vertices[:, 0], transformed_vertices[:, 1], np.zeros(len(vertices))]

    mesh = vedo.Mesh([vedo_vertices_for_display, triangles])
    mesh.lighting('off')

    vtk_cluster_colors_array = numpy_support.numpy_to_vtk(
        num_array=clustered_vertex_colors,
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_cluster_colors_array.SetName("ClusterColors")
    mesh.polydata().GetPointData().SetScalars(vtk_cluster_colors_array)

    plotter.show(mesh, at=0, title=f"KMeans Clustering (k={best_k}) [Elbow Method]")
    plotter.close()
    print("KMeans clustering visualization closed")

    return best_labels, cluster_colors, best_k




def sample_colors(vertices, original_image):
    print("Sampling colors using bilinear interpolation")
    img_width, img_height = original_image.size
    vertex_colors_sampled = []
    
    # Add validation
    print(f"Sampling {len(vertices)} vertices from {img_width}x{img_height} image")
    
    for i, (vx, vy) in enumerate(vertices):
        # Clamp coordinates to image bounds
        vx = np.clip(vx, 0, img_width - 1)
        vy = np.clip(vy, 0, img_height - 1)
        
        # Get the four surrounding pixels for bilinear interpolation
        x0 = int(np.floor(vx))
        y0 = int(np.floor(vy))
        x1 = min(x0 + 1, img_width - 1)
        y1 = min(y0 + 1, img_height - 1)
        
        # Calculate interpolation weights
        wx = vx - x0
        wy = vy - y0
        
        # Get colors of four surrounding pixels
        c00 = np.array(original_image.getpixel((x0, y0)), dtype=np.float32)
        c10 = np.array(original_image.getpixel((x1, y0)), dtype=np.float32)
        c01 = np.array(original_image.getpixel((x0, y1)), dtype=np.float32)
        c11 = np.array(original_image.getpixel((x1, y1)), dtype=np.float32)
        
        # Bilinear interpolation
        c0 = c00 * (1 - wx) + c10 * wx
        c1 = c01 * (1 - wx) + c11 * wx
        interpolated_color = c0 * (1 - wy) + c1 * wy
        
        # Convert to RGB tuple
        color_rgb = (
            int(interpolated_color[0]),
            int(interpolated_color[1]),
            int(interpolated_color[2])
        )
        vertex_colors_sampled.append(color_rgb)
    
    # Add validation
    print(f"Successfully sampled colors for {len(vertex_colors_sampled)} vertices")
    print(f"Color range: R[{min(c[0] for c in vertex_colors_sampled)}-{max(c[0] for c in vertex_colors_sampled)}], "
          f"G[{min(c[1] for c in vertex_colors_sampled)}-{max(c[1] for c in vertex_colors_sampled)}], "
          f"B[{min(c[2] for c in vertex_colors_sampled)}-{max(c[2] for c in vertex_colors_sampled)}]")
    
    return vertex_colors_sampled

def save_visualization_as_svg(vertices, triangles, vertex_colors, output_path):
    """Save mesh visualization as SVG using vertex colors"""
    try:
        # Calculate bounds
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        
        # Create SVG content
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .triangle {{ stroke: #000; stroke-width: 0.5; }}
    </style>
  </defs>
'''
        
        # Add triangles with interpolated colors
        for i, triangle in enumerate(triangles):
            v1, v2, v3 = vertices[triangle]
            # Use average of vertex colors for triangle (simple interpolation)
            color1 = vertex_colors[triangle[0]]
            color2 = vertex_colors[triangle[1]]
            color3 = vertex_colors[triangle[2]]
            avg_color = [
                int((color1[0] + color2[0] + color3[0]) / 3),
                int((color1[1] + color2[1] + color3[1]) / 3),
                int((color1[2] + color2[2] + color3[2]) / 3)
            ]
            hex_color = f"#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}"
            
            # Create polygon points
            points = f"{v1[0]},{v1[1]} {v2[0]},{v2[1]} {v3[0]},{v3[1]}"
            
            svg_content += f'  <polygon points="{points}" fill="{hex_color}" class="triangle" />\n'
        
        svg_content += '</svg>'
        
        # Save SVG file
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        print(f"Visualization saved as SVG: {output_path}")
        
    except Exception as e:
        print(f"Error saving SVG: {e}")

def reconstruct_image_from_mesh(original_image_path, vertices, triangles, labels, save_svg=False, output_path=None):
    try:
        original_image = Image.open(original_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Original image not found at {original_image_path}")
        return
    except Exception as e:
        print(f"Error loading original image: {e}")
        return

    img_width, img_height = original_image.size
    print(f"Original image dimensions: {img_width}x{img_height}")
    print(f"Mesh has {len(vertices)} vertices and {len(triangles)} triangles")
    print(f"Labels range: {min(labels)} to {max(labels)}")

    # Step 1: Sample vertex colors
    vertex_colors_sampled = sample_colors(vertices, original_image)
    print(f"Sampled colors for {len(vertex_colors_sampled)} vertices")

    # Step 2: Setup visualization
    window_width = 1000
    window_height = 1400
    plotter = vedo.Plotter(N=1, axes=0, bg='white', size=(window_width, window_height))

    # Compute mesh bounds and scaling
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    width = max_x - min_x
    height = max_y - min_y

    padding = 0.05
    scale_x = (1 - 2 * padding) * window_width / width
    scale_y = (1 - 2 * padding) * window_height / height
    scale = min(scale_x, scale_y)

    # Transform vertices to fit in visualization window
    transformed_vertices = np.copy(vertices)
    transformed_vertices[:, 0] = scale * (vertices[:, 0] - min_x) + window_width * padding
    transformed_vertices[:, 1] = scale * (vertices[:, 1] - min_y) + window_height * padding
    transformed_vertices[:, 1] = window_height - transformed_vertices[:, 1]  # Flip Y-axis

    # Add Z-coordinate (all zeros for 2D mesh)
    vedo_vertices_for_display = np.c_[transformed_vertices[:, 0], transformed_vertices[:, 1], np.zeros(len(vertices))]

    # Create vedo mesh
    mesh = vedo.Mesh([vedo_vertices_for_display, triangles])
    mesh.lighting('off')

    # Assign vertex colors directly
    vtk_vertex_colors_array = numpy_support.numpy_to_vtk(
        num_array=vertex_colors_sampled,
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_vertex_colors_array.SetName("PerVertexColors")
    mesh_polydata = mesh.polydata()
    mesh_polydata.GetPointData().SetScalars(vtk_vertex_colors_array)
    print("Assigned per-vertex colors to mesh")

    # Show mesh
    plotter.show(mesh, at=0)
    labels, colors, chosen_k = kmeans_cluster_and_visualize_auto_k(
    vertices, triangles, vertex_colors_sampled, k_min=2, k_max=20)
    print(f"Automatically selected k={chosen_k}")


    if save_svg:
        if output_path is None:
            import os
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            output_path = f"{base_name}_mesh_reconstructed.svg"
        save_visualization_as_svg(vertices, triangles, vertex_colors_sampled, output_path)
    
    plotter.close()
    print("Vedo visualization closed")

    # Return reconstructed mesh data
    reconstructed_mesh = {
        "vertices": vertices,
        "triangles": triangles,
        "vertex_colors": vertex_colors_sampled,  # sampled colors for each vertex
        "labels": labels
    }

    return reconstructed_mesh


import numpy as np
import cv2
import sys
import os
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import polygonize
from shapely.prepared import prep
from meshpy.triangle import MeshInfo, build
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from collections import defaultdict
from reconstruct import reconstruct_image_from_mesh

# Converts image to label image
def rgb_to_label(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    flat_rgb = image_rgb.reshape(-1, 3).astype(np.uint32)
    flat_labels = (flat_rgb[:, 0] << 16) | (flat_rgb[:, 1] << 8) | flat_rgb[:, 2] # Converts RGB to label
    unique_vals, inverse = np.unique(flat_labels, return_inverse=True)
    label_img = inverse.reshape(h, w)
    
    
    return label_img, image_rgb

# Extracts segment boundaries from label image by using edge detection
def extract_segment_boundaries(label_img, simplify_tolerance=0.4):
    h, w = label_img.shape
    segment_edges = defaultdict(set)
    h_diff = np.diff(label_img, axis=1)
    h_boundaries = np.where(h_diff != 0)
    for y, x in zip(h_boundaries[0], h_boundaries[1]):
        curr_label = label_img[y, x]
        next_label = label_img[y, x + 1]
        edge = ((x + 1, y), (x + 1, y + 1)) # Edge between current and next pixel
        segment_edges[curr_label].add(edge)
        segment_edges[next_label].add(edge)
    v_diff = np.diff(label_img, axis=0)
    v_boundaries = np.where(v_diff != 0)
    for y, x in zip(v_boundaries[0], v_boundaries[1]):
        curr_label = label_img[y, x]
        next_label = label_img[y + 1, x]
        edge = ((x, y + 1), (x + 1, y + 1)) # Edge between current and next pixel
        segment_edges[curr_label].add(edge)
        segment_edges[next_label].add(edge)
    for y in range(h):
        segment_edges[label_img[y, 0]].add(((0, y), (0, y + 1)))
        segment_edges[label_img[y, w-1]].add(((w, y), (w, y + 1)))
    for x in range(w):
        segment_edges[label_img[0, x]].add(((x, 0), (x + 1, 0)))
        segment_edges[label_img[h-1, x]].add(((x, h), (x + 1, h)))
    # Polygonize and simplify
    segment_boundaries = {} # Stores the boundaries for each label
    for label_val, edges in segment_edges.items():
        lines = [LineString([p1, p2]) for p1, p2 in edges]
        polys = list(polygonize(lines))
        # Simplifies the polygon by removing small edges
        # simplified_polys = [p.simplify(simplify_tolerance, preserve_topology=True) for p in polys if p.is_valid and not p.is_empty]
        # valid_polys = [p for p in simplified_polys if p.is_valid and not p.is_empty]
        valid_polys = [p for p in polys if p.is_valid and not p.is_empty]
        if valid_polys:
            segment_boundaries[label_val] = MultiPolygon(valid_polys)
    
    return segment_boundaries

# Extracts points and segments from boundaries
def extract_points_and_segments(boundaries):
    point_id_map = {} # Maps points to their indices to avoid duplicates
    points = [] # stores the points as a list of tuples of coordinates
    segments = [] # stores the segments as a list of tuples of point indices
    point_index = 0
    # Gets the index of a point by checking if it is in the point_id_map
    def get_point_index(pt):
        nonlocal point_index
        if pt not in point_id_map:
            point_id_map[pt] = point_index
            points.append(pt)
            point_index += 1
        return point_id_map[pt]
    # Extracts points and segments from boundaries by iterating through each polygon
    for multipoly in boundaries.values():
        for poly in multipoly.geoms:
            if not poly.is_valid or poly.is_empty:
                continue
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                i1 = get_point_index(tuple(coords[i]))
                i2 = get_point_index(tuple(coords[i + 1]))
                segments.append((i1, i2))
            for interior in poly.interiors:
                interior_coords = list(interior.coords)
                for i in range(len(interior_coords) - 1):
                    i1 = get_point_index(tuple(interior_coords[i]))
                    i2 = get_point_index(tuple(interior_coords[i + 1]))
                    segments.append((i1, i2))
    return points, segments

def mesh_all_segments(boundaries, image_shape, max_volume=1000.0, min_angle=25.0):
    points, segments = extract_points_and_segments(boundaries)
    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(segments)
    mesh = build(mesh_info, max_volume=max_volume, min_angle=min_angle)
    verts = np.array(mesh.points)
    tris = np.array(mesh.elements)
    return verts, tris

def assign_labels_to_triangles(triangles, vertices, label_img):
    centroids = np.mean(vertices[triangles], axis=1)
    h, w = label_img.shape
    
    labels = np.full(len(triangles), -1, dtype=np.int32)
    # labels stores triangles with the label it is associated with
    for i, centroid in enumerate(centroids):
        # Convert centroid to image coordinates
        x = int(centroid[0])
        y = int(centroid[1])
        
        # Ensure coordinates are within image bounds
        if 0 <= x < w and 0 <= y < h:
            # Get the label from the original image at this pixel
            labels[i] = label_img[y, x]
        else:
            # For triangles outside image bounds, use nearest valid pixel
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            labels[i] = label_img[y, x]
    # stores trinagles segment wise
    segments_dict = {}
    for i, label in enumerate(labels):
        if label not in segments_dict:
            segments_dict[label] = []
        segments_dict[label].append(triangles[i])
    
    return labels.tolist(), segments_dict


def visualize_colored_mesh(vertices, triangles, labels, image_shape):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white', dpi=100)
    h, w = image_shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    labels_array = np.array(labels)
    unique_labels = np.unique(labels_array)
    if -1 in unique_labels:
        unique_labels = unique_labels[unique_labels != -1]
    if len(unique_labels) > 0:
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        color_indices = np.full(labels_array.shape, -1, dtype=int)
        for label, idx in label_to_idx.items():
            color_indices[labels_array == label] = idx
        triangle_colors = np.ones((len(labels_array), 4))
        for idx, label in enumerate(unique_labels):
            triangle_colors[color_indices == idx] = cmap(idx)
        triangle_colors[color_indices == -1] = [0.5, 0.5, 0.5, 1.0]
        triangle_verts = vertices[triangles]
        collection = PolyCollection(
            triangle_verts,
            facecolors=triangle_colors,
            edgecolors='black',
            linewidths=0.2,
            alpha=0.9
        )
        ax.add_collection(collection)
    plt.title("Mesh Visualization")
    plt.tight_layout()
    plt.show()



# to visualize the mesh segment wise
# def visualize_segment_triangles(vertices, segments_dict, image_shape, segment_label=None):
#     h, w = image_shape
    
#     if segment_label is not None:
#         # Visualize only specific segment
#         if segment_label not in segments_dict:
#             print(f"Segment {segment_label} not found!")
#             return
        
#         fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
#         ax.set_xlim(0, w)
#         ax.set_ylim(h, 0)
#         ax.set_aspect('equal')
#         ax.axis('off')
#         ax.set_facecolor('white')
        
#         # Get triangles for this segment
#         segment_triangles = segments_dict[segment_label]
#         triangle_verts = [vertices[tri] for tri in segment_triangles]
        
#         # Create polygon collection for this segment
#         collection = PolyCollection(
#             triangle_verts,
#             facecolors='red',  # Single color for single segment
#             edgecolors='black',
#             linewidths=0.5,
#             alpha=0.8
#         )
#         ax.add_collection(collection)
        
#         plt.title(f"Segment {segment_label} - {len(segment_triangles)} triangles", fontsize=14, pad=20)
#         plt.tight_layout()
#         plt.show()
        
#     else:
#         # Visualize all segments one by one
#         for label in sorted(segments_dict.keys()):
#             fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
#             ax.set_xlim(0, w)
#             ax.set_ylim(h, 0)
#             ax.set_aspect('equal')
#             ax.axis('off')
#             ax.set_facecolor('white')
            
#             # Get triangles for this segment
#             segment_triangles = segments_dict[label]
#             triangle_verts = [vertices[tri] for tri in segment_triangles]
            
#             # Create polygon collection
#             collection = PolyCollection(
#                 triangle_verts,
#                 facecolors='red',  # Single color for each segment
#                 edgecolors='black',
#                 linewidths=0.5,
#                 alpha=0.8
#             )
#             ax.add_collection(collection)
            
#             plt.title(f"Segment {label} - {len(segment_triangles)} triangles", fontsize=14, pad=20)
#             plt.tight_layout()
#             plt.show()

if __name__ == "__main__":
    # Check if image path is provided as command line argument
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 mesh.py <path_to_input_image> [path_to_original_image]")
        print("Example: python3 mesh.py /path/to/your/image.png")
        print("Example: python3 mesh.py /path/to/your/image.png /path/to/original_image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    original_image_path = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Check if input file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)
    
    # Check if original image file exists (if provided)
    if original_image_path and not os.path.exists(original_image_path):
        print(f"Error: Original image file '{original_image_path}' does not exist.")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    
    label_img, rgb_image = rgb_to_label(image_path)
    print('Image loaded')
    boundaries = extract_segment_boundaries(label_img, simplify_tolerance=0.4)
    print('Extracted boundaries')
    vertices, triangles = mesh_all_segments(boundaries, label_img.shape, max_volume=1000.0, min_angle=25.0)
    print('Performed mesh triangulation')
    # segments_dict stores triangles, segment wise, with the label it is associated with
    labels,segments_dict = assign_labels_to_triangles(triangles, vertices, label_img)
    print('Assigned labels to triangles')
    # visualize_segment_triangles(vertices, segments_dict, label_img.shape)
    visualize_colored_mesh(vertices, triangles, labels, label_img.shape)
    
    # Run reconstruction if original image path is provided
    if original_image_path:
        print(f'Reconstruction with original image: {original_image_path}')
        reconstructed_mesh = reconstruct_image_from_mesh(original_image_path, vertices, triangles, labels)
        print('Reconstructed image')

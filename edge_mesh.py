import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import polygonize
from collections import defaultdict, deque
from meshpy.triangle import MeshInfo, build
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from reconstruct import reconstruct_image_from_mesh
from kmeans import kmeans_seg

def extract_boundaries_from_connected_components(label_img, connectivity=8, simplify_tolerance=0.4):
    h, w = label_img.shape
    segment_edges = defaultdict(set)

    # Build edges between pixels of different labels (boundary edges)
    h_diff = np.diff(label_img, axis=1)
    h_boundaries = np.where(h_diff != 0)
    for y, x in zip(h_boundaries[0], h_boundaries[1]):
        label1 = label_img[y, x]
        label2 = label_img[y, x + 1]
        edge = ((x + 1, y), (x + 1, y + 1))
        segment_edges[label1].add(edge)
        segment_edges[label2].add(edge)


    v_diff = np.diff(label_img, axis=0)
    v_boundaries = np.where(v_diff != 0)
    for y, x in zip(v_boundaries[0], v_boundaries[1]):
        label1 = label_img[y, x]
        label2 = label_img[y + 1, x]

        edge = ((x, y + 1), (x + 1, y + 1))
        segment_edges[label1].add(edge)
        segment_edges[label2].add(edge)


    for y in range(h):
        segment_edges[label_img[y, 0]].add(((0, y), (0, y + 1)))
        segment_edges[label_img[y, w - 1]].add(((w, y), (w, y + 1)))
    for x in range(w):
        segment_edges[label_img[0, x]].add(((x, 0), (x + 1, 0)))
        segment_edges[label_img[h - 1, x]].add(((x, h), (x + 1, h)))

    # Polygonize and simplify boundaries
    segment_boundaries = {}
    for label_val, edges in segment_edges.items():
        lines = [LineString([p1, p2]) for p1, p2 in edges]
        polys = list(polygonize(lines))
        #simplified_polys = [p.simplify(simplify_tolerance, preserve_topology=True) for p in polys if p.is_valid and not p.is_empty]
        #valid_polys = [p for p in simplified_polys if p.is_valid and not p.is_empty]
        valid_polys = [p for p in polys if p.is_valid and not p.is_empty]
        if valid_polys:
            segment_boundaries[label_val] = MultiPolygon(valid_polys)

    return segment_boundaries

def extract_points_and_segments(boundaries):
    point_id_map = {}
    points = []
    segments = []
    point_index = 0

    def get_point_index(pt):
        nonlocal point_index
        if pt not in point_id_map:
            point_id_map[pt] = point_index
            points.append(pt)
            point_index += 1
        return point_id_map[pt]

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

def mesh_all_segments(boundaries, image_shape, max_volume=250.0, min_angle=25.0):
    points, segments = extract_points_and_segments(boundaries)
    points = [(round(p[0], 1), round(p[1], 1)) for p in points]

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(segments)
    mesh = build(mesh_info, max_volume=max_volume, min_angle=min_angle)
    verts = np.array(mesh.points)
    tris = np.array(mesh.elements)

    # Limit subdivision per pixel (max 3 triangles per pixel)
    h, w = image_shape
    pixel_triangles = defaultdict(list)

    for i, tri in enumerate(tris):
        centroid = np.mean(verts[tri], axis=0)
        px, py = int(centroid[0]), int(centroid[1])
        if 0 <= px < w and 0 <= py < h:
            pixel_triangles[(px, py)].append(i)

    valid_tris = []
    for pixel, tri_indices in pixel_triangles.items():
        if len(tri_indices) > 3:
            areas = []
            for tri_idx in tri_indices:
                tri = tris[tri_idx]
                v1, v2, v3 = verts[tri]
                area = abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])) / 2
                areas.append((area, tri_idx))
            areas.sort(reverse=True)
            valid_tris.extend([tri_idx for _, tri_idx in areas[:3]])
        else:
            valid_tris.extend(tri_indices)

    return verts, tris[valid_tris]

def assign_labels_to_triangles(triangles, vertices, label_img):
    centroids = np.mean(vertices[triangles], axis=1)
    h, w = label_img.shape
    labels = np.full(len(triangles), -1, dtype=np.int32)

    for i, centroid in enumerate(centroids):
        x = int(centroid[0])
        y = int(centroid[1])
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        labels[i] = label_img[y, x]

    segments_dict = defaultdict(list)
    for i, label in enumerate(labels):
        segments_dict[label].append(triangles[i])

    return labels.tolist(), segments_dict

def visualize_colored_mesh(vertices, triangles, labels, image_shape):
    h, w = image_shape
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

    labels_array = np.array(labels)
    unique_labels = np.unique(labels_array)
    if -1 in unique_labels:
        unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) == 0:
        print("No valid labeled triangles to display.")
        return

    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    color_indices = np.full(labels_array.shape, -1, dtype=int)

    for label, idx in label_to_idx.items():
        color_indices[labels_array == label] = idx

    triangle_colors = np.ones((len(labels_array), 4))
    for idx in range(len(unique_labels)):
        triangle_colors[color_indices == idx] = cmap(idx)
    triangle_colors[color_indices == -1] = [0.5, 0.5, 0.5, 1.0]

    triangle_verts = vertices[triangles]
    collection = PolyCollection(triangle_verts, facecolors=triangle_colors,
                                edgecolors='black', linewidths=0.2, alpha=0.9)
    ax.add_collection(collection)
    plt.title("2D Mesh from Edge Image")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python edge_mesh.py <path_to_edge_image> [path_to_original_image]")
        print("Example: python edge_mesh.py /path/to/edge_image.png")
        print("Example: python edge_mesh.py /path/to/edge_image.png /path/to/original_image.png")
        sys.exit(1)

    edge_path = sys.argv[1]
    original_image_path = sys.argv[2] if len(sys.argv) == 3 else None
    
    if not os.path.exists(edge_path):
        print(f"File '{edge_path}' not found.")
        sys.exit(1)
    
    # Check if original image file exists (if provided)
    if original_image_path and not os.path.exists(original_image_path):
        print(f"Error: Original image file '{original_image_path}' does not exist.")
        sys.exit(1)

    # Load image
    edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edge_img is None:
        print(f"Failed to load image '{edge_path}'.")
        sys.exit(1)

    # Threshold to get binary edge image
    _, binary_edge = cv2.threshold(edge_img, 127, 255, cv2.THRESH_BINARY)

    # Invert edges to get connected regions inside edges
    region_mask = cv2.bitwise_not(binary_edge)

    # Connected components labeling (each connected region gets a unique label)
    num_labels, labels_img = cv2.connectedComponents(region_mask, connectivity=8)
    print(f"Found {num_labels - 1} connected components inside edges.")

    # Extract boundaries and perform mesh triangulation
    boundaries = extract_boundaries_from_connected_components(labels_img, connectivity=8, simplify_tolerance=0.4)
    vertices, triangles = mesh_all_segments(boundaries, labels_img.shape, max_volume=1000.0, min_angle=25.0)
    labels, segments_dict = assign_labels_to_triangles(triangles, vertices, labels_img)

    # Visualize the resulting mesh
    visualize_colored_mesh(vertices, triangles, labels, labels_img.shape)
    
    # Run reconstruction if original image path is provided
    if original_image_path:
        print(f'Running reconstruction with original image: {original_image_path}')
        reconstructed_mesh = reconstruct_image_from_mesh(original_image_path, vertices, triangles, labels)
        print('Reconstructed image')

        vertex_colors = reconstructed_mesh['vertex_colors']
        dtype = np.dtype([
            ('coordinates', np.float64, (2,)),  # x, y coordinates
            ('colors', np.float32, (3,)),       # r, g, b colors
            ('vertex_id', np.int32)             # vertex index
        ])
        
        vertex_data = np.zeros(len(vertices), dtype=dtype)
        vertex_data['coordinates'] = vertices
        vertex_data['colors'] = vertex_colors
        vertex_data['vertex_id'] = np.arange(len(vertices))

        original_image = cv2.imread(original_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        print("Performing kmeans clustering")
        kmeans_seg(vertex_data, vertices, triangles, segments_dict, original_image, labels)
        print("Kmeans clustering with differnt visualisation results")
        

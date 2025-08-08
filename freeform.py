#!/usr/bin/env python3

import numpy as np
import cv2
from shapely.geometry import MultiPolygon, LineString
from shapely.ops import polygonize
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import os
import time
from meshpy.triangle import build, MeshInfo
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import spsolve
from scipy.sparse import vstack, identity
import igl

# Global variables for animation
X, Y = None, None
history_e = None
iter = 1
finished = False

def boundary2str(X, Y, G):
    """Convert boundary information to string"""
    def colour2str(c):
        s = "("
        for i in range(c.shape[0] - 1):
            s += f"{c[i]:.2f},"
        s += f"{c[-1]:.2f})"
        return s

    s = ""
    for i in range(len(X)):
        g = G[X[i]]
        y = Y[i]
        s += f'\tX: {X[i]}, Y: {colour2str(y)}, G: {colour2str(g)}\n'
    return s


def extract_boundaries_from_connected_components(label_img, connectivity=8, simplify_tolerance=0.4):
    """
    Extract boundaries from connected components labeling.
    
    Args:
        label_img: Label image where each connected component has a unique label
        connectivity: Connectivity for connected components (4 or 8)
        simplify_tolerance: Tolerance for polygon simplification
    
    Returns:
        Dictionary mapping segment labels to MultiPolygon boundaries
    """
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

    # Add image boundary edges
    for y in range(h):
        segment_edges[label_img[y, 0]].add(((0, y), (0, y + 1)))
        segment_edges[label_img[y, w - 1]].add(((w, y), (w, y + 1)))
    for x in range(w):
        segment_edges[label_img[0, x]].add(((x, 0), (x + 1, 0)))
        segment_edges[label_img[h - 1, x]].add(((x, h), (x + 1, h)))

    # Polygonize and simplify boundaries
    segment_boundaries = {}
    for label_val, edges in segment_edges.items():
        if label_val == 0:
            continue
        lines = [LineString([p1, p2]) for p1, p2 in edges]
        polys = list(polygonize(lines))
        valid_polys = [p for p in polys if p.is_valid and not p.is_empty]
        if valid_polys:
            segment_boundaries[label_val] = MultiPolygon(valid_polys)

    return segment_boundaries


def mesh_all_segments(boundaries, image_shape, max_volume=500.0, min_angle=30.0):
    """
    Generate triangular mesh for all segments using triangle library.
    
    Args:
        boundaries: Dictionary mapping segment labels to MultiPolygon boundaries
        image_shape: (height, width) of the original image
        max_volume: Maximum triangle area constraint
        min_angle: Minimum angle constraint for triangles
    
    Returns:
        vertices: Array of vertex coordinates (N, 2)
        triangles: Array of triangle indices (M, 3)
    """
    points, segments = extract_points_and_segments(boundaries)
    points = [(round(p[0], 1), round(p[1], 1)) for p in points]

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(segments)
    mesh = build(mesh_info, max_volume=max_volume, min_angle=min_angle)
    verts = np.array(mesh.points)
    tris = np.array(mesh.elements)


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

    print(f"Generated mesh with {len(verts)} vertices and {len(valid_tris)} triangles")
    return verts, tris[valid_tris]

def extract_points_and_segments(boundaries):
    """
    Extract unique points and line segments from polygon boundaries.
    
    Args:
        boundaries: Dictionary mapping segment labels to MultiPolygon boundaries
    
    Returns:
        points: List of unique (x, y) coordinates
        segments: List of (point_index1, point_index2) pairs
    """
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

def sample_vertex_colors_from_image(vertices, original_image):
    """
    Sample colors from original image at vertex positions.
    
    Args:
        vertices: Array of vertex coordinates (N, 2) in image space
        original_image: Original image array (H, W, 3) in RGB format
    
    Returns:
        vertex_colors: Array of colors (N, 3) for each vertex in [0,1] range
    """
    h, w = original_image.shape[:2]
    vertex_colors = []
    
    for vertex in vertices:
        x, y = int(vertex[0]), int(vertex[1])
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        color = original_image[y, x] / 255.0
        vertex_colors.append(color)
    
    return np.array(vertex_colors)


def compute_mesh_laplacian(vertices, vertex_colors, triangles):
    """
    Compute discrete Laplacian for mesh vertices using cotangent weights.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3) in [0,1] range
        triangles: Array of triangle indices (M, 3)
    
    Returns:
        laplacian_values: Array of Laplacian magnitudes (N,)
    """
    # Compute cotangent Laplacian matrix
    L = igl.cotmatrix(vertices, triangles)
    
    # Compute Laplacian for each color channel
    laplacian_values = np.zeros(len(vertices))
    
    for channel in range(3): 
        channel_laplacian = L.dot(vertex_colors[:, channel])
        laplacian_values += np.abs(channel_laplacian)
    
    return laplacian_values


def max_min_pool_mesh(laplacian_values, vertices, radius):
    """
    Implement max/min pooling for mesh vertices.
    
    Args:
        laplacian_values: Array of Laplacian values (N,)
        vertices: Array of vertex coordinates (N, 2)
        radius: Pooling radius
    
    Returns:
        max_pooled: Array of max-pooled values (N,)
        min_pooled: Array of min-pooled values (N,)
    """
    n_vertices = len(vertices)
    max_pooled = np.zeros(n_vertices)
    min_pooled = np.zeros(n_vertices)
    
    for i in range(n_vertices):
        distances = cdist([vertices[i]], vertices)[0]
        neighbors = np.where(distances <= radius)[0]
        
        if len(neighbors) > 0:
            max_pooled[i] = np.max(laplacian_values[neighbors])
            min_pooled[i] = np.min(laplacian_values[neighbors])
        else:
            max_pooled[i] = laplacian_values[i]
            min_pooled[i] = laplacian_values[i]
    
    return max_pooled, min_pooled


def connected_components_clustering_mesh(vertex_indices, vertices, radius):
    """
    Implement connected components clustering for mesh vertices.
    
    Args:
        vertex_indices: Array of vertex indices to cluster
        vertices: Array of vertex coordinates (N, 2)
        radius: Clustering radius
    
    Returns:
        clusters: List of vertex index arrays for each cluster
    """
    if len(vertex_indices) == 0:
        return []
    
    vertex_coords = vertices[vertex_indices]
    visited = np.zeros(len(vertex_indices), dtype=bool)
    clusters = []
    
    def traverse(start_idx):
        """Find connected component starting from start_idx"""
        if visited[start_idx]:
            return []
        
        visited[start_idx] = True
        component = [start_idx]
        queue = [start_idx]
        
        while queue:
            current_idx = queue.pop(0)
            
            # Find neighbors within radius
            distances = cdist([vertex_coords[current_idx]], vertex_coords)[0]
            neighbors = np.where(distances <= radius)[0]
            
            for neighbor_idx in neighbors:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    component.append(neighbor_idx)
                    queue.append(neighbor_idx)
        
        return component
    
    # Find all connected components
    for i in range(len(vertex_indices)):
        if not visited[i]:
            component = traverse(i)
            if len(component) > 0:
                cluster = [vertex_indices[j] for j in component]
                clusters.append(cluster)
    
    return clusters


def find_control_points_mesh_based(vertices, vertex_colors, triangles, top_k=3):
    """
    Find control points using the exact same clustering approach as harmonic.py
    but adapted for mesh space.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3) in [0,1] range
        triangles: Array of triangle indices (M, 3)
        top_k: Number of control points to find per channel
    
    Returns:
        control_point_indices: Array of control point vertex indices
    """
    print(f"Finding control points using mesh-based analysis (top_k={top_k} per channel).")
    
    ps_max_per_channel = []
    ps_min_per_channel = []
    
    for channel in range(3):  
        print(f"Processing channel {channel}...")
        
        # Compute Laplacian for this channel
        L = igl.cotmatrix(vertices, triangles)
        channel_laplacian = L.dot(vertex_colors[:, channel])
        
        # Apply max/min pooling
        max_dim = max(np.max(vertices[:, 0]), np.max(vertices[:, 1]))
        pooling_radius = max_dim // 30
        max_pooled, min_pooled = max_min_pool_mesh(channel_laplacian, vertices, pooling_radius)
        
        # Find top 3 peaks and valleys per channel
        peak_indices = np.argsort(max_pooled)[-3:][::-1]  # Top 3 peaks
        valley_indices = np.argsort(min_pooled)[:3]       # Top 3 valleys
        
        ps_max_per_channel.append(peak_indices)
        ps_min_per_channel.append(valley_indices)
    
    # Step 2: Combine all channels
    ps_max = np.concatenate(ps_max_per_channel, axis=0)  # 9 points (3 per channel)
    ps_min = np.concatenate(ps_min_per_channel, axis=0)  # 9 points (3 per channel)
    
    print(f"Combined points: {len(ps_max)} peaks, {len(ps_min)} valleys")
    
    # Step 3: TWO-STAGE CLUSTERING
    max_dim = max(np.max(vertices[:, 0]), np.max(vertices[:, 1]))
    
    # Stage 1: Separate clustering of peaks and valleys
    radius1 = max_dim // 25  
    print(f"Stage 1 clustering radius: {radius1}")
    
    peak_clusters = connected_components_clustering_mesh(ps_max, vertices, radius1)
    valley_clusters = connected_components_clustering_mesh(ps_min, vertices, radius1)
    
    # Compute cluster centers for peaks and valleys
    peak_centers = []
    for cluster in peak_clusters:
        cluster_vertices = vertices[cluster]
        center_idx = cluster[np.argmin(np.linalg.norm(cluster_vertices - np.mean(cluster_vertices, axis=0), axis=1))]
        peak_centers.append(center_idx)
    
    valley_centers = []
    for cluster in valley_clusters:
        cluster_vertices = vertices[cluster]
        center_idx = cluster[np.argmin(np.linalg.norm(cluster_vertices - np.mean(cluster_vertices, axis=0), axis=1))]
        valley_centers.append(center_idx)
    
    # Combine peaks and valleys
    initial_points = np.concatenate([peak_centers, valley_centers]) if len(peak_centers) > 0 and len(valley_centers) > 0 else (
        peak_centers if len(peak_centers) > 0 else valley_centers
    )
    print(f"After Stage 1: {len(initial_points)} points")
    
    # Stage 2: Combined clustering of all points
    radius2 = max_dim // 15  
    print(f"Stage 2 clustering radius: {radius2}")
    final_clusters = connected_components_clustering_mesh(initial_points, vertices, radius2)
    
    # Compute final centers
    final_indices = []
    for cluster in final_clusters:
        cluster_vertices = vertices[cluster]
        center_idx = cluster[np.argmin(np.linalg.norm(cluster_vertices - np.mean(cluster_vertices, axis=0), axis=1))]
        final_indices.append(center_idx)
    
    final_indices = np.array(final_indices)
    print(f"After Stage 2: {len(final_indices)} points")
    
    if len(final_indices) == 0:
        print("Warning: No control points found. Using random vertices.")
        final_indices = np.random.choice(len(vertices), min(6, len(vertices)), replace=False)
    
    print(f"Final control points: {len(final_indices)}")
    for i, idx in enumerate(final_indices):
        point = vertices[idx]
        color = vertex_colors[idx]
        print(f"  Point {i+1}: Vertex {idx} at ({point[0]:.1f}, {point[1]:.1f}), Color RGB{tuple((color * 255).astype(int))}")
    
    return final_indices


def assign_labels_to_triangles(triangles, vertices, label_img):
    """
    Assign segment labels to triangles based on their centroids.
    
    Args:
        triangles: Array of triangle indices (M, 3)
        vertices: Array of vertex coordinates (N, 2)
        label_img: Label image where each pixel has a segment label
    
    Returns:
        labels: List of labels for each triangle
        segments_dict: Dictionary mapping segment labels to triangle arrays
    """
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

def process_canny_edge_image(edge_path, original_image_path):
    """
    Process Canny edge image and create mesh inside edge boundaries.
    
    Args:
        edge_path: Path to the Canny edge image
        original_image_path: Path to the original image
    
    Returns:
        Dictionary containing mesh data and processing results
    """
    print("=== Processing Canny Edge Image ===")
    
    # Step 1: Load and validate images
    edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edge_img is None:
        print(f"Failed to load edge image '{edge_path}'.")
        return None
    
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"Failed to load original image '{original_image_path}'.")
        return None
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    print(f"Loaded edge image: {edge_img.shape}")
    print(f"Loaded original image: {original_image.shape}")
    
    # Step 2: Process edge image to create regions
    # Threshold to get binary edge image
    _, binary_edge = cv2.threshold(edge_img, 127, 255, cv2.THRESH_BINARY)
    
    # Invert edges to get connected regions inside edges
    region_mask = cv2.bitwise_not(binary_edge)
    
    # Connected components labeling
    num_labels, labels_img = cv2.connectedComponents(region_mask, connectivity=8)
    print(f"Found {num_labels - 1} connected components inside edges.")
    
    # Validate that we have meaningful regions
    if num_labels <= 1:
        print("Warning: No meaningful regions found in edge image.")
        print("This could indicate:")
        print("  - Edge image is all white (no edges detected)")
        print("  - Edge image is all black (no interior regions)")
        print("  - Edge detection failed to create proper boundaries")
        return None
    
    # Step 3: Extract boundaries and create mesh
    print("Extracting boundaries from connected components...")
    boundaries = extract_boundaries_from_connected_components(labels_img, connectivity=8, simplify_tolerance=0.4)
    
    if not boundaries:
        print("Warning: No valid boundaries extracted.")
        return None
    
    print(f"Extracted boundaries for {len(boundaries)} segments")
    
    # Step 4: Generate triangular mesh
    print("Generating triangular mesh...")
    vertices, triangles = mesh_all_segments(boundaries, labels_img.shape, max_volume=1000.0, min_angle=30.0)  
    print(f"Generated mesh with {len(vertices)} vertices and {len(triangles)} triangles")
    
    if len(vertices) == 0:
        print("Error: No vertices generated. Check boundary extraction.")
        return None
    
    # Step 5: Assign labels to triangles
    print("Assigning segment labels to triangles...")
    labels, segments_dict = assign_labels_to_triangles(triangles, vertices, labels_img)
    print(f"Assigned labels to {len(labels)} triangles across {len(segments_dict)} segments")
    
    # Step 6: Sample colors and find control points for entire mesh 
    print("\n=== Processing Entire Mesh ===")
    
    # Sample colors for all vertices
    print("Sampling colors for all vertices...")
    vertex_colors = sample_vertex_colors_from_image(vertices, original_image)
    print(f"Sampled colors for {len(vertex_colors)} vertices")
    
    # Step 6.5: Create mesh-based boundary filtering 
    print("Creating mesh-based boundary filtering...")
    max_dim = max(np.max(vertices[:, 0]), np.max(vertices[:, 1]))
    inset_radius = max_dim // 50  # Same as harmonic.py
    inset_mask = create_mesh_inset_mask(vertices, triangles, inset_radius)
    
    # Filter to internal vertices only
    internal_vertex_indices = np.where(inset_mask)[0]
    internal_vertices = vertices[internal_vertex_indices]
    internal_colors = vertex_colors[internal_vertex_indices]
    
    print(f"Filtered to {len(internal_vertex_indices)} internal vertices out of {len(vertices)} total")
    
    # Find control points for entire mesh 
    print("Finding control points for entire mesh...")
    
    # Start timing for initial control points
    initial_control_start_time = time.time()
    
    # Create filtered triangles that only use internal vertices
    filtered_triangles = filter_triangles_for_vertices(triangles, internal_vertex_indices)
    
    internal_control_indices = find_control_points_mesh_based(
        internal_vertices, 
        internal_colors, 
        filtered_triangles,  # Use filtered triangles for Laplacian computation
        top_k=3  
    )
    
    # Map back to original vertex indices
    control_point_indices = internal_vertex_indices[internal_control_indices]
    init_control_pts = control_point_indices.copy()  # Store initial control points
    
    # End timing for initial control points
    initial_control_end_time = time.time()
    initial_control_time = initial_control_end_time - initial_control_start_time
    
    print(f"Found {len(control_point_indices)} control points for entire mesh")
    print(f"Initial control points creation time: {initial_control_time:.4f} seconds")
    
    # Visualize initial control points 
    print("\n=== Visualizing Initial Control Points ===")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    
    # Show original image as background
    ax.imshow(original_image)
    
    # Get control point coordinates from indices
    control_point_coords = vertices[control_point_indices]
    
    # Plot control points 
    ax.scatter(control_point_coords[:, 0], control_point_coords[:, 1], 
               c='k', s=50, alpha=0.8)
    
    # Add vertex indices as text labels for debugging
    # for i, (x, y) in enumerate(control_point_coords):
    #     ax.text(x, y, f'{control_point_indices[i]}', 
    #             fontsize=8, ha='center', va='bottom', color='white')
    
    ax.set_title(f"Initial Control Points: {len(control_point_indices)} vertices")
    ax.set_aspect('equal')
    plt.show()
    
    # Step 7: Optimize control points using harmonic functions 
    print("\n=== Optimizing Control Points ===")
    
    # Start timing for optimization process
    optimization_start_time = time.time()
    
    # Create vertex mask (only internal vertices valid)
    vert_mask = np.zeros((len(vertices), 1))
    vert_mask[internal_vertex_indices] = 1
    
    # Create Manifold object
    M = Manifold(vertices, triangles, vert_mask, original_image)
    print("Ground Shape:", vertex_colors.shape)
    
    # Set up boundary conditions 
    boundary_indices = control_point_indices
    boundary_values = vertex_colors[boundary_indices]
    Boundary = [boundary_indices, boundary_values]
    Theta_Indx = list(range(len(boundary_indices)))
    
    print("Theta X initial:", boundary_indices)
    
    # Initial solution and energy calculation
    u = M.U(Boundary)
    print("U Shape:", u.shape)
    
    diff = (vertex_colors - u) * M.vert_mask
    print("Initial Diff Shape:", diff.shape)
    for c in range(3): print(f"\t axis[{c}] := {np.sum(diff[:, c] ** 2)}")
    
    e = np.sum(((vertex_colors - u) * M.vert_mask) ** 2)
    print("Initial Energy: ", e)
    
    # optimization without animation
    M.optimize(vertex_colors, Boundary, Theta_Indx, alpha=1e-3, eps=1e-1, deb=False, last_E=e, max_iter=15)
    
    # Get final results after animation
    X, Y = Boundary
    
    optimization_end_time = time.time()
    optimization_time = optimization_end_time - optimization_start_time
    
    # Calculate final energy
    final_energy = np.sum(((vertex_colors - M.U([X, Y])) * M.vert_mask) ** 2)
    print(f"\nOptimization Results:")
    print(f"Initial Energy: {e:.4f}")
    print(f"Final Energy: {final_energy:.4f}")
    print(f"Optimization time: {optimization_time:.4f} seconds")
    print(f"Initial control points completed in : {initial_control_time:.4f} seconds")
    print(f"Total control point processing time: {initial_control_time + optimization_time:.4f} seconds")
    #print(f"Energy Reduction: {((e - final_energy) / e * 100):.2f}%")
    
    # Store results
    mesh_results = {
        'vertex_colors': vertex_colors,
        'control_point_indices': control_point_indices,
        'control_point_coordinates': vertices[control_point_indices],
        'control_point_colors': vertex_colors[control_point_indices],
        'optimized_indices': X,
        'optimized_values': Y,
        'final_solution': M.U([X, Y]),
        'initial_energy': e,
        'final_energy': final_energy,
        'initial_control_time': initial_control_time,
        'optimization_time': optimization_time,
        'total_control_time': initial_control_time + optimization_time
    }
    
    # Add to return dictionary
    result_dict = {
        'vertices': vertices,
        'triangles': triangles,
        'labels': labels,
        'segments_dict': segments_dict,
        'labels_img': labels_img,
        'original_image': original_image,
        'mesh_results': mesh_results,
    }
    
    # Add visualization of control points comparison
    print("\n=== Visualizing Control Points Comparison ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    

    ax1.imshow(original_image)
    initial_control_coords = vertices[init_control_pts]
    
    # Plot initial control points as black circles
    radius = 0.015 * np.sqrt(original_image.shape[0] ** 2 + original_image.shape[1] ** 2)
    for ij in initial_control_coords:
        circle = Circle((ij[0], ij[1]), radius=radius, fill=False, linewidth=2, color='black')
        ax1.add_patch(circle)
    
    ax1.set_title(f"Initial Control Points: {len(control_point_indices)} vertices")
    ax1.set_aspect('equal')
    
    # Final control points (optimized positions)
    ax2.imshow(original_image)
    final_control_coords = vertices[X]
    
    # Plot final control points as green circles
    radius = 0.015 * np.sqrt(original_image.shape[0] ** 2 + original_image.shape[1] ** 2)
    for ij in final_control_coords:
        circle = Circle((ij[0], ij[1]), radius=radius, fill=False, linewidth=2, color='black')
        ax2.add_patch(circle)
    
    # # Add vertex indices as text labels for debugging
    # for i, (x, y) in enumerate(final_control_coords):
    #     ax2.text(x, y, f'{X[i]}', 
    #             fontsize=8, ha='center', va='bottom', color='white')
    
    ax2.set_title(f"Reconstructed Control Points: {len(X)} vertices")
    ax2.set_aspect('equal')
    
    plt.show()
    
    return result_dict


class Manifold:
    def __init__(self, V, F, vert_mask, image):
        self.V = V
        self.Ind = np.arange(V.shape[0])
        self.F = F
        self.A = adjacency(V, F)
        self.L = igl.cotmatrix(V, F)
        self.N = len(V)
        self.vert_mask = vert_mask
        assert len(vert_mask) == len(V)
        self.last_X = None
        self.last_Lambda = None
        self.image = image

    # X, Y defines the K vertices of Theta and respective values.
    def U(self, Boundary):
        X, Y = Boundary
        X_bar = np.setdiff1d(self.Ind, X)
        L_ii = self.L[X_bar, :][:, X_bar]
        L_ib = self.L[X_bar, :][:, X]
        ch = Y.shape[1] if len(Y.shape) > 1 else 1
        u = np.zeros((self.N, ch))
        for c in range(ch):
            u[X_bar, c] = spsolve(-L_ii, L_ib.dot(Y[:, c]))
            u[X, c] = Y[:, c]
        return u

    def dE_dY(self, g, Boundary, Theta_Indx):
        X, Y = Boundary
        K = len(X)
        ch = Y.shape[1] if len(Y.shape) > 1 else 1

        X_bar = np.setdiff1d(self.Ind, X)
        L_ii = self.L[X_bar, :][:, X_bar]
        L_ib = self.L[X_bar, :][:, X]

        u = np.zeros((self.N, ch))
        for c in range(ch):
            u[X_bar, c] = spsolve(-L_ii, L_ib.dot(Y[:, c]))
            u[X, c] = Y[:, c]

        a = -2 * (g - u) * self.vert_mask # row vector of size N
        assert a.shape == g.shape == u.shape == (self.N, ch)
        a = np.concatenate([a[X_bar], a[X]]).reshape(-1, ch) #  column vector of size N

        sort_X = np.sort(X)
        if self.last_X is None:
            self.last_X = sort_X
            self.last_Lambda = spsolve(-L_ii, L_ib.dot(1))
        else:
            if not np.all(sort_X == self.last_X):
                self.last_X = sort_X
                self.last_Lambda = spsolve(-L_ii, L_ib.dot(1))

        Lambda = self.last_Lambda  # spsolve(-L_ii, L_ib.dot(1))  # - inv(L_ii) @ L_ib # matrix of size (N-K) x K
        assert Lambda.shape == (self.N - K, K)

        B = Lambda  # Lambda @ identity(K) # matrix of size N - K x K
        assert B.shape == (self.N - K, K)
        B = vstack([B, identity(K)]) # matrix of size N x K
        assert B.shape == (self.N, K)

        assert a.shape == (self.N, ch)
        d = np.zeros((K, ch))
        for c in range(ch):
            at = a[:, c].T
            d[:, c] = (at @ B).T # (1 x N) * (N x K) = 1 x K   (row vector)
        assert d.shape == (K, ch)
        return d[Theta_Indx]

    def dE_dX(self, g, Boundary, Theta_Indx):
        u = self.U(Boundary)
        E = np.sum(((g-u) * self.vert_mask) **2)

        def diff(i):
            Boundary_ = Boundary  
            X, Y = Boundary_
            assert i < len(X)
            x_i = X[i]
            ds = []
            for v in self.A[x_i]:
                if v == x_i: continue
                X[i] = v # Push x_i to v
                u = self.U(Boundary_)
                E_ = np.sum(((g-u) * self.vert_mask)**2)
                d = np.linalg.norm(self.V[x_i] - self.V[v])
                ds.append([v, (E_ - E) / d])
                X[i] = x_i # Pop x_i

            min_ds = min(ds, key=lambda x: x[1])
            return min_ds if min_ds[1] < 0 else None

        return [diff(i) for i in Theta_Indx]

    def step(self, g, Boundary, Theta_Indx, alpha):
        dY = self.dE_dY(g, Boundary, Theta_Indx)
        dX = self.dE_dX(g, Boundary, Theta_Indx)  # [None] * len(Theta_Indx) #
        X, Y = Boundary
        ch = Y.shape[1] if len(Y.shape) > 1 else 1
        for i in range(len(Theta_Indx)):
            for c in range(ch):
                new_c = Y[Theta_Indx[i], c] - alpha * dY[i, c]
                Y[Theta_Indx[i], c] = np.clip(new_c, 0., 1.)
            if dX[i] is not None:
                X[Theta_Indx[i]] = dX[i][0]
        return X, Y

    def optimize(self, g, Boundary, Theta_Indx, alpha, eps, deb, last_E=np.inf, max_iter=10):
        X, Y = Boundary
        history_size = 5
        history_e = np.array([last_E])
        iter = 1
        while True:
            X, Y = self.step(g, [X, Y], Theta_Indx, alpha)
            u = self.U([X, Y])
            e = np.sum(((g-u) * self.vert_mask)**2)
            Theta_X = X[Theta_Indx] # Vertex indices of Theta
            Theta_Y = Y[Theta_Indx] # Vertex values of Theta

            min_diff = np.min(np.abs(history_e-e)) # min difference between current and previous EMA_E
            print(f'{iter}. E: {e:.4f}, Min Diff with history: {min_diff:.4f}. History Size: {len(history_e)}.')
            if min_diff < eps: break
            if len(history_e) == history_size: history_e = history_e[1:]
            history_e = np.append(history_e, e)
            iter += 1
            if iter > max_iter:
                break

        return X, Y

    def optimize_with_animate(self, g, Boundary, Theta_Indx, alpha, eps, last_E=np.inf, max_iter=10):
        global X, Y, history_e, iter, finished
        X, Y = Boundary
        history_size = 10
        history_e = np.array([last_E])
        iter = 1
        finished = False

        fig, ax = plt.subplots()
        img = ax.imshow(self.image)
        radius = 0.015 * np.sqrt(self.image.shape[0] ** 2 + self.image.shape[1] ** 2)
        circles = [
            Circle((i, j), radius=radius, fill=False, linewidth=2) for i, j in self.V[X, :2]
        ]
        for circle in circles:
            ax.add_patch(circle)

        ax.set(xlim=[0, self.image.shape[1]], ylim=[self.image.shape[0], 0], xlabel='X', ylabel='Y')
        # ax.set_aspect('equal')

        def update(frame):
            global X, Y, history_e, iter, finished
            if not finished:
                X, Y = self.step(g, [X, Y], Theta_Indx, alpha)
                u = self.U([X, Y])
                e = np.sum(((g - u) * self.vert_mask) ** 2)
                Theta_X = X[Theta_Indx]  # Vertex indices of Theta
                Theta_Y = Y[Theta_Indx]  # Vertex values of Theta

                min_diff = np.min(np.abs(history_e - e))  # min difference between current and previous EMA_E
                print(
                    f'{iter}. E: {e:.4f}, Min Diff with history: {min_diff:.4f}. History Size: {len(history_e)}.\n {boundary2str(Theta_X, Theta_Y, g)}')
                if min_diff < eps or iter > max_iter:
                    finished = True
                if len(history_e) == history_size: history_e = history_e[1:]
                history_e = np.append(history_e, e)
                iter += 1

            radius = 0.015 * np.sqrt(self.image.shape[0] ** 2 + self.image.shape[1] ** 2)
            for i, circle in enumerate(circles):
                circle.center = self.V[X[i], :2]
                circle.radius = radius
            img.set_data(self.image)
            return circles + [img]

        ani = FuncAnimation(fig=fig, func=update, frames=100, interval=30)
        plt.show()


def create_mesh_inset_mask(vertices, triangles, radius):
    """
    Create inset mask for mesh vertices (equivalent to harmonic.py's inset function)
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        triangles: Array of triangle indices (M, 3)
        radius: Inset radius (same as harmonic.py: max_dim // 50)
    
    Returns:
        inset_mask: Boolean array indicating internal vertices
    """
    # 1. Detect boundary vertices using triangle count
    triangle_count = np.zeros(len(vertices))
    for tri in triangles:
        for vertex_idx in tri:
            triangle_count[vertex_idx] += 1
    
    # Boundary vertices have fewer triangles
    boundary_threshold = np.mean(triangle_count) * 0.8
    boundary_vertices = triangle_count < boundary_threshold
    
    # 2. Calculate distance from each vertex to nearest boundary vertex
    boundary_coords = vertices[boundary_vertices]
    if len(boundary_coords) > 0:
        # Use smaller batches to avoid memory issues
        batch_size = 1000
        distances = np.full(len(vertices), np.inf)
        
        for i in range(0, len(vertices), batch_size):
            end_idx = min(i + batch_size, len(vertices))
            batch_vertices = vertices[i:end_idx]
            batch_distances = np.min(cdist(batch_vertices, boundary_coords), axis=1)
            distances[i:end_idx] = batch_distances
    else:
        distances = np.full(len(vertices), np.inf)
    
    # 3. Create inset mask
    inset_mask = distances > radius
    
    # Ensure we have enough vertices for processing
    if np.sum(inset_mask) < 100:
        print(f"Warning: Only {np.sum(inset_mask)} internal vertices found. Using all vertices.")
        inset_mask = np.ones(len(vertices), dtype=bool)
    
    return inset_mask


def filter_triangles_for_vertices(triangles, vertex_indices):
    """
    Filter triangles to only include those with all vertices in vertex_indices
    
    Args:
        triangles: Array of triangle indices (M, 3)
        vertex_indices: Array of valid vertex indices
    
    Returns:
        filtered_triangles: Array of filtered triangle indices
    """
    vertex_indices = np.array(vertex_indices)
    vertex_set = set(vertex_indices)
    filtered_triangles = []
    
    for tri in triangles:
        if all(vertex in vertex_set for vertex in tri):
            # Map to local indices using numpy operations
            local_tri = [np.where(vertex_indices == vertex)[0][0] for vertex in tri]
            filtered_triangles.append(local_tri)
    
    return np.array(filtered_triangles)


def adjacency(v, f):
    A = [set() for _ in range(v.shape[0])]
    for i in range(f.shape[0]):
        assert f[i].shape[0] == 3
        for j in range(3):
            A[f[i, j]].add(f[i, (j+1)%3])
            A[f[i, j]].add(f[i, (j+2)%3])
    return A


def visualize_colored_mesh(vertices, triangles, labels, image_shape):
    """
    Visualize the generated mesh with different colors for each segment.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        triangles: Array of triangle indices (M, 3)
        labels: List of labels for each triangle
        image_shape: (height, width) of the original image
    """
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
    plt.title("2D Mesh from Canny Edge Image")
    plt.tight_layout()
    plt.show()





def main():
    """
    Main function to demonstrate the edge mesh processing with vertex-based gradient reconstruction.
    """
    import sys
    import os
    
    if len(sys.argv) != 3:
        print("Usage: python freeform.py <canny_edge_image_path> <original_image_path>")
        print("Example: python freeform.py edge-1.png org_img/image.png")
        sys.exit(1)
    
    edge_path = sys.argv[1]
    original_image_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(edge_path):
        print(f"Error: Edge image file '{edge_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(original_image_path):
        print(f"Error: Original image file '{original_image_path}' does not exist.")
        sys.exit(1)
    
    # Process the images
    result = process_canny_edge_image(edge_path, original_image_path)
    
    
if __name__ == "__main__":
    main()

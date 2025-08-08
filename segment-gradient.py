import numpy as np
import cv2
from shapely.geometry import MultiPolygon, LineString
from shapely.ops import polygonize
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
from meshpy.triangle import build, MeshInfo

# Import radial gradient functionality from gradient.py
import sys
sys.path.append('.')
sys.path.append('GradientExtraction/Radial')
from gradient import MeshConcentric
from general2 import Scale, Rotate, dot, radius, normalize_rad, E, Orth




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

def assign_labels_to_triangles(triangles, vertices, label_img):
    """
    Assign segment labels to triangles based on their centroids.
    Also build vertex-to-segment mapping for faster segment organization.
    
    Args:
        triangles: Array of triangle indices (M, 3)
        vertices: Array of vertex coordinates (N, 2)
        label_img: Label image where each pixel has a segment label
    
    Returns:
        labels: List of labels for each triangle
        segments_dict: Dictionary mapping segment labels to triangle arrays
        vertex_to_segments: Dictionary mapping vertex indices to segment labels
    """
    centroids = np.mean(vertices[triangles], axis=1)
    h, w = label_img.shape
    labels = np.full(len(triangles), -1, dtype=np.int32)
    vertex_to_segments = defaultdict(set)

    for i, centroid in enumerate(centroids):
        x = int(centroid[0])
        y = int(centroid[1])
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        segment_label = label_img[y, x]
        labels[i] = segment_label
        
        # Add all vertices of this triangle to the segment
        triangle = triangles[i]
        for vertex_idx in triangle.flatten():
            vertex_to_segments[int(vertex_idx)].add(segment_label)

    segments_dict = defaultdict(list)
    for i, label in enumerate(labels):
        segments_dict[label].append(triangles[i])

    return labels.tolist(), segments_dict, vertex_to_segments
    
def sample_colors_at_vertices(vertices, original_image):
    """
    Sample colors at each vertex directly from the original image.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        original_image: Original RGB image (H, W, 3)
    
    Returns:
        vertex_colors: Array of colors for each vertex (N, 3)
    """
    h, w = original_image.shape[:2]
    vertex_colors = np.zeros((len(vertices), 3), dtype=np.float32)
    
    for i, vertex in enumerate(vertices):
        x, y = vertex[0], vertex[1]
        
        # Clamp coordinates to image boundaries
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Sample color directly from image
        color = original_image[int(y), int(x)].astype(np.float32) / 255.0
        vertex_colors[i] = color
    
    return vertex_colors

def fit_solid_fill_vertex_based(vertices, vertex_colors, segment_triangles, vertex_to_segments):
    """
    Fit solid fill color to a segment using vertex colors.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3)
        segment_triangles: List of triangle indices for this segment
        vertex_to_segments: Dictionary mapping vertex indices to segment labels
    
    Returns:
        weighted_color: The fitted solid color for the segment
    """
    # Get all vertices for this segment
    segment_vertices = set()
    for triangle in segment_triangles:
        segment_vertices.update(triangle)
    
    # Collect colors with boundary weighting
    colors = []
    weights = []
    
    for vertex_idx in segment_vertices:
        # Check if vertex is on boundary (shared by multiple segments)
        if len(vertex_to_segments[vertex_idx]) > 1:
            weight = 4.0  # Higher weight for boundary vertices
        else:
            weight = 1.0
        
        colors.append(vertex_colors[vertex_idx])
        weights.append(weight)
    
    # Compute weighted average
    weighted_color = np.average(colors, weights=weights, axis=0)
    return weighted_color

def compute_vertex_gradients(vertices, triangles, vertex_colors, p_scale=None):
    """
    Compute gradients at vertices using mesh-based approach.
    """
    print("Computing vertex gradients")
    
    # Create vertex connectivity graph
    vertex_neighbors = defaultdict(set)
    for triangle in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    vertex_neighbors[triangle[i]].add(triangle[j])
    
    vertex_gradients = np.zeros((len(vertices), 3, 2))  # 3 colors, 2 directions
    
    for i, vertex in enumerate(vertices):
        neighbors = list(vertex_neighbors[i])
        
        if len(neighbors) >= 2:
            neighbor_vertices = vertices[neighbors]
            neighbor_colors = vertex_colors[neighbors]
            
            for color_channel in range(3):
                neighbor_diffs = neighbor_vertices - vertex
                color_diffs = neighbor_colors[:, color_channel] - vertex_colors[i, color_channel]
                
                if len(neighbor_diffs) >= 2:
                    try:
                        gradient = np.linalg.lstsq(neighbor_diffs, color_diffs, rcond=None)[0]
                        vertex_gradients[i, color_channel] = gradient
                    except np.linalg.LinAlgError:
                        vertex_gradients[i, color_channel] = np.array([0.0, 0.0])
                else:
                    vertex_gradients[i, color_channel] = np.array([0.0, 0.0])
        else:
            vertex_gradients[i, :, :] = 0.0
    
    
    for channel in range(3):
        mag = np.linalg.norm(vertex_gradients[:, channel, :], axis=1)
        mag[mag < 1e-8] = 1
        vertex_gradients[:, channel, :] = vertex_gradients[:, channel, :] / mag[:, None]
    
    return vertex_gradients, vertex_neighbors

def compute_vertex_gradient_direction(vertex_positions, vertex_colors, vertex_gradients, segment_vertices):
    """
    Compute dominant gradient direction using vertex gradients (following existing codebase approach).
    
    Args:
        vertex_positions: Array of vertex positions for the segment
        vertex_colors: Array of vertex colors for the segment
        vertex_gradients: Pre-computed gradients at all vertices
        segment_vertices: Indices of vertices in this segment
    
    Returns:
        gradient_direction: Angle in radians of the dominant gradient direction
    """
    if len(segment_vertices) < 2:
        return 0.0
    
    # Get gradients for vertices in this segment
    segment_gradients = vertex_gradients[segment_vertices]
    
    # Build structure tensor from gradients (following existing codebase)
    structure_tensor = np.zeros((2, 2))
    
    for vertex_idx in segment_vertices:
        # Get gradients for this vertex
        vertex_grad = vertex_gradients[vertex_idx]  # Shape: (3, 2)
        
        # Combine gradients from all color channels
        for channel in range(3):
            grad = vertex_grad[channel]  # Shape: (2,)
            if np.any(grad != 0):
                # Accumulate into structure tensor
                structure_tensor += np.outer(grad, grad)
    
    # Check if structure tensor has meaningful values
    if np.allclose(structure_tensor, 0):
        return 0.0
    
    # Find dominant eigenvector
    eigenvals, eigenvecs = np.linalg.eigh(structure_tensor)
    dominant_direction = eigenvecs[:, np.argmax(eigenvals)]
    
    return np.arctan2(dominant_direction[1], dominant_direction[0])

def get_reconstruction_error_profile(profile_points, stops, debug=False):
    """
    Compute reconstruction error for a profile with given stops.
    Following the existing codebase approach.
    
    Args:
        profile_points: Array of (position, color) pairs
        stops: List of stop indices
        debug: Debug flag
    
    Returns:
        error: Reconstruction error
    """
    if len(stops) < 2:
        return float('inf')
    
    total_error = 0.0
    
    for i in range(len(stops) - 1):
        start_idx = stops[i]
        end_idx = stops[i + 1]
        
        if start_idx >= end_idx:
            continue
            
        # Get the slice of profile between these stops
        slice_points = profile_points[start_idx:end_idx + 1]
        
        if len(slice_points) < 2:
            continue
        
        # Align to x-axis (following existing codebase)
        start_pos = slice_points[0, 0]
        slice_aligned = slice_points.copy()
        slice_aligned[:, 0] -= start_pos
        
        # Compute error as sum of absolute deviations from linear
        if len(slice_aligned) > 2:
            # Linear interpolation between start and end
            start_color = slice_aligned[0, 1]
            end_color = slice_aligned[-1, 1]
            total_length = slice_aligned[-1, 0] - slice_aligned[0, 0]
            
            if total_length > 0:
                for j in range(1, len(slice_aligned) - 1):
                    t = (slice_aligned[j, 0] - slice_aligned[0, 0]) / total_length
                    interpolated_color = (1 - t) * start_color + t * end_color
                    error = abs(slice_aligned[j, 1] - interpolated_color)
                    total_error += error
            else:
                # All points at same position
                total_error += np.sum(np.abs(slice_aligned[1:-1, 1] - start_color))
    
    return total_error

def find_max_deviation_index(profile_points, start_idx, end_idx):
    """
    Find the index of maximum deviation from linear interpolation.
    Following the existing codebase approach.
    
    Args:
        profile_points: Array of (position, color) pairs
        start_idx: Start index
        end_idx: End index
    
    Returns:
        max_idx: Index of maximum deviation
    """
    if end_idx <= start_idx:
        return start_idx
    
    slice_points = profile_points[start_idx:end_idx + 1]
    
    if len(slice_points) < 3:
        return start_idx
    
    # Align to x-axis
    start_pos = slice_points[0, 0]
    slice_aligned = slice_points.copy()
    slice_aligned[:, 0] -= start_pos
    
    # Find maximum deviation from linear interpolation
    max_deviation = 0.0
    max_idx = start_idx
    
    start_color = slice_aligned[0, 1]
    end_color = slice_aligned[-1, 1]
    total_length = slice_aligned[-1, 0] - slice_aligned[0, 0]
    
    if total_length > 0:
        for j in range(1, len(slice_aligned) - 1):
            t = (slice_aligned[j, 0] - slice_aligned[0, 0]) / total_length
            interpolated_color = (1 - t) * start_color + t * end_color
            deviation = abs(slice_aligned[j, 1] - interpolated_color)
            
            if deviation > max_deviation:
                max_deviation = deviation
                max_idx = start_idx + j
    
    return max_idx

def get_optimized_stops(profile_points, global_error_threshold=10.0, local_error_threshold=8.0, 
                       distance_threshold=20.0, debug=False):
    """
    Get optimized stops for a profile using iterative refinement.
    Following the existing codebase approach.
    
    Args:
        profile_points: Array of (position, color) pairs
        global_error_threshold: Global error threshold
        local_error_threshold: Local error threshold
        distance_threshold: Minimum distance between stops
        debug: Debug flag
    
    Returns:
        stops: List of optimized stop indices
    """
    if len(profile_points) < 2:
        return [0]
    
    # Initialize with start and end
    stops = [0, len(profile_points) - 1]
    error = get_reconstruction_error_profile(profile_points, stops, debug)
    
    while error > global_error_threshold:
        new_stops = [stops[0]]
        
        for i in range(1, len(stops)):
            start_idx = stops[i - 1]
            end_idx = stops[i]
            
            start_pos = profile_points[start_idx, 0]
            end_pos = profile_points[end_idx, 0]
            
            if end_pos - start_pos < distance_threshold:
                new_stops.append(end_idx)
                continue
            
            # Check local error
            local_error = get_reconstruction_error_profile(profile_points, [start_idx, end_idx], debug)
            
            if local_error <= local_error_threshold:
                new_stops.append(end_idx)
            else:
                # Find point of maximum deviation
                max_dev_idx = find_max_deviation_index(profile_points, start_idx, end_idx)
                
                # Check distance constraints
                if (abs(max_dev_idx - start_idx) >= distance_threshold and 
                    abs(end_idx - max_dev_idx) >= distance_threshold):
                    new_stops.append(max_dev_idx)
                
                new_stops.append(end_idx)
        
        if len(new_stops) == len(stops):
            break  # No new stops added
        
        stops = new_stops
        error = get_reconstruction_error_profile(profile_points, stops, debug)
    
    return stops

def consolidate_stops(stops, distance_threshold):
    """
    Consolidate nearby stops.
    Following the existing codebase approach.
    
    Args:
        stops: List of stop indices
        distance_threshold: Distance threshold for consolidation
    
    Returns:
        consolidated_stops: List of consolidated stop indices
    """
    if not stops:
        return []
    
    consolidated_stops = []
    window = []
    
    for stop in stops:
        if not window or stop - window[-1] <= distance_threshold:
            window.append(stop)
        else:
            # Average the window
            avg_stop = int(round(np.mean(window)))
            consolidated_stops.append(avg_stop)
            window = [stop]
    
    if window:
        avg_stop = int(round(np.mean(window)))
        consolidated_stops.append(avg_stop)
    
    return consolidated_stops

def get_stops_from_profiles(profiles, global_error_threshold=10.0, local_error_threshold=8.0, 
                           distance_threshold=20.0, min_distance=20, debug=False):
    """
    Get optimized stops from multi-channel profiles.
    Following the existing codebase approach.
    
    Args:
        profiles: Array of (position, r, g, b) values
        global_error_threshold: Global error threshold
        local_error_threshold: Local error threshold
        distance_threshold: Distance threshold
        min_distance: Minimum distance for consolidation
        debug: Debug flag
    
    Returns:
        stops: List of optimized stop indices
    """
    all_stops = []
    
    # Process each color channel separately
    for channel in range(1, 4):  # R, G, B channels
        profile = profiles[:, [0, channel]]  # position and color channel
        channel_stops = get_optimized_stops(profile, global_error_threshold, 
                                          local_error_threshold, distance_threshold, debug)
        all_stops.extend(channel_stops)
    
    # Sort and consolidate
    all_stops = sorted(list(set(all_stops)))
    consolidated_stops = consolidate_stops(all_stops, min_distance)
    
    return consolidated_stops

def fit_linear_gradient_vertex_based(vertices, vertex_colors, segment_triangles, vertex_to_segments, vertex_gradients):
    """
    Fit linear gradient to a segment using vertex colors and gradients.
    Following the existing codebase approach with proper stop optimization.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3)
        segment_triangles: List of triangle indices for this segment
        vertex_to_segments: Dictionary mapping vertex indices to segment labels
        vertex_gradients: Pre-computed gradients at all vertices
    
    Returns:
        tuple: (pwl_function, gradient_direction) or (None, None) if fitting fails
    """
    # Get unique vertices for this segment efficiently
    segment_vertices = np.unique(np.concatenate(segment_triangles))
    
    if len(segment_vertices) < 3:
        return None, None
    
    # Get vertex positions and colors for this segment
    vertex_positions = vertices[segment_vertices]
    vertex_colors_segment = vertex_colors[segment_vertices]
    
    # Find gradient direction using vertex gradients (following existing approach)
    gradient_direction = compute_vertex_gradient_direction(
        vertex_positions, vertex_colors_segment, vertex_gradients, segment_vertices
    )
    
    # Vectorized projection onto gradient direction
    cos_theta = np.cos(gradient_direction)
    sin_theta = np.sin(gradient_direction)
    
    # Project all vertices at once
    projected_positions = vertex_positions[:, 0] * cos_theta + vertex_positions[:, 1] * sin_theta
    
    # Create dense color mapping (following existing DensePWL approach)
    lambda2colors = defaultdict(list)
    
    for i, lambda_val in enumerate(projected_positions):
        lambda_int = int(round(lambda_val))
        lambda2colors[lambda_int].append(vertex_colors_segment[i])
    
    if len(lambda2colors) < 2:
        return None, None
    
    # Create profiles matrix (position, r, g, b)
    lambda_positions = sorted(lambda2colors.keys())
    profiles = []
    
    for lambda_pos in lambda_positions:
        colors_at_pos = np.array(lambda2colors[lambda_pos])
        dominant_color = np.mean(colors_at_pos, axis=0)  # Use mean as dominant color
        profiles.append([lambda_pos, dominant_color[0], dominant_color[1], dominant_color[2]])
    
    profiles = np.array(profiles)
    
    # Get optimized stops using the existing codebase approach
    optimized_stops = get_stops_from_profiles(profiles, global_error_threshold=10.0, 
                                            local_error_threshold=8.0, distance_threshold=20.0, 
                                            min_distance=20, debug=False)
    
    if len(optimized_stops) < 2:
        return None, None
    
    # Extract colors at optimized stops
    stop_colors = []
    stop_positions = []
    
    for stop_idx in optimized_stops:
        if 0 <= stop_idx < len(profiles):
            stop_positions.append(profiles[stop_idx, 0])
            stop_colors.append(profiles[stop_idx, 1:4])
    
    # Create piecewise linear function with optimized stops
    pwl_function = {
        'lambda_positions': stop_positions,
        'dominant_colors': stop_colors,
        'start_color': stop_colors[0],
        'end_color': stop_colors[-1],
        'start_pos': stop_positions[0],
        'end_pos': stop_positions[-1],
        'direction': gradient_direction,
        'num_stops': len(stop_positions)
    }
    
    return pwl_function, gradient_direction

def compute_reconstruction_error_vertex_based(vertices, vertex_colors, segment_triangles, fill_function, fill_type):
    """
    Compute reconstruction error for a fill function on a segment.
    Optimized version with vectorized operations.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3)
        segment_triangles: List of triangle indices for this segment
        fill_function: The fitted fill function
        fill_type: 'solid' or 'linear'
    
    Returns:
        error: RMS reconstruction error
    """
    # Get unique vertices efficiently
    segment_vertices = np.unique(np.concatenate(segment_triangles))
    original_colors = vertex_colors[segment_vertices]
    
    if fill_type == 'solid':
        predicted_colors = np.full_like(original_colors, fill_function)
    elif fill_type == 'linear':
        vertex_positions = vertices[segment_vertices]
        pwl_func = fill_function
        
        # Vectorized projection and interpolation
        gradient_direction = pwl_func.get('direction', 0.0)
        cos_theta = np.cos(gradient_direction)
        sin_theta = np.sin(gradient_direction)
        
        lambda_vals = vertex_positions[:, 0] * cos_theta + vertex_positions[:, 1] * sin_theta
        
        # Interpolate using dense color stops
        lambda_positions = pwl_func['lambda_positions']
        dominant_colors = pwl_func['dominant_colors']
        
        predicted_colors = np.zeros_like(original_colors)
        
        for i, lambda_val in enumerate(lambda_vals):
            # Find interpolation interval
            if lambda_val <= lambda_positions[0]:
                predicted_colors[i] = dominant_colors[0]
            elif lambda_val >= lambda_positions[-1]:
                predicted_colors[i] = dominant_colors[-1]
            else:
                # Find the interval to interpolate
                for j in range(len(lambda_positions) - 1):
                    if lambda_positions[j] <= lambda_val <= lambda_positions[j + 1]:
                        # Linear interpolation
                        t = (lambda_val - lambda_positions[j]) / (lambda_positions[j + 1] - lambda_positions[j])
                        predicted_colors[i] = (1 - t) * dominant_colors[j] + t * dominant_colors[j + 1]
                        break
    else:
        predicted_colors = original_colors
    
    # Vectorized error computation
    errors = np.linalg.norm(original_colors - predicted_colors, axis=1)
    return np.sqrt(np.mean(errors ** 2))

def fit_fill_function_vertex_based(vertices, vertex_colors, segments_dict, vertex_to_segments, triangles, image_shape):
    """
    Fit appropriate fill function to each segment using vertex-based approach.
    Now supports solid, linear, and radial gradient fitting.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3)
        segments_dict: Dictionary mapping segment labels to triangle arrays
        vertex_to_segments: Dictionary mapping vertex indices to segment labels
        triangles: Array of all triangle indices
        image_shape: Shape of the original image (height, width)
    
    Returns:
        fill_functions: Dictionary mapping segment labels to fill function data
    """
    # Pre-compute vertex gradients for all vertices
    vertex_gradients, vertex_neighbors = compute_vertex_gradients(vertices, triangles, vertex_colors)
    
    fill_functions = {}
    
    # Pre-compute vertex counts for all segments
    segment_vertex_counts = {}
    for segment_label, segment_triangles in segments_dict.items():
        if segment_label == 0 or segment_label == 1:
            continue
        segment_vertices = set()
        for triangle in segment_triangles:
            segment_vertices.update(triangle)
        segment_vertex_counts[segment_label] = len(segment_vertices)
    
    for segment_label, segment_triangles in segments_dict.items():
        if segment_label == 0 or segment_label == 1:  # Skip background
            continue
            
        vertex_count = segment_vertex_counts[segment_label]
        print(f"\n--- Fitting fill function for Segment {segment_label} ---")
        visualize_segment_mesh(segment_label, vertex_to_segments, vertices, vertex_colors, segments_dict, triangles)
        print(f"Number of vertices: {vertex_count}")
        print(f"Number of triangles: {len(segment_triangles)}")
        
        # Decision based on vertex count
        if vertex_count < 10:  # Small segment threshold
            print("Small segment detected, using solid fill")
            solid_color = fit_solid_fill_vertex_based(
                vertices, vertex_colors, segment_triangles, vertex_to_segments
            )
            fill_functions[segment_label] = {
                'type': 'solid',
                'color': solid_color,
                'error': 0.0  # No error computation for small segments
            }
            print(f"Solid fill color: RGB({solid_color[0]:.3f}, {solid_color[1]:.3f}, {solid_color[2]:.3f})")
            continue
        
        # Check if segment has only one color (solid fill)
        segment_vertices = np.unique(np.concatenate(segment_triangles))
        segment_colors = vertex_colors[segment_vertices]
        color_variance = np.var(segment_colors, axis=0)
        total_variance = np.sum(color_variance)
        
        if total_variance < 0.001:  # Very low color variance indicates solid fill
            solid_color = np.mean(segment_colors, axis=0)
            fill_functions[segment_label] = {
                'type': 'solid',
                'color': solid_color,
                'error': 0.0
            }
            print(f"Solid fill color: RGB({solid_color[0]:.3f}, {solid_color[1]:.3f}, {solid_color[2]:.3f})")
            continue
        
        # Try solid fill
        print("Fitting solid fill...")
        solid_color = fit_solid_fill_vertex_based(
            vertices, vertex_colors, segment_triangles, vertex_to_segments
        )
        solid_error = compute_reconstruction_error_vertex_based(
            vertices, vertex_colors, segment_triangles, solid_color, 'solid'
        )
        print(f"Solid fill error: {solid_error:.4f}")
        print(f"Solid fill color: RGB({solid_color[0]:.3f}, {solid_color[1]:.3f}, {solid_color[2]:.3f})")
        
        # Try linear gradient
        print("Fitting linear gradient...")
        linear_function, direction = fit_linear_gradient_vertex_based(
            vertices, vertex_colors, segment_triangles, vertex_to_segments, vertex_gradients
        )
        
        linear_error = float('inf')
        if linear_function is not None:
            linear_error = compute_reconstruction_error_vertex_based(
                vertices, vertex_colors, segment_triangles, linear_function, 'linear'
            )
            print(f"Linear gradient error: {linear_error:.4f}")
            print(f"Gradient direction: {np.degrees(direction):.1f} degrees")
            print(f"Number of optimized stops: {linear_function['num_stops']}")
            print(f"Start color: RGB({linear_function['start_color'][0]:.3f}, {linear_function['start_color'][1]:.3f}, {linear_function['start_color'][2]:.3f})")
            print(f"End color: RGB({linear_function['end_color'][0]:.3f}, {linear_function['end_color'][1]:.3f}, {linear_function['end_color'][2]:.3f})")
        else:
            print("Linear gradient fitting failed")
        
        # Try radial gradient
        print("Fitting radial gradient...")
        radial_function, radial_error = fit_radial_gradient_vertex_based(
            vertices, vertex_colors, segment_triangles, vertex_gradients, image_shape
        )
        
        if radial_function is not None:
            print(f"Radial gradient error: {radial_error:.4f}")
            print(f"Center: ({radial_function['center'][0]:.1f}, {radial_function['center'][1]:.1f})")
            print(f"Focus: ({radial_function['focus'][0]:.1f}, {radial_function['focus'][1]:.1f})")
            print(f"Scale: {radial_function['scale']:.4f}")
            print(f"Rotation: {np.degrees(radial_function['rotate']):.1f}°")
        else:
            print("Radial gradient fitting failed")
        
        # Choose the fill function with the lowest error
        errors = {
            'solid': solid_error,
            'linear': linear_error,
            'radial': radial_error
        }
        
        best_type = min(errors, key=errors.get)
        best_error = errors[best_type]
        
        print(f"\n--- Error Comparison ---")
        print(f"Solid fill error: {solid_error:.4f}")
        print(f"Linear gradient error: {linear_error:.4f}")
        print(f"Radial gradient error: {radial_error:.4f}")
        print(f"Best fit: {best_type.upper()} (error: {best_error:.4f})")
        
        # Assign the best fitting function
        if best_type == 'solid':
            fill_functions[segment_label] = {
                'type': 'solid',
                'color': solid_color,
                'error': solid_error
            }
            print(f"✓ Chosen: SOLID FILL")
        elif best_type == 'linear':
            fill_functions[segment_label] = {
                'type': 'linear',
                'function': linear_function,
                'error': linear_error
            }
            print(f"✓ Chosen: LINEAR GRADIENT")
        elif best_type == 'radial':
            fill_functions[segment_label] = {
                'type': 'radial',
                'function': radial_function,
                'error': radial_error
            }
            print(f"✓ Chosen: RADIAL GRADIENT")
    
    return fill_functions

def fit_radial_gradient_vertex_based(vertices, vertex_colors, segment_triangles, vertex_gradients, image_shape):
    """
    Fit concentric radial gradient to a segment using vertex colors and gradients.
    Uses MeshConcentric class from gradient2.py.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3)
        segment_triangles: List of triangle indices for this segment
        vertex_gradients: Pre-computed gradients at all vertices
        image_shape: Shape of the original image (height, width)
    
    Returns:
        tuple: (radial_function, error) or (None, float('inf')) if fitting fails
    """
    # Get unique vertices for this segment efficiently
    segment_vertices = np.unique(np.concatenate(segment_triangles))
    
    if len(segment_vertices) < 3:
        return None, float('inf')
    
    # Get vertex positions, colors, and gradients for this segment
    vertex_positions = vertices[segment_vertices]
    vertex_colors_segment = vertex_colors[segment_vertices]
    vertex_gradients_segment = vertex_gradients[segment_vertices]
    
    try:
        # Create MeshConcentric instance for this segment
        mesh_concentric = MeshConcentric(
            vertices=vertex_positions,
            vertex_gradients=vertex_gradients_segment,
            vertex_colors=vertex_colors_segment,
            image_shape=image_shape
        )
        
        # Fit radial gradient parameters
        result = mesh_concentric.fit(mode='Fit')
        
        if result is None:
            return None, float('inf')
        
        # Extract parameters
        fHat = result['fHat']
        oHat = result['oHat']
        scale = result['scale']
        rotate = result['rotate']
        T = result['T']
        
        # Create radial function structure
        radial_function = {
            'type': 'radial',
            'fHat': fHat,
            'oHat': oHat,
            'scale': scale,
            'rotate': rotate,
            'T': T,
            'eHat': np.zeros(2),  # Concentric (no eccentricity)
            'Ro': 0,  # Concentric (no outer radius)
            'direction': np.arctan2(fHat[1] - oHat[1], fHat[0] - oHat[0]),
            'center': oHat,
            'focus': fHat
        }
        
        # Compute reconstruction error
        error = compute_radial_reconstruction_error_vertex_based(
            vertices, vertex_colors, segment_triangles, radial_function
        )
        
        return radial_function, error
        
    except Exception as e:
        print(f"Radial gradient fitting failed: {e}")
        return None, float('inf')

def compute_radial_reconstruction_error_vertex_based(vertices, vertex_colors, segment_triangles, radial_function):
    """
    Compute reconstruction error for a radial gradient function on a segment.
    
    Args:
        vertices: Array of vertex coordinates (N, 2)
        vertex_colors: Array of vertex colors (N, 3)
        segment_triangles: List of triangle indices for this segment
        radial_function: The fitted radial gradient function
    
    Returns:
        error: RMS reconstruction error
    """
    # Get unique vertices efficiently
    segment_vertices = np.unique(np.concatenate(segment_triangles))
    original_colors = vertex_colors[segment_vertices]
    vertex_positions = vertices[segment_vertices]
    
    # Extract radial parameters
    fHat = radial_function['fHat']
    oHat = radial_function['oHat']
    T = radial_function['T']
    Tinv = np.linalg.inv(T)
    
    # Transform vertices to radial space
    Pn = vertex_positions - oHat
    PHat = Pn @ Tinv
    
    # Compute radial distances from focus
    fHatn = fHat - oHat
    distances = np.linalg.norm(PHat - fHatn, axis=1)
    
    # Normalize distances to [0, 1] range for color interpolation
    max_distance = np.max(distances)
    if max_distance > 0:
        normalized_distances = distances / max_distance
    else:
        normalized_distances = np.zeros_like(distances)
    
    # Simple radial color interpolation (center to edge)
    # For concentric gradients, we interpolate from center color to edge color
    center_color = np.mean(original_colors, axis=0)  # Use mean as center color
    edge_color = np.mean(original_colors[normalized_distances > 0.8], axis=0) if np.any(normalized_distances > 0.8) else center_color
    
    # Interpolate colors based on distance
    predicted_colors = np.zeros_like(original_colors)
    for i, dist in enumerate(normalized_distances):
        predicted_colors[i] = (1 - dist) * center_color + dist * edge_color
    
    # Vectorized error computation
    errors = np.linalg.norm(original_colors - predicted_colors, axis=1)
    return np.sqrt(np.mean(errors ** 2))

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
    
    edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edge_img is None:
        print(f"Failed to load edge image '{edge_path}'.")
        return None
    
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"Failed to load original image '{original_image_path}'.")
        return None
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Step 2: Process edge image to create regions
    _, binary_edge = cv2.threshold(edge_img, 127, 255, cv2.THRESH_BINARY)
    
    # Close small gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edge = cv2.morphologyEx(binary_edge, cv2.MORPH_CLOSE, kernel)

    # Invert to get filled regions
    region_mask = cv2.bitwise_not(closed_edge)
    
    # Connected components labeling
    num_labels, labels_img = cv2.connectedComponents(region_mask, connectivity=4)
    
    # Validate that we have meaningful regions
    if num_labels <= 1:
        print("Warning: No meaningful regions found in edge image.")
        return None
    
    # Step 3: Extract boundaries and create mesh
    print("Extraction of segment boundaries")
    boundaries = extract_boundaries_from_connected_components(labels_img, connectivity=4, simplify_tolerance=0.4)
    
    if not boundaries:
        print("Warning: No valid boundaries extracted.")
        return None
    
    # Step 4: Generate triangular mesh
    print("Computing 2d mesh triangulation")
    vertices, triangles = mesh_all_segments(boundaries, labels_img.shape, max_volume=100.0, min_angle=30.0)
    
    if len(vertices) == 0:
        print("Error: No vertices generated. Check boundary extraction.")
        return None
    
    # Step 5: Assign labels to triangles
    print("Assigning labels to mesh")
    labels, segments_dict, vertex_to_segments = assign_labels_to_triangles(triangles, vertices, labels_img)
    
    # Get unique segment labels (excluding background)
    unique_segments = set(segments_dict.keys()) - {0}
    print(f"Number of segments: {len(unique_segments)}")
    print(f"Number of labels: {len(segments_dict)}")
    
    # Step 6: Sample colors at vertices
    vertex_colors = sample_colors_at_vertices(vertices, original_image)
    
    # Step 7: Fit fill functions to each segment
    print("\n" + "="*60)
    print("FITTING FILL FUNCTIONS TO SEGMENTS")
    print("="*60)
    
    fill_functions = fit_fill_function_vertex_based(
        vertices, vertex_colors, segments_dict, vertex_to_segments, triangles, original_image.shape[:2]
    )
    
    # Step 8: Print summary of chosen fill functions
    print("\n" + "="*60)
    print("SUMMARY OF CHOSEN FILL FUNCTIONS")
    print("="*60)
    
    solid_count = 0
    linear_count = 0
    radial_count = 0
    
    for segment_label, fill_func in fill_functions.items():
        if fill_func['type'] == 'solid':
            solid_count += 1
            print(f"Segment {segment_label}: SOLID FILL - RGB({fill_func['color'][0]:.3f}, {fill_func['color'][1]:.3f}, {fill_func['color'][2]:.3f})")
        elif fill_func['type'] == 'linear':
            linear_count += 1
            linear_func = fill_func['function']
            print(f"Segment {segment_label}: LINEAR GRADIENT - Direction: {np.degrees(linear_func['direction']):.1f}°")
            print(f"  Start: RGB({linear_func['start_color'][0]:.3f}, {linear_func['start_color'][1]:.3f}, {linear_func['start_color'][2]:.3f})")
            print(f"  End: RGB({linear_func['end_color'][0]:.3f}, {linear_func['end_color'][1]:.3f}, {linear_func['end_color'][2]:.3f})")
        elif fill_func['type'] == 'radial':
            radial_count += 1
            radial_func = fill_func['function']
            print(f"Segment {segment_label}: RADIAL GRADIENT - Center: ({radial_func['center'][0]:.1f}, {radial_func['center'][1]:.1f})")
            print(f"  Focus: ({radial_func['focus'][0]:.1f}, {radial_func['focus'][1]:.1f})")
            print(f"  Scale: {radial_func['scale']:.4f}, Rotation: {np.degrees(radial_func['rotate']):.1f}°")
    
    print(f"\nTotal segments: {len(fill_functions)}")
    print(f"Solid fills: {solid_count}")
    print(f"Linear gradients: {linear_count}")
    print(f"Radial gradients: {radial_count}")
    
    # Step 9: Visualize segments (optional)
    result_dict = {
        'vertices': vertices,
        'triangles': triangles,
        'labels': labels,
        'segments_dict': segments_dict,
        'labels_img': labels_img,
        'original_image': original_image,
        'edge_img': edge_img,
        'boundaries': boundaries,
        'vertex_colors': vertex_colors,
        'fill_functions': fill_functions,
        'vertex_to_segments': vertex_to_segments,
    }
    
    return result_dict


def visualize_segment_mesh(segment_label, vertex_to_segments, vertices, vertex_colors, segments_dict, triangles):
    """
    Visualize a single segment's mesh triangles.
    
    Args:
        segment_label: Label of the segment to visualize
        vertex_to_segments: Dictionary mapping vertex indices to segment labels
        vertices: Array of all vertex coordinates (N, 2)
        vertex_colors: Array of all vertex colors (N, 3)
        segments_dict: Dictionary mapping segment labels to triangle arrays
        triangles: Array of all triangle indices (M, 3)
    """
    # Get image shape from vertices bounds
    h = int(np.max(vertices[:, 1])) + 1
    w = int(np.max(vertices[:, 0])) + 1
    image_shape = (h, w)
    
    # Check if segment exists
    if segment_label not in segments_dict:
        print(f"Segment {segment_label} not found!")
        return
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Get triangles for this segment
    segment_triangles = segments_dict[segment_label]
    triangle_verts = [vertices[tri] for tri in segment_triangles]
    
    # Create polygon collection for this segment
    collection = PolyCollection(
        triangle_verts,
        facecolors='aqua',  # Single color for single segment
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8
    )
    ax.add_collection(collection)
    
    plt.title(f"Segment {segment_label} - {len(segment_triangles)} triangles", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate the edge mesh processing with vertex-based gradient reconstruction.
    """
    import sys
    import os
    
    if len(sys.argv) != 3:
        print("Usage: python mesh_processor.py <canny_edge_image_path> <original_image_path>")
        print("Example: python mesh_processor.py edge-1.png org_img/image.png")
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

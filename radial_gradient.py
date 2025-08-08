#!/usr/bin/env python3
'''
Reconstruction of radial gradient in mesh space. Adapted from GradientExtraction/Radial/general2.py
'''
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import polygonize
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
from meshpy.triangle import build, MeshInfo

# GradientExtraction/Radial imports for stop extraction and SVG generation
import sys
sys.path.append('GradientExtraction/Radial')
from general2 import Scale, Rotate, dot, radius, normalize_rad, E
from misc import laplacian1d
from StopExtraction import get_peaks, cluster, co_linearity
from hyperparam import VERBOSE
from Draw.svg import create_radial_grad_ecc_T

# Orthogonal matrix for gradient operations
Orth = np.array([[0, -1], [1, 0]])


def rgb2hex(rgb):
    """Convert RGB values to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def create_vertex_radial_gradient_svg(cols_rows, oHat, fHat, outer_center, outer_radius, T, stops, output_path=None):
    """
    Create and save radial gradient SVG locally.
    Adapted from create_radial_grad_ecc_T in Draw/svg.py
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output6.svg')
    
    header = '<svg viewBox="0 0 {}, {}" xmlns="http://www.w3.org/2000/svg">'.format(cols_rows[0], cols_rows[1])
    fo = fHat - oHat
    co = outer_center - oHat
    radial_gradient_begin = '\t<radialGradient id="gradient1" gradientUnits="userSpaceOnUse" \n\t\t' \
                            'cx="{}" cy="{}"  fx="{}" fy="{}" r="{}"\n\t\t' \
                            'gradientTransform="matrix({}, {}, {}, {}, {}, {})">'.format(co[0], co[1],
                                                                                         fo[0], fo[1], outer_radius,
                                                                                         T[0, 0], T[0, 1],
                                                                                         T[1, 0], T[1, 1],
                                                                                         oHat[0], oHat[1])
    stops_str = []
    for c, p in zip(*stops):
        stops_str.append('\t<stop offset="{}%" style="stop-color:{}" />'.format(p*100, rgb2hex(c)))
    
    radial_gradient_end = '\t</radialGradient>'
    rect = '\t<rect x="0" y="0" width="{}" height="{}" fill="url(#gradient1)" />'.format(cols_rows[0], cols_rows[1])
    end = '</svg>'

    svg = '\n'.join([header, radial_gradient_begin, *stops_str, radial_gradient_end, rect, end])
    print("----------------------------------")
    print(svg)
    print("----------------------------------")
    
    # Save to local directory
    with open(output_path, 'w') as file:
        file.write(svg)
    
    print(f"SVG saved to: {output_path}")


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
    
    if p_scale is not None:
        vertex_gradients = vertex_gradients / p_scale
        print(f"Normalized gradients by p_scale: {p_scale}")
    
    for channel in range(3):
        mag = np.linalg.norm(vertex_gradients[:, channel, :], axis=1)
        mag[mag < 1e-8] = 1
        vertex_gradients[:, channel, :] = vertex_gradients[:, channel, :] / mag[:, None]
    
    return vertex_gradients, vertex_neighbors


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
    vertices, triangles = mesh_all_segments(boundaries, labels_img.shape, max_volume=0.5, min_angle=10.0)  # Quality mesh
    print(f"Generated mesh with {len(vertices)} vertices and {len(triangles)} triangles")
    
    if len(vertices) == 0:
        print("Error: No vertices generated. Check boundary extraction.")
        return None
    
    # Step 5: Assign labels to triangles
    print("Assigning segment labels to triangles...")
    labels, segments_dict = assign_labels_to_triangles(triangles, vertices, labels_img)
    print(f"Assigned labels to {len(labels)} triangles across {len(segments_dict)} segments")
    # Step 6: Sample colors at vertices
    print("Sampling colors at vertices...")
    vertex_colors = sample_colors_at_vertices(vertices, original_image)
    print(f"Sampled colors for {len(vertices)} vertices")
    
    # Step 7: Compute gradients at vertices
    print("Computing gradients at vertices")
    p_scale = np.max(original_image.shape[:2])
    vertex_gradients, vertex_neighbors = compute_vertex_gradients(vertices, triangles, vertex_colors, p_scale=p_scale)
    print(f"Computed gradients for {len(vertices)} vertices")
    
    if isinstance(vertex_gradients, tuple):
        vertex_gradients = vertex_gradients[0]
    
    print("Performing complete reconstruction with stop extraction...")
    
    vertex_data = {
        'coordinates': vertices,
        'colors': vertex_colors
    }
    
    complete_result = reconstruct_and_save_svg_mesh(
        vertex_data=vertex_data,
        triangles=triangles,
        mode='Fit',
        original_image_shape=original_image.shape,
        segment_boundary=None
    )
    print("Complete reconstruction pipeline finished")
    

    result_dict = {
        'vertices': vertices,
        'triangles': triangles,
        'labels': labels,
        'segments_dict': segments_dict,
        'vertex_colors': vertex_colors,
        'vertex_gradients': vertex_gradients,
        'vertex_neighbors': vertex_neighbors,
        'complete_result': complete_result,
        'image_shape': original_image.shape[:2],
    }
    
    return result_dict

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







def reconstruct_and_save_svg_mesh(vertex_data, triangles=None, mode='Fit', original_image_shape=None, segment_boundary=None):
    """
    Main reconstruction function for mesh data, following the structure of general2.py.
    This function uses only the MeshEccentric class with fit_T method, similar to general2.py.
    
    Args:
        vertex_data: Dictionary containing 'coordinates' and 'colors' for vertices
        triangles: Array of triangle indices (optional)
        mode: 'Fit' or 'Test'
        original_image_shape: Shape of original image
        segment_boundary: Segment boundary for constraint checking
    
    Returns:
        Dictionary containing reconstruction results
    """
    vertices = vertex_data['coordinates']
    vertex_colors = vertex_data['colors']
    
    if triangles is None:
        triangles = []
    
    # Calculate p_scale for coordinate normalization
    p_scale = np.max(original_image_shape) if original_image_shape is not None else np.max([np.max(vertices[:, 0]), np.max(vertices[:, 1])])
    
    vertex_gradients, vertex_neighbors = compute_vertex_gradients(vertices, triangles, vertex_colors, p_scale=p_scale)
    
    if isinstance(vertex_gradients, tuple):
        vertex_gradients = vertex_gradients[0]
    
    if mode == 'Test':
        mesh_eccentric = MeshEccentric(vertices, vertex_gradients, vertex_colors, original_image_shape)
        result = mesh_eccentric.fit_T(mode)
        return result
    else:
        mesh_eccentric = MeshEccentric(vertices, vertex_gradients, vertex_colors, original_image_shape)
        result = mesh_eccentric.fit_T(mode)
        
        if result is None:
            print("Failed to fit parameters")
            return None
        
        fHat = result['fHat']
        eHat = result['eHat']
        oHat = result['oHat']
        Ro = result['Ro']
        scale = result['scale']
        rotate = result['rotate']
        
        print(f"Focus position: {fHat}")
        print(f"Origin position: {oHat}")
        
        vertex_data_struct = {
            'coordinates': vertices,
            'colors': vertex_colors
        }
        
        mask = np.ones(len(vertices))
        
        cols_rows = np.array([np.max(vertices[:, 0]), np.max(vertices[:, 1])])
        VertexStopExtractor(
            vertex_data=vertex_data_struct, 
            mask=mask, 
            fHat=fHat, 
            eHat=eHat, 
            Ro=Ro, 
            scale=scale, 
            rotate=rotate, 
            cols_rows=cols_rows,
            original_image_shape=original_image_shape, 
            segment_boundary=segment_boundary
        ).build_svg()
        
        return {
            'fHat': fHat,
            'eHat': eHat,
            'oHat': oHat,
            'Ro': Ro,
            'scale': scale,
            'rotate': rotate,
            'result': result
        }


class MeshConcentric:
    """
    Concentric radial gradient class for mesh-based data.
    Adapted from general2.py Concentric class.
    """
    def __init__(self, vertices, vertex_gradients, vertex_colors, image_shape):
        self.vertices = vertices
        self.vertex_gradients = vertex_gradients
        self.vertex_colors = vertex_colors
        self.image_shape = image_shape
        
        N = len(vertices)
        self.P = vertices.astype(np.float32)
        self.Gs = np.concatenate([vertex_gradients[:, i, :] for i in range(3)], axis=0)
        self.mask = np.ones(3 * N, dtype=np.float32)
        self.cols_rows = np.array([image_shape[1], image_shape[0]])
        self.p_scale = np.max(self.cols_rows)  
        
        print(f"MeshConcentric: {N} vertices, {len(self.Gs)} gradients, dimensions: {self.cols_rows}")

    def fit(self, mode):
        """
        Reconstructs generalized concentric gradient parameters.
        Space J represents the observable image space. Space I represents the iso-metric unrotated space. fHat are
        defined in I. Once a CONCENTRIC radial gradient is created in space I, we add affine transform with oHat as the origin.
        This transform the radial gradient from space I to J. We compare the gradient of the transformed gradient (which may
        not be concentric) with the observed image gradient Gs.
        :param mode: 'Fit' or 'Test'
        :return:
        """
        from scipy.optimize import least_squares

        self.P = self.P / self.p_scale

        def fun(x):
            oHat = x[:2]
            scale = x[2]
            theta = x[3]

            T = Scale(np.exp(scale)) @ Rotate(theta)
            Tinv = Rotate(-theta) @ Scale(np.exp(-scale))

            Pn = self.P - oHat
            fHatn = oHat - oHat

            PHat = Pn @ Tinv
            CPHat = fHatn
            GHat1 = (PHat - CPHat) @ Orth @ T

            GHat1_tiled = np.tile(GHat1, [3, 1])
            
            # NORMALIZE theoretical gradients to match observed gradients
            mag = np.linalg.norm(GHat1_tiled, axis=1)
            mag[mag < 1e-8] = 1  # Avoid division by zero
            GHat1_normalized = GHat1_tiled / mag[:, None]

            res = dot(self.Gs, GHat1_normalized)
            # NORMALIZE residual by number of vertices to get average residual
            normalized_res = res / len(self.P)
            print("Residue: res:{:.2f}, oHat:{}, Scale:{:.2f}, Rotate:{:.2f}".
                  format(np.sum(np.abs(normalized_res)), np.round(oHat, 2), np.exp(scale), theta))
            return normalized_res

        def pack(o, s, r): 
            return np.concatenate([o, [s, r]])
        
        if mode == 'Test':
            o = np.ones(2) * 0.5  # Use normalized coordinates
            scale = 0.0
            fun(x=pack(o, scale, 0))
            return None
        else:
            log_scale = 0
            theta = 0
            oHat = np.ones(2) * 0.5  
            # Add bounds for rotation: ±π radians (4th parameter)
            bounds = ([-np.inf, -np.inf, -np.inf, -np.pi], [np.inf, np.inf, np.inf, np.pi])
            residue = least_squares(fun=fun, x0=pack(oHat, log_scale, theta), xtol=1e-10, ftol=1e-10)
            scaled_oHat = residue.x[:2] * self.p_scale
            x = np.concatenate([scaled_oHat, np.zeros(2), residue.x[2:], [np.linalg.norm(self.cols_rows*0.4)]])
            return self.extract_parameters(x)

    def extract_parameters(self, x, dump=True):
        print("X: {}".format(np.round(x,3)))
        fHat = x[:2]
        scale, rotate = np.exp(x[4]), x[5]
        T = Scale(scale) @ Rotate(rotate)
        oHat = fHat
        f = (fHat - oHat) @ T + oHat

        print("--------------------------------------------------------------------------")
        print("Width Height:{}".format(self.cols_rows))
        print("fHat:{}, oHat:{}".format(fHat, oHat))
        print("Scale: {:.2f}, Rotate: {:.2f}deg".format(scale, np.rad2deg(normalize_rad(rotate))))
        print("Transform : matrix({}, {}, {}, {}, {}, {})".format(*np.round(T.flatten(), 4), 0, 0))
        print("--------------------------------------------------------------------------")

        if dump:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1], c=self.vertex_colors, s=1)
            plt.plot(f[0], f[1], 'bo', markersize=10, label='Focus')
            plt.plot(oHat[0], oHat[1], 'ro', markersize=10, label='Origin')
            plt.legend()
            plt.title("Concentric Radial Gradient")
            plt.axis('equal')
            plt.show()
        
        return {
            'img': self.vertex_colors, 
            'mask': self.mask, 
            'cols_rows': self.cols_rows, 
            'fHat': fHat, 
            'eHat': np.zeros(2), 
            'oHat': oHat, 
            'Ro': 0, 
            'T': T, 
            'scale': scale, 
            'rotate': rotate
        }


class MeshEccentric:
    """
    Eccentric radial gradient class for mesh-based data.
    Adapted from general2.py Eccentric class.
    """
    def __init__(self, vertices, vertex_gradients, vertex_colors, image_shape):
        self.vertices = vertices
        self.vertex_gradients = vertex_gradients
        self.vertex_colors = vertex_colors
        self.image_shape = image_shape
        
        # Inline data preparation (equivalent to prepare_mesh_data_for_radial_reconstruction)
        N = len(vertices)
        self.P = vertices.astype(np.float32)  # Vertex coordinates
        self.Gs = np.concatenate([vertex_gradients[:, i, :] for i in range(3)], axis=0)  # Flattened gradients
        self.mask = np.ones(3 * N, dtype=np.float32)  # All vertices valid
        self.cols_rows = np.array([image_shape[1], image_shape[0]])  # width, height
        self.p_scale = np.max(self.cols_rows)
        
        print(f"MeshEccentric: {N} vertices, {len(self.Gs)} gradients, dimensions: {self.cols_rows}")

    def fit(self, mode):
        """
        Reconstructs generalized radial gradient parameters in iso-metric unrotated space.
        We compare the color gradient of the created radial image with the observed image gradient Gs.
        :param mode: 'Fit' or 'Test'
        :return:
        """
        from scipy.optimize import least_squares
        
        self.P = self.P / self.p_scale
        GsT = self.Gs @ Orth

        def center_Hat(PHat, fHat, eHat):
            r = radius(PHat, fHat, eHat)
            return fHat[None, :] + r[:, None] * eHat

        def fun(x):
            fHat, eHat = x[:2], x[2:4]

            PHat = self.P

            cPHat = center_Hat(PHat, fHat, eHat)
            GHat = PHat - cPHat

            GHat = np.tile(GHat, [3, 1])

            res = dot(GsT, GHat)
            print("Residue Sum:{:.2f}, fHat:{} Ecc:{}".format(np.sum(np.abs(res)), np.round(fHat, 2), np.round(eHat,2)))
            return res

        f = np.ones(2) * 0.5  
        e = np.zeros(2)

        def pack(o, e):
            return np.concatenate([o, e])

        if mode == 'Test':
            fun(x=pack(f, e))
            return None
        else:
            residue = least_squares(fun=fun, x0=pack(f, e), xtol=1e-10, ftol=1e-10)
            f = residue.x[:2] * self.p_scale  
            e = residue.x[2:]
            x = np.concatenate([f, e, [0, 0, 0.6]])
            return self.extract_parameters(x)

    def get_outer_radius(self, eHat, frac):
        return np.linalg.norm(self.cols_rows * frac)

    def fit_T(self, mode):
        """
        Reconstructs generalized radial gradient parameters.
        Space J represents the observable image space. Space I represents the iso-metric unrotated space. fHat, eHat are
        defined in I. Once a CONCENTRIC radial gradient is created in space I, we add affine transform with oHat as the origin.
        This transform the radial gradient from space I to J. We compare the gradient of the transformed gradient (which may
        not be concentric) with the observed image gradient Gs.
        :param mode: 'Fit' or 'Test'
        :return:
        """
        from scipy.optimize import least_squares
        
        self.P = self.P / self.p_scale
        frac = 0.6
        
        def fun(x):
            fHat = x[:2]
            eHat = x[2:4]
            scale, rotate = x[4], normalize_rad(x[5])

            T = Scale(np.exp(scale)) @ Rotate(rotate)
            Tinv = Rotate(-rotate) @ Scale(np.exp(-scale))

            outer_radius = self.get_outer_radius(eHat, frac) / self.p_scale
            oHat = fHat + outer_radius * eHat

            Pn = self.P - oHat
            fHatn = fHat - oHat

            PHat = Pn @ Tinv
            RPHat = radius(PHat, fHatn, eHat)
            CPHat = fHatn + RPHat[:, None] * eHat
            GHat1 = (PHat - CPHat) @ Orth @ T
            GHat1 = np.tile(GHat1, [3, 1])

            res = dot(self.Gs, GHat1)
            print("Residue: res:{:.2f}, fHat:{}, oHat:{}, eHat:{}, Radius:{:.4f}, Scale:{:.4f}, Rotate:{:.4f}".
                  format(np.sum(np.abs(res)), np.round(fHat, 2), np.round(oHat, 2), np.round(eHat, 2),
                         outer_radius, np.exp(scale), rotate))
            return res

        def pack(f, e, s, t, frac):
            return np.concatenate([f, e, [s, t]])

        if mode == 'Test':
            fHat = np.array([15.015, 56.224])
            eHat = np.array([0.524, -0.559])
            s = 0.60
            t = np.deg2rad(30)
            res = fun(x=pack(fHat, eHat, s=s, t=t, frac=frac))
            print("Residue: {}".format(np.sum(np.abs(res))))
            return None
        else:
            fHat = np.ones(2) * 0.5  
            eHat = np.zeros(2)
            s, t = 0, 0
            residue = least_squares(fun=fun, x0=pack(fHat, eHat, s=s, t=t, frac=frac), xtol=1e-5, ftol=1e-5)
            
            fHat = residue.x[:2] * self.p_scale  
            eHat = residue.x[2:4]
            scale, rotate = residue.x[4], residue.x[5]
            if len(residue.x) == 7:
                frac = residue.x[6]
            x = np.concatenate([fHat, eHat, [scale, rotate, frac]])
            return self.extract_parameters(x=x)

    def extract_parameters(self, x, dump=True):
        print("X:", np.round(x,3))
        fHat, eHat = x[:2], x[2:4]
        scale, rotate = np.exp(x[4]), x[5]
        Ro = self.get_outer_radius(eHat, x[6])
        T = Scale(scale) @ Rotate(rotate)
        oHat = fHat + eHat * Ro
        f = (fHat - oHat) @ T + oHat

        print("--------------------------------------------------------------------------")
        print("Width Height:{}, Outer radius:{}, fraction:{}".format(self.cols_rows, Ro, x[6]))
        print("fHat:{}, oHat:{}".format(fHat, oHat))
        print("Radius:{:.2f}, Scale: {:.2f}, Rotate: {:.2f}deg".format(Ro, scale, np.rad2deg(normalize_rad(rotate))))
        print("Transform : matrix({}, {}, {}, {}, {}, {})".format(*np.round(T.flatten(), 4), 0, 0))
        print("--------------------------------------------------------------------------")

        if dump:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1], c=self.vertex_colors, s=1)
            plt.plot(f[0], f[1], 'bo', markersize=10, label='Focus')
            plt.plot(oHat[0], oHat[1], 'ro', markersize=10, label='Origin')
            plt.legend()
            plt.title("Eccentric Radial Gradient")
            plt.axis('equal')
            plt.show()
        
        return {
            'img': self.vertex_colors, 
            'mask': self.mask, 
            'cols_rows': self.cols_rows, 
            'fHat': fHat, 
            'eHat': eHat, 
            'oHat': oHat, 
            'Ro': Ro, 
            'T': T, 
            'scale': scale, 
            'rotate': rotate
        }


class VertexStopExtractor:
    """
    Vertex-based implementation of the StopExtractor class.
    """
    def __init__(self, vertex_data, mask, fHat, eHat, scale, rotate, Ro, cols_rows, original_image_shape=None, segment_boundary=None):
        self.vertex_data = vertex_data
        self.vertices = vertex_data['coordinates']
        self.vertex_colors = vertex_data['colors']
        
        self.mask = mask
        self.fHat = fHat
        self.eHat = eHat
        self.scale = scale
        self.rotate = rotate
        self.Ro = Ro
        self.T = Scale(self.scale) @ Rotate(self.rotate)
        self.Tinv = Rotate(-self.rotate) @ Scale(1 / self.scale)
        self.oHat = self.fHat + self.eHat * self.Ro
        self.cols_rows = cols_rows
        self.original_image_shape = original_image_shape
        self.segment_boundary = segment_boundary
        
        self._create_spatial_index()
    
    def is_valid_vertex(self, vertex_idx):
        """
        Check if a vertex is valid (has mask value > 0).
        """
        if self.mask[vertex_idx] < 1:
            return False
        
        return True
    
    def is_point_in_bounds(self, point):
        """
        Check if a point is within image bounds.
        """
        if self.original_image_shape is not None:
            img_height, img_width = self.original_image_shape[:2]
            return 0 <= point[0] < img_width and 0 <= point[1] < img_height
        else:
            return (0 <= point[0] <= self.cols_rows[0] and 
                   0 <= point[1] <= self.cols_rows[1])
    
    def _create_spatial_index(self):
        """
        Create a spatial index (grid) for O(1) vertex lookup.
        This replaces the expensive O(N) nearest neighbor search.
        """
        print("Creating spatial index for vertex lookup...")
        
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        
        vertex_diffs = np.diff(self.vertices, axis=0)
        avg_vertex_distance = np.mean(np.linalg.norm(vertex_diffs, axis=1))
        grid_cell_size = max(avg_vertex_distance * 2, 1.0)
        
        grid_dims = np.ceil((max_coords - min_coords) / grid_cell_size).astype(int)
        grid_dims = np.maximum(grid_dims, [1, 1])
        
        self.grid = {}
        self.grid_cell_size = grid_cell_size
        self.grid_min_coords = min_coords
        
        for i, vertex in enumerate(self.vertices):
            grid_x = int((vertex[0] - min_coords[0]) / grid_cell_size)
            grid_y = int((vertex[1] - min_coords[1]) / grid_cell_size)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in self.grid:
                self.grid[grid_key] = []
            self.grid[grid_key].append(i)
        
        print(f"Spatial index created: {len(self.grid)} grid cells, cell size: {grid_cell_size:.2f}")
    
    def _find_nearest_vertex_fast(self, point, max_distance=15.0):
        """
        Find nearest vertex using spatial index for O(1) lookup.
        """
        grid_x = int((point[0] - self.grid_min_coords[0]) / self.grid_cell_size)
        grid_y = int((point[1] - self.grid_min_coords[1]) / self.grid_cell_size)

        min_distance = float('inf')
        closest_idx = None
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                grid_key = (grid_x + dx, grid_y + dy)
                if grid_key in self.grid:
                    for vertex_idx in self.grid[grid_key]:
                        distance = np.linalg.norm(self.vertices[vertex_idx] - point)
                        if distance < min_distance and distance < max_distance:
                            min_distance = distance
                            closest_idx = vertex_idx
        
        return closest_idx, min_distance
    
    def weighted_circular_avg(self, p, N=50, ax=None):
        """
        Compute weighted circular average at a point.
        Adapted from StopExtractor.weighted_circular_avg().
        """
        def weights(n, frac=None):
            m = n if frac is None else int(np.ceil(n*frac))
            assert m <= n
            if m < 1: return 0
            a = np.arange(m, 0, -1)**2
            if n > m: a = np.concatenate([np.zeros(n-m), a])
            return a / np.sum(a)
        
        def avg_color(ts):
            half_circ = cp + rp * np.stack([np.cos(ts), np.sin(ts)]).T
            half_circ_T = (half_circ - self.oHat) @ self.T + self.oHat
            
            if ax is not None:
                ax.plot(half_circ_T[:, 0], half_circ_T[:, 1], '--', color='aqua', alpha=0.5)
            
            colors = []
            for sample_point in half_circ_T:
                if not self.is_point_in_bounds(sample_point):
                    continue
                
                closest_idx, distance = self._find_nearest_vertex_fast(sample_point, max_distance=15.0)
                
                if closest_idx is not None and self.is_valid_vertex(closest_idx):
                    colors.append(self.vertex_colors[closest_idx])
            
            colors = np.array(colors)
            if len(colors) == 0: return None
            w = weights(len(colors), 0.7)
            return np.sum(w[:, None] * colors, axis=0)
        
        pHat = (p - self.oHat) @ self.Tinv + self.oHat
        rp = radius(pHat[None, :], self.fHat, self.eHat)
        cp = self.fHat + rp * self.eHat
        
        counter_clock_weighted_avg = avg_color(ts=np.arange(0, np.pi+np.pi/N, np.pi/N))
        clock_weighted_avg = avg_color(ts=np.arange(0, -np.pi-np.pi/N, -np.pi/N))
        
        if counter_clock_weighted_avg is None:
            return clock_weighted_avg
        elif clock_weighted_avg is None:
            return counter_clock_weighted_avg
        else:
            return (counter_clock_weighted_avg + clock_weighted_avg) * 0.5
    
    def get_color_profile(self, start, direction, max_length):
        """
        Extract color profile along a direction.
        Adapted from StopExtractor.get_color_profile() with mask-based filtering.
        """
        profile = []
        coords = []
        ax = None
        
        if VERBOSE:
            fig, ax = plt.subplots(tight_layout=True)
            ax.set_aspect('equal')
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], c=self.vertex_colors, s=1)
        
        consecutive_none_count = 0
        max_consecutive_none = 5
        
        for t in range(max_length):
            p = start + t * direction
            
            if not self.is_point_in_bounds(p):
                print(f"Profile stopped at t={t}: outside image bounds")
                break
            
            axis = ax if VERBOSE and t % 25 == 0 else None
            avg_color_p = self.weighted_circular_avg(p, ax=axis)
            
            if avg_color_p is not None:
                profile.append(avg_color_p)
                coords.append(p)
                consecutive_none_count = 0
            else:
                consecutive_none_count += 1
                if consecutive_none_count >= max_consecutive_none:
                    print(f"Profile stopped at t={t}: too many consecutive None values")
                    break
        
        if VERBOSE:
            ax.axis('off')
            plt.title("Color Profile")
            plt.show()
        
        return np.array(profile), np.array(coords)
    
    def stop_extractor(self, start, direction):
        """
        Extract color stops from color profile.
        Adapted from StopExtractor.stop_extractor() with mask-based filtering.
        """
        max_length = int(np.round(np.linalg.norm(self.cols_rows))*1.5)
        print(f"Using mask-based profile length: {max_length}")
        
        profile, coords = self.get_color_profile(start, direction, max_length)
        
        if len(profile) == 0:
            print("Warning: Empty profile generated. Using fallback approach.")
            profile = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
            coords = np.array([start, start + direction])
        
        b = [np.abs(laplacian1d(profile[:, i])) for i in range(3)]
        peaks = [get_peaks(b[i]) for i in range(3)]
        
        peaks = np.sort(list(set(cluster(b[0], peaks[0]) + 
                           cluster(b[1], peaks[1]) + 
                           cluster(b[2], peaks[2]))))
        
        threshold = 0.005 * np.max(b)
        peaks = co_linearity(profile, peaks, thresold=threshold)
        
        return profile[peaks], coords[peaks]
    
    def build_svg(self):
        """
        Build SVG output with extracted gradient parameters.
        Adapted from StopExtractor.build_svg().
        """
        start = (self.fHat - self.oHat) @ self.T + self.oHat
        direction = (self.oHat - start)
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < E:
            direction = np.array([1, 0])
        else:
            direction = direction / direction_norm
        
        profile, coord_indicies = self.stop_extractor(start, direction)
        
        p = coord_indicies[-1]
        pHat = (p - self.oHat) @ self.Tinv + self.oHat
        outer_radius = radius(pHat[None, :], self.fHat, self.eHat).squeeze()
        outer_center = self.fHat + self.eHat * outer_radius
        
        diff = np.linalg.norm(coord_indicies[1:] - coord_indicies[:-1], axis=1)
        diff = np.insert(diff, 0, 0)
        assert len(diff) == len(profile)
        length = np.cumsum(diff)
        length = length / length[-1]
        
        if VERBOSE:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.vertices[:, 0], self.vertices[:, 1], c=self.vertex_colors, s=1)
            
            plt.plot(self.oHat[0], self.oHat[1], "+", markersize=15, label='Origin')
            plt.plot(start[0], start[1], "*", markersize=15, label='Start')
            plt.plot(coord_indicies[:, 0], coord_indicies[:, 1], '--', label='Profile Path')
            plt.plot(p[0], p[1], 'go', markersize=10, label='Outer Point')
            plt.plot(outer_center[0], outer_center[1], 'g*', markersize=15, label='Outer Center')
            plt.legend()
            plt.title("Outer radius: {:.2f}".format(outer_radius))
            plt.axis('equal')
            plt.show()
        
        # Generate SVG
        cols_rows = np.array([np.max(self.vertices[:, 0]), np.max(self.vertices[:, 1])])
        create_vertex_radial_gradient_svg(cols_rows=cols_rows, oHat=self.oHat, fHat=self.fHat, 
                                        outer_center=outer_center, outer_radius=outer_radius, 
                                        T=self.T, stops=(profile, length))







def main():
    """
    Main function to demonstrate the edge mesh processing with vertex-based gradient reconstruction.
    """
    import sys
    import os
    
    if len(sys.argv) != 3:
        print("Usage: python linear.py <canny_edge_image_path> <original_image_path>")
        print("Example: python linear.py edge-1.png org_img/image.png")
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
    
    if result is not None:
        print("\n=== Radial Gradient Reconstruction Results ===")
        
        # Print complete reconstruction results
        if result.get('complete_result') is not None:
            complete_result = result['complete_result']
            print(f"\n=== Complete Reconstruction Results ===")
            print(f"Focus: ({complete_result['fHat'][0]:.1f}, {complete_result['fHat'][1]:.1f})")
            print(f"Origin: ({complete_result['oHat'][0]:.1f}, {complete_result['oHat'][1]:.1f})")
            print(f"Outer Radius: {complete_result['Ro']:.2f}")
            print(f"Scale: {complete_result['scale']:.4f}")
            print(f"Rotation: {np.rad2deg(complete_result['rotate']):.2f}°")
            print(f"SVG file generated: output6.svg")
        else:
            print("Complete reconstruction failed or not performed.")
    else:
        print("Processing failed. Check input images and try again.")
    
if __name__ == "__main__":
    main()




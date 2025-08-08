import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import cv2
from mesh import assign_labels_to_triangles

def get_segment_vertices_and_colors(vertices, segments_dict, original_rgb_image):
    """
    For each segment in segments_dict, extract the unique vertices and their colors from the original image.

    Parameters:
    - vertices: np.array of shape (N, 2 or 3), float coordinates of mesh vertices
    - segments_dict: dict of {segment_label: list of triangles (np.array of indices)}
    - original_rgb_image: np.array (H, W, 3), uint8 RGB image (original colors)
    
    Returns:
    - segment_vertices: dict mapping segment_label -> np.array of unique vertices coordinates for that segment
    - segment_vertex_colors: dict mapping segment_label -> np.array of vertex colors (RGB uint8) for those vertices
    """
    h, w, _ = original_rgb_image.shape
    segment_vertices = {}
    segment_vertex_colors = {}

    for seg_label, tri_list in segments_dict.items():
        tri_array = np.array(tri_list)
        vert_indices = np.unique(tri_array.flatten())

        verts = vertices[vert_indices]

        # Round vertices to nearest pixel coordinates for sampling
        px_coords = np.clip(np.round(verts).astype(int), [[0,0]], [[w-1, h-1]])

        # Sample colors directly from original RGB image
        colors = original_rgb_image[px_coords[:, 1], px_coords[:, 0], :]

        segment_vertices[seg_label] = verts
        segment_vertex_colors[seg_label] = colors

    return segment_vertices, segment_vertex_colors


def find_optimal_k_colors(colors, max_k=8):
    """Find optimal number of clusters for color clustering using elbow method"""
    if len(colors) < 3:
        return 1
    
    max_k = min(max_k, len(colors) - 1)
    if max_k < 2:
        return 1
    
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        if k == 1:
            inertias.append(np.sum((colors - np.mean(colors, axis=0))**2))
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(colors)
            inertias.append(kmeans.inertia_)
    
    # Use elbow method to find optimal k
    try:
        kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = kn.elbow if kn.elbow else 2
    except:
        optimal_k = 2
    
    # Ensure optimal_k is reasonable
    optimal_k = max(1, min(optimal_k, max_k))
    
    return optimal_k


def cluster_segment_colors(segment_vertices, segment_vertex_colors, segments_dict, max_k=8):
    """Perform k-means clustering on vertex colors for each segment using elbow method"""
    clustered_data = {}
    
    for seg_label in segment_vertices.keys():
        vertices = segment_vertices[seg_label]
        colors = segment_vertex_colors[seg_label]
        
        if len(colors) == 0:
            clustered_data[seg_label] = None
            continue
        


        mean_color = np.mean(colors, axis=0)
        if np.all(mean_color < 30):  
            print(f"Segment {seg_label}: Skipping black boundary segment")
            clustered_data[seg_label] = None
            continue
        
        # Find optimal k using elbow method
        optimal_k = find_optimal_k_colors(colors, max_k)
        print(f"Segment {seg_label}: Optimal k = {optimal_k}")
        
        if optimal_k == 1:
            cluster_labels = np.zeros(len(colors), dtype=int)
            cluster_centers = np.mean(colors, axis=0).reshape(1, -1)
        else:
            # Perform k-means clustering on colors with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(colors)
            cluster_centers = kmeans.cluster_centers_
        
        # Calculate mean color for each cluster
        cluster_colors = []
        for i in range(optimal_k):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                mean_color = np.mean(colors[cluster_mask], axis=0)
                cluster_colors.append(tuple(map(int, mean_color)))
            else:
                cluster_colors.append((0, 0, 0))  # Default black
        
        clustered_data[seg_label] = {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_colors': cluster_colors,
            'optimal_k': optimal_k,
            'vertex_indices': np.unique(np.array(segments_dict[seg_label]).flatten())
        }
    
    return clustered_data


def visualize_clustered_mesh(vertices, triangles, segments_dict, clustered_data, image_shape, labels=None, original_rgb=None):
    """Visualize the mesh with clustered colors assigned to segments"""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white', dpi=100)
    h, w = image_shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Calculate dominant color for each segment 
    segment_dominant_colors = {}
    
    for segment_label in segments_dict.keys():
        if segment_label not in clustered_data or clustered_data[segment_label] is None:
            if original_rgb is not None:
                tri_list = segments_dict[segment_label]
                tri_array = np.array(tri_list)
                vert_indices = np.unique(tri_array.flatten())
                verts = vertices[vert_indices]
                px_coords = np.clip(np.round(verts).astype(int), [[0,0]], [[w-1, h-1]])
                colors = original_rgb[px_coords[:, 1], px_coords[:, 0], :]
                mean_color = np.mean(colors, axis=0)
                
                if np.all(mean_color < 30):  
                    segment_dominant_colors[segment_label] = (0, 0, 0)  # Keep black
                    continue
            
            segment_dominant_colors[segment_label] = (128, 128, 128)  # Default gray
            continue
        
        cluster_data = clustered_data[segment_label]
        cluster_colors = cluster_data['cluster_colors']
        cluster_labels = cluster_data['cluster_labels']
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        dominant_cluster = unique_labels[np.argmax(counts)]
        
        dominant_color = cluster_colors[dominant_cluster]
        segment_dominant_colors[segment_label] = dominant_color
    
    if labels is not None:
        labels_array = np.array(labels)
        unique_labels = np.unique(labels_array)
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) > 0:
            triangle_colors = np.ones((len(labels_array), 3)) 
            
            for label in unique_labels:
                if label in segment_dominant_colors:
                    dominant_color = segment_dominant_colors[label]
                    color_rgb = [
                        dominant_color[0] / 255.0,
                        dominant_color[1] / 255.0,
                        dominant_color[2] / 255.0
                    ]
                    triangle_colors[labels_array == label] = color_rgb
                else:
                    triangle_colors[labels_array == label] = [0.5, 0.5, 0.5]
            
            triangle_colors[labels_array == -1] = [0.5, 0.5, 0.5]
            
            from matplotlib.tri import Triangulation
            triangulation = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
            
            triangle_verts = vertices[triangles]
            collection = PolyCollection(
                triangle_verts,
                facecolors=triangle_colors,
                edgecolors='none',
                linewidths=0.0,
                alpha=1.0
            )
            ax.add_collection(collection)
        else:
            print("No valid labeled triangles to display.")
            return
    else:
        print("No labels provided for visualization.")
        return
    
    plt.title("Mesh Visualization with K-means Clustered Colors")
    plt.tight_layout()
    plt.show()



def kmeans_seg(vertex_colors, vertices, triangles, segments_dict, original_rgb, labels=None):
    """Main function to perform k-means clustering on vertex colors segment-wise"""
    print("=== Starting Segment-wise Color K-means Clustering ===")
    
    # Get segment-wise vertices and colors
    print("Extracting segment-wise data")
    segment_vertices, segment_vertex_colors = get_segment_vertices_and_colors(
        vertices, segments_dict, original_rgb
    )
    
    # Perform clustering on each segment
    print("Performing color clustering")
    clustered_data = cluster_segment_colors(segment_vertices, segment_vertex_colors, segments_dict, max_k=20)
    
    # Print clustering results
    for segment_label in segments_dict.keys():
        if segment_label in clustered_data and clustered_data[segment_label] is not None:
            cluster_result = clustered_data[segment_label]
            print(f"Segment {segment_label}: {cluster_result['optimal_k']} clusters")
            print(f"  Cluster colors: {cluster_result['cluster_colors']}")
        else:
            print(f"Segment {segment_label}: No clustering (empty segment)")
    
    # Visualize the results
    print("\n=== Visualizing Clustered Mesh ===")
    # Get image shape from original RGB image
    h, w, _ = original_rgb.shape
    image_shape = (h, w)
    
    visualize_clustered_mesh(vertices, triangles, segments_dict, clustered_data, image_shape, labels, original_rgb)
    

    
    print("=== Color K-means Clustering Complete ===")
    return clustered_data
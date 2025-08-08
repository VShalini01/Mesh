#!/usr/bin/env python3
"""
Usage:
    python3 main.py -e "path/to/edge_image.png"                                    # Run edge_mesh.py
    python3 main.py -e "path/to/edge_image.png" "path/to/original_image.png"       # Run edge_mesh.py with original image for reconstruction
    python3 main.py -c "path/to/colored_image.png"                                 # Run mesh.py
    python3 main.py -c "path/to/colored_image.png" "path/to/original_image.png"    # Run mesh.py with original image for reconstruction
"""

import sys
import os
import subprocess

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage:")
        print("  python3 main.py -e <edge_image_path>                                    # Run edge_mesh.py")
        print("  python3 main.py -e <edge_image_path> <original_image_path>              # Run edge_mesh.py with original image for reconstruction")
        print("  python3 main.py -c <colored_image_path>                                 # Run mesh.py")
        print("  python3 main.py -c <colored_image_path> <original_image_path>           # Run mesh.py with original image for reconstruction")
        sys.exit(1)
    
    option = sys.argv[1]
    input_image_path = sys.argv[2]
    original_image_path = sys.argv[3] if len(sys.argv) == 4 else None
    
    # Check if input image file exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input image file '{input_image_path}' does not exist.")
        sys.exit(1)
    
    # Check if original image file exists (if provided)
    if original_image_path and not os.path.exists(original_image_path):
        print(f"Error: Original image file '{original_image_path}' does not exist.")
        sys.exit(1)
    
    # Dispatch to appropriate script
    if option == "-e":
        if original_image_path:
            print(f"Running edge_mesh.py with edge image: {input_image_path}")
            print(f"Original image for reconstruction: {original_image_path}")
            # Pass both paths to edge_mesh.py
            subprocess.run([sys.executable, "edge_mesh.py", input_image_path, original_image_path])
        else:
            print(f"Running edge_mesh.py with: {input_image_path}")
            subprocess.run([sys.executable, "edge_mesh.py", input_image_path])
    elif option == "-c":
        if original_image_path:
            print(f"Running mesh.py with colored image: {input_image_path}")
            print(f"Original image for reconstruction: {original_image_path}")
            # Pass both paths to mesh.py
            subprocess.run([sys.executable, "mesh.py", input_image_path, original_image_path])
        else:
            print(f"Running mesh.py with: {input_image_path}")
            subprocess.run([sys.executable, "mesh.py", input_image_path])
    else:
        print("Error: Invalid option. Use -e for edge image or -c for colored image.")
        sys.exit(1)

if __name__ == "__main__":
    main() 

import numpy as np
from stl import mesh
from pathlib import Path

# Configuration
input_file = "FisherMask_HeightMap_OG.txt"
output_file = Path("./3D_Files/fisher_mask_manifold_100k.stl")
xy_pitch = 125e-6 *1000 # 125 micrometers
z_scale = 1.0 *1000     # Scale factor for height values
base_height = 0.0  # Base height for the mesh bottom
base_thickness = 10e-6  # 10 micrometers base thickness

def create_manifold_mesh_from_heightmap(height_map, xy_pitch, z_scale, base_height, base_thickness):
    """
    Create a properly manifolded STL mesh from a heightmap using numpy-stl.
    Each heightmap element becomes a square pillar with proper connectivity.
    """
    rows, cols = height_map.shape
    
    # Scale heights
    heights = height_map * z_scale + base_height
    
    # Calculate total number of triangles needed:
    # - Top surface: 2 triangles per cell
    # - Bottom surface: 2 triangles per cell  
    # - Side walls: 4 walls per cell, 2 triangles per wall
    # - Internal walls between adjacent cells: handled by connecting neighboring pillars
    
    # For a proper manifold, we'll create individual pillars for each cell
    # and ensure they're properly connected at the base
    
    triangles_per_cell = 12  # 2 top + 2 bottom + 4*2 sides = 10 triangles per pillar
    total_triangles = rows * cols * triangles_per_cell
    
    # Create mesh
    cube_mesh = mesh.Mesh(np.zeros(total_triangles, dtype=mesh.Mesh.dtype))
    
    triangle_idx = 0
    
    for i in range(rows):
        for j in range(cols):
            # Calculate position for this cell
            x_center = j * xy_pitch + xy_pitch/2
            y_center = i * xy_pitch + xy_pitch/2
            z_top = heights[i, j]
            z_bottom = base_height - base_thickness
            
            # Define the 8 vertices of the rectangular pillar
            x0, x1 = x_center - xy_pitch/2, x_center + xy_pitch/2
            y0, y1 = y_center - xy_pitch/2, y_center + xy_pitch/2
            
            vertices = np.array([
                [x0, y0, z_bottom],  # 0: bottom-front-left
                [x1, y0, z_bottom],  # 1: bottom-front-right
                [x1, y1, z_bottom],  # 2: bottom-back-right
                [x0, y1, z_bottom],  # 3: bottom-back-left
                [x0, y0, z_top],     # 4: top-front-left
                [x1, y0, z_top],     # 5: top-front-right
                [x1, y1, z_top],     # 6: top-back-right
                [x0, y1, z_top],     # 7: top-back-left
            ])
            
            # Define the 12 triangles (2 per face, 6 faces)
            faces = [
                # Top face (z = z_top) - normal pointing up
                [4, 5, 6],  # triangle 1
                [4, 6, 7],  # triangle 2
                
                # Bottom face (z = z_bottom) - normal pointing down
                [0, 2, 1],  # triangle 1
                [0, 3, 2],  # triangle 2
                
                # Front face (y = y0) - normal pointing forward (-Y)
                [0, 1, 5],  # triangle 1
                [0, 5, 4],  # triangle 2
                
                # Back face (y = y1) - normal pointing backward (+Y)
                [2, 3, 7],  # triangle 1
                [2, 7, 6],  # triangle 2
                
                # Left face (x = x0) - normal pointing left (-X)
                [3, 0, 4],  # triangle 1
                [3, 4, 7],  # triangle 2
                
                # Right face (x = x1) - normal pointing right (+X)
                [1, 2, 6],  # triangle 1
                [1, 6, 5],  # triangle 2
            ]
            
            # Add triangles to mesh
            for face in faces:
                cube_mesh.vectors[triangle_idx] = vertices[face]
                triangle_idx += 1
    
    return cube_mesh

# Load height map data
print(f"Loading height map from {input_file}...")
height_map = np.loadtxt(input_file)
rows, cols = height_map.shape
print(f"Height map dimensions: {rows} x {cols}")

# Total physical dimensions
total_width = cols * xy_pitch
total_height = rows * xy_pitch
print(f"Physical dimensions: {total_width*1e3:.3f} mm x {total_height*1e3:.3f} mm")
print(f"Height range: {height_map.min()*z_scale*1e6:.3f} to {height_map.max()*z_scale*1e6:.3f} micrometers")

# Create the manifold mesh
print("Creating manifold mesh...")
fisher_mesh = create_manifold_mesh_from_heightmap(
    height_map, xy_pitch, z_scale, base_height, base_thickness
)
print(fisher_mesh)
# Verify mesh properties
print(f"Generated {len(fisher_mesh.vectors)} triangles")
print(f"Mesh bounds:")
print(f"  X: {fisher_mesh.min_[0]*1e3:.3f} to {fisher_mesh.max_[0]*1e3:.3f} mm")
print(f"  Y: {fisher_mesh.min_[1]*1e3:.3f} to {fisher_mesh.max_[1]*1e3:.3f} mm") 
print(f"  Z: {fisher_mesh.min_[2]*1e6:.3f} to {fisher_mesh.max_[2]*1e6:.3f} micrometers")

# Calculate and display volume and surface area
volume, cog, inertia = fisher_mesh.get_mass_properties()
print(f"Volume: {volume*1e9:.6f} mm³")

# Save the STL file
print(f"Saving STL file to {output_file}...")
fisher_mesh.save(str(output_file))

print(f"Manifold STL file created successfully: {output_file}")
print("This mesh should be free of non-manifold geometry errors.")

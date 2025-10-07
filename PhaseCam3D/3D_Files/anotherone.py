import numpy as np
from stl import mesh
from pathlib import Path

# Configuration
input_file = "FisherMask_HeightMap_OG.txt"
output_file = Path("./3D_Files/fisher_mask_manifold_optimized_upnano_1k_scaled_yx_10k_scaled_z.stl")
xy_pitch = 125e-6 * 1000  # 125 micrometers
z_scale = 1.0 * 1000      # Scale factor for height values
base_height = 0.0         # Base height for the mesh bottom
base_thickness = 10e-6  # 10 micrometers base thickness

def create_optimized_mesh_from_heightmap(height_map, xy_pitch, z_scale, base_height, base_thickness):
    """
    Create an optimized manifold STL mesh from a heightmap.
    Creates a continuous surface with vertical walls only where heights differ.
    """
    rows, cols = height_map.shape
    
    # Scale heights
    heights = height_map * z_scale + base_height
    
    triangles = []
    
    def add_triangle(v1, v2, v3):
        """Add a triangle to the list"""
        triangles.append([v1, v2, v3])
    
    # 1. Create top surface triangles
    print("Creating top surface...")
    for i in range(rows):
        for j in range(cols):
            x_center = j * xy_pitch + xy_pitch/2
            y_center = i * xy_pitch + xy_pitch/2
            z_height = heights[i, j]
            
            x0, x1 = x_center - xy_pitch/2, x_center + xy_pitch/2
            y0, y1 = y_center - xy_pitch/2, y_center + xy_pitch/2
            
            # Top surface triangles
            v00 = [x0, y0, z_height]
            v01 = [x1, y0, z_height]
            v10 = [x0, y1, z_height]
            v11 = [x1, y1, z_height]
            
            add_triangle(v00, v01, v11)
            add_triangle(v00, v11, v10)
    
    # 2. Create vertical walls between adjacent cells with different heights
    print("Creating vertical walls...")
    
    # Vertical walls between horizontally adjacent cells
    for i in range(rows):
        for j in range(cols - 1):
            z_left = heights[i, j]
            z_right = heights[i, j + 1]
            
            # Only create wall if heights are different
            if abs(z_left - z_right) > 1e-10:
                x_edge = (j + 1) * xy_pitch
                y_center = i * xy_pitch + xy_pitch/2
                y0, y1 = y_center - xy_pitch/2, y_center + xy_pitch/2
                
                # Create vertical wall
                v_left0 = [x_edge, y0, z_left]
                v_left1 = [x_edge, y1, z_left]
                v_right0 = [x_edge, y0, z_right]
                v_right1 = [x_edge, y1, z_right]
                
                add_triangle(v_left0, v_right1, v_left1)
                add_triangle(v_left0, v_right0, v_right1)
    
    # Horizontal walls between vertically adjacent cells
    for i in range(rows - 1):
        for j in range(cols):
            z_top = heights[i, j]
            z_bottom = heights[i + 1, j]
            
            # Only create wall if heights are different
            if abs(z_top - z_bottom) > 1e-10:
                x_center = j * xy_pitch + xy_pitch/2
                y_edge = (i + 1) * xy_pitch
                x0, x1 = x_center - xy_pitch/2, x_center + xy_pitch/2
                
                # Create vertical wall
                v_top0 = [x0, y_edge, z_top]
                v_top1 = [x1, y_edge, z_top]
                v_bottom0 = [x0, y_edge, z_bottom]
                v_bottom1 = [x1, y_edge, z_bottom]
                
                add_triangle(v_top0, v_top1, v_bottom1)
                add_triangle(v_top0, v_bottom1, v_bottom0)
    
    # 3. Create perimeter walls
    print("Creating perimeter walls...")
    z_bottom = base_height - base_thickness
    
    # Left wall (x = 0)
    for i in range(rows):
        y_center = i * xy_pitch + xy_pitch/2
        z_height = heights[i, 0]
        
        y0, y1 = y_center - xy_pitch/2, y_center + xy_pitch/2
        
        v_top0 = [0, y0, z_height]
        v_top1 = [0, y1, z_height]
        v_bot0 = [0, y0, z_bottom]
        v_bot1 = [0, y1, z_bottom]
        
        add_triangle(v_top0, v_bot0, v_top1)
        add_triangle(v_top1, v_bot0, v_bot1)
    
    # Right wall (x = total_width)
    x_max = cols * xy_pitch
    for i in range(rows):
        y_center = i * xy_pitch + xy_pitch/2
        z_height = heights[i, -1]
        
        y0, y1 = y_center - xy_pitch/2, y_center + xy_pitch/2
        
        v_top0 = [x_max, y0, z_height]
        v_top1 = [x_max, y1, z_height]
        v_bot0 = [x_max, y0, z_bottom]
        v_bot1 = [x_max, y1, z_bottom]
        
        add_triangle(v_top1, v_bot0, v_top0)
        add_triangle(v_bot1, v_bot0, v_top1)
    
    # Front wall (y = 0)
    for j in range(cols):
        x_center = j * xy_pitch + xy_pitch/2
        z_height = heights[0, j]
        
        x0, x1 = x_center - xy_pitch/2, x_center + xy_pitch/2
        
        v_top0 = [x0, 0, z_height]
        v_top1 = [x1, 0, z_height]
        v_bot0 = [x0, 0, z_bottom]
        v_bot1 = [x1, 0, z_bottom]
        
        add_triangle(v_top1, v_bot0, v_top0)
        add_triangle(v_bot1, v_bot0, v_top1)
    
    # Back wall (y = total_height)
    y_max = rows * xy_pitch
    for j in range(cols):
        x_center = j * xy_pitch + xy_pitch/2
        z_height = heights[-1, j]
        
        x0, x1 = x_center - xy_pitch/2, x_center + xy_pitch/2
        
        v_top0 = [x0, y_max, z_height]
        v_top1 = [x1, y_max, z_height]
        v_bot0 = [x0, y_max, z_bottom]
        v_bot1 = [x1, y_max, z_bottom]
        
        add_triangle(v_top0, v_bot0, v_top1)
        add_triangle(v_top1, v_bot0, v_bot1)
    
    # 4. Create bottom surface
    print("Creating bottom surface...")
    for i in range(rows):
        for j in range(cols):
            x_center = j * xy_pitch + xy_pitch/2
            y_center = i * xy_pitch + xy_pitch/2
            
            x0, x1 = x_center - xy_pitch/2, x_center + xy_pitch/2
            y0, y1 = y_center - xy_pitch/2, y_center + xy_pitch/2
            
            # Bottom surface triangles (reversed winding for downward normal)
            v00 = [x0, y0, z_bottom]
            v01 = [x1, y0, z_bottom]
            v10 = [x0, y1, z_bottom]
            v11 = [x1, y1, z_bottom]
            
            add_triangle(v00, v11, v01)
            add_triangle(v00, v10, v11)
    
    # Convert to numpy array and create mesh
    print(f"Creating mesh with {len(triangles)} triangles...")
    triangle_array = np.array(triangles)
    
    # Create the mesh
    optimized_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    optimized_mesh.vectors = triangle_array
    
    return optimized_mesh

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

# Create the optimized mesh
print("Creating optimized manifold mesh...")
fisher_mesh = create_optimized_mesh_from_heightmap(
    height_map, xy_pitch, z_scale, base_height, base_thickness
)

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

print(f"Optimized manifold STL file created successfully: {output_file}")
print("This mesh eliminates redundant triangles and should be free of non-manifold geometry errors.")

# Compare triangle counts
original_count = rows * cols * 12  # Original pillar approach
optimized_count = len(fisher_mesh.vectors)
print(f"Triangle reduction: {original_count} -> {optimized_count} ({100*(original_count-optimized_count)/original_count:.1f}% reduction)")
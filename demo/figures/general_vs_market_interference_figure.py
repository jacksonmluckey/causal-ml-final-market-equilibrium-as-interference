import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Define positions for nodes
w_positions = [(0, 2), (0, 1), (0, 0)]  # Left column (W nodes)
y_positions = [(3, 2), (3, 1), (3, 0)]  # Right column (Y nodes)
p_position = (1.5, -0.8)  # P node position for marketplace

def draw_node(ax, pos, label, color='white'):
    """Draw a circular node"""
    circle = Circle(pos, 0.25, color=color, ec='blue', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=16, 
            fontweight='normal', zorder=4)

def draw_arrow(ax, start, end, color):
    """Draw a line with a triangle arrow head at the end"""
    # Draw the line
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2.5, zorder=1)
    
    # Calculate arrow head
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Arrow head size
    head_length = 0.15
    head_width = 0.1
    
    # Arrow head tip is slightly before the end point to avoid overlapping with nodes
    tip_x = end[0] - dx_norm * 0.28
    tip_y = end[1] - dy_norm * 0.28
    
    # Calculate perpendicular direction
    perp_x = -dy_norm
    perp_y = dx_norm
    
    # Arrow head triangle points
    p1 = (tip_x, tip_y)  # Tip
    p2 = (tip_x - dx_norm * head_length + perp_x * head_width, 
          tip_y - dy_norm * head_length + perp_y * head_width)
    p3 = (tip_x - dx_norm * head_length - perp_x * head_width, 
          tip_y - dy_norm * head_length - perp_y * head_width)
    
    # Draw filled triangle
    triangle = Polygon([p1, p2, p3], facecolor=color, edgecolor=color, zorder=2)
    ax.add_patch(triangle)

def draw_dotted_line(ax, start, end):
    """Draw a dotted vertical line"""
    ax.plot([start[0], end[0]], [start[1], end[1]], 
            'k:', linewidth=1.5, zorder=2)

# ===== LEFT PLOT: GENERAL INTERFERENCE =====
ax1.set_xlim(-0.8, 3.8)
ax1.set_ylim(-1.5, 2.8)
ax1.axis('off')
ax1.set_aspect('equal')

# Title
ax1.text(1.5, 2.6, 'General', fontsize=20, color='black', ha='center', 
         fontweight='normal', style='italic')

# Draw nodes
w_labels = ['W₁', 'W₂', 'Wₙ']
y_labels = ['Y₁', 'Y₂', 'Yₙ']

for i, (pos, label) in enumerate(zip(w_positions, w_labels)):
    draw_node(ax1, pos, label)

for i, (pos, label) in enumerate(zip(y_positions, y_labels)):
    draw_node(ax1, pos, label)

# Draw blue direct arrows
for i in range(3):
    draw_arrow(ax1, w_positions[i], y_positions[i], '#4169E1')

# Draw dotted lines between nodes
draw_dotted_line(ax1, (w_positions[1][0], w_positions[1][1] - 0.25), 
                 (w_positions[2][0], w_positions[2][1] + 0.25))
draw_dotted_line(ax1, (y_positions[1][0], y_positions[1][1] - 0.25), 
                 (y_positions[2][0], y_positions[2][1] + 0.25))

# Draw orange interference arrows (crossing connections) - STRAIGHT
for i in range(3):
    for j in range(3):
        if i != j:  # Skip direct connections
            draw_arrow(ax1, w_positions[i], y_positions[j], '#FF8C00')

# ===== RIGHT PLOT: MARKETPLACE INTERFERENCE =====
ax2.set_xlim(-0.8, 3.8)
ax2.set_ylim(-1.5, 2.8)
ax2.axis('off')
ax2.set_aspect('equal')

# Title
ax2.text(1.5, 2.6, 'Market Equilibrium', fontsize=20, color='black', ha='center', 
         fontweight='normal', style='italic')

# Draw W and Y nodes
for i, (pos, label) in enumerate(zip(w_positions, w_labels)):
    draw_node(ax2, pos, label)

for i, (pos, label) in enumerate(zip(y_positions, y_labels)):
    draw_node(ax2, pos, label)

# Draw P node
draw_node(ax2, p_position, 'P')

# Draw blue direct arrows
for i in range(3):
    draw_arrow(ax2, w_positions[i], y_positions[i], '#4169E1')

# Draw dotted lines between nodes
draw_dotted_line(ax2, (w_positions[1][0], w_positions[1][1] - 0.25), 
                 (w_positions[2][0], w_positions[2][1] + 0.25))
draw_dotted_line(ax2, (y_positions[1][0], y_positions[1][1] - 0.25), 
                 (y_positions[2][0], y_positions[2][1] + 0.25))

# Draw orange arrows through P - straight lines
for i in range(3):
    draw_arrow(ax2, w_positions[i], p_position, '#FF8C00')

for i in range(3):
    draw_arrow(ax2, p_position, y_positions[i], '#FF8C00')

# Add an overall title
fig.suptitle('General vs Market Interference', fontsize=24, fontweight='bold', y=1.02)

# Adjust subplot spacing
plt.subplots_adjust(top=0.92)

# Create legend handles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#4169E1', linewidth=2.5, label='Direct connection'),
    Line2D([0], [0], color='#FF8C00', linewidth=2.5, label='Interference')
]

# Add legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
           bbox_to_anchor=(0.5, -0.02), fontsize=14, frameon=True)
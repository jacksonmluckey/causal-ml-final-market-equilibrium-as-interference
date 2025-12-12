import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from matplotlib.lines import Line2D
from utils import get_figures_path

# Color constants
DIRECT_COLOR = '#4169E1'  # Blue for direct connections
INTERFERENCE_COLOR = '#FF8C00'  # Orange for interference
NODE_EDGE_COLOR = 'blue'
NODE_FILL_COLOR = 'white'

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Define positions for nodes
w_positions = [(0, 2), (0, 1), (0, 0)]  # Left column (W nodes)
y_positions = [(3, 2), (3, 1), (3, 0)]  # Right column (Y nodes)
p_position = (1.5, -0.8)  # P node position for marketplace

def draw_node(ax, pos, label, color=NODE_FILL_COLOR):
    """Draw a circular node"""
    circle = Circle(pos, 0.25, color=color, ec=NODE_EDGE_COLOR, linewidth=2, zorder=3)
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
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(-1.2, 2.8)
ax1.axis('off')
ax1.set_aspect('equal')

# Title
ax1.text(1.5, 2.3, 'General', fontsize=20, color='black', ha='center', fontweight='normal')

# Draw nodes
w_labels = ['W₁', 'W₂', 'Wₙ']
y_labels = ['Y₁', 'Y₂', 'Yₙ']

for i, (pos, label) in enumerate(zip(w_positions, w_labels)):
    draw_node(ax1, pos, label)

for i, (pos, label) in enumerate(zip(y_positions, y_labels)):
    draw_node(ax1, pos, label)

# Draw blue direct arrows
for i in range(3):
    draw_arrow(ax1, w_positions[i], y_positions[i], DIRECT_COLOR)

# Draw dotted lines between nodes
draw_dotted_line(ax1, (w_positions[1][0], w_positions[1][1] - 0.25), 
                 (w_positions[2][0], w_positions[2][1] + 0.25))
draw_dotted_line(ax1, (y_positions[1][0], y_positions[1][1] - 0.25), 
                 (y_positions[2][0], y_positions[2][1] + 0.25))

# Draw orange interference arrows (crossing connections) - STRAIGHT
for i in range(3):
    for j in range(3):
        if i != j:  # Skip direct connections
            draw_arrow(ax1, w_positions[i], y_positions[j], INTERFERENCE_COLOR)

# ===== RIGHT PLOT: MARKETPLACE INTERFERENCE =====
ax2.set_xlim(-0.5, 3.5)
ax2.set_ylim(-1.2, 2.8)
ax2.axis('off')
ax2.set_aspect('equal')

# Title
ax2.text(1.5, 2.3, 'Market Equilibrium', fontsize=20, color='black', ha='center', fontweight='normal',)

# Draw W and Y nodes
for i, (pos, label) in enumerate(zip(w_positions, w_labels)):
    draw_node(ax2, pos, label)

for i, (pos, label) in enumerate(zip(y_positions, y_labels)):
    draw_node(ax2, pos, label)

# Draw P node
draw_node(ax2, p_position, 'P')

# Draw blue direct arrows
for i in range(3):
    draw_arrow(ax2, w_positions[i], y_positions[i], DIRECT_COLOR)

# Draw dotted lines between nodes
draw_dotted_line(ax2, (w_positions[1][0], w_positions[1][1] - 0.25), 
                 (w_positions[2][0], w_positions[2][1] + 0.25))
draw_dotted_line(ax2, (y_positions[1][0], y_positions[1][1] - 0.25), 
                 (y_positions[2][0], y_positions[2][1] + 0.25))

# Draw orange arrows through P - straight lines
for i in range(3):
    draw_arrow(ax2, w_positions[i], p_position, INTERFERENCE_COLOR)

for i in range(3):
    draw_arrow(ax2, p_position, y_positions[i], INTERFERENCE_COLOR)

# Add an overall title
fig.suptitle('General vs Market Interference', fontsize=24, fontweight='bold', y=0.96)

# Create legend handles
legend_elements = [
    Line2D([0], [0], color=DIRECT_COLOR, linewidth=2.5, label='Direct'),
    Line2D([0], [0], color=INTERFERENCE_COLOR, linewidth=2.5, label='Interference')
]

# Add legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
           bbox_to_anchor=(0.5, 0.01), fontsize=14, frameon=True)

# Adjust subplot spacing to minimize whitespace
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.08)

fig.savefig(get_figures_path("general-vs-market-interference.png"), bbox_inches='tight', pad_inches=0.1)
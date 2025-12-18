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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

# Define positions for 6 nodes per column
# Group 1: nodes 1, 2, 3 (top)
# Gap
# Group 2: nodes n-2, n-1, n (bottom)
w_positions = [
    (0, 4.0),   # W₁
    (0, 3.2),   # W₂
    (0, 2.4),   # W₃
    (0, 1.0),   # Wₙ₋₂
    (0, 0.2),   # Wₙ₋₁
    (0, -0.6),  # Wₙ
]

y_positions = [
    (3, 4.0),   # Y₁
    (3, 3.2),   # Y₂
    (3, 2.4),   # Y₃
    (3, 1.0),   # Yₙ₋₂
    (3, 0.2),   # Yₙ₋₁
    (3, -0.6),  # Yₙ
]

p_position = (1.5, 1.7)  # P node position for marketplace (centered)

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

def draw_bracket(ax, x, y_top, y_bottom, label=''):
    """Draw a bracket to show grouping, facing the nodes"""
    bracket_width = 0.15
    bracket_offset = 0.4
    
    x_bracket = x - bracket_offset
    
    # Draw the bracket lines (like [ facing right toward the nodes)
    ax.plot([x_bracket, x_bracket + bracket_width], [y_top, y_top], 'k-', linewidth=2)
    ax.plot([x_bracket, x_bracket], [y_top, y_bottom], 'k-', linewidth=2)
    ax.plot([x_bracket, x_bracket + bracket_width], [y_bottom, y_bottom], 'k-', linewidth=2)
    
    # Add label if provided
    if label:
        y_mid = (y_top + y_bottom) / 2
        ax.text(x_bracket - 0.2, y_mid, label, ha='right', va='center', 
                fontsize=14, fontweight='normal')

# ===== LEFT PLOT: GENERAL INTERFERENCE =====
ax1.set_xlim(-1.0, 3.5)
ax1.set_ylim(-1.2, 4.8)
ax1.axis('off')
ax1.set_aspect('equal')

# Title
ax1.text(1.5, 4.5, 'General', fontsize=20, color='black', ha='center', fontweight='normal')

# Draw nodes
w_labels = [r'$W_1$', r'$W_2$', r'$W_3$', r'$W_{n-2}$', r'$W_{n-1}$', r'$W_n$']
y_labels = [r'$Y_1$', r'$Y_2$', r'$Y_3$', r'$Y_{n-2}$', r'$Y_{n-1}$', r'$Y_n$']

for i, (pos, label) in enumerate(zip(w_positions, w_labels)):
    draw_node(ax1, pos, label)

for i, (pos, label) in enumerate(zip(y_positions, y_labels)):
    draw_node(ax1, pos, label)

# Draw blue direct arrows (all 6 connections)
for i in range(6):
    draw_arrow(ax1, w_positions[i], y_positions[i], DIRECT_COLOR)

# Draw dotted lines to indicate continuation
# (Removed lines between W₂-W₃ and Y₂-Y₃ as requested)

# Draw dotted lines in the gap between groups
draw_dotted_line(ax1, (w_positions[2][0], w_positions[2][1] - 0.35), 
                 (w_positions[3][0], w_positions[3][1] + 0.35))
draw_dotted_line(ax1, (y_positions[2][0], y_positions[2][1] - 0.35), 
                 (y_positions[3][0], y_positions[3][1] + 0.35))

# (Removed lines between Wₙ₋₂-Wₙ₋₁ and Yₙ₋₂-Yₙ₋₁ as requested)

# Draw orange interference arrows ONLY WITHIN GROUPS
# Group 1: indices 0, 1, 2 (W₁, W₂, W₃ to Y₁, Y₂, Y₃)
for i in range(3):
    for j in range(3):
        if i != j:  # Skip direct connections
            draw_arrow(ax1, w_positions[i], y_positions[j], INTERFERENCE_COLOR)

# Group 2: indices 3, 4, 5 (Wₙ₋₂, Wₙ₋₁, Wₙ to Yₙ₋₂, Yₙ₋₁, Yₙ)
for i in range(3, 6):
    for j in range(3, 6):
        if i != j:  # Skip direct connections
            draw_arrow(ax1, w_positions[i], y_positions[j], INTERFERENCE_COLOR)

# Draw brackets to show groupings
# Top bracket for group 1 (W₁, W₂, W₃)
draw_bracket(ax1, w_positions[0][0], w_positions[0][1] + 0.3, w_positions[2][1] - 0.3, 'Group 1')

# Bottom bracket for group 2 (Wₙ₋₂, Wₙ₋₁, Wₙ)
draw_bracket(ax1, w_positions[3][0], w_positions[3][1] + 0.3, w_positions[5][1] - 0.3, r'Group $\frac{n}{3}$')

# ===== RIGHT PLOT: MARKETPLACE INTERFERENCE =====
ax2.set_xlim(-1.0, 3.5)
ax2.set_ylim(-1.2, 4.8)
ax2.axis('off')
ax2.set_aspect('equal')

# Title
ax2.text(1.5, 4.5, 'Market Equilibrium', fontsize=20, color='black', ha='center', fontweight='normal')

# Draw W and Y nodes
for i, (pos, label) in enumerate(zip(w_positions, w_labels)):
    draw_node(ax2, pos, label)

for i, (pos, label) in enumerate(zip(y_positions, y_labels)):
    draw_node(ax2, pos, label)

# Draw P node
draw_node(ax2, p_position, 'P')

# Draw blue direct arrows
for i in range(6):
    draw_arrow(ax2, w_positions[i], y_positions[i], DIRECT_COLOR)

# Draw dotted lines to indicate continuation
# (Removed lines between W₂-W₃ and Y₂-Y₃ as requested)

# Draw dotted lines in the gap
draw_dotted_line(ax2, (w_positions[2][0], w_positions[2][1] - 0.35), 
                 (w_positions[3][0], w_positions[3][1] + 0.35))
draw_dotted_line(ax2, (y_positions[2][0], y_positions[2][1] - 0.35), 
                 (y_positions[3][0], y_positions[3][1] + 0.35))

# (Removed lines between Wₙ₋₂-Wₙ₋₁ and Yₙ₋₂-Yₙ₋₁ as requested)

# Draw orange arrows through P - all 6 W nodes to P, and P to all 6 Y nodes
for i in range(6):
    draw_arrow(ax2, w_positions[i], p_position, INTERFERENCE_COLOR)

for i in range(6):
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

# Save the figure
fig.savefig(get_figures_path("general-vs-market-interference.png"), bbox_inches='tight', pad_inches=0.1)
plt.close()
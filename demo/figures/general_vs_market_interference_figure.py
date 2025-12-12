import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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

def draw_arrow(ax, start, end, color, curved=False, curvature=0.3):
    """Draw an arrow between two points"""
    if curved:
        # Add some curvature for crossing arrows
        style = f"arc3,rad={curvature}"
    else:
        style = "arc3,rad=0"
    
    arrow = FancyArrowPatch(start, end, 
                           arrowstyle='->', 
                           color=color, 
                           linewidth=2.5,
                           connectionstyle=style,
                           zorder=1,
                           mutation_scale=20)
    ax.add_patch(arrow)

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
ax1.text(1.5, 2.6, 'General', fontsize=20, color='blue', ha='center', 
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
# From each W to other Ys
for i in range(3):
    for j in range(3):
        if i != j:  # Skip direct connections
            draw_arrow(ax1, w_positions[i], y_positions[j], '#FF8C00', curved=False)

# ===== RIGHT PLOT: MARKETPLACE INTERFERENCE =====
ax2.set_xlim(-0.8, 3.8)
ax2.set_ylim(-1.5, 2.8)
ax2.axis('off')
ax2.set_aspect('equal')

# Title
ax2.text(1.5, 2.6, 'Market Equilibrium', fontsize=20, color='blue', ha='center', 
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
# From W nodes to P
for i in range(3):
    draw_arrow(ax2, w_positions[i], p_position, '#FF8C00', curved=False)

# From P to Y nodes
for i in range(3):
    draw_arrow(ax2, p_position, y_positions[i], '#FF8C00', curved=False)
    
# For routing the interference arrows/lines to the outside of the figure:
#
## Draw orange arrows through P - routed around the outside
## We'll use custom paths with intermediate points to route around the edges
#
## Create curved paths that go around the outside
#from matplotlib.path import Path
#import matplotlib.patches as mpatches
#
## For arrows from W to P - route around the left side
#for i, w_pos in enumerate(w_positions):
#    # Create a path that curves outward (to the left)
#    mid_x = w_pos[0] - 0.8  # Go further left
#    mid_y = (w_pos[1] + p_position[1]) / 2
#    
#    path = mpatches.FancyBboxPatch((0, 0), 0, 0, visible=False)
#    arrow = FancyArrowPatch(w_pos, p_position,
#                           arrowstyle='->',
#                           color='#FF8C00',
#                           linewidth=2.5,
#                           connectionstyle=f"arc3,rad=0.8",
#                           zorder=1,
#                           mutation_scale=20)
#    ax2.add_patch(arrow)
#
## For arrows from P to Y - route around the right side  
#for i, y_pos in enumerate(y_positions):
#    arrow = FancyArrowPatch(p_position, y_pos,
#                           arrowstyle='->',
#                           color='#FF8C00',
#                           linewidth=2.5,
#                           connectionstyle=f"arc3,rad=0.8",
#                           zorder=1,
#                           mutation_scale=20)
#    ax2.add_patch(arrow)

plt.tight_layout()
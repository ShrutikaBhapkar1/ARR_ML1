import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt

# Define grid boundaries and step sizes


x_start, x_end, x_step = 0, 20, 0.5
y_start, y_end, y_step = 0, 10, 0.5
z_start, z_end, z_step = 0, 10, 1
"""
x_start, x_end, x_step = 0, 20, 1
y_start, y_end, y_step = 0, 10, 1
z_start, z_end, z_step = 0, 10, 1
"""


blocked_cells1 = [(3,3,3),(8,8,8)]




# Define blocked cells
x_dim = int((x_end - x_start) / x_step) + 1
y_dim = int((y_end - y_start) / y_step) + 1
z_dim = int((z_end - z_start) / z_step) + 1


# Split blocked cells into subcells
import random

blocked_subcells1 = []
for cell in blocked_cells1:
    x, y, z = cell
    for dx in [0, 0.5]:
        for dy in [0, 0.5]:
            for dz in [0, 0.5]:
                subcell = [x + dx, y + dy, z + dz]
                blocked_subcells1.append(subcell)

# Randomly select 4 subcells as blocked cells 
blocked_subcells = random.sample(blocked_subcells1, k=8)

print(blocked_subcells)
#blocked_subcells = blocked_subcells
#blocked_subcells = [[3.5, 3.5, 3.5]]


blocked_subcells =[[3.5, 3.5, 3.5], [8, 8, 8], [8, 8, 8.5], [3, 3, 3.5], [8, 8.5, 8], [3, 3, 3], [8.5, 8.5, 8.5], [3, 3.5, 3]]

# Define start and goal cells
start_cell = (0, 0, 0)
goal_cell = (10, 10, 10)

# Compute grid dimensions
x_dim = int((x_end - x_start) / x_step) + 1
y_dim = int((y_end - y_start) / y_step) + 1
z_dim = int((z_end - z_start) / z_step) + 1

# Create grid
grid = np.zeros((x_dim, y_dim, z_dim))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot start cell in yellow
ax.scatter(start_cell[0], start_cell[1], start_cell[2], c='yellow', marker='o', label='Start Cell')

# Plot goal cell in green
ax.scatter(goal_cell[0], goal_cell[1], goal_cell[2], c='green', marker='o', label='Goal Cell')

# Plot blocked subcells in red
x_blocked = []
y_blocked = []
z_blocked = []
for subcell in blocked_subcells:
    x_blocked.append(subcell[0])
    y_blocked.append(subcell[1])
    z_blocked.append(subcell[2])
ax.scatter(x_blocked, y_blocked, z_blocked, c='red', marker='s', label='Blocked Subcells')

# Set plot limits
ax.set_xlim(x_start, x_end)
ax.set_ylim(y_start, y_end)
ax.set_zlim(z_start, z_end)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Grid')

# Add legend
ax.legend()

# Show the plot
plt.show()

# Mark blocked subcells as 1
for subcell in blocked_subcells:
    x, y, z = subcell
    grid[int(x / x_step), int(y / y_step), int(z / z_step)] = 1


def calculate_heuristic(cell, goal):
    # Calculate the Manhattan distance heuristic
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]) + abs(cell[2] - goal[2])


def find_path(start, goal):
    open_set = [(0, start)]
    closed_set = set()
    g_scores = {start: 0}
    parents = {start: None}

    while open_set:
        current_cost, current_cell = heappop(open_set)

        if current_cell == goal:
            path = []
            while current_cell is not None:
                path.append(current_cell)
                current_cell = parents[current_cell]
            path.reverse()
            return path

        closed_set.add(current_cell)

        neighbors = get_neighbors(current_cell)

        for neighbor in neighbors:
            neighbor_cost = g_scores[current_cell] + 1
            if neighbor not in g_scores or neighbor_cost < g_scores[neighbor]:
                g_scores[neighbor] = neighbor_cost
                f_score = neighbor_cost + calculate_heuristic(neighbor, goal)
                heappush(open_set, (f_score, neighbor))
                parents[neighbor] = current_cell

    return []  # No path found


def get_neighbors(cell):
    x, y, z = cell
    neighbors = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                
               for ddx in [0, 0.5]:
                    for ddy in [0, 0.5]:
                        for ddz in [0, 0.5]:
                            new_x = x + dx + ddx
                            new_y = y + dy + ddy
                            new_z = z + dz + ddz
                            print(new_x,new_y,new_z)
                            if (0 <= new_x < x_dim and 0 <= new_y < y_dim and 0 <= new_z < z_dim and [new_x, new_y, new_z] not in blocked_subcells):
                                neighbors.append((new_x, new_y, new_z))
    return neighbors






# Find the shortest path
path = find_path(start_cell, goal_cell)

if path:
    print("Shortest path found:")
    for cell in path:
        print(cell)
else:
    print("No path found.")

print(blocked_subcells)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get grid indices for plotting
x_indices, y_indices, z_indices = np.meshgrid(
    np.arange(x_start, x_end + x_step, x_step),
    np.arange(y_start, y_end + y_step, y_step),
    np.arange(z_start, z_end + z_step, z_step),
    indexing='ij'
)

# Plot start cell in yellow
ax.scatter(start_cell[0], start_cell[1], start_cell[2], c='yellow', marker='o', label='Start Cell')

# Plot goal cell in green
ax.scatter(goal_cell[0], goal_cell[1], goal_cell[2], c='green', marker='o', label='Goal Cell')

# Plot blocked subcells in red
x_blocked = []
y_blocked = []
z_blocked = []
for subcell in blocked_subcells:
    x_blocked.append(subcell[0])
    y_blocked.append(subcell[1])
    z_blocked.append(subcell[2])
ax.scatter(x_blocked, y_blocked, z_blocked, c='red', marker='s', label='Blocked Subcells')

# Plot the path in cyan
if path:
    path_x = [cell[0] for cell in path]
    path_y = [cell[1] for cell in path]
    path_z = [cell[2] for cell in path]
    ax.plot(path_x, path_y, path_z, c='cyan', label='Shortest Path')

# Set plot limits
ax.set_xlim(x_start, x_end)
ax.set_ylim(y_start, y_end)
ax.set_zlim(z_start, z_end)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Grid')

# Add legend
ax.legend()
plt.show()


# Extract x, y, and z coordinates from the path
x_path = [cell[0] * x_step + x_start for cell in path]
y_path = [cell[1] * y_step + y_start for cell in path]
z_path = [cell[2] * z_step + z_start for cell in path]

# Fit a polynomial curve to the path points
order = 3  # Set the desired order of the polynomial curve
coefficients = np.polyfit(x_path, y_path, order)

# Generate the equation of the curve
curve_equation = np.poly1d(coefficients)

# Print the equation
print("Equation of the path curve:")
print(curve_equation)

points_to_check = [[3.5, 3.5, 3.5], [8, 8, 8], [8, 8, 8.5], [3, 3, 3.5], [8, 8.5, 8], [3, 3, 3], [8.5, 8.5, 8.5], [3, 3.5, 3]]

for point in points_to_check:
    x_coordinate = point[0]
    y_coordinate = point[1]
    result = curve_equation(x_coordinate)
    
    if result == y_coordinate:
        print(f"The point {point} satisfies the curve equation.")
    else:
        print(f"The point {point} does not satisfy the curve equation.")




import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt

# Define grid boundaries and step sizes
x_start, x_end, x_step = 0, 20, 1
y_start, y_end, y_step = 0, 10, 1
z_start, z_end, z_step = 0, 10, 1

x_cells = int((x_end - x_start) / x_step) + 1
y_cells = int((y_end - y_start) / y_step) + 1
z_cells = int((z_end - z_start) / z_step) + 1
total_cells = x_cells * y_cells * z_cells

# 15% blocked cell 

num_selected_cells = int(0.1 * total_cells)

selected_indices = np.random.choice(total_cells, num_selected_cells, replace=False)

selected_cells = []
for index in selected_indices:
    x_index = index // (y_cells * z_cells)
    y_index = (index % (y_cells * z_cells)) // z_cells
    z_index = index % z_cells
    x_coord = x_start + x_index * x_step
    y_coord = y_start + y_index * y_step
    z_coord = z_start + z_index * z_step
    selected_cells.append([x_coord, y_coord, z_coord])

print(selected_cells)

blocked_cells = selected_cells

# Define blocked cells
#blocked_cells = [[3, 3, 3], [4, 5, 6], [3, 6, 7], [4, 4, 4], [5, 5, 5], [1, 2, 3]]

# Define start and goal cells
start_cell = (0, 0, 0)
goal_cell = (8, 9, 10)

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

# Plot blocked cells in red
x_blocked = []
y_blocked = []
z_blocked = []
for cell in blocked_cells:
    x_blocked.append(cell[0])
    y_blocked.append(cell[1])
    z_blocked.append(cell[2])
ax.scatter(x_blocked, y_blocked, z_blocked, c='red', marker='s', label='Blocked Cells')

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

# Mark blocked cells as 1
for cell in blocked_cells:
    x, y, z = cell
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
                new_x = x + dx
                new_y = y + dy
                new_z = z + dz
                if (0 <= new_x < x_dim and 0 <= new_y < y_dim and 0 <= new_z < z_dim and
                        grid[new_x, new_y, new_z] != 1 and [new_x, new_y, new_z] not in blocked_cells):
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

# Plot blocked cells in red
x_blocked = []
y_blocked = []
z_blocked = []
for cell in blocked_cells:
    x_blocked.append(cell[0])
    y_blocked.append(cell[1])
    z_blocked.append(cell[2])
ax.scatter(x_blocked, y_blocked, z_blocked, c='red', marker='s', label='Blocked Cells')

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



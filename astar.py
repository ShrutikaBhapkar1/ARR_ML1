
class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.g = float('inf')  
        self.h = 0  
        self.f = float('inf')  
        self.parent = None  

    def __lt__(self, other):
        return self.f < other.f


def heuristic(node, goal):
    return abs(node.row - goal[0]) + abs(node.col - goal[1])


def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    closed_set = set()

    
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    
    start_node.g = 0
    start_node.h = heuristic(start_node, goal)
    start_node.f = start_node.g + start_node.h

    
    open_set.append(start_node)

    while open_set:
        
        current_node = min(open_set, key=lambda x: x.f)

        
        if current_node.row == goal_node.row and current_node.col == goal_node.col:
            path = []
            while current_node:
                path.append((current_node.row, current_node.col))
                current_node = current_node.parent
            return path[::-1] 

       
        open_set.remove(current_node)

       
        closed_set.add((current_node.row, current_node.col))

       
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_row = current_node.row + dr
            neighbor_col = current_node.col + dc

           
            if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
               
                if grid[neighbor_row][neighbor_col] == '#':
                    continue

                neighbor = Node(neighbor_row, neighbor_col)

                
                if (neighbor.row, neighbor.col) in closed_set:
                    continue

                
                tentative_g = current_node.g + 1

                
                if tentative_g < neighbor.g or neighbor not in open_set:
                    neighbor.g = tentative_g
                    neighbor.h = heuristic(neighbor, goal)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node

                    
                    if neighbor not in open_set:
                        open_set.append(neighbor)

   
    return None


def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))


grid = [
    ['#', '0', '#', '0', '0', '0', '0', '0'],
    ['0', '0', '0', '0', '#', '0', '#', '0'],
    ['#', '0', '#', '0', '0', '0', '#', '0'],
    ['#', '0', '#', '#', '0', '#', '#', '0'],
    ['0', '0', '0', '0', '0', '0', '0', '0'],
    ['0', '#', '0', '0', '0', '#', '0', '0'],
    ['0', '0', '0', '#', '#', '0', '0', '0'],
]



print("Grid:")
print_grid(grid)


start_row = int(input("Enter the start row: "))
start_col = int(input("Enter the start column: "))
start = (start_row, start_col)

goal_row = int(input("Enter the goal row: "))
goal_col = int(input("Enter the goal column: "))
goal = (goal_row, goal_col)


path = astar(grid, start, goal)


def is_diagonal(node1, node2):
    return abs(node1[0] - node2[0]) == 1 and abs(node1[1] - node2[1]) == 1


if path:
    print("Path found:")
    print("Original Path:")
    for node in path:
        print(node)
else:
    print("No path found.")




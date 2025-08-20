import heapq


# A* algorithm with turn left, turn right, and move forward actions
def astar(is_freespace, heuristic, start, goal):
    # Directions for turning (left, right, forward)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W (clockwise)

    # Define actions: (action_name, change in direction, change in position)
    actions = [
        (0, -1, 0),  # Turn left (change direction counter-clockwise)
        (1, 1, 0),  # Turn right (change direction clockwise)
        (2, 0, 1),  # Move forward (no direction change, just move)
    ]

    # Initialize open list (priority queue) and closed list (set)
    open_list = []
    closed_list = set()

    # G (cost from start) and F (total cost) dictionaries
    g_cost = {start: 0}
    f_cost = {start: heuristic(start[:2], goal)}

    # Parent mapping for path reconstruction (position, direction, action)
    parent = {start: None}
    action_taken = {start: None}

    # Add start point to open list with priority based on F
    heapq.heappush(open_list, (f_cost[start], start))

    while open_list:
        # Pop the node with the lowest F score from open list
        _, current = heapq.heappop(open_list)

        # If we reached the goal, reconstruct the path
        if current[:2] == goal:
            path = []
            actions_path = []
            while current:
                path.append(current[:2])
                actions_path.append(action_taken[current])
                current = parent[current]
            return actions_path[::-1], path[
                ::-1
            ]  # Return actions and positions in order

        # Add current node to closed list
        closed_list.add(current)

        # Explore neighbors (actions)
        for action, direction_change, move in actions:
            # Calculate new direction after turning
            new_direction = (current[2] + direction_change) % 4

            # If the action is to move forward, update the position
            if action == 2:
                new_x = current[0] + directions[new_direction][0]
                new_y = current[1] + directions[new_direction][1]
                new_pos = (new_x, new_y)
            else:
                new_pos = (current[0], current[1])

            neighbor = (new_pos[0], new_pos[1], new_direction)

            # Skip if the neighbor is out of bounds or not free
            if not is_freespace(neighbor[:2]) or neighbor in closed_list:
                continue

            # Calculate G, F, and update parent if better path is found
            tentative_g = g_cost[current] + (
                1 if action == 2 else 0.5
            )  # Assume turning costs less
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + heuristic(neighbor[:2], goal)
                parent[neighbor] = current
                action_taken[neighbor] = action
                heapq.heappush(open_list, (f_cost[neighbor], neighbor))

    return [], []  # No path found

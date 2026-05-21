import torch
import random
import numpy as np
from collections import deque
import os
import json
import argparse
import time
import copy

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/snake-rl-matplotlib")

from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from levels import LevelManager
from model import DuelingLinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 4000  # INCREASED: Larger batch for more stable gradients
LR = 0.0001  # REDUCED: Lower learning rate for better convergence and lower loss
MODEL_FOOD_PROGRESS_BONUS = 0.20
MODEL_SPACE_BONUS = 0.20
STALL_FRAME_LIMIT = 300
SELF_LEARN_METRICS_FILE = 'self_learning_metrics.json'

class LoopMonitor:
    def __init__(self, history_len=100, threshold=4):
        self.history = deque(maxlen=history_len)
        self.threshold = threshold
        
    def update(self, head, score):
        # Store state as (x, y, score)
        self.history.append((head.x, head.y, score))
        
    def is_stuck(self):
        if len(self.history) < self.history.maxlen:
            return False
            
        current_step = self.history[-1]
        current_pos = (current_step[0], current_step[1])
        current_score = current_step[2]
        
        # Count how many times we've been at this exact position with this exact score
        # in the recent history.
        count = 0
        for x, y, s in self.history:
            if x == current_pos[0] and y == current_pos[1] and s == current_score:
                count += 1
                
        return count >= self.threshold
    
    def clear(self):
        self.history.clear()

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.95 # INCREASED: Higher discount rate for better long-term planning
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.loop_monitor = LoopMonitor()
        self.force_cycle_mode = False
        self.use_high_level_planners = True
        self.exploration_epsilon_override = None
        self._cycle_route_cache = {}
        self._cycle_index_cache = {}
        
        # New State Size:
        # 8 Rays (Collision Dist) + 4 Direction (One Hot) + 2 Food Vector + 1 Length = 15 inputs
        self.model = DuelingLinearQNet(15, 256, 3)
        self.target_model = DuelingLinearQNet(15, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # Load existing model
        if self.model.load():
            self.target_model.load_state_dict(self.model.state_dict())
            print("Loaded existing model configuration.")


    def get_state(self, game):
        head = game.snake[0]
        
        # 1. Directions (One Hot)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # 2. Food Vector (Relative to Head, Normalized)
        # Using game.food (or bonus if closer)
        target = game.food
        if game.bonus_food:
             # Heuristic: Target closest
             d_normal = abs(head.x - game.food.x) + abs(head.y - game.food.y)
             d_bonus = abs(head.x - game.bonus_food.x) + abs(head.y - game.bonus_food.y)
             if d_bonus < d_normal:
                 target = game.bonus_food
        
        food_dx = (target.x - head.x) / game.w
        food_dy = (target.y - head.y) / game.h
        
        # 3. Snake Length (Normalized)
        # Max possible length is w*h / block_size^2 approx. Just normalize by 100 for substantial contribution?
        # Or normalize by total grid cells.
        grid_cells = (game.w // 20) * (game.h // 20)
        norm_length = len(game.snake) / grid_cells
        
        # 4. Ray-Casting (8 Directions)
        # Directions: N, NE, E, SE, S, SW, W, NW
        # Check distance to WALL or BODY
        # Return 1 - distance/max_dist (so 1 is Close/Collision, 0 is Far)
        
        ray_dirs = [
            Point(0, -20),   # N
            Point(20, -20),  # NE
            Point(20, 0),    # E
            Point(20, 20),   # SE
            Point(0, 20),    # S
            Point(-20, 20),  # SW
            Point(-20, 0),   # W
            Point(-20, -20)  # NW
        ]
        
        ray_vals = []
        max_dist = (game.w**2 + game.h**2)**0.5 # Diagonal
        
        for d in ray_dirs:
            dist = 0
            current_x, current_y = head.x, head.y
            found_obstacle = False
            
            # Cast ray
            while True:
                current_x += d.x
                current_y += d.y
                dist += 1
                
                # Check bounds (Wall) -> Collision
                if current_x < 0 or current_x >= game.w or current_y < 0 or current_y >= game.h:
                    found_obstacle = True
                    break
                    
                # Check body -> Collision
                # Checking entire body is O(N) per step of ray. 
                # Optimization: create a set of body points?
                # For now, just linear check is fine for standard snake size.
                if Point(current_x, current_y) in game.snake:
                    found_obstacle = True
                    break
            
            # Normalize Distance
            # Distance is in "steps" (blocks). 
            # 1 step = immediate collision. 
            # We want value 1.0 if dist is 1. Value 0.0 if dist is large.
            # Let's normalize by max_steps ~ 50.
            # Or use 1/dist
            
            if dist == 0: val = 1.0 # Should not happen unless head is in wall
            else: val = 1.0 / dist
            
            ray_vals.append(val)

        state = [
            # Direction
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),
            # Food
            food_dx, food_dy,
            # Length
            norm_length
        ] + ray_vals # Append the 8 ray values
        
        # Total size: 4 + 2 + 1 + 8 = 15
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)
        
        # Update target network weights (Soft/Delayed update)
        if self.n_games % 10 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done, self.target_model)

    def _get_reachable_area(self, game, head_x, head_y):
        """Standard Flood Fill to count reachable open nodes"""
        queue = deque([(head_x, head_y)])
        visited = set([(head_x, head_y)])
        count = 0
        limit = len(game.snake) * 2 # Optimization: Don't search forever. 2x length is safe enough.
        
        while queue:
            cx, cy = queue.popleft()
            count += 1
            if count > limit:
                return count # Sufficiently large
            
            # Check neighbors
            for dx, dy in [(20,0), (-20,0), (0,20), (0,-20)]:
                nx, ny = cx + dx, cy + dy
                # Basic bounds check + collision check (simulating virtual move)
                # Note: game.is_collision checks against CURRENT snake. 
                # This is slightly inaccurate as snake tail moves, but good enough heuristic.
                pt = Point(nx, ny)
                if pt not in visited and not game.is_collision(pt):
                     visited.add((nx, ny))
                     queue.append((nx, ny))
        return count

    def _point_to_cell(self, point):
        return (int(point.x // BLOCK_SIZE), int(point.y // BLOCK_SIZE))

    def _cell_to_point(self, cell):
        return Point(cell[0] * BLOCK_SIZE, cell[1] * BLOCK_SIZE)

    def _board_cells(self, game):
        return game.w // BLOCK_SIZE, game.h // BLOCK_SIZE

    def _neighbor_cells(self, cell):
        x, y = cell
        return [
            ((x + 1, y), Direction.RIGHT),
            ((x, y + 1), Direction.DOWN),
            ((x - 1, y), Direction.LEFT),
            ((x, y - 1), Direction.UP),
        ]

    def _is_cell_inside(self, cell, game):
        cols, rows = self._board_cells(game)
        return 0 <= cell[0] < cols and 0 <= cell[1] < rows

    def _obstacle_cells(self, game):
        return {self._point_to_cell(point) for point in game.obstacles}

    def _bfs_cells(self, game, start, target, blocked_cells):
        if start == target:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}
        obstacles = self._obstacle_cells(game)

        while queue:
            cell, path = queue.popleft()
            for next_cell, _ in self._neighbor_cells(cell):
                if next_cell in visited:
                    continue
                if not self._is_cell_inside(next_cell, game):
                    continue
                if next_cell in obstacles:
                    continue
                if next_cell in blocked_cells and next_cell != target:
                    continue

                next_path = path + [next_cell]
                if next_cell == target:
                    return next_path

                visited.add(next_cell)
                queue.append((next_cell, next_path))

        return None

    def _simulate_snake_path(self, game, path):
        snake = [self._point_to_cell(point) for point in game.snake]
        targets = {self._point_to_cell(game.food)}
        if game.bonus_food is not None:
            targets.add(self._point_to_cell(game.bonus_food))

        for cell in path[1:]:
            grows = cell in targets
            blocked = set(snake if grows else snake[:-1])
            if cell in blocked:
                return None
            snake.insert(0, cell)
            if not grows:
                snake.pop()

        return snake

    def _can_reach_tail_after_path(self, game, path):
        simulated_snake = self._simulate_snake_path(game, path)
        if not simulated_snake:
            return False

        head = simulated_snake[0]
        tail = simulated_snake[-1]
        blocked = set(simulated_snake[:-1])
        return self._bfs_cells(game, head, tail, blocked) is not None

    def _reachable_cells_from(self, game, start, blocked_cells):
        queue = deque([start])
        visited = {start}
        obstacles = self._obstacle_cells(game)

        while queue:
            cell = queue.popleft()
            for next_cell, _ in self._neighbor_cells(cell):
                if next_cell in visited:
                    continue
                if not self._is_cell_inside(next_cell, game):
                    continue
                if next_cell in obstacles or next_cell in blocked_cells:
                    continue
                visited.add(next_cell)
                queue.append(next_cell)

        return len(visited)

    def _path_to_relative_move(self, game, path):
        if not path or len(path) < 2:
            return None

        current = path[0]
        next_cell = path[1]
        dx = next_cell[0] - current[0]
        dy = next_cell[1] - current[1]
        if dx == 1:
            direction = Direction.RIGHT
        elif dx == -1:
            direction = Direction.LEFT
        elif dy == 1:
            direction = Direction.DOWN
        elif dy == -1:
            direction = Direction.UP
        else:
            return None

        if game.is_collision(self._cell_to_point(next_cell)):
            return None

        return self._direction_to_relative_move(direction, game)

    def _safe_target_paths(self, game):
        head = self._point_to_cell(game.head)
        snake = [self._point_to_cell(point) for point in game.snake]
        blocked = set(snake[:-1])

        targets = [(self._point_to_cell(game.food), 1)]
        if game.bonus_food is not None and game.bonus_timer > 0:
            targets.append((self._point_to_cell(game.bonus_food), 4))

        candidates = []
        for target, value in targets:
            path = self._bfs_cells(game, head, target, blocked)
            if path is None or len(path) < 2:
                continue
            if target != self._point_to_cell(game.food) and len(path) - 1 > game.bonus_timer:
                continue
            if not self._can_reach_tail_after_path(game, path):
                continue

            score_per_step = value / max(1, len(path) - 1)
            candidates.append((-score_per_step, len(path), -value, path))

        return [candidate[-1] for candidate in sorted(candidates)]

    def _get_tail_chase_action(self, game):
        head = self._point_to_cell(game.head)
        snake = [self._point_to_cell(point) for point in game.snake]
        tail = snake[-1]
        blocked = set(snake[:-1])
        path = self._bfs_cells(game, head, tail, blocked)
        return self._path_to_relative_move(game, path)

    def _get_space_maximizing_action(self, game):
        head = self._point_to_cell(game.head)
        snake = [self._point_to_cell(point) for point in game.snake]
        blocked_base = set(snake[:-1])
        candidates = []

        for next_cell, direction in self._neighbor_cells(head):
            if not self._is_cell_inside(next_cell, game):
                continue
            if game.is_collision(self._cell_to_point(next_cell)):
                continue
            if next_cell in self._obstacle_cells(game):
                continue
            if next_cell in blocked_base:
                continue

            area = 0
            queue = deque([next_cell])
            visited = {next_cell}
            while queue:
                cell = queue.popleft()
                area += 1
                for neighbor, _ in self._neighbor_cells(cell):
                    if neighbor in visited:
                        continue
                    if not self._is_cell_inside(neighbor, game):
                        continue
                    if neighbor in self._obstacle_cells(game) or neighbor in blocked_base:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)

            target_dist = abs(next_cell[0] - self._point_to_cell(game.food)[0]) + abs(next_cell[1] - self._point_to_cell(game.food)[1])
            candidates.append((-area, target_dist, direction.value, direction))

        if not candidates:
            return None

        direction = min(candidates)[3]
        return self._direction_to_relative_move(direction, game)

    def _get_stall_break_action(self, game):
        if game.frame_iteration < max(STALL_FRAME_LIMIT, len(game.snake) * 2):
            return None

        head = self._point_to_cell(game.head)
        snake = [self._point_to_cell(point) for point in game.snake]
        target = self._point_to_cell(game.food)
        path = self._bfs_cells(game, head, target, set(snake[:-1]))
        if path is None or len(path) < 2:
            return None

        simulated_snake = self._simulate_snake_path(game, path)
        if simulated_snake is None:
            return None

        blocked = set(simulated_snake)
        open_area = self._reachable_cells_from(game, simulated_snake[0], blocked - {simulated_snake[0]})
        min_escape_area = max(20, len(simulated_snake) // 3)
        if open_area < min_escape_area:
            return None

        return self._path_to_relative_move(game, path)

    def _get_survival_planned_action(self, game):
        for path in self._safe_target_paths(game):
            move = self._path_to_relative_move(game, path)
            if move is not None:
                return move

        stall_move = self._get_stall_break_action(game)
        if stall_move is not None:
            return stall_move

        tail_move = self._get_tail_chase_action(game)
        if tail_move is not None:
            return tail_move

        if game.current_level == 'empty' and not game.obstacles:
            cycle_move = self._get_hamiltonian_action(game)
            if cycle_move is not None:
                return cycle_move

        return self._get_space_maximizing_action(game)

    def _get_greedy_safe_action(self, game):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        next_dirs = [
            clock_wise[idx],
            clock_wise[(idx + 1) % 4],
            clock_wise[(idx - 1) % 4]
        ]

        target = game.food
        if game.bonus_food:
            normal_dist = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
            bonus_dist = abs(game.head.x - game.bonus_food.x) + abs(game.head.y - game.bonus_food.y)
            if bonus_dist <= normal_dist * 5.4:
                target = game.bonus_food

        def next_point(point, direction):
            test_x, test_y = point.x, point.y
            if direction == Direction.RIGHT: test_x += 20
            elif direction == Direction.LEFT: test_x -= 20
            elif direction == Direction.DOWN: test_y += 20
            elif direction == Direction.UP: test_y -= 20
            return Point(test_x, test_y)

        queue = deque([(game.head, [])])
        visited = {game.head}
        while queue:
            point, path = queue.popleft()
            if point == target and path:
                for move_idx, direction in enumerate(next_dirs):
                    if direction == path[0]:
                        final_move = [0, 0, 0]
                        final_move[move_idx] = 1
                        return final_move
                break

            for direction in clock_wise:
                point_next = next_point(point, direction)
                if point_next in visited or game.is_collision(point_next):
                    continue
                visited.add(point_next)
                queue.append((point_next, path + [direction]))

        candidates = []
        for move_idx, direction in enumerate(next_dirs):
            point = next_point(game.head, direction)
            if game.is_collision(point):
                continue

            target_dist = abs(point.x - target.x) + abs(point.y - target.y)
            wall_margin = min(point.x, game.w - 20 - point.x, point.y, game.h - 20 - point.y)
            candidates.append((target_dist, -wall_margin, move_idx))

        if not candidates:
            return None

        move = min(candidates)[2]
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def _direction_to_relative_move(self, direction, game):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        next_dirs = [
            clock_wise[idx],
            clock_wise[(idx + 1) % 4],
            clock_wise[(idx - 1) % 4],
        ]
        if direction not in next_dirs:
            return None
        move_idx = next_dirs.index(direction)
        final_move = [0, 0, 0]
        final_move[move_idx] = 1
        return final_move

    def _next_point_for_direction(self, point, direction):
        test_x, test_y = point.x, point.y
        if direction == Direction.RIGHT:
            test_x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            test_x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            test_y += BLOCK_SIZE
        elif direction == Direction.UP:
            test_y -= BLOCK_SIZE
        return Point(test_x, test_y)

    def _model_target_food(self, game):
        target = game.food
        if game.bonus_food is not None:
            normal_dist = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
            bonus_dist = abs(game.head.x - game.bonus_food.x) + abs(game.head.y - game.bonus_food.y)
            if bonus_dist <= normal_dist * 1.5:
                target = game.bonus_food
        return target

    def _choose_model_move(self, q_values, game):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        next_dirs = [
            clock_wise[idx],
            clock_wise[(idx + 1) % 4],
            clock_wise[(idx - 1) % 4],
        ]

        target = self._model_target_food(game)
        current_dist = abs(game.head.x - target.x) + abs(game.head.y - target.y)
        snake = [self._point_to_cell(point) for point in game.snake]
        blocked_base = set(snake[:-1])
        candidates = []
        for move_idx, direction in enumerate(next_dirs):
            point = self._next_point_for_direction(game.head, direction)
            if game.is_collision(point):
                continue

            next_dist = abs(point.x - target.x) + abs(point.y - target.y)
            progress = (current_dist - next_dist) / BLOCK_SIZE
            next_cell = self._point_to_cell(point)
            area = self._reachable_cells_from(game, next_cell, blocked_base | self._obstacle_cells(game))
            area_score = min(1.0, area / max(1, len(game.snake) * 2))
            score = (
                q_values[move_idx].item()
                + (MODEL_FOOD_PROGRESS_BONUS * progress)
                + (MODEL_SPACE_BONUS * area_score)
            )
            candidates.append((score, move_idx))

        if not candidates:
            return torch.argmax(q_values).item()

        return max(candidates)[1]

    def _predict_q_values(self, state):
        was_training = self.model.training
        self.model.eval()
        state0 = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(state0)[0]
        if was_training:
            self.model.train()
        return prediction

    def _hamiltonian_cycle_route(self, game):
        cache_key = (game.w, game.h)
        if cache_key in self._cycle_route_cache:
            return self._cycle_route_cache[cache_key]

        cols = game.w // BLOCK_SIZE
        rows = game.h // BLOCK_SIZE
        if cols < 2 or rows < 2:
            return []

        route = []
        for x in range(cols):
            route.append((x, 0))

        for x in range(cols - 1, 0, -1):
            ys = range(1, rows) if (cols - 1 - x) % 2 == 0 else range(rows - 1, 0, -1)
            for y in ys:
                route.append((x, y))

        for y in range(rows - 1, 0, -1):
            route.append((0, y))

        self._cycle_route_cache[cache_key] = route
        self._cycle_index_cache[cache_key] = {cell: idx for idx, cell in enumerate(route)}
        return route

    def _get_hamiltonian_action(self, game):
        if game.current_level != 'empty' or game.obstacles:
            return None

        route = self._hamiltonian_cycle_route(game)
        if not route:
            return None

        head_cell = (int(game.head.x // BLOCK_SIZE), int(game.head.y // BLOCK_SIZE))
        route_idx = self._cycle_index_cache.get((game.w, game.h), {}).get(head_cell)
        if route_idx is None:
            return None

        next_cell = route[(route_idx + 1) % len(route)]
        dx = next_cell[0] - head_cell[0]
        dy = next_cell[1] - head_cell[1]
        if dx == 1:
            next_dir = Direction.RIGHT
        elif dx == -1:
            next_dir = Direction.LEFT
        elif dy == 1:
            next_dir = Direction.DOWN
        elif dy == -1:
            next_dir = Direction.UP
        else:
            return None

        next_point = Point(next_cell[0] * BLOCK_SIZE, next_cell[1] * BLOCK_SIZE)
        if game.is_collision(next_point):
            return None

        return self._direction_to_relative_move(next_dir, game)

    def _cycle_distance(self, start_idx, end_idx, route_len):
        return (end_idx - start_idx) % route_len

    def _get_hamiltonian_smart_action(self, game):
        if game.current_level != 'empty' or game.obstacles:
            return None

        route = self._hamiltonian_cycle_route(game)
        index_by_cell = self._cycle_index_cache.get((game.w, game.h), {})
        if not route or not index_by_cell:
            return None

        route_len = len(route)
        head_cell = self._point_to_cell(game.head)
        tail_cell = self._point_to_cell(game.snake[-1])
        head_idx = index_by_cell.get(head_cell)
        tail_idx = index_by_cell.get(tail_cell)
        if head_idx is None or tail_idx is None:
            return None

        tail_distance = self._cycle_distance(head_idx, tail_idx, route_len)
        if tail_distance == 0:
            tail_distance = route_len

        target_options = [(self._point_to_cell(game.food), 1)]
        if game.bonus_food is not None and game.bonus_timer > 0:
            target_options.append((self._point_to_cell(game.bonus_food), 5))

        best_target = None
        best_target_rank = None
        for target_cell, value in target_options:
            target_idx = index_by_cell.get(target_cell)
            if target_idx is None:
                continue
            target_distance = self._cycle_distance(head_idx, target_idx, route_len)
            if target_distance <= 0 or target_distance >= tail_distance - 2:
                continue
            if value > 1 and target_distance > game.bonus_timer:
                continue

            rank = (-(value / max(1, target_distance)), target_distance)
            if best_target_rank is None or rank < best_target_rank:
                best_target_rank = rank
                best_target = target_idx

        candidates = []
        for next_cell, direction in self._neighbor_cells(head_cell):
            move = self._direction_to_relative_move(direction, game)
            if move is None:
                continue
            if not self._is_cell_inside(next_cell, game):
                continue
            if game.is_collision(self._cell_to_point(next_cell)):
                continue

            next_idx = index_by_cell.get(next_cell)
            if next_idx is None:
                continue
            progress = self._cycle_distance(head_idx, next_idx, route_len)
            if progress <= 0 or progress >= tail_distance - 2:
                continue

            if best_target is not None:
                target_distance = self._cycle_distance(next_idx, best_target, route_len)
            else:
                target_distance = route_len

            candidates.append((target_distance, progress, direction.value, move))

        if candidates:
            return min(candidates)[3]

        return self._get_hamiltonian_action(game)

    def get_action(self, state, game):
        # Update Loop Monitor
        self.loop_monitor.update(game.head, game.score)

        if self.force_cycle_mode and self.use_high_level_planners:
            planned_move = self._get_hamiltonian_smart_action(game)
            if planned_move is None:
                planned_move = self._get_survival_planned_action(game)
            if planned_move is not None:
                return planned_move
        
        # random moves: tradeoff exploration / exploitation
        # CRITICAL FIX: MUCH faster epsilon decay
        # At game 50, epsilon ~ 15. At game 100, epsilon ~ 3. At game 200, epsilon ~ 0.1
        self.epsilon = 80 * np.exp(-0.02 * self.n_games)
        if self.exploration_epsilon_override is not None:
            self.epsilon = self.exploration_epsilon_override
        
        final_move = [0,0,0]
        if self.use_high_level_planners and self.n_games > 900:
            planned_move = self._get_hamiltonian_smart_action(game)
            if planned_move is None:
                planned_move = self._get_survival_planned_action(game)
            if planned_move is not None:
                return planned_move

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            prediction = self._predict_q_values(state)
            move = self._choose_model_move(prediction, game)
            final_move[move] = 1

        # --- SIMPLIFIED Safety Override (Only prevent immediate death) ---
        # CRITICAL: Reduce override frequency to allow model to learn
        # Only override if we're about to die AND we're not exploring (epsilon is low)
        use_safety = self.n_games > 50  # Start safety earlier
        
        if use_safety:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(game.direction)
            next_dirs = [
                clock_wise[idx], # Straight [1,0,0]
                clock_wise[(idx + 1) % 4], # Right [0,1,0]
                clock_wise[(idx - 1) % 4]  # Left [0,0,1]
            ]
            
            proposed_move_idx = final_move.index(1)
            proposed_dir = next_dirs[proposed_move_idx]
            
            # Check if proposed move leads to immediate collision
            cx, cy = game.head.x, game.head.y
            if proposed_dir == Direction.RIGHT: cx += 20
            elif proposed_dir == Direction.LEFT: cx -= 20
            elif proposed_dir == Direction.DOWN: cy += 20
            elif proposed_dir == Direction.UP: cy -= 20
            
            # Only override if immediate death
            if game.is_collision(Point(cx, cy)):
                # Find safe moves (simple collision check only, no flood fill)
                safe_moves = []
                for i in range(3):
                    check_dir = next_dirs[i]
                    test_x, test_y = game.head.x, game.head.y
                    if check_dir == Direction.RIGHT: test_x += 20
                    elif check_dir == Direction.LEFT: test_x -= 20
                    elif check_dir == Direction.DOWN: test_y += 20
                    elif check_dir == Direction.UP: test_y -= 20
                    
                    if not game.is_collision(Point(test_x, test_y)):
                        safe_moves.append(i)
                
                # Override only if there are safe alternatives
                if safe_moves:
                    # Pick safe move with highest Q-value
                    q_values = self._predict_q_values(state)
                    
                    # Choose best safe move
                    best_safe_move = max(safe_moves, key=lambda m: q_values[m].item())
                    final_move = [0, 0, 0]
                    final_move[best_safe_move] = 1

        return final_move


def evaluate_model_only(games=25, level='random', width=640, height=480, seed=20260521, agent=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if agent is None:
        agent = Agent()

    previous_n_games = agent.n_games
    previous_force_cycle = agent.force_cycle_mode
    previous_planner_setting = agent.use_high_level_planners
    previous_training = agent.model.training
    previous_epsilon_override = agent.exploration_epsilon_override

    agent.n_games = 1000
    agent.force_cycle_mode = False
    agent.use_high_level_planners = False
    agent.exploration_epsilon_override = 0.0
    agent.model.eval()

    game = SnakeGameAI(w=width, h=height, render=False)

    scores = []
    steps = []
    with torch.no_grad():
        for game_idx in range(games):
            game_seed = seed + game_idx
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
            game.set_level(level)
            step_count = 0

            while True:
                state = agent.get_state(game)
                move = agent.get_action(state, game)
                _, done, score = game.play_step(move)
                step_count += 1

                if done:
                    scores.append(score)
                    steps.append(step_count)
                    agent.loop_monitor.clear()
                    break

    result = {
        "model_only_mean_score": round(float(np.mean(scores)), 4),
        "model_only_max_score": int(max(scores)),
        "model_only_median_score": round(float(np.median(scores)), 4),
        "scores": scores,
        "mean_steps": round(float(np.mean(steps)), 4),
        "games": len(scores),
        "level": level,
        "board": [width, height],
        "planner": "disabled_safety_only",
    }

    agent.n_games = previous_n_games
    agent.force_cycle_mode = previous_force_cycle
    agent.use_high_level_planners = previous_planner_setting
    agent.exploration_epsilon_override = previous_epsilon_override
    if previous_training:
        agent.model.train()
    else:
        agent.model.eval()

    return result


def _self_learning_reward(agent, game, env_reward, done, dist_before, score_before):
    if done:
        return -25.0

    reward = float(env_reward)
    dist_after = game._get_closest_food_dist()
    distance_delta = (dist_before - dist_after) / BLOCK_SIZE
    reward += 0.35 * distance_delta

    if game.score > score_before:
        reward += 5.0
    else:
        head_cell = agent._point_to_cell(game.head)
        snake = [agent._point_to_cell(point) for point in game.snake]
        blocked = set(snake[:-1]) | agent._obstacle_cells(game)
        area = agent._reachable_cells_from(game, head_cell, blocked)
        area_target = max(1, len(game.snake) * 2)
        area_score = min(1.0, area / area_target)
        reward += 0.25 * area_score
        if area < len(game.snake):
            reward -= 1.5

    return reward


def self_learn(
    episodes=200,
    eval_games=25,
    eval_interval=25,
    level='random',
    width=640,
    height=480,
    seed=20260521,
    save_eval_games=25,
    min_improvement=1.0,
    restore_patience=2,
    min_exploration=6.0,
    exploration_decay=0.006,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = Agent()
    agent.use_high_level_planners = False
    agent.force_cycle_mode = False
    agent.n_games = 0
    agent.model.train()

    game = SnakeGameAI(w=width, h=height, render=False)
    game.set_level(level)

    save_eval_games = max(eval_games, save_eval_games)
    best_eval = evaluate_model_only(
        games=save_eval_games,
        level=level,
        width=width,
        height=height,
        seed=seed,
        agent=agent,
    )
    best_mean = best_eval["model_only_mean_score"]
    best_max = best_eval["model_only_max_score"]
    best_state = copy.deepcopy(agent.model.state_dict())
    failed_evals = 0
    print(json.dumps({"event": "initial_eval", **best_eval}, ensure_ascii=False), flush=True)

    start_time = time.time()
    scores_window = deque(maxlen=50)
    losses_window = deque(maxlen=50)

    for episode in range(1, episodes + 1):
        agent.exploration_epsilon_override = max(
            min_exploration,
            80 * np.exp(-exploration_decay * episode),
        )
        while True:
            state_old = agent.get_state(game)
            dist_before = game._get_closest_food_dist()
            score_before = game.score
            final_move = agent.get_action(state_old, game)
            env_reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            reward = _self_learning_reward(agent, game, env_reward, done, dist_before, score_before)

            loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            losses_window.append(loss)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                agent.loop_monitor.clear()
                scores_window.append(score)
                break

        if episode % eval_interval == 0 or episode == episodes:
            eval_result = evaluate_model_only(
                games=eval_games,
                level=level,
                width=width,
                height=height,
                seed=seed,
                agent=agent,
            )
            save_eval_result = eval_result
            if eval_games < save_eval_games:
                save_eval_result = evaluate_model_only(
                    games=save_eval_games,
                    level=level,
                    width=width,
                    height=height,
                    seed=seed,
                    agent=agent,
                )

            improved = (
                save_eval_result["model_only_mean_score"] >= best_mean + min_improvement
                or (
                    save_eval_result["model_only_mean_score"] >= best_mean
                    and save_eval_result["model_only_max_score"] > best_max
                )
            )
            if improved:
                best_mean = save_eval_result["model_only_mean_score"]
                best_max = save_eval_result["model_only_max_score"]
                best_state = copy.deepcopy(agent.model.state_dict())
                failed_evals = 0
                agent.model.save()
                restored = False
            else:
                failed_evals += 1
                restored = failed_evals >= restore_patience
                if restored:
                    agent.model.load_state_dict(best_state)
                    agent.target_model.load_state_dict(best_state)
                    agent.trainer = QTrainer(agent.model, lr=LR, gamma=agent.gamma)
                    agent.memory.clear()
                    agent.loop_monitor.clear()
                    failed_evals = 0

            report = {
                "event": "self_learn_eval",
                "episode": episode,
                "train_recent_mean": round(float(np.mean(scores_window)), 4) if scores_window else 0.0,
                "recent_loss": round(float(np.mean(losses_window)), 6) if losses_window else 0.0,
                "saved": improved,
                "restored_best": restored,
                "best_mean": best_mean,
                "best_max": best_max,
                "exploration_epsilon": round(float(agent.exploration_epsilon_override), 4),
                "elapsed_sec": round(time.time() - start_time, 2),
                **eval_result,
            }
            if save_eval_result is not eval_result:
                report["save_eval_mean_score"] = save_eval_result["model_only_mean_score"]
                report["save_eval_max_score"] = save_eval_result["model_only_max_score"]
            with open(SELF_LEARN_METRICS_FILE, "a") as f:
                f.write(json.dumps(report, ensure_ascii=False) + "\n")
            print(json.dumps(report, ensure_ascii=False), flush=True)
            agent.model.train()

    return {"best_mean": best_mean, "best_max": best_max}


def train(target_level=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    # Rolling window for mean score calculation
    scores_window = deque(maxlen=50) 
    
    agent = Agent()
    # Headless training (render=False) for speed
    game = SnakeGameAI(render=True)

    # Curriculum Learning State
    possible_levels = LevelManager.LEVELS
    current_level_idx = 0
    stagnation_counter = 0
    stagnation_limit = 50 # If no improvement for 50 games, switch level
    
    # Load training state (n_games, best_mean_score, curriculum)
    best_mean_score = 0
    if os.path.exists('model/training_state.json'):
        with open('model/training_state.json', 'r') as f:
            state_data = json.load(f)
            agent.n_games = state_data.get('n_games', 0)
            best_mean_score = state_data.get('best_mean_score', 0)
            
            # Load Curriculum State
            current_level_idx = state_data.get('current_level_idx', 0)
            stagnation_counter = state_data.get('stagnation_counter', 0)
            
            print(f"Resumed training from Game {agent.n_games}, Best Mean: {best_mean_score}")
            print(f"Resumed Level: {possible_levels[current_level_idx]} (Stagnation: {stagnation_counter})")

    # Override if target_level is set
    if target_level:
        if target_level not in possible_levels:
            print(f"Error: Level '{target_level}' not found. Available: {possible_levels}")
            return
        current_level_idx = possible_levels.index(target_level)
        print(f"Forcing Training on Level: {target_level}")
    
    # Set initial level
    game.set_level(possible_levels[current_level_idx])
    # print(f"Starting Level: {possible_levels[current_level_idx]}") # Already printed above
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, game)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.loop_monitor.clear()

            scores_window.append(score)
            mean_score = sum(scores_window) / len(scores_window)

            saved = False
            # Only save if mean score is better than all-time best and we have enough samples
            if mean_score > best_mean_score and len(scores_window) >= 20:
                best_mean_score = mean_score
                agent.model.save()
                
                # Save training state
                state_data = {
                    'n_games': agent.n_games,
                    'best_mean_score': best_mean_score,
                    'current_level_idx': current_level_idx,
                    'stagnation_counter': stagnation_counter
                }
                with open('model/training_state.json', 'w') as f:
                    json.dump(state_data, f)
                
                saved = True
            
            # Print training progress with epsilon
            print('Game', agent.n_games, 'Score', score, 'Mean', f'{mean_score:.2f}', 
                  'Best Mean', f'{best_mean_score:.2f}', 'Epsilon', f'{agent.epsilon:.2f}',
                  'Loss', f'{loss:.4f}', 'SAVED' if saved else '')

            if saved:
                 stagnation_counter = 0 # Reset counter on improvement
            else:
                 stagnation_counter += 1
                
            # Curriculum Switch (Only if no target level is forced)
            if not target_level and stagnation_counter >= stagnation_limit:
                 print(f"Stagnation detected ({stagnation_limit} games without new best mean). Switching Level.")
                 current_level_idx = (current_level_idx + 1) % len(possible_levels)
                 next_level = possible_levels[current_level_idx]
                 game.set_level(next_level)
                 print(f"New Level: {next_level}")
                 stagnation_counter = 0

            plot_scores.append(score)
            total_score += score
            cumulative_mean = total_score / agent.n_games
            plot_mean_scores.append(cumulative_mean)
            plot(plot_scores, plot_mean_scores, loss)

def test(target_level=None):
    """Runs the game with UI using the current best model (Greedy/Low Epsilon)"""
    agent = Agent()
    agent.model.eval()
    game = SnakeGameAI(render=True)
    
    if target_level:
        game.set_level(target_level)
        print(f"Testing on Level: {target_level}")

    # Force low epsilon for testing (mostly exploitation)
    agent.n_games = 1000 
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old, game)
        reward, done, score = game.play_step(final_move)
        
        if done:
            game.reset()
            print('Game Over. Score:', score)

def test_levels():
    """Runs one game for each level type to verify performance"""
    agent = Agent()
    agent.model.eval()
    game = SnakeGameAI(render=True)
    agent.n_games = 1000 # Force low epsilon
    
    for level_name in LevelManager.LEVELS:
        print(f"--- Testing Level: {level_name} ---")
        game.set_level(level_name)
        
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            reward, done, score = game.play_step(final_move)
            
            if done:
                print(f"Level {level_name} Finished. Score: {score}")
                break # Move to next level

def target500(target_score=500, render=False, seed=1, max_steps=200000):
    """Runs a deterministic empty-board survival route until the target score."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = Agent()
    agent.n_games = 1000
    agent.force_cycle_mode = True
    agent.model.eval()

    game = SnakeGameAI(render=render)
    game.set_level('empty')

    steps = 0
    with torch.no_grad():
        while game.score < target_score and steps < max_steps:
            state = agent.get_state(game)
            final_move = agent.get_action(state, game)
            _, done, score = game.play_step(final_move)
            steps += 1

            if done:
                result = {
                    "achieved": False,
                    "score": score,
                    "target_score": target_score,
                    "steps": steps,
                    "level": game.current_level,
                    "board": [game.w, game.h],
                    "seed": seed,
                }
                print(json.dumps(result, ensure_ascii=False))
                return result

    result = {
        "achieved": game.score >= target_score,
        "score": game.score,
        "target_score": target_score,
        "steps": steps,
        "level": game.current_level,
        "board": [game.w, game.h],
        "seed": seed,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result

if __name__ == '__main__':
    # Default to train, but allows simple toggle or CLI later
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'model_eval':
        parser = argparse.ArgumentParser(description='Measure model-only Snake performance.')
        parser.add_argument('--games', type=int, default=25)
        parser.add_argument('--level', default='random', choices=LevelManager.LEVELS)
        parser.add_argument('--width', type=int, default=640)
        parser.add_argument('--height', type=int, default=480)
        parser.add_argument('--seed', type=int, default=20260521)
        args = parser.parse_args(sys.argv[2:])
        print(json.dumps(evaluate_model_only(
            games=args.games,
            level=args.level,
            width=args.width,
            height=args.height,
            seed=args.seed,
        ), ensure_ascii=False))

    elif len(sys.argv) > 1 and sys.argv[1] == 'selflearn':
        parser = argparse.ArgumentParser(description='Train the model from self-play with high-level planners disabled.')
        parser.add_argument('--episodes', type=int, default=200)
        parser.add_argument('--eval-games', type=int, default=25)
        parser.add_argument('--eval-interval', type=int, default=25)
        parser.add_argument('--level', default='random', choices=LevelManager.LEVELS)
        parser.add_argument('--width', type=int, default=640)
        parser.add_argument('--height', type=int, default=480)
        parser.add_argument('--seed', type=int, default=20260521)
        parser.add_argument('--save-eval-games', type=int, default=25)
        parser.add_argument('--min-improvement', type=float, default=1.0)
        parser.add_argument('--restore-patience', type=int, default=2)
        parser.add_argument('--min-exploration', type=float, default=6.0)
        parser.add_argument('--exploration-decay', type=float, default=0.006)
        args = parser.parse_args(sys.argv[2:])
        print(json.dumps(self_learn(
            episodes=args.episodes,
            eval_games=args.eval_games,
            eval_interval=args.eval_interval,
            level=args.level,
            width=args.width,
            height=args.height,
            seed=args.seed,
            save_eval_games=args.save_eval_games,
            min_improvement=args.min_improvement,
            restore_patience=args.restore_patience,
            min_exploration=args.min_exploration,
            exploration_decay=args.exploration_decay,
        ), ensure_ascii=False))
    
    # Check for parallel training mode
    elif '--parallel' in sys.argv:
        from parallel_trainer import ParallelTrainer
        
        num_workers = None
        target_level = None
        
        # Parse workers argument
        if '--workers' in sys.argv:
            idx = sys.argv.index('--workers')
            if idx + 1 < len(sys.argv):
                try:
                    num_workers = int(sys.argv[idx + 1])
                except ValueError:
                    print("Error: --workers must be followed by a number")
                    sys.exit(1)
        
        # Parse level argument
        if len(sys.argv) > 1:
            for i, arg in enumerate(sys.argv):
                if arg not in ['--parallel', '--workers'] and not arg.isdigit() and arg.endswith('.py') == False:
                    if i > 0 and sys.argv[i-1] != '--workers':
                        target_level = arg
                        break
        
        print("=" * 50)
        print("PARALLEL TRAINING MODE")
        print("=" * 50)
        trainer = ParallelTrainer(num_workers=num_workers, target_level=target_level)
        trainer.train()
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        if len(sys.argv) > 2:
             test(target_level=sys.argv[2])
        else:
             test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test_level':
        test_levels()
    elif len(sys.argv) > 1 and sys.argv[1] == 'target500':
        render = '--render' in sys.argv
        target_score = 500
        seed = 1
        max_steps = 200000

        if '--target' in sys.argv:
            idx = sys.argv.index('--target')
            if idx + 1 < len(sys.argv):
                target_score = int(sys.argv[idx + 1])
        if '--seed' in sys.argv:
            idx = sys.argv.index('--seed')
            if idx + 1 < len(sys.argv):
                seed = int(sys.argv[idx + 1])
        if '--max-steps' in sys.argv:
            idx = sys.argv.index('--max-steps')
            if idx + 1 < len(sys.argv):
                max_steps = int(sys.argv[idx + 1])

        target500(target_score=target_score, render=render, seed=seed, max_steps=max_steps)
    elif len(sys.argv) > 2 and sys.argv[1] == 'train':
        train(target_level=sys.argv[2])
    else:
        train()

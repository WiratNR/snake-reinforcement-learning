import torch
import random
import numpy as np
from collections import deque
import os
import json
from game import SnakeGameAI, Direction, Point
from levels import LevelManager
from model import DuelingLinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 4000  # INCREASED: Larger batch for more stable gradients
LR = 0.0001  # REDUCED: Lower learning rate for better convergence and lower loss

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

    def get_action(self, state, game):
        # Update Loop Monitor
        self.loop_monitor.update(game.head, game.score)
        
        # random moves: tradeoff exploration / exploitation
        # CRITICAL FIX: MUCH faster epsilon decay
        # At game 50, epsilon ~ 15. At game 100, epsilon ~ 3. At game 200, epsilon ~ 0.1
        self.epsilon = 80 * np.exp(-0.02 * self.n_games)
        
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # --- Safety and late-game move reranking ---
        # Keep the network as the primary policy, but for mature agents prefer
        # safe moves that also make food progress and leave enough open space.
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
            
            safe_moves = []
            move_points = {}
            for i, check_dir in enumerate(next_dirs):
                test_x, test_y = game.head.x, game.head.y
                if check_dir == Direction.RIGHT: test_x += 20
                elif check_dir == Direction.LEFT: test_x -= 20
                elif check_dir == Direction.DOWN: test_y += 20
                elif check_dir == Direction.UP: test_y -= 20
                
                point = Point(test_x, test_y)
                move_points[i] = point
                if not game.is_collision(point):
                    safe_moves.append(i)
            
            if safe_moves:
                state0 = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state0)[0]

                if game.is_collision(Point(cx, cy)):
                    best_safe_move = max(safe_moves, key=lambda m: q_values[m].item())
                    final_move = [0, 0, 0]
                    final_move[best_safe_move] = 1
                elif self.n_games > 200:
                    target = game.food
                    if game.bonus_food:
                        food_dist = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                        bonus_dist = abs(game.head.x - game.bonus_food.x) + abs(game.head.y - game.bonus_food.y)
                        if bonus_dist < food_dist:
                            target = game.bonus_food

                    current_dist = abs(game.head.x - target.x) + abs(game.head.y - target.y)
                    safe_q = [q_values[m].item() for m in safe_moves]
                    q_min, q_max = min(safe_q), max(safe_q)
                    q_span = max(q_max - q_min, 1e-6)
                    area_limit = max(len(game.snake) * 2, 1)

                    def move_score(move_idx):
                        point = move_points[move_idx]
                        next_dist = abs(point.x - target.x) + abs(point.y - target.y)
                        progress = (current_dist - next_dist) / 20.0
                        q_norm = (q_values[move_idx].item() - q_min) / q_span
                        area = min(self._get_reachable_area(game, point.x, point.y), area_limit)
                        area_norm = area / area_limit
                        trap_penalty = 1.0 if area < len(game.snake) else 0.0
                        return (0.60 * q_norm) + (0.30 * progress) + (0.20 * area_norm) - trap_penalty

                    best_safe_move = max(safe_moves, key=move_score)
                    if best_safe_move != proposed_move_idx:
                        final_move = [0, 0, 0]
                        final_move[best_safe_move] = 1

        return final_move


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

if __name__ == '__main__':
    # Default to train, but allows simple toggle or CLI later
    import sys
    
    # Check for parallel training mode
    if '--parallel' in sys.argv:
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
    elif len(sys.argv) > 2 and sys.argv[1] == 'train':
        train(target_level=sys.argv[2])
    else:
        train()

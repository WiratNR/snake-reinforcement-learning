import torch
import random
import numpy as np
from collections import deque
import os
import json
from game import SnakeGameAI, Direction, Point
from levels import LevelManager
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.target_model = Linear_QNet(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target net not trained directly
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # Load existing model
        if self.model.load():
            self.target_model.load_state_dict(self.model.state_dict())
            print("Loaded existing model configuration.")


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Determine target food (Normal or Bonus)
        food_target = game.food
        if game.bonus_food is not None:
            dist_normal = abs(head.x - game.food.x) + abs(head.y - game.food.y)
            dist_bonus = abs(head.x - game.bonus_food.x) + abs(head.y - game.bonus_food.y)
            # Simple heuristic: if bonus is present, target the closer one
            if dist_bonus < dist_normal:
                food_target = game.bonus_food

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location (Target)
            food_target.x < game.head.x,  # food left
            food_target.x > game.head.x,  # food right
            food_target.y < game.head.y,  # food up
            food_target.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

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

        return final_move

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
        # random moves: tradeoff exploration / exploitation
        # Epsilon Decay: Exponential decay to ensure long-term exploration
        # At game 1000, epsilon ~ 29. At game 2000, epsilon ~ 10.
        self.epsilon = 80 * np.exp(-0.001 * self.n_games)
        
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # --- Advanced Planning (Safety Overrides) ---
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        next_dirs = [
            clock_wise[idx], # Straight [1,0,0]
            clock_wise[(idx + 1) % 4], # Right [0,1,0]
            clock_wise[(idx - 1) % 4]  # Left [0,0,1]
        ]
        
        proposed_move_idx = final_move.index(1)
        safe_moves = []
        
        # 1. Identify all truly safe moves using Lookahead + Flood Fill
        for i in range(3):
            check_dir = next_dirs[i]
            cx, cy = game.head.x, game.head.y
            if check_dir == Direction.RIGHT: cx += 20
            elif check_dir == Direction.LEFT: cx -= 20
            elif check_dir == Direction.DOWN: cy += 20
            elif check_dir == Direction.UP: cy -= 20
            
            # (1) Immediate Collision Check
            if not game.is_collision(Point(cx, cy)):
                # (2) Free-space Estimation (Flood Fill)
                # Only strictly required if we are near obstacles, but good to run.
                # Heuristic: If reachable area < snake length, it's a trap.
                area = self._get_reachable_area(game, cx, cy)
                if area > len(game.snake): 
                     safe_moves.append(i)
        
        # 2. If proposed move is NOT in safe_moves, override it
        if proposed_move_idx not in safe_moves:
            if safe_moves:
                # Pick the safe move that the model prefers (highest Q), or random safe
                # For simplicity, pick random safe or the first one.
                # Ideal: Pick safe move with highest prediction?
                # Let's just pick random safe to avoid getting stuck in loops.
                new_move_idx = random.choice(safe_moves)
                final_move = [0, 0, 0]
                final_move[new_move_idx] = 1
            else:
                 # No safe moves? We are dead. Keep original move.
                 pass

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    # Rolling window for mean score calculation
    scores_window = deque(maxlen=50) 
    
    agent = Agent()
    # Headless training (render=False) for speed
    game = SnakeGameAI(render=False)

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
                stagnation_counter = 0 # Reset counter on improvement
            else:
                stagnation_counter += 1
                
            # Curriculum Switch
            if stagnation_counter >= stagnation_limit:
                 print(f"Stagnation detected ({stagnation_limit} games without new best mean). Switching Level.")
                 current_level_idx = (current_level_idx + 1) % len(possible_levels)
                 next_level = possible_levels[current_level_idx]
                 game.set_level(next_level)
                 print(f"New Level: {next_level}")
                 stagnation_counter = 0

            print('Game', agent.n_games, 'Score', score, 'Mean', mean_score, 'Best Mean', best_mean_score, 'Loss', loss, 'Saved' if saved else '')

            plot_scores.append(score)
            total_score += score
            cumulative_mean = total_score / agent.n_games
            plot_mean_scores.append(cumulative_mean)
            plot(plot_scores, plot_mean_scores, loss)

def test():
    """Runs the game with UI using the current best model (Greedy/Low Epsilon)"""
    agent = Agent()
    game = SnakeGameAI(render=True)
    
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
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test_level':
        test_levels()
    else:
        train()

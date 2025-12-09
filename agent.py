import torch
import random
import numpy as np
from collections import deque
import os
import json
from game import SnakeGameAI, Direction, Point
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
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # Load existing model
        if self.model.load():
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
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    # Rolling window for mean score calculation
    scores_window = deque(maxlen=50) 
    
    agent = Agent()
    game = SnakeGameAI()

    # Load training state (n_games, best_mean_score)
    best_mean_score = 0
    if os.path.exists('model/training_state.json'):
        with open('model/training_state.json', 'r') as f:
            state_data = json.load(f)
            agent.n_games = state_data.get('n_games', 0)
            best_mean_score = state_data.get('best_mean_score', 0)
            print(f"Resumed training from Game {agent.n_games}, Best Mean: {best_mean_score}")
    
    # If model exists but no state file (legacy), we rely on model weights but n_games is 0 (high exploration)
    # Ideally should sync, but for now this handles the restart case.
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

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
                    'best_mean_score': best_mean_score
                }
                with open('model/training_state.json', 'w') as f:
                    json.dump(state_data, f)
                
                saved = True

            print('Game', agent.n_games, 'Score', score, 'Mean', mean_score, 'Best Mean', best_mean_score, 'Loss', loss, 'Saved' if saved else '')

            plot_scores.append(score)
            # Plotting the cumulative mean for the graph still, or could switch to rolling
            total_score += score
            cumulative_mean = total_score / agent.n_games
            plot_mean_scores.append(cumulative_mean)
            # plot(plot_scores, plot_mean_scores) # Updated helper to plot loss is needed
            # For now passing None for loss until helper is updated
            plot(plot_scores, plot_mean_scores, loss)


if __name__ == '__main__':
    train()

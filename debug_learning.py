"""
Debug script to test if basic training works
This will run a simple sequential training to verify the learning mechanism
"""
import torch
import numpy as np
from collections import deque
from game import SnakeGameAI
from agent import Agent
import matplotlib.pyplot as plt

def test_basic_learning():
    """Test if agent can learn in simple sequential mode"""
    print("=" * 60)
    print("TESTING BASIC LEARNING")
    print("=" * 60)
    
    agent = Agent()
    game = SnakeGameAI(render=False)  # Headless for speed
    
    scores = []
    losses = []
    epsilons = []
    
    print("\nRunning 50 games to test learning...")
    print("Game | Score | Epsilon | Loss | Mean Score")
    print("-" * 60)
    
    for game_num in range(50):
        agent.n_games = game_num
        game.reset()
        game_score = 0
        game_loss = 0
        steps = 0
        
        while True:
            # Get state and action
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            
            # Perform move
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            # Train short memory
            loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
            game_loss = loss
            
            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)
            
            steps += 1
            if done:
                game_score = score
                break
        
        # Train long memory
        agent.train_long_memory()
        
        scores.append(game_score)
        losses.append(game_loss)
        epsilons.append(agent.epsilon)
        
        mean_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        
        print(f"{game_num:4d} | {game_score:5d} | {agent.epsilon:7.2f} | {game_loss:.4f} | {mean_score:.2f}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Analysis
    first_10_mean = np.mean(scores[:10])
    last_10_mean = np.mean(scores[-10:])
    
    print(f"\nFirst 10 games mean score: {first_10_mean:.2f}")
    print(f"Last 10 games mean score: {last_10_mean:.2f}")
    print(f"Improvement: {last_10_mean - first_10_mean:.2f}")
    
    print(f"\nFirst epsilon: {epsilons[0]:.2f}")
    print(f"Last epsilon: {epsilons[-1]:.2f}")
    
    print(f"\nFirst loss: {losses[0]:.4f}")
    print(f"Last loss: {losses[-1]:.4f}")
    
    # Check if learning occurred
    if last_10_mean > first_10_mean + 1:
        print("\n✅ LEARNING DETECTED! Model is improving.")
    else:
        print("\n❌ NO LEARNING! Model is not improving.")
        print("\nPossible issues:")
        print("1. Rewards might be too sparse")
        print("2. Learning rate might be wrong")
        print("3. Model architecture issue")
        print("4. State representation issue")
    
    return scores, losses, epsilons

def test_model_updates():
    """Test if model weights actually change during training"""
    print("\n" + "=" * 60)
    print("TESTING MODEL WEIGHT UPDATES")
    print("=" * 60)
    
    agent = Agent()
    
    # Get initial weights
    initial_weights = agent.model.get_weights()
    initial_sum = sum(np.sum(np.abs(w)) for w in initial_weights.values())
    
    print(f"\nInitial weight sum: {initial_sum:.4f}")
    
    # Train on dummy data
    game = SnakeGameAI(render=False)
    for _ in range(100):
        state = agent.get_state(game)
        action = [1, 0, 0]
        reward = 10
        next_state = agent.get_state(game)
        done = False
        
        agent.remember(state, action, reward, next_state, done)
    
    # Train
    agent.train_long_memory()
    
    # Get new weights
    new_weights = agent.model.get_weights()
    new_sum = sum(np.sum(np.abs(w)) for w in new_weights.values())
    
    print(f"After training weight sum: {new_sum:.4f}")
    print(f"Weight change: {abs(new_sum - initial_sum):.4f}")
    
    if abs(new_sum - initial_sum) > 0.001:
        print("\n✅ WEIGHTS ARE UPDATING! Model is training.")
    else:
        print("\n❌ WEIGHTS NOT CHANGING! Training is not working.")

if __name__ == '__main__':
    # Test 1: Check if weights update
    test_model_updates()
    
    # Test 2: Check if learning occurs
    scores, losses, epsilons = test_basic_learning()
    
    print("\n" + "=" * 60)
    print("Debug test complete!")
    print("=" * 60)

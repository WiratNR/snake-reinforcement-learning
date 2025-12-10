"""
Extended learning test - 200 games to see clear learning progression
"""
import torch
import numpy as np
from collections import deque
from game import SnakeGameAI
from agent import Agent

def test_extended_learning():
    """Test learning over 200 games"""
    print("=" * 70)
    print("EXTENDED LEARNING TEST (200 GAMES)")
    print("=" * 70)
    
    agent = Agent()
    game = SnakeGameAI(render=False)
    
    scores = []
    losses = []
    
    print("\nGame | Score | Epsilon | Loss | Mean(10) | Mean(50)")
    print("-" * 70)
    
    for game_num in range(200):
        agent.n_games = game_num
        game.reset()
        game_score = 0
        game_loss = 0
        
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
            game_loss = loss
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                game_score = score
                break
        
        agent.train_long_memory()
        
        scores.append(game_score)
        losses.append(game_loss)
        
        mean_10 = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        mean_50 = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
        
        # Print every 10 games
        if game_num % 10 == 0 or game_num < 10:
            print(f"{game_num:4d} | {game_score:5d} | {agent.epsilon:7.2f} | {game_loss:6.2f} | {mean_10:6.2f} | {mean_50:6.2f}")
    
    print("\n" + "=" * 70)
    print("LEARNING ANALYSIS")
    print("=" * 70)
    
    # Compare different periods
    period_1 = scores[0:50]    # Games 0-49
    period_2 = scores[50:100]  # Games 50-99
    period_3 = scores[100:150] # Games 100-149
    period_4 = scores[150:200] # Games 150-199
    
    print(f"\nPeriod 1 (Games 0-49):     Mean = {np.mean(period_1):.2f}, Max = {max(period_1)}")
    print(f"Period 2 (Games 50-99):    Mean = {np.mean(period_2):.2f}, Max = {max(period_2)}")
    print(f"Period 3 (Games 100-149):  Mean = {np.mean(period_3):.2f}, Max = {max(period_3)}")
    print(f"Period 4 (Games 150-199):  Mean = {np.mean(period_4):.2f}, Max = {max(period_4)}")
    
    improvement = np.mean(period_4) - np.mean(period_1)
    print(f"\nTotal Improvement: {improvement:.2f} ({improvement/max(np.mean(period_1), 0.01)*100:.1f}%)")
    
    if improvement > 2:
        print("\n✅ STRONG LEARNING! Model is improving significantly.")
    elif improvement > 0.5:
        print("\n✅ LEARNING DETECTED! Model is improving.")
    else:
        print("\n⚠️  WEAK LEARNING. Model needs more tuning.")
    
    # Save the model
    agent.model.save('model_test.pth')
    print("\nModel saved to model_test.pth")

if __name__ == '__main__':
    test_extended_learning()

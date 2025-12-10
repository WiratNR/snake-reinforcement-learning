"""
Deep diagnostic script to find the root cause
"""
import torch
import numpy as np
from model import DuelingLinearQNet, QTrainer

def test_gradient_flow():
    """Test if gradients are actually flowing"""
    print("=" * 60)
    print("TEST 1: GRADIENT FLOW")
    print("=" * 60)
    
    model = DuelingLinearQNet(15, 256, 3)
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    
    # Get initial weights
    initial_params = [p.clone() for p in model.parameters()]
    
    # Create dummy data
    state = np.random.rand(10, 15)
    action = np.array([[1,0,0]] * 10)
    reward = np.array([10.0] * 10)
    next_state = np.random.rand(10, 15)
    done = [False] * 10
    
    # Train
    loss = trainer.train_step(state, action, reward, next_state, done)
    
    # Check if weights changed
    changed = False
    for i, p in enumerate(model.parameters()):
        if not torch.allclose(p, initial_params[i]):
            changed = True
            break
    
    print(f"Loss: {loss:.4f}")
    print(f"Weights changed: {changed}")
    
    if changed:
        print("✅ Gradients are flowing!")
    else:
        print("❌ Gradients NOT flowing - CRITICAL BUG!")
    
    return changed

def test_q_value_learning():
    """Test if Q-values actually change with training"""
    print("\n" + "=" * 60)
    print("TEST 2: Q-VALUE LEARNING")
    print("=" * 60)
    
    model = DuelingLinearQNet(15, 256, 3)
    target_model = DuelingLinearQNet(15, 256, 3)
    target_model.load_state_dict(model.state_dict())
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    
    # Test state
    test_state = np.random.rand(15)
    
    # Initial Q-values
    with torch.no_grad():
        initial_q = model(torch.FloatTensor(test_state)).numpy()
    
    print(f"Initial Q-values: {initial_q}")
    
    # Train on positive reward for action 0
    for _ in range(100):
        state = np.random.rand(15)
        action = [1, 0, 0]  # Always action 0
        reward = 10.0  # High reward
        next_state = np.random.rand(15)
        done = False
        
        trainer.train_step(state, action, reward, next_state, done, target_model)
    
    # Final Q-values
    with torch.no_grad():
        final_q = model(torch.FloatTensor(test_state)).numpy()
    
    print(f"Final Q-values:   {final_q}")
    print(f"Change in Q[0]:   {final_q[0] - initial_q[0]:.4f}")
    
    if final_q[0] > initial_q[0] + 1:
        print("✅ Q-values are learning!")
    else:
        print("❌ Q-values NOT learning properly!")

def test_action_selection():
    """Test if model can learn to prefer one action"""
    print("\n" + "=" * 60)
    print("TEST 3: ACTION PREFERENCE LEARNING")
    print("=" * 60)
    
    model = DuelingLinearQNet(15, 256, 3)
    target_model = DuelingLinearQNet(15, 256, 3)
    target_model.load_state_dict(model.state_dict())
    trainer = QTrainer(model, lr=0.001, gamma=0.9)
    
    # Train: Action 1 always gets reward 10, others get -1
    print("Training: Action 1 → +10, Others → -1")
    
    for _ in range(500):
        state = np.random.rand(15)
        action_idx = np.random.randint(0, 3)
        action = [0, 0, 0]
        action[action_idx] = 1
        
        reward = 10.0 if action_idx == 1 else -1.0
        next_state = np.random.rand(15)
        done = False
        
        trainer.train_step(state, action, reward, next_state, done, target_model)
        
        if _ % 100 == 0:
            target_model.load_state_dict(model.state_dict())
    
    # Test: Which action does it prefer?
    test_state = torch.FloatTensor(np.random.rand(1, 15))
    with torch.no_grad():
        q_values = model(test_state)[0]
    
    preferred_action = torch.argmax(q_values).item()
    
    print(f"\nQ-values: {q_values.numpy()}")
    print(f"Preferred action: {preferred_action}")
    
    if preferred_action == 1:
        print("✅ Model learned to prefer the rewarded action!")
    else:
        print("❌ Model did NOT learn action preference!")

if __name__ == '__main__':
    test_gradient_flow()
    test_q_value_learning()
    test_action_selection()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

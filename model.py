import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class DuelingLinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # LayerNorm works with batch_size=1
        self.dropout1 = nn.Dropout(0.2)  # Dropout to prevent overfitting
        
        # Value stream (deeper with normalization)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream (deeper with normalization)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # Handle both 1D and 2D inputs
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if squeeze_output:
            q_values = q_values.squeeze(0)
            
        return q_values

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        
        if os.path.exists(file_name):
            try:
                self.load_state_dict(torch.load(file_name))
                print(f"Model loaded from {file_name}")
                return True
            except RuntimeError:
                print("Note: Incompatible model found (likely due to architecture change). Starting fresh training session.")
                return False
        return False
    
    def get_weights(self):
        """Get model weights for sharing between processes"""
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights):
        """Set model weights from shared weights dictionary"""
        state_dict = {k: torch.tensor(v) for k, v in weights.items()}
        self.load_state_dict(state_dict)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Huber Loss: More robust than MSE, less sensitive to outliers
        self.criterion = nn.SmoothL1Loss()  # Huber Loss

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Double DQN Logic
                # 1. Select best action using ONLINE model
                best_action = torch.argmax(self.model(next_state[idx].unsqueeze(0))).item()
                
                # 2. Evaluate that action using TARGET model
                if target_model:
                     # target_model usually in eval mode, but just in case
                     with torch.no_grad():
                         target_q_values = target_model(next_state[idx].unsqueeze(0))
                         Q_new = reward[idx] + self.gamma * target_q_values[0][best_action]
                else:
                     # CRITICAL FIX: Fallback to standard DQN if no target model
                     with torch.no_grad():
                         Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        
        # Gradient Clipping: Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss.item()


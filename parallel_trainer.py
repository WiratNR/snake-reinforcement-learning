import torch
import random
import numpy as np
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Manager
import os
import json
import time
from game import SnakeGameAI
from levels import LevelManager
from model import DuelingLinearQNet, QTrainer
from agent import Agent, MAX_MEMORY, BATCH_SIZE, LR

class WorkerProcess:
    """Individual worker process that collects experiences"""
    
    def __init__(self, worker_id, experience_queue, stats_queue, weights_queue, 
                 stop_flag, target_level=None):
        self.worker_id = worker_id
        self.experience_queue = experience_queue
        self.stats_queue = stats_queue
        self.weights_queue = weights_queue
        self.stop_flag = stop_flag
        self.target_level = target_level
        
    def run(self):
        """Main worker loop"""
        # Create local agent and game (headless for workers)
        agent = Agent()
        game = SnakeGameAI(render=False)
        
        if self.target_level:
            game.set_level(self.target_level)
        
        games_played = 0
        
        while not self.stop_flag.value:
            # Check for model weight updates
            if not self.weights_queue.empty():
                try:
                    weights = self.weights_queue.get_nowait()
                    agent.model.set_weights(weights)
                    agent.target_model.set_weights(weights)
                except:
                    pass
            
            # CRITICAL FIX: Update agent's n_games for proper epsilon decay
            agent.n_games = games_played
            
            # Play one game episode
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, game)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            # Send experience to shared queue
            try:
                self.experience_queue.put_nowait(
                    (state_old, final_move, reward, state_new, done)
                )
            except:
                # Queue full, skip this experience
                pass
            
            if done:
                # Send statistics
                try:
                    self.stats_queue.put_nowait({
                        'worker_id': self.worker_id,
                        'score': score,
                        'reward': game.total_reward,
                        'frame_iteration': game.frame_iteration
                    })
                except:
                    pass
                
                game.reset()
                agent.loop_monitor.clear()
                games_played += 1

def worker_process_fn(worker_id, experience_queue, stats_queue, weights_queue, 
                      stop_flag, target_level):
    """Function to run in separate process"""
    worker = WorkerProcess(worker_id, experience_queue, stats_queue, 
                          weights_queue, stop_flag, target_level)
    worker.run()

class ParallelTrainer:
    """Manages parallel training with multiple worker processes"""
    
    def __init__(self, num_workers=None, target_level=None):
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        self.num_workers = num_workers
        self.target_level = target_level
        
        # Shared queues
        self.experience_queue = Queue(maxsize=10000)
        self.stats_queue = Queue(maxsize=1000)
        
        # Weight queues for each worker
        self.weights_queues = [Queue(maxsize=1) for _ in range(num_workers)]
        
        # Stop flag
        self.stop_flag = Value('i', 0)
        
        # Main agent and trainer
        self.agent = Agent()
        self.game = SnakeGameAI(render=True)  # Main process shows UI
        
        if self.target_level:
            self.game.set_level(self.target_level)
        
        # Training state
        self.best_mean_score = 0
        self.load_training_state()
        
        # Workers
        self.workers = []
        
        print(f"Parallel Trainer initialized with {self.num_workers} workers")
    
    def load_training_state(self):
        """Load training state from file"""
        if os.path.exists('model/training_state.json'):
            with open('model/training_state.json', 'r') as f:
                state_data = json.load(f)
                self.agent.n_games = state_data.get('n_games', 0)
                self.best_mean_score = state_data.get('best_mean_score', 0)
                print(f"Resumed training from Game {self.agent.n_games}, Best Mean: {self.best_mean_score}")
    
    def save_training_state(self):
        """Save training state to file"""
        state_data = {
            'n_games': self.agent.n_games,
            'best_mean_score': self.best_mean_score
        }
        with open('model/training_state.json', 'w') as f:
            json.dump(state_data, f)
    
    def start_workers(self):
        """Start all worker processes"""
        for i in range(self.num_workers):
            p = Process(
                target=worker_process_fn,
                args=(i, self.experience_queue, self.stats_queue, 
                      self.weights_queues[i], self.stop_flag, self.target_level)
            )
            p.start()
            self.workers.append(p)
        print(f"Started {self.num_workers} worker processes")
    
    def stop_workers(self):
        """Stop all worker processes"""
        self.stop_flag.value = 1
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        print("All workers stopped")
    
    def broadcast_weights(self):
        """Broadcast current model weights to all workers"""
        weights = self.agent.model.get_weights()
        for queue in self.weights_queues:
            # Clear old weights
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
            # Send new weights
            try:
                queue.put_nowait(weights)
            except:
                pass
    
    def collect_experiences(self, batch_size=100):
        """Collect experiences from workers"""
        experiences = []
        timeout = 0.05  # Shorter timeout for faster response
        start_time = time.time()
        
        while len(experiences) < batch_size:
            # Timeout after 2 seconds to avoid hanging
            if time.time() - start_time > 2:
                break
                
            try:
                exp = self.experience_queue.get(timeout=timeout)
                experiences.append(exp)
            except:
                # Queue empty, break if we have some experiences
                if len(experiences) > 0:
                    break
                time.sleep(0.01)
        
        return experiences
    
    def collect_stats(self):
        """Collect statistics from workers"""
        stats = []
        while not self.stats_queue.empty():
            try:
                stat = self.stats_queue.get_nowait()
                stats.append(stat)
            except:
                break
        return stats
    
    def train(self):
        """Main parallel training loop"""
        from helper import plot
        
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        scores_window = deque(maxlen=50)
        
        # CRITICAL FIX: Add memory replay buffer for main process
        memory = deque(maxlen=MAX_MEMORY)
        
        # Start workers
        self.start_workers()
        
        # Broadcast initial weights
        self.broadcast_weights()
        
        update_interval = 5  # Broadcast weights more frequently
        training_steps = 0
        last_loss = 0
        
        # Use smaller batch for faster training cycles
        train_batch_size = 256  # Smaller than BATCH_SIZE for faster updates
        
        try:
            while True:
                # Collect experiences from workers and add to memory
                # Collect smaller batches more frequently
                experiences = self.collect_experiences(batch_size=100)
                
                # CRITICAL FIX: Store experiences in memory buffer
                for exp in experiences:
                    memory.append(exp)
                
                # Train on memory if we have enough samples
                if len(memory) > train_batch_size:
                    # Sample from memory for training
                    mini_sample = random.sample(memory, train_batch_size)
                    states, actions, rewards, next_states, dones = zip(*mini_sample)
                    
                    last_loss = self.agent.trainer.train_step(
                        states, actions, rewards, next_states, dones, 
                        self.agent.target_model
                    )
                    
                    training_steps += 1
                    
                    # Update learning rate scheduler based on loss
                    self.agent.trainer.scheduler.step(last_loss)
                    
                    # Update target network periodically (INCREASED to 20 for stability)
                    if training_steps % 20 == 0:
                        self.agent.target_model.load_state_dict(self.agent.model.state_dict())
                    
                    # Broadcast weights to workers more frequently
                    if training_steps % update_interval == 0:
                        self.broadcast_weights()
                
                # Collect and process statistics
                stats = self.collect_stats()
                for stat in stats:
                    score = stat['score']
                    self.agent.n_games += 1
                    
                    scores_window.append(score)
                    mean_score = sum(scores_window) / len(scores_window)
                    
                    saved = False
                    # Save if mean score improved
                    if mean_score > self.best_mean_score and len(scores_window) >= 20:
                        self.best_mean_score = mean_score
                        self.agent.model.save()
                        self.save_training_state()
                        saved = True
                    
                    # Print stats more frequently with epsilon
                    # Calculate epsilon for display (same formula as agent)
                    epsilon = max(0.05, 0.8 * np.exp(-0.005 * self.agent.n_games))
                    print(f'Game {self.agent.n_games} | Worker {stat["worker_id"]} | '
                          f'Score {score} | Mean {mean_score:.2f} | '
                          f'Best Mean {self.best_mean_score:.2f} | '
                          f'Epsilon {epsilon:.2f} | '
                          f'Loss {last_loss:.4f} | '
                          f'Memory {len(memory)} | '
                          f'{"SAVED" if saved else ""}')
                    
                    plot_scores.append(score)
                    total_score += score
                    cumulative_mean = total_score / self.agent.n_games
                    plot_mean_scores.append(cumulative_mean)
                    
                    # Update plot
                    if self.agent.n_games % 10 == 0:
                        plot(plot_scores, plot_mean_scores, last_loss)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self.stop_workers()
            print("Training completed")

if __name__ == '__main__':
    import sys
    
    num_workers = None
    target_level = None
    
    # Parse arguments
    if '--workers' in sys.argv:
        idx = sys.argv.index('--workers')
        if idx + 1 < len(sys.argv):
            num_workers = int(sys.argv[idx + 1])
    
    if '--level' in sys.argv:
        idx = sys.argv.index('--level')
        if idx + 1 < len(sys.argv):
            target_level = sys.argv[idx + 1]
    
    trainer = ParallelTrainer(num_workers=num_workers, target_level=target_level)
    trainer.train()

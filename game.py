import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from levels import LevelManager

pygame.init()
font = pygame.font.SysFont('arial', 25)

# Reset
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    
    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render
        # init display
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.total_reward = 0
        self.food = None
        self.bonus_food = None
        self.obstacles = []
        self._place_obstacles()
        self._place_food()
        self.frame_iteration = 0
        
    def _place_obstacles(self):
        # Use external LevelManager to get a random level
        # IMPORTANT: 'Point' in levels.py might be distinct from 'Point' here if not handled carefully
        # But namedtokens are compatible if fields match.
        raw_obstacles = LevelManager.get_random_level(self.w, self.h)
        self.obstacles = []
        # Convert to local Point just in case, though they are likely compatible
        for p in raw_obstacles:
            pt = Point(p.x, p.y)
            if pt not in self.snake and pt != self.head:
                self.obstacles.append(pt)

    def _place_food(self):
        # 1. Place Normal Food (Always)
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles or self.food == self.bonus_food:
            self._place_food()
            return

        # 2. Try to place Bonus Food (20% chance if not exists)
        if self.bonus_food is None and random.random() < 0.2:
            self._place_bonus_food()
            
    def _place_bonus_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.bonus_food = Point(x, y)
        if self.bonus_food in self.snake or self.bonus_food in self.obstacles or self.bonus_food == self.food:
            self._place_bonus_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
    def _get_closest_food_dist(self):
        dist_normal = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        if self.bonus_food:
            dist_bonus = abs(self.head.x - self.bonus_food.x) + abs(self.head.y - self.bonus_food.y)
            return min(dist_normal, dist_bonus)
        return dist_normal

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
        # 2. move
        # Track distance to CLOSEST food (Normal or Bonus)
        dist_before = self._get_closest_food_dist()
        
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Base Time Penalty (C)
        reward = -0.01
        
        # 3. check if game over
        game_over = False
        # Increased frame limit to 150 to allow for detours around obstacles
        if self.is_collision() or self.frame_iteration > 150*len(self.snake):
            game_over = True
            reward = -1 # Basic Die (A)
            return reward, game_over, self.score
            
        # 4. Check for Eating Data
        # Normal Food
        if self.head == self.food:
            self.score += 1
            reward = 1 # Basic Eat Food (A)
            self._place_food()
        # Bonus Food
        elif self.bonus_food is not None and self.head == self.bonus_food:
            self.score += 3
            reward = 3 # Scaled Bonus
            self.bonus_food = None
        else:
            self.snake.pop()
            
            # Distance-based shaping (B)
            dist_after = self._get_closest_food_dist()
            
            if dist_after < dist_before:
                reward += 0.1 # Closer
            else:
                reward += -0.05 # Further
                
            # Area / Free-space rewards (D)
            # Simple heuristic: Reward if head has > 2 empty neighbors (not walls/body/obstacles)
            # This encourages staying in open space
            safe_neighbors = 0
            check_dirs = [Point(0, 20), Point(0, -20), Point(20, 0), Point(-20, 0)]
            for d in check_dirs:
                neighbor = Point(self.head.x + d.x, self.head.y + d.y)
                if not self.is_collision(neighbor):
                    safe_neighbors += 1
            
            if safe_neighbors >= 3:
                reward += 0.005 # Small reward for being in open space
            
        self.total_reward += reward
        
        # 5. update ui and clock
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        # hits obstacles
        if pt in self.obstacles:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        # Draw Normal Food (Red)
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw Bonus Food (Yellow)
        if self.bonus_food is not None:
             pygame.draw.rect(self.display, (255, 255, 0), pygame.Rect(self.bonus_food.x, self.bonus_food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        for pt in self.obstacles:
            pygame.draw.rect(self.display, (128, 128, 128), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

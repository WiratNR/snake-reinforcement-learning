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
            self.clock = pygame.time.Clock()
        self.current_level = 'random'
        self.reset()
        
    def set_level(self, level_name):
        self.current_level = level_name
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y)]
        
        self.score = 0
        self.total_reward = 0
        self.food = None
        self.food = None
        self.bonus_food = None
        self.bonus_timer = 0
        self.obstacles = []
        self._place_obstacles()
        self._place_food()
        self.frame_iteration = 0
        
    def _place_obstacles(self):
        # Use external LevelManager
        raw_obstacles = LevelManager.get_level(self.current_level, self.w, self.h)
        self.obstacles = []
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

        # 2. Try to place Bonus Food (30% chance if not exists)
        if self.bonus_food is None and random.random() < 0.3:
            self._place_bonus_food()
            
    def _place_bonus_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.bonus_food = Point(x, y)
        self.bonus_timer = 150 # Bonus lasts long enough for planned routes
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

        return reward, game_over, self.score
    
    def _is_approaching_wall(self, head, action):
        # A simple check: if we are close to a wall and moving towards it
        # This is a bit complex to do perfectly without raycasting here, but we can do a basic check
        w, h = self.w, self.h
        
        # Current head position
        x, y = head.x, head.y
        
        # Predict next position based on action? 
        # Actually this is called AFTER move in original code, but we want to know if the move *was* approaching.
        # But wait, we are strictly modifying `play_step`.
        
        # Let's check if we are simply "close to wall".
        # Margin
        margin = 2 * BLOCK_SIZE
        near_left = x < margin
        near_right = x > w - margin
        near_up = y < margin
        near_down = y > h - margin
        
        # If we are near a wall, and we just moved towards it... 
        # This state is hard to deduce just from "action" without strictly parsing direction history.
        # Simpler: Just penalize if very close to wall? No, that punishes just being near content.
        # User said: "-1 when entering wall proximity" roughly.
        # Let's stick to "Distance to food" logic mostly, and maybe penalty for moving AWAY from food which is often moving towards wall in corners.
        return False # Placeholder if needed, but I'll implement logic inline

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
        
        # 3. check if game over
        reward = 0
        game_over = False
        
        # Increased frame limit to 100*len (standard) or keep 150
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10  # REDUCED: Less harsh death penalty
            return reward, game_over, self.score
            
        # 3.5 Check Bonus Expiration
        if self.bonus_food:
            self.bonus_timer -= 1
            if self.bonus_timer <= 0:
                self.bonus_food = None
                
        # 4. Check for Eating Data
        # Normal Food
        if self.head == self.food:
            self.score += 1
            reward = 10  # Eat Food reward
            self._place_food()
        # Bonus Food
        elif self.bonus_food is not None and self.head == self.bonus_food:
            self.score += 5
            reward = 20  # INCREASED: Better bonus reward
            self.bonus_food = None
        else:
            self.snake.pop()
            
            # BALANCED: Distance-based shaping
            dist_after = self._get_closest_food_dist()
            
            if dist_after < dist_before:
                reward = 1  # INCREASED: Reward for getting closer
            else:
                reward = -1  # SAME: Small penalty for moving away
                
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

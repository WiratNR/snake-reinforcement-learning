import random
from collections import namedtuple

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20

class LevelManager:
    
    LEVELS = ['random', 'box', 'cross', 'border', 'empty']

    @staticmethod
    def get_level(name, w, h):
        if name == 'random': return LevelManager.random_obstacles(w, h)
        if name == 'box': return LevelManager.box_obstacles(w, h)
        if name == 'cross': return LevelManager.cross_obstacles(w, h)
        if name == 'border': return LevelManager.border_obstacles(w, h)
        if name == 'empty': return []
        return [] # Default empty

    @staticmethod
    def get_random_level(w, h):
        """Randomly selects a level pattern"""
        name = random.choice(LevelManager.LEVELS)
        return LevelManager.get_level(name, w, h)

    @staticmethod
    def random_obstacles(w, h):
        """Old logic: 1-3 random blocks"""
        obstacles = []
        num_obstacles = random.randint(1, 3)
        for _ in range(num_obstacles):
            x = random.randint(0, (w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
            y = random.randint(0, (h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            obstacles.append(Point(x, y))
        return obstacles

    @staticmethod
    def box_obstacles(w, h):
        """A box shape in the middle"""
        obstacles = []
        center_x, center_y = w // 2, h // 2
        
        # 4 blocks around center
        # Left wall
        obstacles.append(Point(center_x - 3*BLOCK_SIZE, center_y - BLOCK_SIZE))
        obstacles.append(Point(center_x - 3*BLOCK_SIZE, center_y))
        obstacles.append(Point(center_x - 3*BLOCK_SIZE, center_y + BLOCK_SIZE))
        
        # Right wall
        obstacles.append(Point(center_x + 3*BLOCK_SIZE, center_y - BLOCK_SIZE))
        obstacles.append(Point(center_x + 3*BLOCK_SIZE, center_y))
        obstacles.append(Point(center_x + 3*BLOCK_SIZE, center_y + BLOCK_SIZE))
        
        return obstacles

    @staticmethod
    def cross_obstacles(w, h):
        """Cross shape"""
        obstacles = []
        center_x, center_y = w // 2, h // 2
        
        # Horizontal line
        for i in range(-2, 3):
            if abs(i) <= 1:
                continue # Leave spawn and adjacent escape cells open
            obstacles.append(Point(center_x + i*BLOCK_SIZE, center_y))
            
        # Vertical line
        for i in range(-2, 3):
            if abs(i) <= 1:
                continue
            obstacles.append(Point(center_x, center_y + i*BLOCK_SIZE))
            
        return obstacles

    @staticmethod
    def border_obstacles(w, h):
        """Random blocks near the edges"""
        obstacles = []
        # Place 4 blocks near corners but not exactly in corner
        margin = 3 * BLOCK_SIZE
        
        obstacles.append(Point(margin, margin))
        obstacles.append(Point(w - margin, margin))
        obstacles.append(Point(margin, h - margin))
        obstacles.append(Point(w - margin, h - margin))
        
        return obstacles

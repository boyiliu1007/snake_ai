import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = [0, 100, 255]
GREEN = (0, 255, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10
BOARDWIDTH = 240
BOARDHEIGHT = 240
BOARDGRIDS = BOARDWIDTH * BOARDHEIGHT // (BLOCK_SIZE**2)

#reset
#reward
#play(action) -> direction
#game_iteration
#is_collision

class SnakeGameAI:
    def __init__(self, w=BOARDWIDTH, h=BOARDHEIGHT):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.snakeBoard = np.zeros((BOARDHEIGHT//BLOCK_SIZE, BOARDWIDTH//BLOCK_SIZE))
        self.reset()
        
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.prev_direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.prev_head = self.head
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.hasnt_eat = 0
        # print(self.snakeBoard.shape, self.snakeBoard)
        for i in range(len(self.snake)):
            if i == 0:
                # set head to be 512
                self.snakeBoard[int(self.snake[i].y//BLOCK_SIZE)][int(self.snake[i].x//BLOCK_SIZE)] = 512
            else:
                # set body to be 256 - i
                self.snakeBoard[int(self.snake[i].y//BLOCK_SIZE)][int(self.snake[i].x//BLOCK_SIZE)] = 256 - i

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        self.snakeBoard[int(y//BLOCK_SIZE)][int(x//BLOCK_SIZE)] = 1024
        if self.food in self.snake:
            self.snakeBoard[int(y//BLOCK_SIZE)][int(x//BLOCK_SIZE)] = 0
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        
        # 2. move
        reward = 0
        self._move(action) # update the head
        if(self.prev_direction != self.direction and len(self.snake) < 5):
            self.prev_direction = self.direction
            reward -= 2

        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            reward -= 210 / len(self.snake)
            return reward, game_over, self.score
        
        # if snake loops for too long, give penalty
        if self.frame_iteration > 100*len(self.snake) and game_over == False:
            game_over = True
            reward -= 210 / len(self.snake)
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            # eat the food
            reward += (len(self.snake)) * 100
            self.score += 1
            self._place_food()
            self.hasnt_eat = 0
        else:
            if self.hasnt_eat > self.h * self.w / (BLOCK_SIZE**2):
                reward += -30/len(self.snake)
            if np.linalg.norm(np.array(self.head) - np.array(self.food)) < np.linalg.norm(np.array(self.prev_head) - np.array(self.food)):
                reward += 30/len(self.snake)
            else:
                reward += -30/len(self.snake)
            self.snake.pop()
            self.hasnt_eat += 1
        
        # 5. update ui and clock
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
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        currBlue = BLUE.copy()
        drawHeadfFlag = True
        self.snakeBoard = np.zeros((BOARDHEIGHT//BLOCK_SIZE, BOARDWIDTH//BLOCK_SIZE))
        i = 0
        for pt in self.snake:
            if drawHeadfFlag:
                self.snakeBoard[int(pt.y//BLOCK_SIZE)][int(pt.x//BLOCK_SIZE)] = 512
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                drawHeadfFlag = False
            else:
                currBlue[-1] -= 1
                self.snakeBoard[int(pt.y//BLOCK_SIZE)][int(pt.x//BLOCK_SIZE)] = 256 - i
                pygame.draw.rect(self.display, tuple(currBlue), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
            i += 1
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        self.snakeBoard[int(self.food.y//BLOCK_SIZE)][int(self.food.x//BLOCK_SIZE)] = 1024
        # print(self.snakeBoard.shape, self.snakeBoard)
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        clockwise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP
        ]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]

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
            
    def get_snake_board(self):
        return self.snakeBoard

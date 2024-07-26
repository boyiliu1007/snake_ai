import torch as pt
import numpy as np
import random
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE, BOARDGRIDS
from model import Linear_QNet, QTrainer
from helper import plot
import torch.nn.functional as F

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, training, keep_training) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        # print(BOARDGRIDS*3)
        self.model = Linear_QNet(BOARDGRIDS + 11, 1024, 3)
        if not training or keep_training:
            self.model.load()
            # for name, param in self.model.named_parameters():
            #         print(f'Parameter name: {name}')
            #         print(f'Parameter shape: {param.shape}')
            #         print(param)
            #         print('---')

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    
    def get_state(self, game) -> np.ndarray:
        # print(f'game.get_snake_board() shape: {game.get_snake_board().shape}')
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

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

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        snake_board = np.array(game.get_snake_board(), dtype=int)
        # snake_board_tensor = pt.tensor(snake_board, dtype = pt.float32).permute(2, 0, 1).unsqueeze(0)
        # pooled_tensor = F.max_pool2d(snake_board_tensor, kernel_size=2, stride=2)
        # snake_board = pooled_tensor.squeeze(0).permute(1, 2, 0).numpy().flatten()
        if snake_board.max() > 0:
            snake_board = snake_board / snake_board.max()

        state_array = np.array(state, dtype=int) * 10

        # Flatten the snake board and concatenate with the state
        combined_state = np.concatenate((snake_board.flatten(), state_array))
        # print(f'combined_state: {combined_state}')
        return combined_state

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))   


    def train_long_memory(self) -> None:
        # random sampling from memory to break unrelated correlations observed by the model
        if len(self.memory) > BATCH_SIZE:
            print("Training long memory")
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            print(len(self.memory), BATCH_SIZE)
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) -> int:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = pt.tensor(state, dtype=pt.float)
            prediction = self.model(state0)
            move = pt.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train(training=True, keep_training=False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0
    total_reward = 0

    agent = Agent(training=training, keep_training=keep_training)
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        total_reward += reward
        
        state_new = agent.get_state(game)
        
        # train short memory
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            if (not training) and (not keep_training):
                game.reset()
                continue
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record_score and training == True:
                record_score = score
                agent.model.save()
                # for name, param in agent.model.named_parameters():
                #     print(f'Parameter name: {name}')
                #     print(f'Parameter shape: {param.shape}')
                #     print(param)
                #     print('---')
            
            mean_score = total_score / agent.n_games
            print(f'Game {agent.n_games}, Score: {score}, Reward:{total_reward}, Record: {record_score}, Mean: {mean_score}')
            total_reward = 0
            # plot_scores.append(score)
            total_score += score
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train(training=True, keep_training=False)


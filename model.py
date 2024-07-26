import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear2 = nn.Linear(int(hidden_size), output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        pt.save(self.state_dict(), file_name)
    
    def load(self, file_name='model.pth'):
        file_name = os.path.join('./model', file_name)
        self.load_state_dict(pt.load(file_name))

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, game_over):
        state = pt.tensor(np.array(state), dtype=pt.float)
        next_state = pt.tensor(np.array(next_state), dtype=pt.float)
        action = pt.tensor(np.array(action), dtype=pt.long)
        reward = pt.tensor(np.array(reward), dtype=pt.float)
        game_over = pt.tensor(np.array(game_over), dtype=pt.float)

        if len(state.shape) == 1:
            # ex: if shape is [5], then we want it to become [5,1]
            state = pt.unsqueeze(state, 0)
            next_state = pt.unsqueeze(next_state, 0)
            action = pt.unsqueeze(action, 0)
            reward = pt.unsqueeze(reward, 0)
            game_over = (game_over, )
        
        # 1. predicted Q values with current state
        pred = self.model(state)
        # print(pred.shape)
        # print(reward.shape)
        # print("========")
        target = pred.clone()
        
        for idx in range(len(game_over)):
            # print("game_over", game_over)
            Qnew = reward[idx]
            if not game_over[idx]:
                Qnew = reward[idx] + self.gamma * pt.max(self.model(next_state[idx]))

            target[idx][pt.argmax(action).item()] = Qnew

        # 2. Q_new = R + y * max(next_predicted Q value) => only do this if not done
        # pred.clone()
        # print("done")
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


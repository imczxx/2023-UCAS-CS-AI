# Algorithm: Double Q Learning

# TODO: True Q learning
# TODO: save model & load model
# TODO: Dueling DQN
# TODO: Prioritized Experience Replay

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_net = DQNNet(state_size, action_size).to(device)
        self.target_net = DQNNet(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.01)
        self.gamma = 0.95
        self.epsilon = 0.2
        self.replay_size = 250
        self.replay_batchsize = 50
        self.iteration_per_train = 5
        self.replay = ReplayBuffer(self.replay_size)

    def act(self, state, action_mask): # temp for training
        action_mask = np.array(action_mask)
        if random.random() < self.epsilon:
            action = random.choice(np.where(action_mask==1)[0])
            bet = np.random.randint(0, 1000)
        else:
            state = torch.tensor(state).unsqueeze(0).to(device)
            state = state.float()
            q_value, bet = self.q_net.forward(state)
            q_value = q_value.squeeze().detach().cpu().numpy()
            legal_q_values = [q_value[i] for i in np.where(action_mask==1)[0]]
            max_q_value = np.max(legal_q_values)
            max_q_indices = [i for i, q in enumerate(legal_q_values) if q == max_q_value]
            action = random.choice(max_q_indices)
        return action, bet

    def feed(self, transition):
        (state, action, reward, next_state) = transition
        state = np.array(state).astype(np.float32)
        action = np.array(action).astype(np.float32)
        reward = np.array(reward).astype(np.float32)
        next_state = np.array(next_state).astype(np.float32)
        transition = (state, action, reward, next_state)
        self.replay.append(transition)
        if len(self.replay.buffer) == self.replay_size:
            self.train()
    
    def train(self): # to modify
        for i in range(self.iteration_per_train):
            state_batch, action_batch, reward_batch, next_state_batch = self.replay.sample(self.replay_batchsize)

            state_batch = torch.tensor(state_batch).to(device)
            action_batch = torch.tensor(action_batch).to(device)
            bet_batch = action_batch[:, 4].unsqueeze(1)
            action_batch = action_batch[:, :4]
            reward_batch = torch.tensor(reward_batch).to(device)
            next_state_batch = torch.tensor(next_state_batch).to(device)

            q_values, bet = self.q_net.forward(state_batch)
            action_batch = torch.tensor(action_batch, dtype=torch.int64).to(device)
            action_batch = action_batch.argmax(dim=1)
            q_values = q_values.gather(1, action_batch.unsqueeze(1))

            next_q_values, _ = self.target_net(next_state_batch)
            next_q_values.squeeze(0)
            next_q_values = next_q_values.max(1)[0].detach()

            reward_batch.squeeze(0).squeeze(0)
            target_q_values = reward_batch + self.gamma * next_q_values

            loss = nn.MSELoss()(q_values, target_q_values)
            for i in range(self.replay_batchsize):
                if bet_batch[i] != 0:
                    loss += nn.MSELoss()(bet[i], bet_batch[i])

            print(f"loss: {loss}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.replay.buffer.clear()

class DQNNet(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(num_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )
        self.bet_net = nn.Sequential(
            nn.Linear(num_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return self.action_net(state), self.bet_net(state)

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return np.stack(state), action, reward, np.stack(next_state)
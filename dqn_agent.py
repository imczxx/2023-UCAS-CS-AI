# 算法: Double DQN + Dueling DQN + Prioritized Experience Replay

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "dqn_model.pth"
loss_path = "loss.csv"


class DQNAgent:
    def __init__(self, state_size, action_size, mode):
        self.state_size = state_size
        self.action_size = action_size
        self.deep_q_net = DQNNet(state_size, action_size).to(device)
        self.target_net = DQNNet(state_size, action_size).to(device) # Double DQN的目标网络
        self.optimizer = optim.Adam(self.deep_q_net.parameters(), lr=0.01)
        self.gamma = 0.95
        self.epsilon = 0.2
        self.replay_size = 1000
        self.replay_batchsize = 200
        self.iteration_per_train = 5
        self.param_update_interval = 2
        self.train_cnt = 0
        self.replay_buffer = ReplayBuffer(self.replay_size)
        self.mode = mode
        self.target_net.load_state_dict(self.deep_q_net.state_dict())
        if self.mode == "pk":
            self.load(model_path)
        self.losses = []

    def act(self, state, action_mask):
        action_mask = np.array(action_mask)
        # 训练时使用epsilon-greedy策略
        if self.mode == "train" and random.random() < self.epsilon:
            action = random.choice(np.where(action_mask == 1)[0])
        else:
            # 选取Q值最大的合法动作
            state = torch.tensor(state).unsqueeze(0).to(device)
            state = state.float()
            self.deep_q_net.eval()
            q_value = self.deep_q_net.forward(state)
            q_value = q_value.detach().cpu().numpy().squeeze(0)
            legal_indices = np.where(action_mask == 1)[0]
            q_value_legal = [q_value[i] for i in legal_indices]
            max_q_value = np.max(q_value_legal)
            max_q_indices = [
                i for i, q in zip(legal_indices, q_value_legal) if q == max_q_value
            ]
            action = random.choice(max_q_indices)
        return action

    def feed(self, transition):
        (state, action, reward, next_state) = transition
        state = np.array(state).astype(np.float32)
        action = np.array(action).astype(np.float32)
        reward = np.array(reward).astype(np.float32)
        next_state = np.array(next_state).astype(np.float32)
        transition = (state, action, reward, next_state)
        self.replay_buffer.append(transition)
        # replay buffer满后开始训练
        if self.replay_buffer.len() >= self.replay_size:
            self.train()

    def train(self):
        self.deep_q_net.train()
        self.replay_buffer.update_priorities(
            self.deep_q_net, self.target_net, self.gamma
        )
        losses_per_train = []
        for i in range(self.iteration_per_train):
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                probabilities_batch,
            ) = self.replay_buffer.sample(self.replay_batchsize)

            state_batch = torch.tensor(state_batch).to(device)
            action_batch = torch.tensor(action_batch).to(device)
            action_batch = action_batch[:, :4]
            reward_batch = torch.tensor(reward_batch).to(device)
            next_state_batch = torch.tensor(next_state_batch).to(device)

            q_value_batch = self.deep_q_net.forward(state_batch)
            action_batch = torch.tensor(action_batch, dtype=torch.int64).to(device)
            action_batch = action_batch.argmax(dim=1)
            q_value_batch = q_value_batch.gather(1, action_batch.unsqueeze(1))
            q_value_batch = q_value_batch.squeeze()

            next_q_value_batch = self.target_net.forward(next_state_batch)
            next_q_value_batch.squeeze(0)
            next_q_value_batch = next_q_value_batch.max(1)[0].detach()

            reward_batch = reward_batch.squeeze()
            target_q_value_batch = reward_batch + self.gamma * next_q_value_batch

            td_error = abs(target_q_value_batch - q_value_batch)

            probabilities_batch = torch.tensor(probabilities_batch).to(device)
            # Prioritized Experience Replay需要根据样本的抽样概率修正样本的学习率
            # 这里采取对计算得到的误差进行调整，其效果与修正样本的学习率相同
            td_error_pow = torch.pow(td_error, 2)
            td_error_pow *= 1 / (self.replay_size * probabilities_batch)

            loss = td_error_pow.sum()
            losses_per_train.append(loss)

            print(f"loss: {loss}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.replay_buffer.clear()
        self.train_cnt += 1
        self.losses.append(
            np.mean([loss.detach().cpu().numpy() for loss in losses_per_train])
        )
        df = pd.DataFrame({"Loss": self.losses})
        # df.to_csv(loss_path, index=False) # 保存loss数据到文件中，不过loss并没有什么参考价值，还是得看对战结果(

        if self.train_cnt % self.param_update_interval == 0:
            self.target_net.load_state_dict(self.deep_q_net.state_dict())
            self.save(model_path)

    def save(self, path):
        torch.save(self.deep_q_net.state_dict(), path)

    def load(self, path):
        self.deep_q_net.load_state_dict(torch.load(path))


class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # Dueling DQN的优势头
        self.a_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        # Dueling DQN的状态价值头
        self.v_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        a = self.a_net(state)
        v = self.v_net(state)
        # Dueling DQN，根据状态价值头和优势头的结果计算Q值
        return v + a - a.mean()


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.priorities = None

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        # 根据优先级采样
        probabilities = self.priorities / self.priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state = zip(*batch)
        probabilities = probabilities[indices]
        return state, action, reward, next_state, probabilities

    # Prioritized Experience Replay，根据TD误差设置优先级
    def update_priorities(self, deep_q_net, target_net, gamma):
        batch = list(self.buffer)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)
        state_batch = torch.tensor(state_batch).to(device)
        action_batch = torch.tensor(action_batch).to(device)
        action_batch = action_batch[:, :4]
        reward_batch = torch.tensor(reward_batch).to(device)
        next_state_batch = torch.tensor(next_state_batch).to(device)

        q_value_batch = deep_q_net.forward(state_batch)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).to(device)
        action_batch = action_batch.argmax(dim=1)
        q_value_batch = q_value_batch.gather(1, action_batch.unsqueeze(1))
        q_value_batch = q_value_batch.squeeze()

        next_q_value_batch = target_net.forward(next_state_batch)
        next_q_value_batch = next_q_value_batch.max(1)[0].detach()

        reward_batch = reward_batch.squeeze(1)
        target_q_value_batch = reward_batch + gamma * next_q_value_batch

        self.priorities = (
            abs(target_q_value_batch - q_value_batch).detach().cpu().numpy()
        )

    def len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

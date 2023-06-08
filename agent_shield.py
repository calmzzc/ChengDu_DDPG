import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

from model import Actor, Critic
from memory import ReplayBuffer


class DDPG_Shield:
    def __init__(self, state_dim, action_dim, cfg, line):
        self.device = cfg.device
        self.line = line
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # 软更新参数
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0]

    def update(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        # reward = torch.FloatTensor(np.float64(reward)).unsqueeze(1).to(self.device)
        # reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        # done = torch.FloatTensor(np.float64(done)).unsqueeze(1).to(self.device)
        # done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)

        next_action = self.get_safe_next_action(next_state, next_action)

        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))

    def get_safe_next_action(self, next_state, next_action):
        key_list = []
        key = 0
        for j in self.line.speed_limit.keys():
            key_list.append(j)
        for i in range(len(next_state)):
            for k in range(len(key_list) - 1):
                a = next_state[i][3]
                if key_list[k] <= next_state[i][3] < key_list[k + 1]:
                    key = key_list[k]
            limit_speed = self.line.speed_limit[key]
            temp_next_next_state_speed = next_state[i][1] * next_state[i][1] + 2 * next_action[i] * self.line.delta_distance
            if temp_next_next_state_speed < 0:
                temp_next_next_state_speed = 0.01
            next_next_state_speed = math.sqrt(temp_next_next_state_speed)
            if next_next_state_speed > limit_speed / 3.6:
                max_action = (limit_speed / 3.6 * limit_speed / 3.6 - next_state[i][1] * next_state[i][1]) / (2 * self.line.delta_distance)
                if max_action > 1:
                    max_action = 1
                if max_action < -1:
                    max_action = -1
                next_action[i] = max_action
        return next_action

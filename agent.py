#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2021-09-16 00:55:30
@Discription:
@Environment: python 3.7.7
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Actor, Critic
from memory import ReplayBuffer
import multiprocessing
import matplotlib.pyplot as plt


class DDPG:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)

        self.integrate_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.integrate_optimizer = optim.Adam(self.integrate_actor.parameters(), lr=5 * cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # 软更新参数
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # state = state.repeat(2, 1)
        action = self.actor(state)
        return np.round(action.detach().cpu().numpy()[0, 0], 2)

    def eval_choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # state = state.repeat(2, 1)
        action = self.integrate_actor(state)
        return np.round(action.detach().cpu().numpy()[0, 0], 2)

    def update(self):  # update全部要改，现在actor永远输出的都是1
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
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

    def save_int(self, path):
        torch.save(self.integrate_actor.state_dict(), path + 'int_checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))

    def load_int(self, path):
        self.actor.load_state_dict(torch.load(path + 'int_checkpoint.pt'))

    def update_shield(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_node_list = self.memory.sample_new(self.batch_size)
        state_node_tuple = tuple(state_node.state for state_node in state_node_list)
        # origin_action_tuple = tuple(state_node.action for state_node in state_node_list)
        # next_state_node_list = self.memory.sample_new(self.batch_size)
        next_state_node_tuple = tuple(state_node.next_state for state_node in state_node_list)
        # action_state_node_list = self.memory.sample_new(self.batch_size)
        action_state_node_tuple = tuple(state_node.action for state_node in state_node_list)
        # reward_state_node_list = self.memory.sample_new(self.batch_size)
        reward_state_node_tuple = tuple(state_node.current_reward for state_node in state_node_list)
        # done_state_node_list = self.memory.sample_new(self.batch_size)
        done_state_node_tuple = tuple(state_node.done for state_node in state_node_list)

        # 转变为张量
        state = torch.FloatTensor(state_node_tuple).to(self.device)
        next_state = torch.FloatTensor(next_state_node_tuple).to(self.device)
        action = torch.FloatTensor(action_state_node_tuple).to(self.device)
        reward = torch.FloatTensor(reward_state_node_tuple).to(self.device)
        done = torch.FloatTensor(done_state_node_tuple).unsqueeze(1).to(self.device)

        for epoch in range(100):
            origin_action = self.actor(state)
            action_loss = nn.MSELoss()(action, origin_action)
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            self.actor_optimizer.step()
        # plt.plot(origin_action.detach().cpu().numpy())
        # plt.plot(action.detach().cpu().numpy())
        # plt.show()

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()

        origin_next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, origin_next_action.detach())
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

    def approximate(self, node_list, length):
        state_node_list = node_list[-int(length - 1):]
        state_node_tuple = tuple(state_node.state for state_node in state_node_list)
        action_state_node_tuple = tuple(state_node.action for state_node in state_node_list)
        state = torch.FloatTensor(state_node_tuple).to(self.device)
        action = torch.FloatTensor(action_state_node_tuple).to(self.device)
        for epoch in range(50000):
            origin_action = self.integrate_actor(state)
            action_loss = nn.MSELoss()(action, origin_action)
            self.integrate_optimizer.zero_grad()
            action_loss.backward()
            self.integrate_optimizer.step()
        # plt.plot(origin_action.detach().cpu().numpy())
        # plt.plot(action.detach().cpu().numpy())
        # plt.show()
        # for param, integrate_param in zip(self.actor.parameters(), self.integrate_actor.parameters()):
        #     param.data.copy_(integrate_param.data)
        print(action_loss)

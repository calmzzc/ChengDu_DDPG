import numpy as np


class StateNode:
    def __init__(self, state, step, line, agent, episode, ou_noise, train_flag, train_model):
        self.line = line  # 区间
        self.train_model = train_model  # 列车模型
        self.agent = agent  # 神经网络
        self.ou_noise = ou_noise  # OU噪声
        self.state = state  # 当前状态 ,0是速度，1是时间
        self.acc = 0  # 当前加速度
        self.action = np.array(0).reshape(1)  # 当前牵引制动力请求百分比
        self.step = step  # 当前阶段
        self.episode = episode  # 当前幕数
        self.current_reward = 0  # 当前奖励
        self.current_q = 0  # 当前Q值
        self.last_node_state = 0  # 前一个节点的状态
        self.last_node_action = 0  # 前一个节点的动作
        self.last_node_acc = 0  # 前一个节点的加速度
        self.next_state = np.zeros(2)  # 状态转移后的状态
        self.t_power = 0  # 状态转移过程的牵引能耗
        self.re_power = 0  # 状态转移过程产生的再生制动能量
        self.train_flag = train_flag  # 训练测试标志位
        self.comfort_punish = 0  # 舒适度惩罚标志位
        self.speed_punish = 0  # 超速惩罚标志位
        self.ave_v = 0  # 本阶段平均速度

    def get_last_node(self, node_list):  # 获取上一个节点的状态、动作加速度
        if len(node_list) != 1:
            self.last_node_state = node_list[self.step - 1].state
            self.last_node_action = node_list[self.step - 1].action
            self.last_node_acc = node_list[self.step - 1].acc
        else:
            self.last_node_state = np.zeros(2)
            self.last_node_action = np.array(0).reshape(1)
            self.last_node_acc = 0

    def get_action(self):  # 选择动作
        if self.train_flag:
            if self.episode > 100:
                self.action = self.agent.choose_action(self.state)
                self.action = np.array(self.action).reshape(1)
                self.action = self.ou_noise.get_action(self.action, self.step)
            else:
                self.action = np.array(np.random.uniform(-1, 1)).reshape(1)
        else:
            self.action = self.agent.choose_action(self.state)
            self.action = np.array(self.action).reshape(1)

    def reshape_action(self):  # 重整动作
        low_bound = -1
        upper_bound = 1
        # 重整当前动作
        self.action = low_bound + (self.action + 1.0) * 0.5 * (upper_bound - low_bound)
        self.action = np.clip(self.action, low_bound, upper_bound)
        # 重整上一个节点的动作，这句话好像可以不要
        self.last_node_action = low_bound + (self.last_node_action + 1.0) * 0.5 * (upper_bound - low_bound)
        self.last_node_action = np.clip(self.last_node_action, low_bound, upper_bound)

    def get_gradient(self):  # 获取当前位置坡度
        pass

    def get_curve(self):  # 获取当前位置曲率
        pass

    def get_ave_v(self):  # 获取平均速度
        self.ave_v = 0.5 * (self.state[0] + self.last_node_state[0])

    def get_current_tra_acc(self):  # 计算当前牵引加速度
        self.train_model.get_max_traction_force(self.state[0])  # 当前车辆的最大牵引力
        tra_force = self.train_model.max_traction_force * self.action  # 当前输出的牵引力
        pass

    def get_current_b_acc(self):  # 计算当前制动加速度
        self.train_model.get_max_brake_force(self.state[0])
        bra_force = self.train_model.max_brake_force * abs(self.action)
        pass

    def get_current_q(self):  # 获取当前Q值
        self.current_q = self.agent.target_critic(self.state, self.action.detach())

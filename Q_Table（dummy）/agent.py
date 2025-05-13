import numpy as np


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # dimension of action
        self.state_dim = state_dim
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma # 衰减系数
        self.epsilon = 0.9
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        """
        根据 epsilon-greedy 策略选择动作。
        :param state: 当前状态，可能是一个元组 (x, y)
        :return: 动作索引
        """
        # 如果 state 是元组或列表，映射为一维索引
        if isinstance(state, (tuple, list)):
            state_index = state[0] * self.state_dim[1] + state[1]
        else:
            # 如果 state 已经是一维整数索引，直接使用
            state_index = state

        # epsilon-greedy 策略
        self.sample_count += 1
        if np.random.random() < self.epsilon:
            # 随机选择动作
            action = np.random.choice(self.action_dim)
        else:
            # 选择当前状态下 Q 值最大的动作
            action = np.argmax(self.Q_table[state_index])
        return action

    def update(self, state, action, reward, next_state, done):
        """
               更新 Q 表格。
               :param state: 当前状态，格式为 (x, y) 的二维坐标
               :param action: 执行的动作
               :param reward: 动作带来的奖励
               :param next_state: 下一状态，格式为 (x, y) 的二维坐标
               :param done: 是否为终止状态
               """
        # 将二维坐标映射为一维索引
        state_index = state[0] * self.state_dim[1] + state[1]
        next_state_index = next_state[0] * self.state_dim[1] + next_state[1]

        if done:
            # 如果是终止状态，目标 Q 值为即时奖励
            target_q = reward
        else:
            # Q-learning 更新公式
            target_q = reward + self.gamma * np.max(self.Q_table[next_state_index])

        # 更新 Q 表格
        self.Q_table[state_index, action] += self.lr * (target_q - self.Q_table[state_index, action])
    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")

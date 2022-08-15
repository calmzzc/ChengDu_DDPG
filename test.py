from line import Section
from train_model import Train
from StateNode import StateNode
import numpy as np

section = "Section1"
line = Section[section]
train = Train()
state = np.zeros(2)
state[0] = np.array(0).reshape(1)
state[1] = np.array(10).reshape(1)
state_node = StateNode(state, 5, line, 0, 0, 0, 0, train)
for i in range(50):
    state_node.action = np.random.uniform(-1, -0.5)
    state_node.get_acc()
    state_node.next_state[1] = state_node.state[1] + state_node.acc * 0.5
    state_node.next_state[0] = state_node.state[0] + 0.5
    state_node.get_power()

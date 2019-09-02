import os
import torch as T
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class DDQN(nn.Module):
    def __init__(self, alpha, n_actions, chkpt_name, input_dims, chkpt_dir='tmp/ddqn'):
        super(DDQN, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        # DDQN calculates value and advanatges both
        self.Value = nn.Linear(128,1)
        self.Advanatge = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, chkpt_name+'dueling_dqn')

    def forward(self, state):
        l1 = self.relu(self.fc1(state))
        l2 = self.relu(self.fc2(l1))
        Val = self.Value(l2)
        Adv = self.Advanatge(l2)

        return Val, Adv

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkpt_file))


class Agent(object):
    def __init__(self, gamma, epsilon, alpha, n_actions, input_dims, mem_size, batch_size, eps_min = 0.01,
                 eps_dec=5e-7, replace=1000, chkpt_dir='tmp/ddqn'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.replace_target_count = replace

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DDQN(alpha, n_actions, input_dims=input_dims, chkpt_name='q_eval',
                           chkpt_dir=chkpt_dir)
        self.q_next = DDQN(alpha, n_actions, input_dims=input_dims, chkpt_name='q_next',
                           chkpt_dir=chkpt_dir)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):

        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis, :]
            state = T.tensor(observation).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):

        if (self.replace_target_count is not None and
            self.learn_step_counter%self.replace_target_count==0):
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.epsilon

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state).to(self.q_eval.device)
        new_state = T.tensor(new_state).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        reward = T.tensor(reward).to(self.q_eval.device)
        done = T.tensor(done).to(self.q_eval.device)

        V_s, A_s = self.q_eval.forward(state)
        V_s_, A_s_ = self.q_next.forward(state)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1,
                                            keepdim=True))).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q_next = T.add(V_s_ ,(A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_target = reward + self.gamma * T.max(q_next, dim=1)[0].detach()
        q_target[done] = 0.0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()








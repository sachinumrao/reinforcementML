# Import dependencies
import torch as T
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np

# Deep-Q network implementation : Neural Network as function approximator 
# for value function
class DeepQModel(nn.Module):
    def __init__(self, alpha):
        super(DeepQModel, self).__init__()

        # Convolutional layers for model
        self.conv1 = nn.Conv2D(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2D(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2D(64, 128, 3)

        # Fully connected layers for model
        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, 6)

        # Define optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        # Check for compute devices
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # Put model onto compute device
        self.to(self.device)

    def forward(self, obs):
        # Convert observation (image) to tensor
        obs = T.Tensor(obs).to(self.device)

        # Reshape image
        obs = obs.view(-1, 1, 185, 195)

        # Pass observation through convolutional layers
        out = F.relu(self.conv1(obs))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        # Reshape (flatten) the convolutional output before passing it to 
        # fully connected layers
        out = out.view(-1, 128*19*8)

        # Pass through fully connected layers
        out = F.relu(self.fc1(out))
        actions = self.fc2(out)

        return actions


# Implement RL agent
class Agent(object):
    def __init__(self, gamma, eps, alpha, max_mem, eps_end=0.05,
                    replace=10000, action_space=[0, 1, 2, 3, 4, 5]):

        self.gamma = gamma
        self.eps = eps
        self.eps_end = eps_end
        self.action_space = action_space
        self.mem_size = max_mem
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.mem_counter = 0
        self.replace_target_count = replace
        self.Q_eval = DeepQModel(alpha)
        self.Q_next = DeepQModel(alpha)

    def store_transition(self, state, action, reward, state_):
        if self.mem_counter < self.mem_size:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.mem_counter % self.mem_size] = [state,
                action, reward, state_]

        self.mem_counter += 1

    def choose_action(self, obs):
        rand = np.random.random()
        actions = self.Q_eval.forward(obs)

        if rand < 1 - self.eps:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space)

        self.steps += 1

        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()

        # Copy Q_Eval to Q_Next 
        if self.replace_target_count is not None and \
            self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # Select data from memory 
        if self.mem_counter + batch_size < self.mem_size:
            mem_start = int(np.random.choice(range(self.mem_counter)))
        else:
            mem_start = int(np.random.choice(range(self.mem_counter - \
                batch_size - 1)))

        mini_batch = self.memory(mem_start:mem_start+batch_size)
        memory = np.array(mini_batch)

        # Q-Learning algorithm
        q_pred = self.Q_eval.forward(
            list(memory[:,0][:])).to(self.Q_eval.device)

        q_next = self.Q_next.forward(
            list(memory[:,3][:])).to(self.Q_eval.device)

        max_action = T.argmax(q_next, dim=1).to(self.Q_eval.device)
        reward = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)

        q_target = q_pred
        q_target[:, max_action] = reward + self.gamma * T.max(q_next[1])

        # Reduce eps (exploration factor)
        if self.steps > 500:
            if self.eps - 1e-4 > self.eps_end:
                self.eps -= 1e-4
            else:
                self.eps = self.eps_end

        # Calculate loss
        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        
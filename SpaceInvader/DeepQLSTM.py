# Import dependencies
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Deep-Q network implementation : Neural Network as function approximator
# for value function


class DeepQModel(nn.Module):
    def __init__(self, alpha, batch_size, seq_len):
        super(DeepQModel, self).__init__()

        # Define params
        self.batch_size = batch_size
        self.seq_len = seq_len

        # N = batch_size + seq_len - 1

        # Convolutional layers for model
        self.conv1 = nn.Conv2d(3, 32, 3)
        # Pool layer
        self.pool = nn.MaxPool2d(2, 2)
        # design of conv layer 2
        self.conv2 = nn.Conv2d(32, 64, 3)
        # design of conv layer 3
        self.conv3 = nn.Conv2d(64, 128, 3)

        # GRU layers
        self.input_dim = 128*21*10
        self.hidden_dim = 128
        self.n_layers = 1

        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          self.n_layers, batch_first=True)

        # Fully connected layers for model
        self.fc1 = nn.Linear(self.hidden_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 6)

        # Define optimizer and loss

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        # Check for compute devices
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Put model onto compute device
        self.to(self.device)

    def forward(self, obs, hidden):
        # Convert observation (image) to tensor
        self.batch_size = obs.shape[0]
        obs = T.Tensor(obs).to(self.device)

        # Reshape image
        obs = obs.view(-1, 3, 185, 95)

        # Pass observation through convolutional layers
        out = self.pool(F.relu(self.conv1(obs)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))

        # Reshape (flatten) the convolutional output before passing it to
        # fully connected layers
        out = out.view(-1, 128*21*10)

        # Unroll tensor to convert output into sequence form
        out = self.unroll_tensor(out)

        # Pass through fully connected layers
        out, hidden = self.gru(out, hidden)
        out = out[:, -1, :]

        # Pass through fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        actions = self.fc4(out)
        return actions, hidden

    def init_hidden(self, batch_size):
        batch_size = batch_size - 9
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size,
                            self.hidden_dim).zero_().to(self.device)

        return hidden

    def unroll_tensor(self, obs):
        window = self.seq_len
        step = 1
        new_obs = obs.unfold(0, window, step)
        new_obs = new_obs.permute(0, 2, 1)
        return new_obs


# Implement RL agent
class Agent(object):
    def __init__(self, img_counts, batch_size, seq_len, gamma, eps, 
                 alpha, max_mem, eps_end=0.05,
                 replace=10000, action_space=[0, 1, 2, 3, 4, 5]):

        self.img_counts = img_counts
        self.batch_size = batch_size
        self.seq_len = seq_len
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
        self.Q_eval = DeepQModel(alpha, self.batch_size, self.seq_len)
        self.Q_next = DeepQModel(alpha, self.batch_size, self.seq_len)

    def store_transition(self, state, action, reward, state_):
        if self.mem_counter < self.mem_size:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.mem_counter % self.mem_size] = \
                [state, action, reward, state_]

        self.mem_counter += 1

    def choose_action(self, obs):
        rand = np.random.random()
        with T.no_grad():
            obs = T.Tensor(obs).to(self.Q_eval.device)
            hidden = self.Q_eval.init_hidden(obs.shape[0])
            actions, _ = self.Q_eval(obs, hidden)

        if rand < 1 - self.eps:
            action = actions.argmax(1).item()
        else:
            action = np.random.choice(self.action_space)

        self.steps += 1
        return action

    def learn(self):
        self.Q_eval.optimizer.zero_grad()

        # Copy Q_Eval to Q_Next
        if self.replace_target_count is not None and \
                self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # Iterate over to find mini_batch data
        for i in range(0, self.mem_counter - self.img_counts,):
            mini_batch = self.memory[i:i+self.img_counts]

        # Q-Learning algorithm

        # Extract input data "state" and "state_" from mini batch data
        inp_data = [i[0] for i in mini_batch]
        next_inp_data = [i[3] for i in mini_batch]

        # Convert state and state_ to torch tensors
        inp_data = T.from_numpy(np.array(inp_data)).float()
        next_inp_data = T.from_numpy(np.array(next_inp_data)).float()

        # Pass find q-values for state and state_
        hidden = self.Q_eval.init_hidden(inp_data.shape[0])

        q_pred, _ = self.Q_eval(inp_data, hidden)
        q_next, _ = self.Q_next(next_inp_data, hidden)

        # print("next input data shape: ", next_inp_data.shape)
        # print("next output data shape", q_next.shape)

        # Select the max action
        max_action = T.argmax(q_next, dim=1).to(self.Q_eval.device)

        # Extract reward from batch_data and convert to torch tensor
        n = self.img_counts - self.seq_len + 1
        real_batch = mini_batch[-n:]

        reward = [i[2] for i in real_batch]
        reward = T.Tensor(reward).to(self.Q_eval.device)

        q_target = q_pred

        # print(q_pred.shape)
        # print(q_next.shape)
        q_target[:, max_action] = reward + self.gamma * T.max(q_next)

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

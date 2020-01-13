import numpy as np
import gym

class Agent:

    def __init__(self, num_states, num_actions, eps, gamma, alpha, dynaq_steps=0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = eps
        self.dynaq_steps = dynaq_steps
        self.gamma = gamma
        self.alpha = alpha

        # specific for state discretization
        self.INTMAX = 10000
        self.state_low = [-2.4, -self.INTMAX, -0.2094, -self.INTMAX]
        self.state_high = [2.4, self.INTMAX, 0.2094, self.INTMAX]

        self.bin_size = [10, 11, 10, 11]

        self.step_sizes = [0.48, 1, 0.04188, 1]

        # initialize Q table
        self.Q = np.zeros(self.num_states, self.num_actions)

    def state_to_id(self, s):
        
        bin1 = (s[0] - self.state_low[0])//self.step_sizes[0]

        bin3 = (s[2] - self.state_low[2])//self.step_sizes[2]

        if s[1] < -5:
            bin2 = 0
        elif s[1] > 5:
            bin2 = 11
        else:
            bin2 = (s[1] - (-5))//self.step_sizes[1]

        if s[3] < -5:
            bin4 = 0
        elif s[3] > 5:
            bin4 = 11
        else:
            bin4 = (s[3] - (-5))//self.step_sizes[3]

        sid =  

        return sid

    def reset_epsiode_history(self):
        self.curr_state_id = []
        self.curr_reward = []
        self.curr_action = []
        self.next_state_id = []

    def simulate(self, env, curr_s):

        curr_sid = self.state_to_id(curr_s)

        done = False
        while not done:
            self.reset_epsiode_history()
            curr_a = self.sample_action(curr_sid)
            next_s, curr_r, done, info = env.step(curr_a)

            next_sid = self.state_to_id(next_s)
            # store steps
            self.curr_state_id.append(curr_sid)
            self.curr_reward.append(curr_r)
            self.curr_action.append(curr_a)
            self.next_state_id.append(next_sid)

    def sample_action(self, curr_sid):

        # select epsilon greedy
        e = np.random.random()
        if e < self.eps:
            curr_a = np.random.randint(low=0, high=2)
        else:
            curr_a = np.argmax(self.Q[curr_sid,:])

        return curr_a

    def update_Q_table(self):

        for i in range(len(self.curr_state_id)):
            curr_sid = self.curr_state_id[i]
            next_sid = self.next_state_id[i]
            curr_r = self.curr_reward[i]
            curr_a = self.curr_action[i]

            self.Q[curr_sid, curr_a] +=  self.alpha*(curr_r + self.gamma* np.max(self.Q[next_sid]) - self.Q[curr_sid, curr_a])


    def dyna_Q_update(self):

        for _ in range(self.dynaq_steps):
            i = np.random.randint(low=0, high=len(self.next_state_id))
            curr_sid = self.curr_state_id[i]
            next_sid = self.next_state_id[i]
            curr_r = self.curr_reward[i]
            curr_a = self.curr_action[i]

            self.Q[curr_sid, curr_a] +=  self.alpha*(curr_r + self.gamma* np.max(self.Q[next_sid]) - self.Q[curr_sid, curr_a])





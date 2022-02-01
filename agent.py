import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import time

class LogitsNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,hidden)
        self.fc2 = nn.Linear(hidden,output_dim)
        self.relu = nn.ReLU()

    def forward(self,state):

        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ValueNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim = 1, hidden = 64):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(eps, discount, lr_, input_dim, output_dim, env, MC = True):
    Episodes = eps
    gamma = discount
    lr = lr_

    model_logit = LogitsNet(input_dim, output_dim)
    optimizer_logit = Adam(model_logit.parameters(), lr=lr)

    model_value = ValueNet(input_dim)
    optimizer_value = Adam(model_value.parameters(), lr=lr)

    all_rewards = []
    for episode in range(Episodes):
        done = False

        s = env.reset()

        logp = []
        actions = []
        rewards = []
        s_prime = []
        states = [s]

        while not done:
            logits = model_logit(torch.FloatTensor(s))
            pi = Categorical(logits = logits)
            action = pi.sample()
            log_prob = pi.log_prob(action)
            action = action.item()

            s_prime,reward,done,_ = env.step(action)
            logp.append(log_prob)
            rewards.append(reward)

            s = s_prime
            states.append(s)
            if episode % 100 == 0:
                env.render()

        all_rewards.append(np.sum(rewards))
        if episode % 100 == 0:

            print(f'Episode {episode} Score: {np.sum(rewards)}')


        action_net_td = []
        value_net_td = []
        for t in range(len(rewards)):
            G_t = 0
            pw = 0
            if MC:
                for r in rewards[t:]:
                    G_t = G_t + gamma**pw * r
                    pw += 1

                action_net_td.append(G_t - model_value(torch.from_numpy(states[t]).float()))
                value_net_td.append((model_value(torch.from_numpy(states[t]).float()) - G_t)**2)

            else:
                if t != len(rewards) - 2:
                    action_net_td.append(rewards[t] + gamma*model_value(torch.from_numpy(states[t + 1]).float()) - model_value(torch.from_numpy(states[t]).float()))
                    value_net_td.append((model_value(torch.from_numpy(states[t]).float()) - rewards[t] - gamma*model_value(torch.from_numpy(states[t + 1]).float()))**2)
                else:
                    action_net_td.append(rewards[t+1] + gamma*model_value(torch.from_numpy(states[t + 1]).float()) - model_value(torch.from_numpy(states[t+1]).float()))
                    value_net_td.append((model_value(torch.from_numpy(states[t]).float()) - rewards[t] + gamma*model_value(torch.from_numpy(states[t]).float()))**2)


        loss = - torch.stack(logp)@torch.stack(action_net_td)

        optimizer_logit.zero_grad()

        loss.backward()

        optimizer_logit.step()

        value_net_tds = torch.stack(value_net_td).sum()

        optimizer_value.zero_grad()

        value_net_tds.backward()

        optimizer_value.step()
    return all_rewards

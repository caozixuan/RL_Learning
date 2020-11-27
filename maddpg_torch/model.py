import torch
from torch import nn

from memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_1, n_hidden_2):
        super(Actor, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, action_dim), nn.Tanh())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_1, n_hidden_2):
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim + action_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, 1))

    def forward(self, sa):
        sa = sa.reshape(sa.size()[0], sa.size()[1] * sa.size()[2])
        x = self.layer1(sa)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


LR_C = 1e-3
LR_A = 1e-3


def soft_update(net_target, net, tau):
    for target_param, param in zip(net_target.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def get_array(array):
    res = [x for y in array for x in y]
    return res


class DDPGAgent(object):

    def __init__(self, index, memory_size, batch_size, gamma, state_global, action_global, local=False):
        self.memory = Memory(memory_size)
        self.state_dim = state_global[index]
        self.action_dim = action_global[index]
        self.Actor = Actor(self.state_dim, self.action_dim, 100, 100).to(device)
        if not local:
            self.Critic = Critic(sum(state_global), sum(action_global), 100, 100).to(device)
        else:
            self.Critic = Critic(self.state_dim, self.action_dim, 100, 100).to(device)
        self.Actor_target = Actor(self.state_dim, self.action_dim, 100, 100).to(device)
        if not local:
            self.Critic_target = Critic(sum(state_global), sum(action_global), 100, 100).to(device)
        else:
            self.Critic_target = Critic(self.state_dim, self.action_dim, 100, 100).to(device)
        self.critic_train = torch.optim.Adam(self.Critic.parameters(), lr=LR_C)
        self.actor_train = torch.optim.Adam(self.Actor.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.5
        self.local = local

    def act(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor(s)[0].detach()


class MADDPG(object):
    def __init__(self, n, state_global, action_global, gamma, memory_size):
        self.n = n
        self.gamma = gamma
        self.memory = Memory(memory_size)
        self.agents = [DDPGAgent(index, 1600, 400, 0.5, state_global, action_global) for index in range(0, n)]

    def update_agent(self, sample, index):
        observations, actions, rewards, next_obs, dones = sample
        curr_agent = self.agents[index]
        curr_agent.critic_train.zero_grad()
        all_target_actions = []
        for i in range(0, self.n):
            action = curr_agent.Actor_target(next_obs[:, i])
            all_target_actions.append(action)
        test = torch.cat(all_target_actions, dim=0).to(device).reshape(actions.size()[0], actions.size()[1],
                                                                       actions.size()[2])
        target_vf_in = torch.cat((next_obs, test), dim=2)
        target_value = rewards[:, index] + self.gamma * curr_agent.Critic_target(target_vf_in).squeeze(dim=1)
        vf_in = torch.cat((observations, actions), dim=2)
        actual_value = curr_agent.Critic(vf_in).squeeze(dim=1)
        vf_loss = curr_agent.loss_td(actual_value, target_value.detach())

        vf_loss.backward()
        curr_agent.critic_train.step()

        curr_agent.actor_train.zero_grad()
        curr_pol_out = curr_agent.Actor(observations[:, index])
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i in range(0, self.n):
            if i == index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(self.agents[i].Actor(observations[:, i]).detach())
        vf_in = torch.cat((observations,
                           torch.cat(all_pol_acs, dim=0).to(device).reshape(actions.size()[0], actions.size()[1],
                                                                            actions.size()[2])), dim=2)
        pol_loss = -torch.mean(curr_agent.Critic(vf_in))
        pol_loss.backward()
        curr_agent.actor_train.step()

    def update(self, sample):
        for index in range(0, self.n):
            self.update_agent(sample, index)

    def update_all_agents(self):
        for agent in self.agents:
            soft_update(agent.Critic_target, agent.Critic, agent.tau)
            soft_update(agent.Actor_target, agent.Actor, agent.tau)

    def add_data(self, s, a, r, s_, done):
        self.memory.add(s, a, r, s_, done)

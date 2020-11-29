import torch
from torch import nn

from memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Actor被设定为一个三层全连接神经网络，输出为(-1,1)
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


# Critic被设定为一个三层全连接神经网络，输出为一个linear值(这里不使用tanh函数是因为原始的奖励没有取值范围的限制)
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_1, n_hidden_2):
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim + action_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, 1))

    def forward(self, sa):
        sa = sa.reshape(sa.size()[0], sa.size()[1] * sa.size()[2])   # 对于传入的s(环境)和a(动作)要展平
        x = self.layer1(sa)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 梯度下降参数
LR_C = 1e-3
LR_A = 1e-3


# 在DDPG中，训练网络的参数不是直接复制给目标网络的，而是一个软更新的过程，也就是 v_new = (1-tau) * v_old + tau * v_new
def soft_update(net_target, net, tau):
    for target_param, param in zip(net_target.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DDPGAgent(object):

    def __init__(self, index, memory_size, batch_size, gamma, state_global, action_global, local=False):
        self.hidden = 200
        self.memory = Memory(memory_size)
        self.state_dim = state_global[index]
        self.action_dim = action_global[index]
        self.Actor = Actor(self.state_dim, self.action_dim, self.hidden, self.hidden).to(device)
        # local决定是用局部信息还是全局信息，也决定是DDPG算法还是MADDPG算法
        if not local:
            self.Critic = Critic(sum(state_global), sum(action_global), self.hidden, self.hidden).to(device)
        else:
            self.Critic = Critic(self.state_dim, self.action_dim, self.hidden, self.hidden).to(device)
        self.Actor_target = Actor(self.state_dim, self.action_dim, self.hidden, self.hidden).to(device)
        if not local:
            self.Critic_target = Critic(sum(state_global), sum(action_global), self.hidden, self.hidden).to(device)
        else:
            self.Critic_target = Critic(self.state_dim, self.action_dim, self.hidden, self.hidden).to(device)
        self.critic_train = torch.optim.Adam(self.Critic.parameters(), lr=LR_C)
        self.actor_train = torch.optim.Adam(self.Actor.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.5
        self.local = local

    # 输出确定行为
    def act(self, s):
        return self.Actor(s)

    # 输出带噪声的行为
    def act_prob(self, s):
        a = self.Actor(s)
        noise = torch.normal(mean=0.0, std=torch.Tensor(size=([len(a)])).fill_(0.02)).to(device)
        a_noise = a + noise
        return a_noise


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
        # 根据局部观测值输出动作目标网络的动作
        for i in range(0, self.n):
            action = curr_agent.Actor_target(next_obs[:, i])
            all_target_actions.append(action)
        action_target_all = torch.cat(all_target_actions, dim=0).to(device).reshape(actions.size()[0], actions.size()[1],
                                                                       actions.size()[2])
        target_vf_in = torch.cat((next_obs, action_target_all), dim=2)
        # 计算在目标网络下，基于贝尔曼方程得到当前情况的评价
        target_value = rewards[:, index] + self.gamma * curr_agent.Critic_target(target_vf_in).squeeze(dim=1)
        vf_in = torch.cat((observations, actions), dim=2)
        actual_value = curr_agent.Critic(vf_in).squeeze(dim=1)
        # 计算针对Critic的损失函数
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
        # DDPG中针对Actor的损失函数
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

    def save_model(self, episode):
        for i in range(0, self.n):
            model_name_c = "Critic_Agent" + str(i) + "_" + str(episode) + ".pt"
            model_name_a = "Actor_Agent" + str(i) + "_" + str(episode) + ".pt"
            torch.save(self.agents[i].Critic_target, 'model_tag/' + model_name_c)
            torch.save(self.agents[i].Actor_target, 'model_tag/' + model_name_a)

    def load_model(self, episode):
        for i in range(0, self.n):
            model_name_c = "Critic_Agent" + str(i) + "_" + str(episode) + ".pt"
            model_name_a = "Actor_Agent" + str(i) + "_" + str(episode) + ".pt"
            self.agents[i].Critic_target = torch.load("model_tag/" + model_name_c)
            self.agents[i].Critic = torch.load("model_tag/" + model_name_c)
            self.agents[i].Actor_target = torch.load("model_tag/" + model_name_a)
            self.agents[i].Actor = torch.load("model_tag/" + model_name_a)

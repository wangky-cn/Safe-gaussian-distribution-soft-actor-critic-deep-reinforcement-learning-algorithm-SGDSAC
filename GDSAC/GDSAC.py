import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from torch.distributions import Normal
from TrackedTractorENV import TractorEnv
import os
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.layers = nn.ModuleList()
        input_dim = state_dim
        for h in hidden_width:
            self.layers.append(nn.Linear(input_dim, h))
            input_dim = h

        self.mean_layer = nn.Linear(hidden_width[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_width[-1], action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        for layer in self.layers:
            x = F.relu(layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        if deterministic:
            a = mean
        else:
            a = dist.rsample()

        if with_logprob:
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)

        return a, log_pi

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, minlogv, maxlogv):
        super(Critic, self).__init__()

        self.minlogv = minlogv
        self.maxlogv = maxlogv

        self.layers = nn.ModuleList()
        input_dim = state_dim + action_dim
        for h in hidden_width:
            self.layers.append(nn.Linear(input_dim, h))
            input_dim = h
        self.mean = nn.Linear(input_dim, 1)
        self.log_var = nn.Linear(input_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        for layer in self.layers:
            x = F.gelu(layer(x))
        mean = self.mean(x)
        log_var = self.log_var(x)
        log_var = torch.clamp(log_var, self.minlogv, self.maxlogv)
        var = torch.exp(log_var)

        return mean, var

class NoveltyReward:
    def __init__(self, state_size=7, latent_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, state_size),
            nn.Tanh()
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    def train(self, replay_buffer, num_epochs=5, batch_size1=300, batch_size2=60):
        dataset, _, _, _, _, _ = replay_buffer.sample(batch_size1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size2, shuffle=True)

        for epoch in range(num_epochs):
            for data in dataloader:
                states = data[0].to(self.device)
                encoded = self.encoder(states)
                decoded = self.decoder(encoded)
                loss = self.criterion(decoded, states)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    def get_novelty_score(self, state):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            encoded = self.encoder(state)
            decoded = self.decoder(encoded)
            score = 0.01 * self.criterion(decoded, state).item()
        return score
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device):
        self.max_size = int(2e6)
        self.device = device
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.c = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, s, a, r, c, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.c[self.count] = c
        self.s_[self.count] = s_
        self.not_done[self.count] = 1.0 - dw
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(self.device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(self.device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(self.device)
        batch_c = torch.tensor(self.c[index], dtype=torch.float).to(self.device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(self.device)
        not_done = torch.tensor(self.not_done[index], dtype=torch.float).to(self.device)

        return batch_s, batch_a, batch_r, batch_c, batch_s_, not_done

class QP:
    def __init__(self):
        self.amin = -2.5
        self.amax = 2.5
        self.Q = 18 * 3600

        self.xmax = 0.845
        self.Rmax = 0.12
        self.lammax = 0.00005
        self.kmax = 8

        self.xmin = 0.355
        self.Rmin = 0.11
        self.lammin = 10
        self.kmin = 10

    def solve_qp_high(self, x, action, v):
        is_safe = True
        phi = 0.0
        u_opt = action
        f_x = -v / (2 * self.Rmax * self.Q)
        g_x = 1 / (2 * self.Rmax * self.Q)

        if x >= 0.75:
            u_r = np.sqrt(v ** 2 - 4 * self.Rmax * action)
            x_dot = f_x + g_x * u_r
            phi = x ** 2 - self.xmax ** 2 + self.kmax * x_dot

            if phi > 0:
                is_safe = False
                Q = matrix(np.array([2.0]), tc='d')
                p = matrix(np.array([-2.0 * u_r]), tc='d')
                A = matrix(np.array([[2 * x * g_x + self.lammax * self.kmax * g_x], [-1.0], [1.0]]), tc='d')
                b = matrix(np.array(
                    [-2 * x * f_x - self.lammax * self.kmax * f_x - self.lammax * x ** 2 + self.lammax * self.xmax ** 2,
                     0.0, v ** 2]),
                           tc='d')
                sol = solvers.qp(Q, p, A, b)
                u_opt = sol['x'][0]
                u_opt = (v ** 2 - u_opt ** 2) / (4 * self.Rmax)
                # print("Optimal u:", u_opt)
        return u_opt, is_safe, phi

    def solve_qp_low(self, x, action, v):
        is_safe = True
        phi = 0.0
        u_opt = action
        f_x = -v / (2 * self.Rmin * self.Q)
        g_x = 1 / (2 * self.Rmin * self.Q)

        if x <= 0.4:
            u_r = np.sqrt(v ** 2 - 4 * self.Rmin * action)
            x_dot = f_x + g_x * u_r
            phi = self.xmin ** 2 - x ** 2 - self.kmin * x_dot

            if phi > 0:
                is_safe = False
                Q = matrix(np.array([2.0]), tc='d')
                p = matrix(np.array([-2.0 * u_r]), tc='d')
                A = matrix(np.array([[-2 * x * g_x - self.lammin * self.kmin * g_x], [-1.0]]), tc='d')
                b = matrix(np.array(
                    [2 * x * f_x + self.lammin * self.kmin * f_x + self.lammin * x ** 2 - self.lammin * self.xmin ** 2,
                     0.0]), tc='d')
                sol = solvers.qp(Q, p, A, b)
                u_opt = sol['x'][0]
                u_opt = (v ** 2 - u_opt ** 2) / (4 * self.Rmin)
                # print("Optimal u:", u_opt)
        return u_opt, is_safe, phi

    def solve_qp(self, org_a, env):
        a = org_a[0]
        v = env.model.BatV.value
        Pice = min(max(env.model.pENGFBK.value + org_a, 0.0), 35.0) * 1000.0
        Pbat = float(env.model.PwrDemd.value - Pice)
        soc = env.model.SOC.value
        if soc < 0.6:
            u_opt, is_safe, phi = self.solve_qp_low(soc, Pbat, v)
        else:
            u_opt, is_safe, phi = self.solve_qp_high(soc, Pbat, v)
        if not is_safe:
            a = (env.model.PwrDemd.value - u_opt) / 1000.0 - env.model.pENGFBK.value
            a = min(max(a, -2.5), 2.5)

        return a, u_opt, is_safe, phi

class Logger(object):
    def __init__(self, log_dir='./log', frequency=500):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.frequency = frequency
        self.steps = {
            'critic': 0,
            'actor': 0,
            'episode': 0
        }
    def _log_scalars(self, scalars, prefix, step_type):
        for tag, value in scalars.items():
            self.writer.add_scalar(f'{prefix}/{tag}', value, self.steps[step_type])
        self.steps[step_type] += 1
    def log_critic(self, loss):
        scalars = {
            'reward critic1': loss[0],
            'reward critic1 Q': loss[1],
            'reward critic1 V': loss[2],
            'reward critic2': loss[3],
            'reward critic2 Q': loss[4],
            'reward critic2 V': loss[5],
        }
        self._log_scalars(scalars, 'Critic', 'critic')

    def log_actor(self, loss):
        scalars = {
            'actor': loss[0],
            'alpha': loss[1],
            'alpha value': loss[2],
        }
        self._log_scalars(scalars, 'Actor', 'actor')

    def log_reward(self, reward, cost, SoC, fuel):
        scalars = {
            'reward': reward,
            'cost': cost,
            'SoC': SoC,
            'fuel': fuel
        }
        self._log_scalars(scalars, 'Episode', 'episode')

    def close(self):
        self.writer.close()

class LoggerVisualizer:
    def __init__(self, log_dir='./log', output_excel='training_logs.xlsx'):
        self.log_dir = log_dir
        self.output_excel = output_excel

    def load_event_data(self):
        # 加载 TensorBoard 事件文件
        event_data = event_accumulator.EventAccumulator(self.log_dir)
        event_data.Reload()

        return event_data

    def save_to_excel(self):
        event_data = self.load_event_data()

        # 创建三个数据帧分别保存 critic, actor, reward 相关数据
        critic_df = pd.DataFrame(columns=['Step',
                                          'reward_critic1', 'reward_critic1_Q', 'reward_critic1_V', 'reward_critic2', 'reward_critic2_Q', 'reward_critic2_V'])
        actor_df = pd.DataFrame(columns=['Step', 'actor', 'alpha', 'alpha_value'])
        reward_df = pd.DataFrame(columns=['Step', 'reward', 'cost', 'SoC', 'fuel'])

        # 解析 critic 数据
        critic_tags = [
                       'Critic/reward critic1', 'Critic/reward critic1 Q', 'Critic/reward critic1 V',
                       'Critic/reward critic2', 'Critic/reward critic2 Q', 'Critic/reward critic2 V']

        critic_data = {}
        for tag in critic_tags:
            steps = [e.step for e in event_data.Scalars(tag)]
            values = [e.value for e in event_data.Scalars(tag)]
            critic_data[tag] = values

        critic_df['Step'] = steps
        for i, tag in enumerate(critic_tags):
            # 移除空格并替换为下划线
            critic_df[tag.split('/')[-1].replace(' ', '_')] = critic_data[tag]

        # 解析 actor 数据
        actor_tags = ['Actor/actor', 'Actor/alpha', 'Actor/alpha value']

        actor_data = {}
        for tag in actor_tags:
            steps = [e.step for e in event_data.Scalars(tag)]
            values = [e.value for e in event_data.Scalars(tag)]
            actor_data[tag] = values

        actor_df['Step'] = steps
        for i, tag in enumerate(actor_tags):
            actor_df[tag.split('/')[-1].replace(' ', '_')] = actor_data[tag]

        # 解析 reward 数据
        reward_tags = ['Episode/reward', 'Episode/cost', 'Episode/SoC', 'Episode/fuel']

        reward_data = {}
        for tag in reward_tags:
            steps = [e.step for e in event_data.Scalars(tag)]
            values = [e.value for e in event_data.Scalars(tag)]
            reward_data[tag] = values

        reward_df['Step'] = steps
        for i, tag in enumerate(reward_tags):
            reward_df[tag.split('/')[-1].replace(' ', '_')] = reward_data[tag]

        # 保存数据到 Excel
        with pd.ExcelWriter(self.output_excel) as writer:
            critic_df.to_excel(writer, sheet_name='Critic', index=False)
            actor_df.to_excel(writer, sheet_name='Actor', index=False)
            reward_df.to_excel(writer, sheet_name='Reward', index=False)

class SDSAC(object):
    def __init__(self, state_dim, action_dim, max_action, device, Logger, use_actor_tag=False, use_current_estimation=False,
                 Qcritic_3sig=True, hidden_width=[400,300,200], batch_size=120, GAMMA=0.99, TAU = 0.005, alphalr = 5e-5,
                 qlr=5e-5, alr=5e-5, clr=5e-5, tau_b=0.1, policy_freq=2, ubc_fac=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.logger = Logger
        self.use_actor_tag = use_actor_tag
        self.use_current_estimation = use_current_estimation

        fa = "TA" if self.use_actor_tag else "CA"
        fq = "TQ" if self.use_current_estimation else "CQ"
        print(f"{fa}{fq}")

        self.Qcritic_3sig = Qcritic_3sig
        self.hidden_width = hidden_width
        self.batch_size = batch_size
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.alphalr = alphalr
        self.qlr = qlr
        self.alr = alr
        self.clr = clr
        self.tau_b = tau_b
        self.policy_freq = policy_freq
        self.ubc_fac = ubc_fac
        self.var_max = np.exp(10)
        self.var_min = np.exp(-8)

        self.init_mean_std()
        self.init_alpha_optimizers()
        self.init_networks()

        self.learn_times = 0
        self.alpha_value = self.alpha

    def init_mean_std(self):
        self.mean_std0 = -1
        self.mean_std1 = -1

    def init_alpha_optimizers(self):
        self.target_entropy = -self.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alphalr)

    def init_networks(self):
        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alr)
        print(self.actor)

        self.reward_critic1 = Critic(state_dim, action_dim, self.hidden_width, -8, 10).to(self.device)
        self.reward_critic_target1 = copy.deepcopy(self.reward_critic1).to(self.device)
        self.reward_critic_optimizer1 = torch.optim.Adam(self.reward_critic1.parameters(), lr=self.qlr)
        print(self.reward_critic1)

        self.reward_critic2 = Critic(state_dim, action_dim, self.hidden_width, -8, 10).to(self.device)
        self.reward_critic_target2 = copy.deepcopy(self.reward_critic2).to(self.device)
        self.reward_critic_optimizer2 = torch.optim.Adam(self.reward_critic2.parameters(), lr=self.qlr)
        print(self.reward_critic2)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def choose_action(self, s, deterministic=False):
        s = torch.FloatTensor(s.reshape(1, -1)).to(self.device)
        a, _ = self.actor(s, deterministic, False)
        return a.cpu().data.numpy().flatten()

    def gradient_clipping(self, model_parameters, clip_val):
        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=clip_val, norm_type=2)

    def soft_update_target_network(self, network, target_network):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def get_values(self, batch_s, batch_a, critic):
        mean, var = critic(batch_s, batch_a)
        var = torch.clamp(var, min=self.var_min, max=self.var_max)
        return mean, var

    def update_sliding_std(self, current_QV1, current_QV2):
        self.mean_std0 = self.update_mean(current_QV1, self.mean_std0)
        self.mean_std1 = self.update_mean(current_QV2, self.mean_std1)

    def update_mean(self, current_value, mean_value):
        if mean_value == -1.0:
            return torch.mean(torch.sqrt(current_value.detach()))
        else:
            return (1 - self.tau_b) * mean_value + self.tau_b * torch.mean(torch.sqrt(current_value.detach()))

    def min_var_combination(self,mean1,var1,mean2,var2):
        inv_var1 = 1.0 / var1
        inv_var2 = 1.0 / var2
        total_inv_var = inv_var1 + inv_var2
        weight1 = inv_var1 / total_inv_var
        weight2 = inv_var2 / total_inv_var
        mean = weight1 * mean1 + weight2 * mean2
        var = weight1 ** 2 * var1 + weight2 ** 2 * var2
        return mean, var

    def get_target_reward_values(self,std,batch_r,not_done,target_Q,current_Q,next_Q,next_QV,log_pi_):
        td_bound = 3 * std if self.Qcritic_3sig else None
        difference = torch.clamp(target_Q - current_Q, -td_bound, td_bound) if self.Qcritic_3sig else target_Q - current_Q
        target_Q = (current_Q + difference).detach()
        current_Q = target_Q if self.use_current_estimation else current_Q
        target_QV = (
                (batch_r - not_done * self.alpha.detach() * log_pi_) ** 2
                - current_Q ** 2
                + not_done * 2 * self.GAMMA * (batch_r - self.alpha.detach() * log_pi_) * next_Q
                + not_done * self.GAMMA ** 2 * next_QV
                + not_done * self.GAMMA ** 2 * next_Q ** 2
        )
        target_QV = torch.clamp(target_QV.detach(), min=self.var_min, max=self.var_max)
        return target_Q, target_QV

    def critic_loss(self, batch_s, batch_a, batch_r, batch_s_, not_done):

        current_Q1, current_QV1 = self.get_values(batch_s, batch_a, self.reward_critic1)
        current_Q2, current_QV2 = self.get_values(batch_s, batch_a, self.reward_critic2)

        self.update_sliding_std(current_QV1, current_QV2)

        with torch.no_grad():
            if self.use_actor_tag:
                batch_a_, log_pi_ = self.actor_target(batch_s_)
            else:
                batch_a_, log_pi_ = self.actor(batch_s_)


            next_Q1, next_QV1 = self.get_values(batch_s_, batch_a_, self.reward_critic_target1)
            next_Q2, next_QV2 = self.get_values(batch_s_, batch_a_, self.reward_critic_target2)
            next_Q, next_QV = self.min_var_combination(next_Q1, next_QV1, next_Q2, next_QV2)
            target_Q = batch_r + not_done * self.GAMMA * (next_Q - self.alpha * log_pi_)
            target_Q1, target_QV1 = self.get_target_reward_values(self.mean_std0, batch_r, not_done, target_Q,
                                                                  current_Q1,
                                                                  next_Q, next_QV, log_pi_)
            target_Q2, target_QV2 = self.get_target_reward_values(self.mean_std1, batch_r, not_done, target_Q,
                                                                  current_Q2,
                                                                  next_Q, next_QV, log_pi_)

        qloss1 = F.mse_loss(current_Q1, target_Q1)
        vloss1 = torch.mean((torch.sqrt(current_QV1) - torch.sqrt(target_QV1)) ** 2)
        reward_critic_loss1 = qloss1 + vloss1
        qloss2 = F.mse_loss(current_Q2, target_Q2)
        vloss2 = torch.mean((torch.sqrt(current_QV2) - torch.sqrt(target_QV2)) ** 2)
        reward_critic_loss2 = qloss2 + vloss2

        if self.learn_times % self.logger.frequency == 0:
            log_data = (
                    reward_critic_loss1.item(), qloss1.item(), vloss1.item(),
                    reward_critic_loss2.item(), qloss2.item(), vloss2.item())
            self.logger.log_critic(log_data)

        return reward_critic_loss1, reward_critic_loss2

    def actor_loss(self, batch_s):

        a, log_pi = self.actor(batch_s)

        actor_Q1, actor_QV1 = self.get_values(batch_s, a, self.reward_critic1)
        actor_Q2, actor_QV2 = self.get_values(batch_s, a, self.reward_critic2)
        actor_Q, actor_QV = self.min_var_combination(actor_Q1, actor_QV1, actor_Q2, actor_QV2)
        actor_Qstd = torch.sqrt(actor_QV)
        ubc = actor_Q + self.ubc_fac * actor_Qstd

        actor_loss = self.alpha.detach() * log_pi - ubc
        actor_loss = actor_loss.mean()

        alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()

        if self.learn_times % self.logger.frequency == 0:
            log_data = (actor_loss.item(), alpha_loss.item(), self.alpha_value)
            self.logger.log_actor(log_data)

        return actor_loss, alpha_loss

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, _, batch_s_, not_done = relay_buffer.sample(self.batch_size)

        reward_critic_loss1, reward_critic_loss2 = \
            self.critic_loss(batch_s, batch_a, batch_r, batch_s_, not_done)

        self.reward_critic_optimizer1.zero_grad()
        reward_critic_loss1.backward()
        self.reward_critic_optimizer1.step()

        self.reward_critic_optimizer2.zero_grad()
        reward_critic_loss2.backward()
        self.reward_critic_optimizer2.step()

        if self.learn_times % self.policy_freq == 0:
            for params in self.reward_critic1.parameters():
                params.requires_grad = False
            for params in self.reward_critic2.parameters():
                params.requires_grad = False

            actor_loss, alpha_loss = self.actor_loss(batch_s)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha_value = self.alpha.item()

            for params in self.reward_critic1.parameters():
                params.requires_grad = True
            for params in self.reward_critic2.parameters():
                params.requires_grad = True

            self.soft_update_target_network(self.reward_critic1, self.reward_critic_target1)
            self.soft_update_target_network(self.reward_critic2, self.reward_critic_target2)
            if self.use_actor_tag:
                self.soft_update_target_network(self.actor, self.actor_target)

        self.learn_times += 1


if __name__ == '__main__':
    env = TractorEnv()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger = Logger()
    visualizer = LoggerVisualizer()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = -max_action
    max_episode_steps = env.model.TotalStep

    kwargs = {
        "state_dim": state_dim, # State dimension
        "action_dim": action_dim, # Action dimension
        "max_action": max_action, # Maximum action
        "device": device, # "CPU" or "GPU"
        "Logger": Logger, # logger
        "use_actor_tag": False, # Use the target actor network?
        "use_current_estimation": False, # Use the estimated current mean to calculate the target variance?
        "Qcritic_3sig": True, # Does the reward critic network use the 3 sigma principle?
        "hidden_width": [400,300,200], # Network nodes
        "batch_size": 100, # Batch size
        "GAMMA": 0.99, # Discount factor
        "TAU": 0.005, # Target network update rate
        "alphalr": 5e-5, # Alpha learning rate
        "qlr": 5e-5, # Reward critic learning rate
        "alr": 5e-5, # Actor learning rate
        "tau_b": 0.1, # Sliding sigma update rate
        "policy_freq": 2, # Frequency of delayed policy updates
        "ubc_fac": 0.08, # UBC factor
    }

    agent = SDSAC(**kwargs)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device)
    Novelty = NoveltyReward(state_dim, 5, learning_rate=0.001)
    QP = QP()

    max_train_steps = 1000
    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0
    random_steps = 5

    for t in range(int(max_train_steps)):
        s, done = env.reset(), False
        episode_steps = 0
        episode_reward = 0.0
        episode_cost = 0.0

        while not done:
            episode_steps += 1
            if t <= random_steps:
                org_a = (max_action - min_action) * np.random.random(env.action_space.shape) + min_action
            else:
                org_a = agent.choose_action(s)

            a, u_opt, is_safe, phi = QP.solve_qp(org_a, env)

            s_, r, c, done, _ = env.step(a)

            state_tensor = torch.FloatTensor(s).to(device)
            r_ = Novelty.get_novelty_score(state_tensor)

            if is_safe:
                replay_buffer.add(s, org_a, r+r_, c, s_, float(done))

            s = s_
            episode_reward += r
            episode_cost += c

            if t > random_steps:
                agent.learn(replay_buffer)

            if done:
                Novelty.train(replay_buffer, num_epochs=6, batch_size1=600, batch_size2=100)

                print("episode:", t,
                      "| SoC: %.3f" % s[2],
                      "| reward: %.3f" % episode_reward,
                      "| cost: %.3f" % episode_cost)

                Logger.log_reward(episode_reward, episode_cost, s[2], float(env.model.ComFu.value))
    Logger.close()
    visualizer.save_to_excel()

import os
import Networks
import torch
import numpy as np
import time
import shutil
import torch.nn as nn
import torch.nn.functional as F
import replay_buffer
from torch.autograd import Variable


class PIDAgent:
    def __init__(self, state_dim: int, action_dim: int, action_limits: torch.tensor,device:str = None):
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_limits.to(self.device)
        self.critic, self.critic_optimizer, self.actor = self._build_networks()
        self.noise = OrnsteinUhlenbeckActionNoise(action_dim, mu=0, theta=0.15, sigma=0.2)
        self.replay_buffer = replay_buffer.ReplayBuffer(10000, self.device)

    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim):
            self.state_dim = state_dim
            self.action_dim = action_dim

            self.P=0.1
            self.I=0
            self.D=0.01


        def forward(self, state):
            #action = torch.zeros(state.shape[0],self.action_dim,device=self.device)
            action = -self.P*state[:,2] -self.D*state[:,3]# - self.D*state[:,3]
            
            return action.unsqueeze(1)

    def _build_networks(self):
        actor = self.Actor(self.state_dim, self.action_dim)
        critic = Networks.Critic(self.state_dim, self.action_dim)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001, weight_decay=0.001)
        
        critic.to(self.device)
        return critic, critic_optimizer, actor

    def select_actions(self, observation: torch.tensor, exploration: bool = True, exploration_mask=None):
        # action: tensor (batch_size x action_dim), observation: tensor (batch_size x state_dim)
        action = self.actor.forward(observation).to(self.device)
        if exploration:
            exploration_mask = torch.ones_like(
                action) if exploration_mask is None else exploration_mask
            n_observation = observation.shape[0]
            #move normal noise withing action limits
            noise = self.noise.sample_n(n_observation).to(self.device)* exploration_mask
            #scaled_noise = noise * (self.action_lim[:,1]-self.action_lim[:,0])/2 + (self.action_lim[:,0]+self.action_lim[:,1])/2
            action = action + noise
            
        return action*(self.action_lim[:,1]-self.action_lim[:,0])/2 + (self.action_lim[:,0]+self.action_lim[:,1])/2

    def add_experience(self, state, action, reward, next_state, done):
        # Input: state (tensor), action (tensor), reward (float), next_state (tensor), done (bool)
        action = action.detach()
        normalized_action = (action -(self.action_lim[:,0]+self.action_lim[:,1])/2)/(self.action_lim[:,1]-self.action_lim[:,0])*2
        self.replay_buffer.store({"action": normalized_action, "state": state,
                                 "reward": reward, "next_state": next_state, "done": done})

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        models_dir = './Models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        actor_file = f'{models_dir}/{episode_count}_actor.pt'
        critic_file = f'{models_dir}/{episode_count}_critic.pt'
        torch.save(self.actor_target.state_dict(), actor_file)
        torch.save(self.critic_target.state_dict(), critic_file)
        print('Models saved successfully')

    def learn(self, batch_size=128, gamma=0.8):
        """
        Trains the actor and critic networks
        :param batch_size: batch size for training
        :param gamma: discount factor
        :return:
        """
        if self.replay_buffer.get_total_count() < batch_size:
            return

        memory_dict = self.replay_buffer.sample(batch_size)
        state = memory_dict['state'].detach()
        action = memory_dict['action'].detach()
        reward = memory_dict['reward'].detach()
        next_state = memory_dict['next_state'].detach()
        done = memory_dict['done'].detach()

        # Update critic

        critic_loss = self.critic.loss(state, action, reward, next_state, done, done,
                                         self.actor, self.critic, gamma, self.device)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        

        # Soft update target networks
        #soft_update(self.critic_target, self.critic, 0.001)

    def estimate_Q(self, state, action):
        action = action.detach()
        normalized_action = (action -(self.action_lim[:,0]+self.action_lim[:,1])/2)/(self.action_lim[:,1]-self.action_lim[:,0])*2
        return self.critic(state, normalized_action)

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        actor_file = f'./Models/{episode}_actor.pt'
        critic_file = f'./Models/{episode}_critic.pt'

        if os.path.exists(actor_file) and os.path.exists(critic_file):
            self.actor.load_state_dict(torch.load(actor_file))
            self.critic.load_state_dict(torch.load(critic_file))
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)
            print('Models loaded successfully')
        else:
            print(f"Error: One or both model files do not exist. Could not load models.")

############################################################################################################
# Utility Functions
############################################################################################################


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def tic():
    global tic_start_time
    tic_start_time = time.time()


def toc():
    global tic_start_time
    return time.time() - tic_start_time
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab


class OrnsteinUhlenbeckActionNoise:
    """
    A class for generating Ornstein-Uhlenbeck noise to add exploration
    in the action space of a Reinforcement Learning agent.
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2,batch_size=1):
        """
        Constructor for the Ornstein-Uhlenbeck noise class.

        Args:
            action_dim (int): Dimensionality of the action space.
            mu (float): Mean of the noise process. Default is 0.
            theta (float): A parameter that governs the rate of mean reversion.
                Default is 0.15.
            sigma (float): The standard deviation of the noise process.
                Default is 0.2.
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.bath_size = batch_size
        self.X = torch.ones(self.bath_size,self.action_dim) * self.mu

    def reset(self):
        """
        Reset the Ornstein-Uhlenbeck noise process.
        """
        self.X = torch.ones(self.bath_size,self.action_dim) * self.mu

    def sample(self):
        """
        Sample from the Ornstein-Uhlenbeck noise process.

        Returns:
            ndarray: An array of noise values for the action space.
        """
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * torch.randn(self.action_dim)
        self.X[0] = self.X[0] + dx
        return self.X[0]
      
      
    def sample_n(self,n):
       """
       Sample from the Ornstein-Uhlenbeck noise process.
        Returns:
           ndarray: An array of noise values for the action space.
       """
       dx = self.theta * (self.mu - self.X)
       dx = dx + self.sigma * torch.randn(n,self.action_dim)
       self.X = self.X + dx
       return self.X
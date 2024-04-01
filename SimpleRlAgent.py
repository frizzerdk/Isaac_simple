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

class SimpleRlAgent:
    def __init__(self, state_dim, action_dim,action_limits):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_limits
        self.latest_episode = 0
        self.actor, self.actor_target, self.actor_optimizer, self.critic, self.critic_target, self.critic_optimizer = self._build_networks()
        self.noise = OrnsteinUhlenbeckActionNoise(action_dim)
        self.replay_buffer = replay_buffer.ReplayBuffer(100000, self.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
    def _build_networks(self):
        actor = Networks.Actor(self.state_dim, self.action_dim, self.action_lim)
        actor_taget = Networks.Actor(self.state_dim, self.action_dim, self.action_lim)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
        critic = Networks.Critic(self.state_dim, self.action_dim)
        critic_target = Networks.Critic(self.state_dim, self.action_dim)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
        
        hard_update(actor_taget, actor)
        hard_update(critic_target, critic)
		
        return actor, actor_taget, actor_optimizer, critic, critic_target, critic_optimizer

    def select_actions(self, observation: torch.tensor, exploration: bool = True):

        action = self.actor.forward(observation) # action: tensor (batch_size x action_dim), observation: tensor (batch_size x state_dim)
        if exploration:
            new_action = action + self.noise.sample() * self.action_lim[:,1]
            return new_action
        return action.cpu().data.numpy()

    def add_experience(self, state, action, reward, next_state, done):
        # Input: state (tensor), action (tensor), reward (float), next_state (tensor), done (bool)
        # Output: None
        self.replay_buffer.store({"action": action, "state": state, "reward": reward, "next_state": next_state, "done": done})

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
        torch.save(self.target_actor.state_dict(), actor_file)
        torch.save(self.target_critic.state_dict(), critic_file)
        print('Models saved successfully')

    def learn(self, batch_size, gamma):
        """
        Trains the actor and critic networks
        :param batch_size: batch size for training
        :param gamma: discount factor
        :return:
        """
        if self.replay_buffer.get_total_count() < batch_size:
            return

        memory_dict = self.replay_buffer.sample(batch_size)
        state = memory_dict['state']
        action = memory_dict['action']
        reward = memory_dict['reward']
        next_state = memory_dict['next_state']
        done = memory_dict['done']

        # Update critic
        self.critic_optimizer.zero_grad()
        target_action = self.actor_target.forward(next_state).detach()
        target_value = torch.squeeze(self.critic_target.forward(next_state, target_action).detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        predicted_value = torch.squeeze(self.critic.forward(state, action))
        critic_loss = F.smooth_l1_loss(predicted_value, expected_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic.forward(state, self.actor.forward(state)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        soft_update(self.target_actor, self.actor, 0.001)
        soft_update(self.target_critic, self.critic, 0.001)

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
            hard_update(self.target_actor, self.actor)
            hard_update(self.target_critic, self.critic)
            print('Models loaded successfully')
        else:
            print(f"Error: One or both model files do not exist. Could not load models.")

############################################################################################################
### Utility Functions
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

	def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
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
		self.X = torch.ones(self.action_dim) * self.mu

	def reset(self):
		"""
        Reset the Ornstein-Uhlenbeck noise process.
        """
		self.X = torch.ones(self.action_dim) * self.mu

	def sample(self):
		"""
        Sample from the Ornstein-Uhlenbeck noise process.

        Returns:
            ndarray: An array of noise values for the action space.
        """
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * torch.randn(self.action_dim)
		self.X = self.X + dx
		return self.X
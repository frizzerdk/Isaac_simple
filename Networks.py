import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    print(v)
    return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        # Set the dimensions of the state and action
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.out_dim = 1
        self.register_buffer('state_dim', torch.tensor(state_dim))
        self.register_buffer('action_dim', torch.tensor(action_dim))
        self.register_buffer('out_dim', torch.tensor(1))

        # Layer 1
        # Layer s1 state
        self.fcs1 = nn.Linear(state_dim, 16)
        self.fcs1.activation = nn.LeakyReLU()
        # Layer s2 state 
        #layer 1 dim 
        self.fcs2 = nn.Linear(self.fcs1.out_features, 8)
        self.fcs2.activation = nn.LeakyReLU()
        # Layer a1 action
        self.fca1 = nn.Linear(action_dim, 8)
        self.fca1.activation = nn.LeakyReLU()

        # Layer 2 from state and action
        self.fc2 = nn.Linear(self.fca1.out_features+self.fcs2.out_features, self.fca1.out_features+self.fcs2.out_features)
        self.fc2.activation = nn.LeakyReLU()

        # Output layer
        self.fc3 = nn.Linear(self.fc2.out_features, self.out_dim)
        self.fc3.weight.data.uniform_(-EPS, EPS)

        # Other layers
        self.dropout = nn.Dropout(0.05)

        self.bn1 = nn.BatchNorm1d(self.fcs1.out_features)
        self.bn2 = nn.BatchNorm1d(self.fca1.out_features)
        self.activations = []
        self.td_error = 0

    def forward(self, state, action,save_activations=False):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """

        s1 = self.fcs1(state)
        s1 = self.fcs1.activation(s1)
        if save_activations:
            self.activations.append(s1.clone())
        do_bn = s1.size(0) > 1 and False
        if do_bn:
            s1 = self.bn1(s1)

        s2 = self.fcs2(s1)
        s2 = self.fcs2.activation(s2)
        if save_activations:
            self.activations.append(s2.clone())
        if do_bn:
            s2 = self.bn2(s2)

        a1 = self.fca1(action)
        a1 = self.fca1.activation(a1)
        if save_activations:
            self.activations.append(a1.clone())
        if do_bn:
            a1 = self.bn2(a1)

        x = torch.cat((s2, a1), dim=1)

        x = self.fc2(x)
        x = self.fc2.activation(x)
        if save_activations:
            self.activations.append(x.clone())
        x = self.fc3(x)
        reward = x
        return reward

    def get_activations(self, state, action):
        self.activations = []
        prediction = self.forward(state, action,save_activations=True)
        return prediction,self.activations

    def loss(self, state, action, reward, next_state, done, truncated, target_actor, target_critic, gamma, device,
             l2_reg_coeff=0.0, is_weights=None):
        done = done.to(device)
        truncated = truncated.to(device)
        a2 = target_actor.forward(next_state).detach().to(device)
        next_val = torch.squeeze(target_critic.forward(next_state, a2).detach().to(device))
        next_val = next_val.squeeze(-1).unsqueeze(-1)
        done = done.squeeze(-1).unsqueeze(-1)
        reward = reward.squeeze(-1).unsqueeze(-1)
        y_expected = (reward + gamma * next_val) * (1 - done) + reward * done
        prediction, layer_activations = self.get_activations(state, action)
        y_predicted = prediction.to(device)

        if is_weights is not None:
            is_weights = is_weights.unsqueeze(-1)
            weighted_td_error = is_weights * (y_predicted - y_expected).abs()
            loss_critic = F.smooth_l1_loss(weighted_td_error, torch.zeros_like(weighted_td_error))
        else:
            loss_critic = F.smooth_l1_loss(y_predicted, y_expected)

        # critic_reg = torch.tensor(0.0).to(device)
        # critic_reg.to(device)
        # for activation in layer_activations:
        #     activation = F.relu(activation.abs() - 10, inplace=True).to(device)
        #     critic_reg += torch.norm(activation, 2)
        # critic_reg = l2_reg_coeff * critic_reg
        td_error = (y_predicted - y_expected).abs()
        self.td_error = td_error.detach()

        return loss_critic #+ critic_reg

    def validate(self, state, action, reward,next_state, target_actor, target_critic, gamma):
        """
        returns average loss of critic network over validation set
        :param state: Input state (Torch Variable : [n,batch_size,state_dim])
        :param action: Input Action (Torch Variable : [n,batch_size,action_dim])
        :param reward: Target reward (Torch Variable : [n,batch_size,1])
        :param next_state: Input next state (Torch Variable : [n,batch_size,state_dim])
        :param target_actor: Target actor network (slow/exploit Actor)
        :param target_critic: Target critic network (slow/exploit Critic)
        :param gamma: Discount factor (float)
        :return: average loss (float)
        """
        # Calculate the average loss over the validation set
        average_loss = 0
        samples = 0
        val_batches = zip(state, action, reward,next_state)
        with torch.no_grad():
            for s1_batch, a1_batch, r1_batch, s2_batch in val_batches:
                loss_critic = self.loss(s1_batch, a1_batch, r1_batch, s2_batch, target_actor, target_critic, gamma)
                average_loss += loss_critic.item()
                samples = s1_batch.size(0)
        average_loss = average_loss / samples
        return average_loss


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.action_lim = action_lim
        self.register_buffer('action_dim', torch.tensor(action_dim))
        self.register_buffer('state_dim', torch.tensor(state_dim))
        self.fc1 = nn.Linear(state_dim, 8)
        self.fc1.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, 4)
        self.fc2.activation = nn.LeakyReLU()
        self.fc3 = nn.Linear(self.fc2.out_features, 2)
        self.fc3.activation = nn.LeakyReLU()
        self.fc4 = nn.Linear(self.fc3.out_features, action_dim)
        self.fc4.activation = nn.Tanh()
        self.fc4.weight.data.uniform_(-EPS,EPS)
        self.l2_reg_coeff = 0.1

        self.bn1 = nn.BatchNorm1d(self.fc1.out_features)
        self.bn2 = nn.BatchNorm1d(self.fc2.out_features)
        self.bn3 = nn.BatchNorm1d(self.fc3.out_features)

        self.activations = []
        



    def forward(self, state,save_activations=False):
        do_bn = state.dim() > 1 and False

        # layer 1
        x = self.fc1(state)
        x = self.fc1.activation(x)
        if do_bn:
            x = self.bn1(x)
        if save_activations:
            self.activations.append(x.clone())

        # layer 2
        x = self.fc2(x)
        x = self.fc2.activation(x)
        if do_bn:
            x = self.bn2(x)
        if save_activations:
            self.activations.append(x.clone())

        # layer 3
        x = self.fc3(x)
        x = self.fc3.activation(x)
        if do_bn:
            x = self.bn3(x)
        if save_activations:
            self.activations.append(x.clone())

        # layer 4
        x = self.fc4(x)
        action = self.fc4.activation(x)

        action = action  # b x action_dim * action_lim + action_lim

        return action

    def get_activations(self, state):
        """
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: action and activations of actor network (Torch Variable : [n,action_dim] , list of torch variables)
        """
        self.activations = []
        action = self.forward(state, save_activations=True)
        return action, self.activations

    def loss(self, state,done, critic,device='cpu',l2_reg_coeff=0.0):
        """
        returns loss of actor network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param done: done flag (Torch Variable : [n,1] )
        :param critic: Critic network
        :param device: device to run on
        :param l2_reg_coeff: coefficient for l2 regularization
        """
        # Calculate the loss
        done = done.to(device)
        pred_a1, layer_activations = self.get_activations(state) # predicted action
        pred_a1.to(device)
        loss_actor = -1 * torch.sum(critic.forward(state, pred_a1).to(device)*(1-done)) # calculate based on action predicted by actor
        actor_reg = torch.tensor(0.0).to(device)
        actor_reg.to(device)
        for activation in layer_activations:
            activation=F.relu(activation.abs()-4, inplace=True).to(device)
            actor_reg += torch.norm(activation, 2)
        actor_reg = l2_reg_coeff * actor_reg
        return loss_actor+actor_reg

    def validate(self, state, critic):
        """
        returns average loss of actor network over validation set
        :param state: Input state (Torch Variable : [n,batch_size,state_dim])
        :param action: Input Action (Torch Variable : [n,batch_size,action_dim])
        :param critic: Critic network
        :return: average loss (float)
        """
        # Calculate the average loss over the validation set
        average_loss = 0
        samples = 0
        val_batches = zip(state)
        with torch.no_grad():
            for s1_batch, in val_batches:
                loss_actor = self.loss(s1_batch, critic)
                average_loss += loss_actor.sum().item()
                samples = s1_batch.size(0)
        average_loss = average_loss / samples
        return average_loss



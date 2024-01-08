import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
from itertools import chain

import torch.nn.functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.

        layers = []
        prev_layer_size = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_layer_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            # Update the previous layer size for the next iteration
            prev_layer_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_layer_size, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        return self.model(s)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.epsilon = 0.05 # Exploration rate
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 

        self.model = NeuralNetwork(self.state_dim, self.action_dim + 1, self.hidden_size,
                                   self.hidden_layers, activation='relu').to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        
        if state.shape == (3,):
            state = state.reshape((1, 3))
        mean, action , log_prob = torch.zeros(state.shape[0]), torch.zeros(state.shape[0]), torch.ones(state.shape[0])

        out = self.model(state)
        mean = out[:, 0:self.action_dim]
        log_std = out[:, -1]

        # clamp log_std between LOG_STD_MIN and LOG_STD_MAX
        log_std = self.clamp_log_std(log_std)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean) # + Normal(0.0, lambdaaa).rsample())   # Ensure actions are in range [-1, 1]
            #log_prob = torch.ones(self.action_dim)  # Placeholder for deterministic policy
            log_prob = torch.full((self.action_dim,), float('-inf'))  # Initialize log_prob with -inf
        else:
            normal_dist = Normal(mean, std.reshape(state.shape[0], self.action_dim))
            z = normal_dist.rsample()
            action = torch.tanh(z)
            log_prob = normal_dist.log_prob(z).sum(dim=-1) - (2 * (np.log(2) - z - torch.nn.functional.softplus(-2 * z))).sum(dim=-1).reshape(state.shape[0], self.action_dim)
            # implement epsilon-greedy exploration
            # if np.random.uniform() > self.epsilon:
            #     normal_dist = Normal(mean, std.reshape(state.shape[0], self.action_dim))
            #     z = normal_dist.rsample()
            #     action = torch.tanh(z)
            #     log_prob = normal_dist.log_prob(z).sum(dim=-1) - (2 * (np.log(2) - z - torch.nn.functional.softplus(-2 * z))).sum(dim=-1).reshape(state.shape[0], self.action_dim)
            # else:
            #     # random exploration within the action space
            #     action = torch.tanh(torch.randn_like(mean))
        
        #action = action.view(-1).detach().numpy()

        assert(action.shape == (self.action_dim,) and \
            log_prob.shape == (self.action_dim,)) or (action.shape == (state.shape[0], 1) and \
                                                      log_prob.shape[0], 1), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.q1 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size,
                                   self.hidden_layers, activation='relu').to(self.device)
        self.q2 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size,
                                    self.hidden_layers, activation='relu').to(self.device)
        # Initialize optimizers for each critic network separately
        #self.optimizer1 = optim.Adam(self.model1.parameters(), lr=self.critic_lr)
        #self.optimizer2 = optim.Adam(self.model2.parameters(), lr=self.critic_lr)

        # Define a single optimizer for both critic networks
        self.q_parameters = chain(self.q1.parameters(), self.q2.parameters())
        self.optimizer = optim.Adam(self.q_parameters, lr=self.critic_lr)

        # define target networks
        self.q1_target = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size,
                                   self.hidden_layers, activation='relu').to(self.device)
        self.q2_target = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size,
                                    self.hidden_layers, activation='relu').to(self.device)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        self.gamma = 0.99
        self.tau = 0.005
        self.counter = 0
        self.critic_target_update_freq = 1
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        self.actor = Actor(hidden_size=64, hidden_layers=5, actor_lr=5e-4,
                           state_dim=self.state_dim, action_dim=self.action_dim,
                           device=self.device)
        
        self.critic = Critic(hidden_size=100, hidden_layers=5, critic_lr=5e-4,
                             state_dim=self.state_dim, action_dim=self.action_dim,
                             device=self.device)
        
        self.critic_target_update(self.critic.q1, self.critic.q1_target, tau = 0, soft_update = False)
        self.critic_target_update(self.critic.q2, self.critic.q2_target, tau = 0, soft_update = False)

        self.temperature = TrainableParameter(init_param=0.1, lr_param=5e-4,
                                              train_param=True, device=self.device)
        
        self.target_entropy = -self.action_dim  # Target entropy for SAC algorithm


    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        # If working with stochastic policies, you should sample an action, otherwise,
        # you should return the deterministic action for the state s.
        state = torch.FloatTensor(s).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.get_action_and_log_prob(state, deterministic = not train)
        action = action.view(-1).detach().numpy()

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def clip_gradient(self, net: nn.Module) -> None:
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        torch.autograd.set_detect_anomaly(True)
        self.counter += 1

        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        batch = self.memory.sample(self.batch_size)
        # states, actions, rewards, next_states = batch
        states, actions, rewards, next_states = [torch.tensor(data, dtype=torch.float32, device=self.device) for data in batch]

        # Compute Q-values from the critic for the current states and actions
        # Concatenate states and actions
        current_inputs = torch.cat((states, actions), dim=-1)
        current_Q1, current_Q2 = self.critic.q1(current_inputs), self.critic.q2(current_inputs)
        
        # Compute Q-values from the critic for the next states and actions
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.get_action_and_log_prob(next_states, deterministic = False)
            #next_actions = torch.tensor(next_actions, dtype=torch.float32).unsqueeze(1)

            # Concatenate next_states and actions
            next_inputs = torch.cat((next_states, next_actions), dim=-1)
            target_Q1, target_Q2 = self.critic.q1_target(next_inputs), self.critic.q2_target(next_inputs)
            target_V = torch.min(target_Q1, target_Q2) - self.temperature.get_param().detach() * next_log_prob
            target_Q = rewards + self.gamma * target_V
            #target_Q = target_Q.detach()

        # Compute MSE loss
        #critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)) / 2
        critic_loss1 = torch.mean((current_Q1 - target_Q)**2)
        critic_loss2 = torch.mean((current_Q2 - target_Q)**2)
        # Optimize the critics
        critic_loss = critic_loss1 + critic_loss2
        self.run_gradient_update_step(self.critic, critic_loss)

        # Freeze critic parameters
        for param in self.critic.q_parameters:
            param.requires_grad = False
        
        # Update actor network and alpha
        actions, log_prob = self.actor.get_action_and_log_prob(states, deterministic = False)
        #actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)

        inputs = torch.cat((states, actions), dim=-1)
        actor_Q1, actor_Q2 = self.critic.q1(inputs), self.critic.q2(inputs)
        
        actor_Q = torch.min(actor_Q1, actor_Q2)
        alpha = self.temperature.get_param()
        actor_loss = - torch.mean( - alpha.detach() * log_prob + actor_Q)
        # Optimiza the actor
        self.run_gradient_update_step(self.actor, actor_loss)

        # Unfreeze critic parameters
        for param in self.critic.q_parameters:
            param.requires_grad = True

        # Update entropy parameter
        self.temperature.optimizer.zero_grad()
        alpha_loss = (alpha * 
                          (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward(retain_graph=True)
        self.temperature.optimizer.step()

        # Update target networks
        if self.counter % self.critic_target_update_freq == 0:
            with torch.no_grad():
                self.critic_target_update(self.critic.q1, self.critic.q1_target, tau = self.tau, soft_update = True)
                self.critic_target_update(self.critic.q2, self.critic.q2_target, tau = self.tau, soft_update = True)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
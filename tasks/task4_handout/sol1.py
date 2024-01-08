import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

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
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 

        self.model = NeuralNetwork(self.state_dim, 2*self.action_dim, self.hidden_size,
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
        x = self.model(state)
        mean, log_std = torch.split(x, self.action_dim, dim=-1)
    
        log_std = self.clamp_log_std(log_std)
        std = torch.exp(log_std)
        normal_dist = Normal(mean, std)

        if deterministic:
            #lambdaaa = 0.4
            action = torch.tanh(mean) # + Normal(0.0, lambdaaa).rsample())   # Ensure actions are in range [-1, 1]
            log_prob = torch.ones(self.action_dim)  # Placeholder for deterministic policy
        else:
            lambdaaa = 0.2
            z = normal_dist.rsample() + Normal(0.0, lambdaaa).rsample()
            action = torch.tanh(z)  # Ensure actions are in range [-1, 1]
            log_prob = normal_dist.log_prob(action).sum(axis=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1)
        
        action = action.view(-1).detach().numpy()

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
        self.model1 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size,
                                   self.hidden_layers, activation='relu').to(self.device)
        self.model2 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size,
                                    self.hidden_layers, activation='relu').to(self.device)
        # Initialize optimizers for each critic network separately
        #self.optimizer1 = optim.Adam(self.model1.parameters(), lr=self.critic_lr)
        #self.optimizer2 = optim.Adam(self.model2.parameters(), lr=self.critic_lr)

        # Define a single optimizer for both critic networks
        self.optimizer = optim.Adam(
            list(self.model1.parameters()) + list(self.model2.parameters()),
            lr=self.critic_lr
        )
    
class Value:
    def __init__(self, hidden_size: int, hidden_layers: int, value_lr: int, 
                 state_dim: int = 3, device: torch.device = torch.device('cpu')):
        super(Value, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.value_lr = value_lr
        self.state_dim = state_dim
        self.device = device
        self.setup_value()

    def setup_value(self):
        self.model_value = NeuralNetwork(self.state_dim, 1, self.hidden_size,
                                         self.hidden_layers, activation='relu').to(self.device)
        self.target_model_value = NeuralNetwork(self.state_dim, 1, self.hidden_size,
                                                self.hidden_layers, activation='relu').to(self.device)
        
        self.optimizer = optim.Adam(self.model_value.parameters(), lr=self.value_lr)

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
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        self.actor = Actor(hidden_size=256, hidden_layers=2, actor_lr=3e-4,
                           state_dim=self.state_dim, action_dim=self.action_dim,
                           device=self.device)
        
        self.critic = Critic(hidden_size=256, hidden_layers=2, critic_lr=3e-4,
                             state_dim=self.state_dim, action_dim=self.action_dim,
                             device=self.device)
        
        self.value = Value(hidden_size=256, hidden_layers=2, value_lr=3e-4,
                           state_dim=self.state_dim, device=self.device)
        
        self.temperature = TrainableParameter(init_param=1.0, lr_param=3e-4,
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
        action, _ = self.actor.get_action_and_log_prob(state, deterministic = not train)

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor, retain_graph: bool = False):
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

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        torch.autograd.set_detect_anomaly(True)

        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = batch

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Compute Q-values from the critic for the current states and actions
        # Concatenate states and actions
        inputs = torch.cat((states, actions), dim=1)

        # Compute Q-values from the critic network
        q1_values = self.critic.model1(inputs)
        q2_values = self.critic.model2(inputs)
        q_values = torch.min(q1_values, q2_values)
        v_values = self.value.model_value(states)
        target_value = self.value.target_model_value(next_states)
        target_q_values = rewards + self.gamma * target_value  # Discount factor: self.gamma = 0.99
        critic1_loss = 0.5 * F.mse_loss(q1_values, target_q_values.detach())
        critic2_loss = 0.5 * F.mse_loss(q2_values, target_q_values.detach())
        critic_loss = torch.add(critic1_loss, critic2_loss)

        self.run_gradient_update_step(self.critic, critic_loss)

        # Compute actor loss using SAC objective
        mean_actions, log_prob = self.actor.get_action_and_log_prob(states, deterministic = False)
        mean_actions = torch.tensor(mean_actions, dtype=torch.float32).unsqueeze(1)
        
        # Compute entropy
        alpha = self.temperature.get_param()  # Use entropy temperature parameter
        #entropy = -(log_prob + self.actor.action_dim * (np.log(2 * np.pi) + 1))  # Compute entropy
        alpha_loss = (alpha * 
                          (-log_prob - self.target_entropy).detach()).mean()
    
        self.temperature.optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.temperature.optimizer.step()

        actor_loss = (-q_values.detach() - alpha * 
                          (-log_prob - self.target_entropy).detach())  # Incorporate entropy into actor loss
        # Update actor network
        self.run_gradient_update_step(self.actor, actor_loss)

        # Update critic network using MSE loss
        with torch.no_grad():
            # next_actions, next_log_prob = self.actor.get_action_and_log_prob(next_states, deterministic = False)
            # next_actions = torch.tensor(next_actions, dtype=torch.float32).unsqueeze(1)
            # next_inputs = torch.cat((states, next_actions), dim=1)
            new_inputs = torch.cat((states, mean_actions), dim=1)
            new_q1_values = self.critic.model1(new_inputs)
            new_q2_values = self.critic.model2(new_inputs)
            new_q_values = torch.min(new_q1_values, new_q2_values)
            target_value_fun = new_q_values - log_prob
        value_loss = 0.5 * F.mse_loss(v_values, target_value_fun.detach())
        self.run_gradient_update_step(self.value, value_loss)

        # Update target networks
        self.critic_target_update(self.value.model_value, self.value.target_model_value, tau=0.005, soft_update=True)

        # Clear gradients
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        self.temperature.optimizer.zero_grad()


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
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

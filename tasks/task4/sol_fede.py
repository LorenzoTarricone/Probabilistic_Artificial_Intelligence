import torch
import torch.optim as optim
from torch.distributions import Normal, Independent
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from itertools import chain
from utils import ReplayBuffer, get_env, run_episode

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
        
        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        #self.drop_out = nn.Dropout(p=0.8)
        #self.bnorm = nn.BatchNorm1d(hidden_size)
        
        self.fc_final = nn.Linear(hidden_size, output_dim)
        
        self.hidden_layers = hidden_layers
        self.activation = activation

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        
        for i in range(self.hidden_layers-1):
            if i==0:
                s = self.fc_in(s)
            else:    
                s = self.fc(s)
            #s = self.drop_out(s)
            #s = self.bnorm(s)
            if self.activation == 'relu':
                s = nn.functional.relu(s)
            elif self.activation == 'sigmoid':
                s = nn.functional.sigmoid(s)
                
        s = self.fc_final(s)
        
        return s
        #pass?
    
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
        self.net = NeuralNetwork(self.state_dim, self.action_dim + 1, self.hidden_size, self.hidden_layers, 'relu')
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.actor_lr)
        
        pass

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
        if state.shape == (3,):
            state = state.reshape((1, 3))
        mu, action , log_prob = torch.zeros(state.shape[0]), torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        net_out = self.net(state)
        mu = net_out[:, 0:self.action_dim]
        log_std = net_out[:, -1]
                
        std = self.clamp_log_std(log_std)
        std = torch.exp(log_std)
        # Pre-squash distribution and sample 
        pi_distribution = Normal(mu, std.reshape(state.shape[0], self.action_dim))
        #print(pi_distribution)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = mu
        else:
            action = pi_distribution.rsample()
            
        
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding 
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        
        log_prob = pi_distribution.log_prob(action)
        log_prob -= (2*(np.log(2) - action - torch.nn.functional.softplus(-2*action))).sum(dim=1).reshape((state.shape[0], self.action_dim))
        action = torch.tanh(action)
        #action = self.act_limit * action ?
        assert action.shape == (state.shape[0], self.action_dim) and \
             log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
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
        self.q1 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        self.q2 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        #define optimizers
        self.q_params = chain(self.q1.parameters(), self.q2.parameters())
        self.optimizer = optim.Adam(self.q_params, lr=self.critic_lr)
        #define target networks
        self.q1_target = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        self.q2_target = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        #freeze target networks
        '''for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False '''    
        
        pass

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temperature parameter for SAC algorithm.
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
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()
        
        self.gamma = 0.99
        self.alpha = 0.1
        self.tau = 0.005

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need. 
        self.critic = Critic(100, 5, 1e-3, self.state_dim, self.action_dim, self.device)
        self.actor = Actor(64, 2, 1e-3, self.state_dim, self.action_dim, self.device)
        #self.critic.setup_critic()
        #self.actor.setup_actor()
        
        self.critic_target_update(self.critic.q1, self.critic.q1_target, 0, False) 
        self.critic_target_update(self.critic.q2, self.critic.q2_target, 0, False)
        
        pass

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        #action = np.random.uniform(-1, 1, (1,))
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(torch.as_tensor(s, dtype=torch.float32), train)
        action = action.numpy()
        action = action[0]
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

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        
        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        
        q1 = self.critic.q1(torch.cat([s_batch,a_batch],1))
        q2 = self.critic.q2(torch.cat([s_batch,a_batch],1))

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a_prime, logp_a_prime = self.actor.get_action_and_log_prob(s_prime_batch, False)
                      
                
            # Target Q-values
            #print(s_prime_batch.shape, a2.shape)
            q1_pi_targ = self.critic.q1_target(torch.cat([s_prime_batch, a_prime],1))
            q2_pi_targ = self.critic.q2_target(torch.cat([s_prime_batch, a_prime],1))
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r_batch + self.gamma * (q_pi_targ - self.alpha * logp_a_prime)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2) #.mean()?
        loss_q2 = ((q2 - backup)**2)
        critic_loss = loss_q1 + loss_q2

        # TODO: Implement Critic(s) update here.
        
        self.run_gradient_update_step(self.critic, critic_loss)
        
        # TODO: Implement Policy update here
        
        #freeze Q-networks
        for p in self.critic.q_params:
            p.requires_grad = False

        #compute loss pi
        pi, logp_pi = self.actor.get_action_and_log_prob(s_batch, False)
        q1_pi = self.critic.q1(torch.cat([s_batch, pi], dim=-1))
        q2_pi = self.critic.q2(torch.cat([s_batch, pi], dim=-1))
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        policy_loss = (self.alpha * logp_pi - q_pi) #.mean()?

        self.run_gradient_update_step(self.actor, policy_loss)
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.critic.q_params:
            p.requires_grad = True
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            self.critic_target_update(self.critic.q1, self.critic.q1_target, self.tau, True) 
            self.critic_target_update(self.critic.q2, self.critic.q2_target, self.tau, True)
            
        

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

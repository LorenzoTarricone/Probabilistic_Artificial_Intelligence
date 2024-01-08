# Reinforcement Learning for Inverted Pendulum Control

This project presents the development of a reinforcement learning agent using the Soft Actor-Critic (SAC) algorithm to swing up and stabilize an inverted pendulum. 
The pendulum, initially in a downward position (angle of r), is controlled to reach and maintain an upward position (angle of 0). 
The SAC algorithm was selected for its efficiency and stability, employing a dual Q-network within the Critic class and entropy maximisation to enhance 
learning stability. 
The gaussian agent controls the pendulum using a motor capable of applying torques in the range of \[-1, 1\].

# Probabilistic Artificial Intelligenc

Repository for the ETH course _Probabilistic Artificial Intelligence_ of Professor Andreas Krause in fall 2023. 

Group composed by: **Lucia Pezzetti**, **Federico Sartore** and **Lorenzo Tarricone**

Graded 6.0/6.0

## Tasks:
1. __Gaussian Process Regression for Air Pollution Prediction__: Addressing the critical issue of air pollution, this project focuses on predicting PM2.5 concentration in a city. Using Gaussian Process regression, we aim to model and forecast pollution levels in areas without direct measurements. The goal is to identify low-pollution residential zones, aiding urban planning for healthier living environments.

2. __Bayesian Neural Networks for Satellite Image Classification__: Implemented Bayesian Neural Networks (BNNs) using SWA-Gaussian (SWAG) technique, drawing insights from Maddox et al.'s "A Simple Baseline for Bayesian Uncertainty in Deep Learning." Addressed complexities like mixed land types, seasonal variations, and uncertainty in image classification for real-world applicability. Demonstrated the crucial role of uncertainty awareness in handling large-scale satellite data.

3. __Maximizing Drug Candidate Effectiveness with Bayesian Optimization__: Leveraging Bayesian optimization to fine-tune drug candidate features, optimizing bioavailability while considering synthesizability constraints. This task navigates complex evaluations to enhance drug effectiveness in discovery processes.

4. __Reinforcement Learning for Inverted Pendulum Control__: Developing an AI-driven system using off-policy RL algorithms (like DDPG or SAC) to maneuver an inverted pendulum from a downward to an upward position. This project revolves around AI applications in control systems.

## Get Started

1. Install and start Docker.
   
2. Once you have implemented your solution, run the checker in Docker:
   - On Linux, run bash runner.sh. In some cases, you might need to enable Docker for your user if you see a Docker permission denied error.
   - On MacOS, run bash runner.sh. Docker might by default restrict how much memory your solution may use. Running over the memory limit will result in docker writing "Killed" to the terminal. If you encounter out-of-memory issues you can increase the limits as described in the Docker Desktop for Mac user manual. Running over the memory limit will result in docker writing "Killed" to the terminal.
   - On Windows, open a PowerShell, change the directory to the handout folder, and run docker build --tag task1 .; docker run --rm -u $(id -u):$(id -g) -v "$(pwd):/results" task1 (replace task1 with the desired task).

3. If available, set the variable EXTENDED_EVALUATION in solution.py to True to create a plot visualizing the model.

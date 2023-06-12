# Final-Year-Project

Undergraduate Dissertation (University of Malta) 2020-2023 - 'Autonomous Drone Control using Reinforcement Learning'

## Abstract
<em>This project aims to develop a system for autonomous drone control that
focuses on the problem of drone obstacle avoidance. This is crucial for
safe drone deployment in industries like search and rescue, package de-
livery, and infrastructure inspections. The focus is on developing a rein-
forcement learning-based solution for unmanned aerial vehicles (UAVs) to
navigate through cluttered environments with static or moving obstacles.
To achieve this, AirSim simulated drone physics, and the UnReal engine
created the virtual environment. A depth sensor captured environmental
data, used as input for the reinforcement learning algorithm. The algorithm
learned from observations, with depth imagery being the most significant.
A Convolutional Neural Network (CNN) processed the depth imagery to ex-
tract relevant features. Additional inputs included velocity, distance from
the goal, and action history. An Artificial Neural Network (ANN) processed
the actions before combining them with the imagery. Four RL algorithms
were trained in a static environment, and the best two were trained in a
dynamic environment. These algorithims were: Deep Q-Network (DQN),
Double Deep Q-Network (DDQN), Proximal Policy Optimization (PPO),
and Trust Region Policy Optimisation (TRPO). The best performance came
from the DDQN algorithm, reaching the target goals 93% of the time in
environments with static obstacles and achieving an average of 84.5% in
environments with dynamic obstacles.</em>

## Demo - Four Rl algorithms (two static, two continuous) in a static environment
For the static environment, a closed corridor was constructed with a
length of 61.3 m, width of 21.8 m, and height of 15 m. Obstacles in the
form of columns with a diameter of 1m were placed every 10.8 m for a total
of four levels. The drone had to navigate through the corridor, avoiding the
obstacles and gaps between them to pass the test. The environment was
designed to be randomized at each episode to achieve generalization. The
vertical and horizontal columns could be shifted uniformly between -3m
and +3m programmatically from their original location dictated by a parent
column. 

https://github.com/kparnis3/Final-Year-Project/assets/81303628/895aa11d-dada-4919-a11e-0982d2ae68dc


## Demo - Best two Rl algorithms in a dynamic environment
For the dynamic environment, the same corridor dimensions were used,
but instead of columns, cubes and squished cylinders of various sizes were
used as obstacles. The obstacles were programmed to move from one point
to another at a constant speed, taking 20 seconds in real time to reach their
destination. Two approaches were taken to test generalization: one with
randomized obstacle order and one with a familiar training environment but
shuffled obstacle order. 

https://github.com/kparnis3/Final-Year-Project/assets/81303628/efeaea3b-fcf5-4495-9204-3f4f831f29ae


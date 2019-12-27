# Chapter 17: Robot Learning in Simulation (Project 4)
## Description:
Example of Sawyer robot learning to reach the target with paralleled Soft Actor-Critic (SAC) algorithm, using PyRep for Sawyer robot simulation and game building. The environment is wrapped into OpenAI Gym format.
<p align="center">
<img src="https://github.com/deep-reinforcement-learning-book/Chapter17-Robot-Learning-in-Simulation/blob/master/figures/reacher.gif" width="40%">
</p>

## Dependencies:
* [V-REP 3.6.2](http://www.coppeliarobotics.com/previousVersions)
* [PyRep](https://github.com/deep-reinforcement-learning-book/PyRep)
* PyTorch

Note:
* The later version of V-REP 3.6.2 is renamed CoppeliaSim after verison 4.0.0, which may have some incompatible issues with PyRep during the process of this project, so we suggest to use V-REP 3.6.2 [here](http://www.coppeliarobotics.com/previousVersions) and the maintained PyRep in our repository.
* The official repository of PyRep is [here](https://github.com/stepjam/PyRep), but we maintain a stable version [here](https://github.com/deep-reinforcement-learning-book/PyRep) in our repository for supporting V-REP 3.6.2, please use the version we provide ([here](https://github.com/deep-reinforcement-learning-book/PyRep)) for avoiding unnecessary incompatibility.

## Contents:
* `arms/`: object models of arms;
* `hands/`: object models of grippers;
* `objects/`: models of other objects in the scene;
* `scenes/`: built scenes for Sawyer robot grasping;
* `figures/`: figures for displaying;
* `model/`: the model after training, and two pre-trained models with different reward functions;
* `data/`: reward logs of with different reward functions;
* `sawyer_grasp_env_boundingbox.py`: script of Sawyer robot grasping environment;
* `sac_learn.py`: pralleled Soft Actor-Critic algorithm for solving Sawyer robot grasping task;
* `reward_log.npy`: log of episode reward during training;
* `plot.ipynb`: displaying the learning curves.


## Usage:
0. First check the environment can run successfully:

    `$ python sawyer_grasp_env_boundingbox.py`

    If it works properly with VRep called to run a scene, with Sawyer robot arm moving randomly, then go to next step; otherwise check the dependencies for necessary packages and versions.
1. Run `$ python sac_learn.py --train` for training the policy

2. Run `$ python sac_learn.py --test` for testing the trained policy, remember to change the `trained_model_path`, which is default to be the trained model we provided.

3. The training process will provide a `reward_log.npy` file for recording the reward value during training, which can be displayed with `$ jupyter notebook` in a new terminal, choose `plot.ipynb`and Shift+Enter to run the first cell, shown as follows:
<p align="center">
<img src="https://github.com/deep-reinforcement-learning-book/Chapter17-Robot-Learning-in-Simulation/blob/master/figures/training.png" width="80%">
</p>

## Authors:
[Zihan Ding](https://github.com/quantumiracle), [Yanhua Huang](https://github.com/Officium)


## Citing:

```
@misc{DeepReinforcementLearning-Chapter17-RobotLearninginSimulation,
  author = {Zihan Ding, Yanhua Huang},
  title = {Chapter17-RobotLearninginSimulation},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/deep-reinforcement-learning-book/Chapter17-Robot-Learning-in-Simulation}},
}
```

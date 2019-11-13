# Project2: RL-for-Robotics
## Description:
Example of Sawyer robot learning to reach the target with paralleled SAC algorithm, using PyRep for Sawyer robot simulation and game building. The environment is wrapped into OpenAI Gym format.
## Dependencies:
* [V-REP](http://www.coppeliarobotics.com/)
* [PyRep](https://github.com/stepjam/PyRep)
* PyTorch

## Usage:
0. First check the environment can run successfully:

    `$ python reacher_sawyer_env_boundingbox.py`

    If it works properly with VRep called to run a scene, with Sawyer robot arm moving randomly, then go to next step; otherwise check the dependencies for necessary packages and versions.
1. Run `$ python sac_learn.py --train` for training the policy

2. Run `$ python sac_learn.py --test` for testing the trained policy, remember to change the `trained_model_path`, which is default to be the trained model we provided.

3. The training process will provide a `reward_log.npy` file for recording the reward value during training, which can be displayed with `$ jupyter notebook` in a new terminal, choose `plot.ipynb`and Shift+Enter, as follows:
<p align="center">
<img src="https://github.com/deep-reinforcement-learning-book/Project2-RL-for-Robotics/blob/master/figures/training.png" width="80%">
</p>

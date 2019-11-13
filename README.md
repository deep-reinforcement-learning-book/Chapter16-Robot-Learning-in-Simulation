# Project2: RL-for-Robotics
## Description:

## Dependencies:

## Usage:
1. Run `python sac_learn.py --train` for training the policy

2. Run `python sac_learn.py --test` for testing the trained policy, remember to change the `trained_model_path`, which is default to be the trained model we provided.

3. The training process will provide a `reward_log.npy` file for recording the reward value during training, which can be displayed with `jupyter notebook` in a new terminal, choose `plot.ipynb`and Shift+Enter, as follows:
![training.png](https://github.com/deep-reinforcement-learning-book/Project2-RL-for-Robotics/blob/master/figures/training.png)

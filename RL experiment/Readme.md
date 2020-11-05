# STOR893FinalProject
This is the code for STOR893 Final Project

## Requirements
For RL experiment:
The experiment is performed on Python 3.8.x with these python packages:
* [tensorflow](https://github.com/tensorflow/tensorflow) == 1.14.0
* [numpy](http://www.numpy.org/) == 1.12.1
* [tflearn](http://tflearn.org/) == 0.3.2
* [gym](https://github.com/openai/gym) == 0.9.1
* [mujoco-py](https://github.com/openai/mujoco-py) == 0.5.7
Note that mujoco-py CANNOT be simply installed by importing the conda environment file. A license is required from https://www.roboti.us/license.html. After downloading the mujoco131, license & key and extract them in your device, you also need to add the environment path. For windows user, you may also need this trouble shooting: https://stackoverflow.com/questions/38766267/python-binding-for-mujoco-physics-library-using-mujoco-py-package

## Scripts explanation:
For RL experiment:
There are 5 python scripts:
* DDPG.py - This is the main script to run experiment for all methods.
* ActorNetwork.py - This script implements actor network with deterministic weights. This is the actor network for usual DDPG.
* CriticNetwork.py - This script implements critic network. This is the critic network for usual DDPG, and is used by all methods.
* Replay_buffer.py - This implements the usual first-in-first-out replay buffer.
* plot_Swimmer.py - This is for plotting the result. 

### How to train an agent:
For RL experiment:
Train an agent is done via running DDPG.py.
For example, to train with adam for Swimmer-v1, run
```
python DDPG.py --method adam --env_name Swimmer-v1
```
You can change the method value to the followings: 'adam', 'adagrad', 'sgd', 'rmsprop'.
If the name of environment is not given, the default task is Pendulum-v0, which is also a simple task from OpenAI Gym.

### How to plot the result:
For RL experiment:
The result reported in the paper is provided in 'result/Swimmer-v1' directory.
Run
```
python plot_Swimmer.py
```
The script loads test returns in .npy files.

#### Acknowledgement:
The general frame of this implementation is largely based on a nice DDPG implementation from the paper: https://arxiv.org/pdf/1806.04854.pdf
Github: https://github.com/emtiyaz/vadam/tree/master/tensorflow_RL

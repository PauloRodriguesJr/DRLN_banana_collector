[//]: # (Image References)


# Udacity's Value-Based Methods Project: Banana Collector


<img title="Banana Collector trained Agent" alt="Alt text" src="docs/banana-collector-agent.gif">

This repository contains material related to Udacity's Value-based Methods project.

### Project

* [Navigation](https://github.com/udacity/Value-based-methods/tree/main/p1_navigation): In the first project, it was trained an agent to collect yellow bananas while avoiding blue bananas.

## Environment Description


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

	In this project, it is provided a _environment.yml_ file that can be used to create the environment with all its dependencies. Tested on a Windows x86 machine.


2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 


## The Solution

To solve the environment, it was implemented a Double DQN algorithm, where the **local network** chooses the best next action, as the vanilla DQN does, but the **target network** evaluates its action value. 
The target equation for Double DQN is shown below:

 $Y^{Double Q}_t=R_{t+1}+\gamma Q(S_{t+1},argmax_a Q(S_{t+1},a;\theta_t);\theta_t')$


## Results

The agent was able to solve the environment before running 665 
episodes.

Below, it is shown a demonstration of the performance of the trained agent.

<img title="Banana Collector trained Agent" alt="Alt text" src="docs/banana-collector-agent.gif">


<!-- It was also plotted the agent's learning curve averaging 10 different seeds: -->


For more details, check the notebook Navigation.ipynb provided in the repository.

## Base references
* [Deep Q-Network](https://github.com/udacity/Value-based-methods/tree/main/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.

* [Double DQN paper](https://arxiv.org/abs/1509.06461): Deep Reinforcement Learning with Double Q-learning.


<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
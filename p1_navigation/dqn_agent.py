import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # < ORIGINAL GYM DATA INPUT>
        # Type of state <class 'numpy.ndarray'>
        # Type of action <class 'numpy.int64'>
        # Type of n_state <class 'numpy.ndarray'>
        # Type of done <class 'bool'>
        # Type of reward <class 'numpy.float64'>
        # <class 'torch.Tensor'>

        # print(f"Type of state {type(state)}")
        # print(f"Type of action {type(action)}")
        # print(f"Type of n_state {type(next_state)}")
        # print(f"Type of done {type(done)}")
        # print(f"Type of reward {type(reward)}")

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # Originally built (DQN notebook)
        """
            Type of state pre modification<class 'numpy.ndarray'>
            Type of state post modification<class 'torch.Tensor'>
            Shape of state input: torch.Size([1, 8])
            Type of action_values <class 'tuple'>
            Shape of action_values torch.Size([1, 4])
        """
        # Versus
        # ml agents environment output
        """
            Type of state pre modification<class 'numpy.ndarray'>
            Type of state post modification<class 'torch.Tensor'>
            Shape of state input: torch.Size([1, 37])
            Type of action_values <class 'torch.Tensor'>
            Shape of action_values 2 items
            Shape of action_values torch.Size([1, 4])
            ## CONCLUSÂO: Para todas as tuplas, minha política é pegar o primeiro elemento. O segundo parece ser apenas um zero!!
        """

        # print(f"Type of state pre modification{type(state)}", flush=True)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # print(f"Type of state post modification{type(state)}", flush=True)
        # print(f"Shape of state input: {state.size()}", flush=True)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            # print(f"Type of action_values {type(action_values)}", flush=True)
            # [print(f"Action_values {(action_value)}", flush=True)
            #  for action_value in action_values]
            # print(f"Shape of action_values {(action_values[0].size())}", flush=True)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values[0].cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, double=True):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        if not double:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(
                next_states)[0].detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states)[0].gather(1, actions)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        else:
            # get_argmax of
            best_action_local_next = self.qnetwork_local(
                next_states)[0].detach().argmax(dim=1).unsqueeze(1)

            # Q_targets_next = self.qnetwork_target(
            #     next_states)[0].detach().max(1)[0].unsqueeze(1)
            # print(
            #     f"dim of Q_targets_next{((Q_targets_next.size()))}", flush=True)
            # print(f"Q_targets_next {(Q_targets_next)}", flush=True)

            # Type of tensor <class 'torch.Tensor'>
            # Outputtensor([171])
            # print(
            #     f"Type of tensor {((best_action_local_next.size()))}", flush=True)
            # print(f"Output{(best_action_local_next)}", flush=True)

            # Type of tensor <class 'torch.Tensor'>
            # Outputtensor([[3],
            #         [3],
            #         [3],
            #         [3],
            #         [3],
            #         [3],

            # print(
            #     f"Type of tensor {((actions.size()))}", flush=True)
            # print(f"Output{(actions)}", flush=True)

            # Tá meio invertido aqui o que é oque. Relaxa que isso funciona e eu vou conseguir fazer funcionar!
            # Mas sera que ta mesmo? ele aprendeu e funcionou...
            # Bora so confirmar isso. Se tiver ok, ta ok mesmo, vou submeter e e isso ai

            Q_targets_next = self.qnetwork_target(
                next_states)[0].gather(1, best_action_local_next)

            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states)[0].gather(1, actions)

            loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

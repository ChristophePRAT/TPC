# Session #2 - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
This is the second session of the DQN project. We will be implementing the DQN algorithm as described in the paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al.

The environment we will be using is the Classic Control environment named CartPole.

## Installation

To install the required dependencies, you can use the following command:

```bash
cd .. && uv sync
```

## Running the code
To run the code, you can use the following command:

```bash
uv run train.py
```

### Custom hyperparameters
You can also specify custom hyperparameters by passing them as arguments to the `train.py` script. Here is the specification:

```bash
usage: train.py [-h] [--visible] [--episodes EPISODES] [--max_steps MAX_STEPS] [--learning_rate LEARNING_RATE] [--gamma GAMMA] [--epsilon EPSILON]
                [--epsilon_decay EPSILON_DECAY] [--epsilon_min EPSILON_MIN] [--update_target_every UPDATE_TARGET_EVERY] [--buffer_size BUFFER_SIZE]
                [--batch_size BATCH_SIZE]

Train a DQN agent on CartPole-v1

options:
  -h, --help            show this help message and exit
  --visible             Whether to render the environment
  --episodes EPISODES   Number of training episodes
  --max_steps MAX_STEPS
                        Maximum steps per episode
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer
  --gamma GAMMA         Discount factor for future rewards
  --epsilon EPSILON     Initial epsilon for epsilon-greedy action selection
  --epsilon_decay EPSILON_DECAY
                        Decay rate for epsilon
  --epsilon_min EPSILON_MIN
                        Minimum epsilon value
  --update_target_every UPDATE_TARGET_EVERY
                        Number of episodes between target network updates
  --buffer_size BUFFER_SIZE
                        Maximum size of the replay buffer
  --batch_size BATCH_SIZE
                        Batch size for training
```

## Further reading

- [Medium article explaining the implementation](https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae)

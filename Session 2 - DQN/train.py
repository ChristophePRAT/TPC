import torch
import gymnasium as gym
from model import DQN
from replay_buffer import ReplayBuffer
import replay_buffer
import argparse

parser = argparse.ArgumentParser(description="Train a DQN agent on CartPole-v1")
parser.add_argument("--visible", action="store_true", help="Whether to render the environment")
parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for epsilon-greedy action selection")
parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Decay rate for epsilon")
parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum epsilon value")
parser.add_argument("--update_target_every", type=int, default=5, help="Number of episodes between target network updates")
parser.add_argument("--buffer_size", type=int, default=10000, help="Maximum size of the replay buffer")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
args = parser.parse_args()



def train():
    env = gym.make("CartPole-v1", render_mode="human" if args.visible else None)

    # Hyperparameters
    learning_rate = args.learning_rate

    gamma = args.gamma

    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    epsilon_min = args.epsilon_min

    update_target_every = args.update_target_every

    num_episodes = args.episodes
    max_steps = args.max_steps

    buffer_size = args.buffer_size
    batch_size = args.batch_size
    # Initialize DQN and target network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    dqn = DQN(state_dim, action_dim)
    target_dqn = DQN(state_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

    # Training loop


    replay_buffer = ReplayBuffer(max_size=buffer_size)

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        total_steps = 0

        mean_loss = 0.0

        for step in range(max_steps):
            # Epsilon-greedy action selection
            if torch.rand(1).item() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(obs, dtype=torch.float32)
                    q_values = dqn(state_tensor)
                    action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            # Store transition in replay buffer
            replay_buffer.add((obs, action, reward, next_obs, terminated or truncated))

            # Sample a batch of transitions from the replay buffer
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.bool)

                # Compute target Q-values
                with torch.no_grad():
                    target_q_values = target_dqn(next_states_tensor).max(1)[0].unsqueeze(1)
                    target_q_values[dones_tensor] = 0.0
                    target_q_values = rewards_tensor + gamma * target_q_values

                # Compute current Q-values
                # Forward the states through the DQN and gather the Q-values corresponding to the taken actions
                current_q_values = dqn(states_tensor).gather(1, actions_tensor)

                # Compute loss and update DQN
                loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)


                mean_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())


            obs = next_obs
            total_steps += 1
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            if terminated or truncated:
                break

        mean_loss /= total_steps

        print(f"Episode {episode}, Loss: {mean_loss:.4f}, Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    train()

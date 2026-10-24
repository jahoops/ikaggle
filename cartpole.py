import gym
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gym.make('CartPole-v1', render_mode=None)

# Simple policy that takes random actions
def basic_policy(observation):
    return env.action_space.sample()

# Track scores across episodes
scores = []
num_episodes = 100

# Run episodes
for episode in range(num_episodes):
    observation = env.reset()[0]
    episode_score = 0
    
    while True:
        action = basic_policy(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_score += reward
        
        if done or truncated:
            scores.append(episode_score)
            break

env.close()

# Plot results
plt.plot(scores)
plt.title('CartPole Performance')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()

print(f"Average score: {np.mean(scores):.2f}")
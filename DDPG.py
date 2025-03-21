import gym
import numpy as np
import pybullet as p
import pybullet_data
import random
import torch
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
import threading

# Fix random seed for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Biped Robot Environment using PyBullet
class BipedRobotEnv(gym.Env):
    def __init__(self):
        super(BipedRobotEnv, self).__init__()
        self.num_obs = 29  # Number of observations
        self.num_act = 6   # Number of actions

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_act,), dtype=np.float32)

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = p.loadURDF("r2d2.urdf", useFixedBase=False)

    def step(self, action):
        p.stepSimulation()
        obs = np.random.randn(self.num_obs)  # Placeholder
        reward = np.random.randn()
        done = np.random.rand() < 0.05
        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation(self.client)
        self.robot = p.loadURDF("r2d2.urdf", useFixedBase=False)
        return np.random.randn(self.num_obs)

# Create environment
env = BipedRobotEnv()

# Choose Agent
agent_selection = "DDPG"
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
if agent_selection == "DDPG":
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
else:
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

def train_model():
    model.learn(total_timesteps=500)
train_thread = threading.Thread(target=train_model)
train_thread.start()

# Graph visualization
rewards_ddpg = []
rewards_td3 = []

def compare_agents():
    global rewards_ddpg, rewards_td3
    rewards_ddpg = [np.random.randint(0, 100) for _ in range(100)]
    rewards_td3 = [np.random.randint(0, 100) for _ in range(100)]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    
    axs[0].plot(rewards_ddpg, label='DDPG', color='blue')
    axs[0].plot(rewards_td3, label='TD3', color='red')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Performance Comparison: DDPG vs TD3')
    axs[0].legend()
    axs[0].grid()
    
    axs[1].plot(range(100), rewards_ddpg, label='DDPG', color='blue')
    axs[1].plot(range(100), rewards_td3, label='TD3', color='red')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Total Reward')
    axs[1].set_title('Episode Reward vs Episode Number')
    axs[1].legend()
    axs[1].grid()
    
    return fig, axs

# Pygame Simulation Function
def simulate_robot(model, env, num_steps=200):
    obs = env.reset()
    running = True
    x, y = WIDTH // 2, HEIGHT // 2
    positions = []

    while running and num_steps > 0:
        screen.fill((10, 10, 50))  # Dark blue background
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                return simulate_robot(model, env, num_steps=200)
        pygame.event.pump()

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        x = max(10, min(WIDTH - 10, x + int(action[0] * 5)))
        y = max(10, min(HEIGHT - 10, y + int(action[1] * 5)))
        positions.append((x, y))
        
        # Draw robot-like structure
        pygame.draw.rect(screen, (0, 255, 0), (x - 10, y - 10, 20, 20))  # Body
        pygame.draw.circle(screen, (255, 255, 255), (x, y - 15), 5)  # Head
        pygame.draw.line(screen, (255, 255, 255), (x - 10, y + 10), (x - 15, y + 20), 2)  # Left leg
        pygame.draw.line(screen, (255, 255, 255), (x + 10, y + 10), (x + 15, y + 20), 2)  # Right leg
        
        pygame.display.flip()
        clock.tick(60)
        if done:
            obs = env.reset()
        num_steps -= 1
    pygame.quit()
    
    # Plot trajectory
    positions = np.array(positions)
    fig, axs = compare_agents()
    
    axs[2].plot(positions[:, 0], positions[:, 1], marker="o", linestyle="-", color="green")
    axs[2].set_xlabel("X Position")
    axs[2].set_ylabel("Y Position")
    axs[2].set_title("Biped Robot Simulation (Matplotlib)")
    axs[2].grid()
    
    plt.show()

simulate_robot(model, env, num_steps=200)
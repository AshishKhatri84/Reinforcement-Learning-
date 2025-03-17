import pygame
import random
import numpy as np
import gym
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Constants
WIDTH, HEIGHT = 800, 400
GROUND_Y = HEIGHT - 50  # Ground position
ROBOT_WIDTH, ROBOT_HEIGHT = 40, 60
STEP_SIZE = 20  # Pixels per step
BG_SCROLL_SPEED = 10  # Background movement speed

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
RED = (255, 50, 50)
DARK_GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
BROWN = (139, 69, 19)
GRAY = (192, 192, 192)
YELLOW = (255, 255, 0)

# Background elements
clouds = [(i * 200, random.randint(30, 100)) for i in range(4)]
trees = [(i * 150, GROUND_Y - 70) for i in range(6)]
houses = [(i * 300, GROUND_Y - 90) for i in range(3)]


# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create the screen object
clock = pygame.time.Clock()  # Initialize the clock for controlling the frame rate

# Biped Environment
class BipedEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)  # Move left, stay, move right
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.state = np.zeros(6)
        self.path = []
        self.bg_offset = 0
        self.step_count = 0

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=(6,))
        self.path = [(WIDTH // 2, GROUND_Y - ROBOT_HEIGHT // 2)]
        self.bg_offset = 0
        self.step_count = 0
        return self.state

    def step(self, action):
        reward = np.random.rand()
        done = np.random.rand() > 0.9
        self.state = np.random.uniform(-1, 1, size=(6,))

        # Move background in opposite direction to simulate walking
        if action == 0:  # Move left
            self.bg_offset += BG_SCROLL_SPEED
        elif action == 2:  # Move right
            self.bg_offset -= BG_SCROLL_SPEED

        # Loop background elements
        for i in range(len(clouds)):
            clouds[i] = ((clouds[i][0] - BG_SCROLL_SPEED // 3) % WIDTH, clouds[i][1])

        for i in range(len(trees)):
            trees[i] = ((trees[i][0] - BG_SCROLL_SPEED) % WIDTH, trees[i][1])

        for i in range(len(houses)):
            houses[i] = ((houses[i][0] - BG_SCROLL_SPEED) % WIDTH, houses[i][1])

        # Keep track of steps for animation
        self.step_count += 1

        return self.state, reward, done, {}

    def render(self):
        screen.fill(LIGHT_GREEN)  # Background color for ground

        # Draw moving elements
        self.draw_sky()
        self.draw_ground()
        self.draw_path()
        self.draw_robot(WIDTH // 2, GROUND_Y - ROBOT_HEIGHT, self.step_count)

        pygame.display.flip()

    def draw_sky(self):
        """Draws sky elements like clouds"""
        for cloud in clouds:
            pygame.draw.ellipse(screen, WHITE, (cloud[0], cloud[1], 40, 20))
            pygame.draw.ellipse(screen, WHITE, (cloud[0] + 20, cloud[1] - 5, 50, 25))

    def draw_ground(self):
        """Draws the ground with looping trees and houses"""
        pygame.draw.rect(screen, DARK_GREEN, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))  # Grass

        for tree in trees:
            pygame.draw.rect(screen, BROWN, (tree[0] - 5, tree[1] + 20, 10, 30))  # Trunk
            pygame.draw.circle(screen, DARK_GREEN, (tree[0], tree[1]), 20)  # Leaves

        for house in houses:
            pygame.draw.rect(screen, GRAY, (house[0] - 20, house[1], 40, 40))  # Walls
            pygame.draw.polygon(screen, RED, [(house[0] - 25, house[1]), (house[0] + 25, house[1]), (house[0], house[1] - 20)])  # Roof
            pygame.draw.rect(screen, YELLOW, (house[0] - 5, house[1] + 10, 10, 10))  # Window

    def draw_path(self):
        """Draws a fading path of past movements"""
        for i in range(len(self.path) - 1):
            fade_factor = i / len(self.path)
            faded_color = (255, int(50 + fade_factor * 150), int(50 + fade_factor * 150))
            pygame.draw.line(screen, faded_color, self.path[i], self.path[i + 1], 3)

    def draw_robot(self, x, y, step_count):
        """ Draws a walking humanoid robot """
        pygame.draw.rect(screen, BLUE, (x - 12, y, 24, 35))  # Body
        pygame.draw.circle(screen, BLUE, (x, y - 15), 12)  # Head
        pygame.draw.circle(screen, WHITE, (x - 4, y - 18), 3)
        pygame.draw.circle(screen, WHITE, (x + 4, y - 18), 3)

        # Arms
        pygame.draw.line(screen, BLACK, (x - 15, y + 5), (x - 25, y + 15), 5)
        pygame.draw.line(screen, BLACK, (x + 15, y + 5), (x + 25, y + 15), 5)

        # Alternating legs for walking effect
        if step_count % 2 == 0:
            pygame.draw.line(screen, BLACK, (x - 5, y + 35), (x - 10, y + 55), 5)
            pygame.draw.line(screen, BLACK, (x + 5, y + 35), (x + 10, y + 50), 5)
        else:
            pygame.draw.line(screen, BLACK, (x - 5, y + 35), (x - 10, y + 50), 5)
            pygame.draw.line(screen, BLACK, (x + 5, y + 35), (x + 10, y + 55), 5)


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_size = action_size  # Number of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table to store Q-values

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0)

    def update_q_value(self, state, action, reward, next_state):
        max_next_q = max(self.get_q_value(next_state, a) for a in range(self.action_size))
        current_q = self.get_q_value(state, action)
        # Update Q-value using the Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(tuple(state), action)] = new_q

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))  # Exploration: random action
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
            return np.argmax(q_values)  # Exploitation: best action


# Pygame Animation Loop with Q-learning
def run_simulation(agent, env, episodes=10, steps_per_episode=30):
    state = env.reset()
    episode_rewards = []  # List to store rewards for each episode
    
    for episode in range(episodes):
        total_reward = 0
        for step in range(steps_per_episode):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state  # Update state for the next step

            env.render()
            clock.tick(10)  # Control animation speed

            total_reward += reward

            if done:
                state = env.reset()
                break

        episode_rewards.append(total_reward)  # Append total reward of the current episode
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    # Plot the rewards after simulation
    plt.plot(range(1, episodes + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode vs Total Reward')
    plt.show()


# Initialize
env = BipedEnv()
agent = QLearningAgent(action_size=env.action_space.n)

# Run simulation
run_simulation(agent, env)

pygame.quit()
import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pygame_gui

# Pygame Initialization
pygame.init()

# Grid and Window parameters
GRID_SIZE, CELL_SIZE = 4, 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 5

# Q-learning Parameters
ALPHA, GAMMA, EPSILON, EPISODES = 0.1, 0.9, 0.1, 500
MAX_SIMULATIONS = 2  # Set to 2 simulations

# Colors (RGB format)
colors = {
    "W": (255, 255, 255), "B": (0, 0, 0), "D": (100, 100, 100),
    "S": (200, 200, 200), "G": (144, 238, 144), "H": (240, 128, 128),
    "P": (255, 165, 0), "F": (173, 216, 230)
}

# Actions (Up, Down, Left, Right)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Initialize pygame window for simulation
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
pygame.display.set_caption("Frozen Lake - Q-learning Robot")
clock = pygame.time.Clock()

# Preload font
font = pygame.font.Font(None, 28)

# Initialize pygame_gui
manager = pygame_gui.UIManager((WINDOW_SIZE, WINDOW_SIZE + 50))

# Flag to indicate when to plot graphs
plot_graphs_flag = False
episode_rewards = []
exploit_counts = []
q_table = None

def plot_graphs():
    global episode_rewards, exploit_counts, q_table
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    plots = [
        ("Episode vs Reward", "Episodes", "Reward", range(EPISODES), episode_rewards, "blue"),
        ("Exploitation Frequency Over Time", "Episodes", "Exploitation Count", range(EPISODES), exploit_counts, "red")
    ]

    for i, (title, xlabel, ylabel, x_data, y_data, color) in enumerate(plots):
        axs[0, i].plot(x_data, y_data, label=title, color=color)
        axs[0, i].set_title(title)
        axs[0, i].set_xlabel(xlabel)
        axs[0, i].set_ylabel(ylabel)
        axs[0, i].legend()

    # Q-table Heatmap
    max_q_values = np.max(q_table, axis=2)
    sns.heatmap(max_q_values, annot=True, cmap="coolwarm", ax=axs[1, 0])
    axs[1, 0].set_title("Q-value Heatmap")

    # Max Q-values Table
    axs[1, 1].table(cellText=np.round(max_q_values, 2), loc='center', cellLoc='center')
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Max Q-values")

    plt.tight_layout()
    plt.show(block=False)

def reset_simulation():
    global q_table, holes, reward_grid, path, simulation_count, start, goal, plot_graphs_flag, episode_rewards, exploit_counts

    q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    start, goal = (0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)

    # Generate random holes avoiding start and goal
    holes = set()
    while len(holes) < 3:
        hole = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if hole not in {start, goal}:
            holes.add(hole)
    holes = list(holes)

    # Reward grid setup
    reward_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
    reward_grid[goal] = 100
    for hole in holes:
        reward_grid[hole] = -100

    episode_rewards = []
    exploit_counts = []

    # Q-learning Training
    for episode in range(EPISODES):
        state = start
        total_reward = 0
        exploit_count = 0
        while state != goal:
            x, y = state
            action = np.random.randint(4) if np.random.rand() < EPSILON else np.argmax(q_table[x, y])
            if np.random.rand() >= EPSILON:
                exploit_count += 1
            dx, dy = actions[action]
            next_state = (x + dx, y + dy) if (0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and (x + dx, y + dy) not in holes) else state

            reward = reward_grid[next_state]
            q_table[x, y, action] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[x, y, action])

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        exploit_counts.append(exploit_count)

    # Extract best path
    path, state = [start], start
    while state != goal:
        x, y = state
        action = np.argmax(q_table[x, y])
        dx, dy = actions[action]
        next_state = (x + dx, y + dy)

        if next_state == state or next_state in holes or not (0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE):
            break

        path.append(next_state)
        state = next_state

    simulation_count = 0

    # Set flag to plot graphs
    plot_graphs_flag = True

# Initialize first simulation
reset_simulation()

def draw_grid():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell_type = "S" if (i, j) == start else "G" if (i, j) == goal else "H" if (i, j) in holes else "P" if (i, j) in path else "F"
            pygame.draw.rect(screen, colors[cell_type], rect)
            pygame.draw.rect(screen, colors["B"], rect, 2)

            # Display rewards and Q-values
            reward_text = font.render(str(int(reward_grid[i, j])), True, colors["B"])
            q_value_text = font.render(f"{np.max(q_table[i, j]):.2f}", True, colors["B"])
            screen.blit(reward_text, (j * CELL_SIZE + 10, i * CELL_SIZE + 10))
            screen.blit(q_value_text, (j * CELL_SIZE + 10, i * CELL_SIZE + 40))

def draw_robot(x, y, step):
    center_x, center_y = y * CELL_SIZE + 50, x * CELL_SIZE + 50
    pygame.draw.rect(screen, colors["D"], (center_x - 20, center_y - 25, 40, 50), border_radius=10)

    eye_color = colors["W"] if step % 6 else colors["B"]
    pygame.draw.circle(screen, eye_color, (center_x - 10, center_y - 10), 5)
    pygame.draw.circle(screen, eye_color, (center_x + 10, center_y - 10), 5)

    leg_offset = 5 if step % 2 == 0 else -5
    pygame.draw.line(screen, colors["B"], (center_x - 15, center_y + 20), (center_x - 10, center_y + 35 + leg_offset), 4)
    pygame.draw.line(screen, colors["B"], (center_x + 15, center_y + 20), (center_x + 10, center_y + 35 - leg_offset), 4)

    pygame.draw.line(screen, colors["B"], (center_x - 20, center_y - 5), (center_x - 30, center_y + leg_offset), 4)
    pygame.draw.line(screen, colors["B"], (center_x + 20, center_y - 5), (center_x + 30, center_y - leg_offset), 4)

def draw_restart_button():
    button_rect = pygame.Rect(WINDOW_SIZE // 4, WINDOW_SIZE + 10, WINDOW_SIZE // 2, 30)
    pygame.draw.rect(screen, colors["D"], button_rect, border_radius=5)
    text = font.render("Restart Simulation", True, colors["W"])
    screen.blit(text, (WINDOW_SIZE // 4 + 15, WINDOW_SIZE + 15))

def move_robot():
    global simulation_count
    for step, (x, y) in enumerate(path):
        screen.fill(colors["W"])
        draw_grid()
        draw_robot(x, y, step)
        pygame.display.update()
        clock.tick(FPS)
    simulation_count += 1

# Game loop
running = True
while running:
    time_delta = clock.tick(FPS) / 1000.0
    screen.fill(colors["W"])
    draw_grid()

    if simulation_count < MAX_SIMULATIONS:
        draw_robot(*start, 0)
        pygame.display.update()
        pygame.time.delay(1000)
        move_robot()
    else:
        draw_restart_button()

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if simulation_count >= MAX_SIMULATIONS and (WINDOW_SIZE // 4 <= x <= 3 * WINDOW_SIZE // 4) and (WINDOW_SIZE + 10 <= y <= WINDOW_SIZE + 40):
                reset_simulation()

        manager.process_events(event)

    # Check if it's time to plot graphs
    if plot_graphs_flag:
        plot_graphs()
        plot_graphs_flag = False

    manager.update(time_delta)
    manager.draw_ui(screen)

    pygame.display.update()

pygame.quit()
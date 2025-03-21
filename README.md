# Reinforcement Learning Project: Multi-Environment Exploration
## Purpose:  
This repository contains various reinforcement learning simulations, including **Q-learning, SARSA, Deep Deterministic Policy Gradient (DDPG) and Deep Q-Networks (DQN)**, applied to environments like **Frozen Lake, CartPole, GridWorld, Tic-Tac-Toe, and Bipedal Simulation**. The purpose of this application is to provide interactive simulations with **visualization, agent training, and comparative analysis** of different reinforcement learning algorithms.

## Project Description:  
The project consists of multiple reinforcement learning implementations, covering:  
- **Tabular methods (Q-learning, SARSA)**
- **Deep Reinforcement Learning (DQN for CartPole)**
- **Grid-based and dynamic simulations (Frozen Lake, Bipedal Simulation)**
- **Comparative analysis of Q-learning vs SARSA**
- **DDPG-based Biped Robot Simulation**

These implementations help understand **exploration vs exploitation**, reward learning, and decision-making processes in AI agents.

## Steps:  
1. Install required dependencies:  
   ```bash
   pip install numpy matplotlib pygame seaborn gym stable-baselines3 pybullet
   ```  
2. Run individual simulations as per the instructions below.  
3. Modify parameters like **learning rate, discount factor, and exploration rate** to observe different behaviors.  
4. Analyze performance using **graphs, heatmaps, and animations**.

# Technologies Used:  
1. **Python 3.6+**  
2. **OpenAI Gym** – For environment simulation  
3. **Pygame** – For rendering animations  
4. **Matplotlib & Seaborn** – For plotting and data visualization  
5. **Stable-Baselines3** – For DDPG and TD3 implementations  
6. **PyBullet** – For physics-based biped simulation  

## Features of the Code:  
1. **Reinforcement Learning Algorithms:**  
   - Q-learning  
   - SARSA  
   - Deep Q-Networks (DQN)  
   - DDPG & TD3 for Biped Robot Simulation  
2. **GUI Components & Visualizations:**  
   - **Frozen Lake & GridWorld**: Animated agent movements with **heatmaps & Q-tables**  
   - **Bipedal Simulation**: Real-time rendering of agent walking in a **dynamic environment**  
   - **Tic Tac Toe AI**: Reinforcement learning-based gameplay  
   - **Biped Robot Simulation**: DDPG & TD3 agents walking in a physics-based environment  
3. **Comparative Analysis:**  
   - Q-learning vs SARSA performance comparison  
   - Reward vs Episode graphs  
   - Exploration vs Exploitation trends  
   - DDPG vs TD3 Performance Comparison  

## How to Run the Applications:  

### 1️⃣ **Frozen Lake Q-Learning**  
```bash
python "Bipedal in Frozen Lake.py"
```
- Animates agent movement  
- Displays **Q-table, heatmap, and episode vs reward graph**  

### 2️⃣ **Bipedal Simulation**  
```bash
python "Bipedal Simulation.py"
```
- Simulates a bipedal robot with **Q-learning-based movement**  

### 3️⃣ **DDPG & TD3 Biped Robot Simulation**  
```bash
python "Biped Robot Sim.py"
```
- Simulates a physics-based **biped robot** using **DDPG & TD3**  
- Displays **Episode Reward vs Episode Number, Performance Comparison, and Trajectory Graph**  

### 4️⃣ **Jupyter Notebook Experiments**  
```bash
jupyter notebook
```
- Open **CartPole using DQN.ipynb, GridWorld.ipynb, SARSA.ipynb, or Q_Learning_vs_Sarsa.ipynb** and execute step by step.  

## Notes:  
- Modify **learning parameters** inside the scripts for better tuning.  
- Use **exploration vs exploitation graphs** to analyze agent behavior.  
- Future enhancements include **policy-based reinforcement learning methods**.  

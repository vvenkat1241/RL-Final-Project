import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

MAX_STEPS = 512

class Agent():
    def __init__(self,
                 number: int,
                 info_grid: np.ndarray,
                 target_grid: np.ndarray,
                 start_pos: np.ndarray,
                 total_targets: int):
        self.number = number
        self.pos = start_pos
        self.info_grid = info_grid
        self.target_grid = target_grid
        self.targets_left = total_targets

        self.grid_size = info_grid.shape[0]
    
    def update_agent_position(self,
                              action: np.ndarray):
        if action == 0:                                             # Down
            self.pos[0] = max(0, self.pos[0] - 1)
        elif action == 1:                                           # Up
            self.pos[0] = min(self.grid_size - 1, self.pos[0] + 1)
        elif action == 2:                                           # Left
            self.pos[1] = max(0, self.pos[1] - 1)
        elif action == 3:                                           # Right
            self.pos[1] = min(self.grid_size - 1, self.pos[1] + 1)

class MAS_env(gym.Env):
    def __init__(self,
                 grid_size: int = 20, 
                 obs_size: int = 5, 
                 num_agents: int = 8, 
                 num_targets: int = 6):
        super(MAS_env, self).__init__()

        self.grid_size = grid_size
        self.obs_size = obs_size
        self.n_agents = num_agents
        self.n_targets = num_targets
        self.targets_left = num_targets

        self.info_grid = self.generate_info_grid()
        self.target_grid, self.targets_pos = self.generate_targets()
        self.agents, self.agent_grid = self.generate_agents()
        

        self.sensor = self.generate_sensor()

        self.action_space = spaces.Discrete(5)  # Five discrete actions: up, down, left, right, stay
        self.observation_space = spaces.Box(low=0,
                                            high=1, 
                                            shape=(obs_size, obs_size), 
                                            dtype=np.float32)
        self.step_counter = 0
        self.terminated = False
        self.truncated = False

    def generate_info_grid(self):
        info_grid = np.zeros((self.grid_size, self.grid_size))

        row, col = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing="ij")
        n_gaussians = np.random.randint(low=5,
                                        high=10)
        gaussian_means = np.random.randint(low=0,
                                           high=self.grid_size,
                                           size=(n_gaussians, 2))
        gaussian_stddevs = np.random.uniform(low=1,
                                             high=min(self.grid_size // 2, 10),
                                             size=n_gaussians)
        
        for mean, stddev in zip(gaussian_means, gaussian_stddevs):
            gaussian = np.exp(-((row - mean[0])**2 + (col - mean[1])**2) / (2 * stddev**2))
            info_grid += gaussian
        
        info_grid /= info_grid.sum()

        return info_grid

    def generate_targets(self):
        targets_grid = np.zeros((self.grid_size, self.grid_size))
        flat_probs = self.info_grid.copy().flatten()
        targets_indices = np.random.choice(self.grid_size*self.grid_size,
                                           size=self.n_targets,
                                           p=flat_probs)
        targets_coordinates = np.unravel_index(targets_indices, (self.grid_size, self.grid_size))

        targets_pos = []

        for row, col in zip(targets_coordinates[0], targets_coordinates[1]):
            targets_grid[row, col] = 1
            targets_pos.append([row, col])

        return targets_grid, np.asarray(targets_pos)

    def generate_agents(self):
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        agents_pos = np.random.randint(low=0,
                                       high=self.grid_size,
                                       size=(self.n_agents, 2))
        agents = [Agent(n,
                        self.info_grid,
                        self.target_grid,
                        agent_pos,
                        self.n_targets)
                    for n, agent_pos in enumerate(agents_pos)]
        
        for agent in agents:
            agent_grid[agent.pos] = agent.number

        return agents, agent_grid

    def generate_sensor(self):
        row, col = np.meshgrid(np.arange(self.obs_size), np.arange(self.obs_size), indexing="ij")
        sensor = np.exp(-((row-2)**2 + (col-2)**2) / (2))
        sensor /= sensor.sum()

        return sensor

    def step(self, 
             actions: np.ndarray):
        self.step_counter += 1

        rewards = []
        next_states_1 = []
        next_states_2 = []

        for agent, action in zip(self.agents, actions):
            self.agent_grid[agent.pos] = 0
            agent.update_agent_position(action)
            self.agent_grid[agent.pos] = agent.number

            reward, observation = self.update_info_get_reward(agent)
            rewards.append(reward)

            next_states_1.append(observation)
            next_states_2.append([agent.pos[0], agent.pos[1], self.step_counter, agent.targets_left])

        if self.step_counter >= MAX_STEPS:  # Define MAX_STEPS as the threshold
            self.truncated = True

        if self.targets_left <= 0:
            self.terminated = True

        
        return np.asarray(next_states_1), np.asarray(next_states_2), np.asarray(rewards), self.terminated, self.truncated, None

    def update_info_get_reward(self, 
                               agent: Agent):

        row_min, row_max = max(0, agent.pos[1]-2), min(self.grid_size, agent.pos[1]+3)
        col_min, col_max = max(0, agent.pos[0]-2), min(self.grid_size, agent.pos[0]+3)

        self.info_grid[row_min:row_max, col_min:col_max] *= (1 - self.sensor[row_min-(agent.pos[1]-2):row_max + self.obs_size-(agent.pos[1]+3),
                                                                             col_min-(agent.pos[0]-2):col_max + self.obs_size-(agent.pos[0]+3)])
        
        temp = agent.info_grid[row_min:row_max, col_min:col_max].copy()

        sensor_visible = self.sensor[row_min-(agent.pos[1]-2):row_max + self.obs_size-(agent.pos[1]+3),
                                     col_min-(agent.pos[0]-2):col_max + self.obs_size-(agent.pos[0]+3)]

        agent.info_grid[row_min:row_max, col_min:col_max] *= (1 - sensor_visible)
        
        reward = (temp - agent.info_grid[row_min:row_max, col_min:col_max]).sum()

        targets_in_sensor = np.where(agent.target_grid[row_min:row_max, col_min:col_max] == 1)

        for row, col in zip(targets_in_sensor[0], targets_in_sensor[1]):
            if np.random.random() < sensor_visible[row, col]:
                if self.target_grid[row_min+row, col_min+col] == 1:
                    self.target_grid[row_min+row, col_min+col] = -1
                    self.targets_left -= 1

                # Move below part into conditional if you want agent to learn what other agents have done
                reward += 5
                agent.target_grid[row_min+row, col_min+col] = -1 
                agent.targets_left -= 1

        observation = np.zeros((self.obs_size, self.obs_size))
        observation[row_min-(agent.pos[1]-2):row_max + self.obs_size-(agent.pos[1]+3),
                    col_min-(agent.pos[0]-2):col_max + self.obs_size-(agent.pos[0]+3)] = agent.info_grid[row_min:row_max, col_min:col_max]

        return reward, observation

    def get_agent_views(self, agent_id):
        view = np.zeros((self.grid_size, self.grid_size))  # Adjust size to match grid_size
        x, y = self.agent_pos[agent_id]
        window = self.info_grid[x - 2:x + 3, y - 2:y + 3]
        view[x - 2:x + 3, y - 2:y + 3] = window  # Update view with window from info_grid
        return view

    def reset(self):
        self.info_grid = self.generate_info_grid()
        self.target_grid, self.targets_pos = self.generate_targets()
        self.agents, self.agent_grid = self.generate_agents()

        self.step_counter = 0
        self.targets_left = self.n_targets
        self.terminated = False
        self.truncated = False

        states_1 = []
        states_2 = []

        for agent in self.agents:
            row_min, row_max = max(0, agent.pos[1]-2), min(self.grid_size, agent.pos[1]+3)
            col_min, col_max = max(0, agent.pos[0]-2), min(self.grid_size, agent.pos[0]+3)

            observation = np.zeros((self.obs_size, self.obs_size))
            observation[row_min-(agent.pos[1]-2):row_max + self.obs_size-(agent.pos[1]+3),
                        col_min-(agent.pos[0]-2):col_max + self.obs_size-(agent.pos[0]+3)] = agent.info_grid[row_min:row_max, col_min:col_max]
            
            states_1.append(observation)
            states_2.append([agent.pos[0], agent.pos[1], self.step_counter, agent.targets_left])

        return np.asarray(states_1), np.asarray(states_2)

    def render(self, mode='human'):
        plt.figure(figsize=(24, 20))
        plt.subplot(2, 2, 1)
        plt.imshow(self.info_grid, cmap='Reds', origin='lower')
        plt.colorbar(label='Information Value')
        plt.title('Main Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(visible=True, which="major")

        # Plot targets
        for target_pos in self.targets_pos:
            if self.target_grid[target_pos[0], target_pos[1]] == 1:
                plt.scatter(target_pos[0], target_pos[1], c='g', marker='o', s=20)
            else:
                plt.scatter(target_pos[0], target_pos[1], c='y', marker='o', s=20)

        for agent in self.agents:
            plt.scatter(agent.pos[0], agent.pos[1], c='b', marker='o', s=20)

        plt.tight_layout()
        plt.show()
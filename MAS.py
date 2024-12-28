import torch
from torch import nn
import gymnasium as gym
from torch import multiprocessing as mp

from tqdm import tqdm
# import optuna
import numpy as np
from collections import deque, namedtuple
import random
import argparse
from MAS_Environment import MAS_env
# import wandb

parser = argparse.ArgumentParser(description="Hyperparameter parser")
parser.add_argument("--asynchronous",
                    type=str,
                    help="asynchronous")
args = parser.parse_args()

# ENV_NAME        = "Walker2d-v4"

env = MAS_env(grid_size=32,
              obs_size=5,
              num_agents=8,
              num_targets=12)

N_STATES        = env.observation_space.shape[0]
N_ACTIONS       = env.action_space.n
del env

N_HIDDEN        = 128

if args.asynchronous == "True":
    # N_WORKERS       = mp.cpu_count()
    N_WORKERS       = 4
    print(f"Running A3C with {N_WORKERS} workers")
else:
    N_WORKERS       = 1
    print("Running A2C")

N_AGENTS        = 4

LR_MASTER       = 1e-4
LR_WORKER       = 1e-4

DISCOUNT_FACTOR = 0.99

MAX_EPISODES    = 100
N_STEP          = 512
# N_STEP          = 10

GRAD_CLIP       = None
TERMINATION_REWARD = 200
REWARD_SCALING  = 1e-11
RETAIN_GRAPH    = True

DEVICE          = torch.device("cpu")
DTYPE           = torch.float32

# if args.lr is not None:
#     LR = args.lr

# if args.df is not None:
#     DISCOUNT_FACTOR = args.df

# print(LR, DISCOUNT_FACTOR)

Transition = namedtuple(typename="Transition",
                        field_names=("policy_distribution",
                                     "action",
                                     "reward",
                                     "value_current_state"))

class ReplayBuffer(object):

    def __init__(self):
        self.buffer = deque([],
                            maxlen=N_STEP)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size=N_STEP):
        return random.sample(self.buffer,
                             k=batch_size)

    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer.clear()

class SharedAdam(torch.optim.Adam):
    def __init__(self,
               params,
               lr=LR_MASTER,
               betas=(0.9, 0.99),
               eps=1e-8,
               weight_decay=0):
        super(SharedAdam, self).__init__(params,
                                       lr=lr,
                                       betas=betas,
                                       eps=eps,
                                       weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):

    def __init__(self):

        super(ActorCritic, self).__init__()
        
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(3, 3)),
            nn.BatchNorm2d(num_features=8,
                           eps=1e-5,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(3, 3)),
            nn.BatchNorm2d(num_features=16,
                           eps=1e-5,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),
            # nn.Conv2d(in_channels=1,
            #           out_channels=8,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1),
            # nn.MaxPool2d(kernel_size=2,
            #     stride=2),
            # nn.Conv2d(in_channels=8,
            #           out_channels=16,
            #           kernel_size=2,
            #           stride=1,
            #           padding=1),
            # nn.MaxPool2d(kernel_size=2,
            #              stride=2)
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(68, N_HIDDEN),
            nn.Tanh(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(N_HIDDEN, N_ACTIONS),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(N_HIDDEN, 1)

    def forward(self, x_1, x_2):
        x_1 = self.shared_conv(x_1)
        x_1 = torch.flatten(x_1,
                            1)
        # print(x_1.shape)
        # print(x_2.shape)
        x = torch.cat((x_1, x_2),
                       dim=1)
        # print(x.shape)
        x = self.shared_fc(x)
        return self.actor(x), self.critic(x)
    
class A3CMaster():

    def __init__(self):

        self.global_model = ActorCritic().to(DEVICE).to(DTYPE)
        self.global_model.share_memory()

        self.global_optimizer = SharedAdam(self.global_model.parameters())

        self.episode_rewards = []
        self.episode_losses = []
        self.episode_steps = []
        self.episode_targets = []

        self.global_queue = mp.Queue()
        self.global_step_counter = mp.Value("i", 0)
    
    def train(self):

        workers = [A3CWorker(self.global_model,
                             self.global_optimizer,
                             self.global_step_counter,
                             self.global_queue)
                   for _ in range(N_WORKERS)]

        [worker.start() for worker in workers]

        done = 0
        while done < N_WORKERS:
            r, l, t, targs = self.global_queue.get()

            if r is not None:
                self.episode_rewards.append(r)
                self.episode_losses.append(l)
                self.episode_steps.append(t)
                self.episode_targets.append(targs)

            else:
                done += 1
            
            # if sum(self.episode_rewards[-100:]) >= 50000:
            #     done = N_WORKERS
            #     [worker.terminate() for worker in workers]
            #     print("Achieved target score, stopping training early")
                
        print("Exit Loop")

        # [worker.join() for worker in workers]

        return self.episode_rewards, self.episode_losses, self.episode_steps, self.episode_targets
        
class A3CWorker(mp.Process):

    def __init__(self,
                 global_model,
                 global_optimizer,
                 globa_step_counter,
                 global_queue):

        super(A3CWorker, self).__init__()

        self.env = MAS_env(grid_size=32,
                           obs_size=5,
                           num_agents=8,
                           num_targets=12)
        self.local_model = ActorCritic().to(DEVICE).to(DTYPE)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(),
                                                lr=LR_WORKER)

        self.global_model = global_model
        self.global_optimizer = global_optimizer

        self.global_step_counter = globa_step_counter
        self.global_queue = global_queue

        self.replay_buffer = ReplayBuffer()

        self.step_counter = 0

    def run(self):

        for _ in tqdm(range(MAX_EPISODES)):
            self.synchronize_models()

            state_1, state_2 = self.env.reset()
            # print(state_1.shape)
            # print(state_2.shape)
            state_1 = torch.tensor(state_1,
                                   dtype=DTYPE,
                                   device=DEVICE).unsqueeze(1)
            state_2 = torch.tensor(state_2,
                                   dtype=DTYPE,
                                   device=DEVICE)
            # print(state_1.shape)
            # print(state_2.shape)
            
            episode_reward = 0
            episode_loss = 0

            done = False

            start_step = self.step_counter

            while not done:
                policy, value_current_state = self.local_model.forward(state_1, state_2)
                policy_distribution = torch.distributions.Categorical(policy)
                action = policy_distribution.sample()

                next_state_1, next_state_2, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())

                self.global_step_counter.value += 1

                self.step_counter += 1

                next_state_1 = torch.tensor(next_state_1,
                                            dtype=DTYPE,
                                            device=DEVICE).unsqueeze(1)
                next_state_2 = torch.tensor(next_state_2,
                                            dtype=DTYPE,
                                            device=DEVICE)

                episode_reward = episode_reward + reward.sum()

                reward *= REWARD_SCALING
                # print(f"Reward: {reward.shape}")
                # print(f"Value: {value_current_state.shape}")

                if done:
                    reward += TERMINATION_REWARD
                reward = torch.tensor(reward,
                                      dtype=DTYPE,
                                      device=DEVICE).unsqueeze(1)

                self.replay_buffer.push(policy_distribution,
                                        action,
                                        reward,
                                        value_current_state)

                state_1, state_2 = next_state_1, next_state_2

                done = terminated or truncated

                if done or (self.step_counter - start_step == N_STEP):
                    episode_loss += self.accumulate_gradients(terminated,
                                                              next_state_1,
                                                              next_state_2,
                                                              start_step)
                    # if episode_reward > 50:
                    #     [self.asynchronous_update() for _ in range(100)]
                    # else:
                    #     self.asynchronous_update()
                    self.asynchronous_update()
                    start_step = self.step_counter
                
            self.global_queue.put((episode_reward, episode_loss, self.env.step_counter, self.env.targets_left))
        
        self.global_queue.put((None, None, None, None))

    def accumulate_gradients(self,
                             terminated,
                             next_state_1,
                             next_state_2,
                             start_step):
        
        memory = Transition(*zip(*self.replay_buffer.sample(self.replay_buffer.__len__())))

        update_loss = 0

        if terminated:
            R = 0
        else:
            _, value_next_state = self.local_model.forward(next_state_1, next_state_2)
            R = value_next_state
        
        for t in reversed(range(self.step_counter - start_step)):
            R *= DISCOUNT_FACTOR
            # print(f"Reward memory: {memory.reward[t].shape}")
            R += memory.reward[t]

            advantage = R - memory.value_current_state[t]
            log_policy = -memory.policy_distribution[t].log_prob(memory.action[t])

            actor_loss = log_policy*advantage
            critic_loss = 0.5 * torch.square(advantage)

            loss = (actor_loss + critic_loss).sum()
            loss.backward(retain_graph=RETAIN_GRAPH)
            update_loss += loss.item()
        
        if GRAD_CLIP is not None:
            nn.utils.clip_grad_value_(self.local_model.parameters(),
                                        clip_value=GRAD_CLIP)
        
        self.replay_buffer.reset()

        return update_loss

    def asynchronous_update(self):

        self.global_optimizer.zero_grad()

        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone().detach()
            else:
                global_param.grad += local_param.grad.clone().detach()
        
        self.global_optimizer.step()
    
    def synchronize_models(self):
        self.local_optimizer.zero_grad()
        self.local_model.load_state_dict(self.global_model.state_dict())

if __name__ == "__main__":
    trainer = A3CMaster()
    episode_rewards, episode_losses, episode_steps, episode_targets = trainer.train()
    torch.save(episode_rewards,
               "a3c_mas_rewards.pkl")
    torch.save(episode_losses,
               "a3c_mas_losses.pkl")
    torch.save(episode_steps,
               "a3c_mas_steps.pkl")
    torch.save(episode_targets,
               "a3c_mas_targets.pkl")
    torch.save(trainer.global_model.state_dict(),
               "a3c_mas_checkpoint.pkl")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
    
    # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="my-awesome-project",
    #     # name=f"lr_{LR}df{DISCOUNT_FACTOR}gc{GRAD_CLIP}",
    #     name="run",
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate_master": LR_MASTER,
    #     "learning_rate_worker": LR_WORKER,
    #     "discount_factor": DISCOUNT_FACTOR,
    #     "grad_clip": GRAD_CLIP,
    #     "n_hidden": N_HIDDEN,
    #     "n_step": N_STEP,
    #     "epochs": MAX_EPISODES*N_WORKERS,
    #     "termination_reward": TERMINATION_REWARD,
    #     "retain_graph": RETAIN_GRAPH
    #     }
    # )
    
    # for r, l, t in zip(episode_rewards, episode_losses, episode_steps):
    #     wandb.log({"reward": r,
    #                 "loss": l,
    #                 "step": t})
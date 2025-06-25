import os

import numpy as np
import torch
import doko_minimal_changed_rewards_env
from gymnasium import spaces
from tqdm import tqdm

from agilerl.utils.utils import create_population
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

from agilerl.algorithms import IPPO
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the network configuration
NET_CONFIG = {
      "encoder_config": {'hidden_size': [32, 32]},  # Network head hidden size
      "head_config": {'hidden_size': [32]}      # Network head hidden size
  }

# TODO: Define the initial hyperparameters for IPPO, this is just copypaste from MADDPG or MATD3
# Define the initial hyperparameters
INIT_HP = {
    "POP_SIZE": 4,
    "ALGO": "IPPO",  # Algorithm
    # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    "CHANNELS_LAST": False,
    "BATCH_SIZE": 32,  # Batch size
    "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
    "EXPL_NOISE": 0.1,  # Action noise scale
    "MEAN_NOISE": 0.0,  # Mean action noise
    "THETA": 0.15,  # Rate of mean reversion in OU noise
    "DT": 0.01,  # Timestep for OU noise
    "LR_ACTOR": 0.001,  # Actor learning rate
    "LR_CRITIC": 0.001,  # Critic learning rate
    "GAMMA": 0.95,  # Discount factor
    "MEMORY_SIZE": 100000,  # Max memory buffer size
    "LEARN_STEP": 64,  # Learning frequency
    "TAU": 0.01,  # For soft update of target parameters
    "POLICY_FREQ": 2,  # Policy frequnecy
    "LR": 0.001, # NEW
    "GAE_LAMBDA": 0.95, # NEW
    "ACTION_STD_INIT": 0, # NEW 
    "CLIP_COEF": 0.2, # NEW
    "ENT_COEF": 0.01, # NEW 
    "VF_COEF": 0.5, # NEW
    "MAX_GRAD_NORM": 0.5, # NEW
    "TARGET_KL": None, # NEW
    "UPDATE_EPOCHS": 4, # NEW

}

num_envs = 8
# Define the simple speaker listener environment as a parallel environment
env = AsyncPettingZooVecEnv(
    [
        lambda: turn_based_aec_to_parallel(doko_minimal_changed_rewards_env.env(render_mode = "ansi"))
        for _ in range(num_envs)
    ]
)
env.reset()

# Configure the multi-agent algo input arguments
observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
action_spaces = [env.single_action_space(agent) for agent in env.agents]


# Append number of agents and agent IDs to the initial hyperparameter dictionary
INIT_HP["N_AGENTS"] = env.num_agents
INIT_HP["AGENT_IDS"] = env.agents

# create population
agent_pop = create_population(
    algo="IPPO",  # RL algorithm
    observation_space = observation_spaces,  # State dimension
    action_space = action_spaces,  # Action dimension
    net_config=NET_CONFIG,  # Network configuration
    INIT_HP=INIT_HP,  # Initial hyperparameters
    population_size=INIT_HP["POP_SIZE"],  # Population size
    num_envs=num_envs,  # Number of vectorized envs
    device=device,
)

# Instantiate a tournament selection object (used for HPO)
tournament = TournamentSelection(
    tournament_size=2,  # Tournament selection size
    elitism=True,  # Elitism in tournament selection
    population_size=INIT_HP["POP_SIZE"],  # Population size
    eval_loop=1,  # Evaluate using last N fitness scores
)

# Instantiate a mutations object (used for HPO)
mutations = Mutations(
    no_mutation=0.2,  # Probability of no mutation
    architecture=0.2,  # Probability of architecture mutation
    new_layer_prob=0.2,  # Probability of new layer mutation
    parameters=0.2,  # Probability of parameter mutation
    activation=0,  # Probability of activation function mutation
    rl_hp=0.2,  # Probability of RL hyperparameter mutation
    mutation_sd=0.1,  # Mutation strength
    rand_seed=1,
    device=device,
)




trained_pop, pop_fitnesses = train_multi_agent_on_policy(
    env=env,                              # Gym-style environment
    env_name="ParallelDoko",  # Environment name
    pop=agent_pop,  # Population of agents
    swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
    max_steps=200000,  # Max number of training steps
    evo_steps=10000,  # Evolution frequency
    eval_steps=None,  # Number of steps in evaluation episode
    eval_loop=1,  # Number of evaluation episodes
    target=200.,  # Target score for early stopping
    tournament=None,  # Tournament selection object
    mutation=None,  # Mutations object
    wb=True,  # Weights and Biases tracking
)
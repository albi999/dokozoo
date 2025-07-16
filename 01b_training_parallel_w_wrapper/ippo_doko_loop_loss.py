import os
import numpy as np
import torch
import tb_doko_env_V5
from gymnasium import spaces
from conversions import turn_based_aec_to_parallel
# from pettingzoo.utils import turn_based_aec_to_parallel
from tqdm import tqdm

# from agilerl.algorithms import IPPO
from ippo import IPPO
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_envs = 8    
env = AsyncPettingZooVecEnv(
    [
        lambda: turn_based_aec_to_parallel(tb_doko_env_V5.env())
        for _ in range(num_envs)
    ]
)
env.reset()

# Configure the multi-agent algo input arguments
observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
action_spaces = [env.single_action_space(agent) for agent in env.agents]
agent_ids = [agent_id for agent_id in env.agents]


NET_CONFIG = {
    "encoder_config" : {
        "latent_dim": 128, # Latent dimension outputted by underlying feature extractors
        "min_latent_dim": 64, # Minimum latent dimension when mutating
        "max_latent_dim": 256, # Maximum latent dimension when mutating
        "mlp_config": {
            "hidden_size": [256, 128],
            "activation": "ReLU",
        },
        "vector_space_mlp": True # Process vector observations with an MLP
    },
    "head_config" : {
        "hidden_size": [64, 64], # Two layers of 64 nodes each
        "min_mlp_nodes": 32, # minimum number of nodes in the MLP when mutating
        "max_mlp_nodes": 128, # maximum number of nodes in the MLP when mutating
    }
}

""" CNN_CONFIG = {
    "channel_size": [32, 32], # CNN channel size
    "kernel_size": [8, 4],   # CNN kernel size
    "stride_size": [4, 2],   # CNN stride size
}

NET_CONFIG = {
    "encoder_config": {
      "latent_dim": 32,
      # Config for nested EvolvableCNN objects
      "cnn_config": CNN_CONFIG,
      # Config for nested EvolvableMLP objects
      "mlp_config": {
          "hidden_size": [32, 32]
      },
      "vector_space_mlp": True # Process vector observations with an MLP
    },
    "head_config": {'hidden_size': [32]}  # Network head hidden size
  } """


agent = IPPO(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    net_config = NET_CONFIG,
    device=device,
    batch_size=128, 
    learn_step=1280 # TODO change
)

# print(agent.actors)
# print(agent.critics)

# losses
min_losses = np.array([np.inf for i in range(5)], dtype=np.float64)

# Define training loop parameters
max_steps = 10000  # Max steps
pbar = tqdm(total=max_steps)    
while agent.steps[-1] < max_steps:
    obs, info  = env.reset() # Reset environment at start of episode
    # print("ippo_doko_loop")
    # print(f"obs: {obs}")
    scores = np.zeros((num_envs, len(agent.shared_agent_ids)))
    completed_episode_scores = []
    steps = 0
    for _ in range(agent.learn_step):
        states = {agent_id: [] for agent_id in agent.agent_ids}
        actions = {agent_id: [] for agent_id in agent.agent_ids}
        log_probs = {agent_id: [] for agent_id in agent.agent_ids}
        entropies = {agent_id: [] for agent_id in agent.agent_ids}
        rewards = {agent_id: [] for agent_id in agent.agent_ids}
        dones = {agent_id: [] for agent_id in agent.agent_ids}
        values = {agent_id: [] for agent_id in agent.agent_ids}

        done = {agent_id: np.zeros(num_envs) for agent_id in agent.agent_ids}

        for _ in range(-(agent.learn_step // -num_envs)):
            
            # print("################# [OBS] ###################")
            # print(obs)

            # print("################# [INFO] ###################")
            # print(info)

            # Get next action from agent
            action, log_prob, entropy, value = agent.get_action(
                obs=obs, infos=info
            )
            
            
            # print("################# [ACTION] ###################")
            # print(action)


            # ignore hallucinated actions
            # get active_agents from info_dict
            first_agent = agent.agent_ids[0]
            active_agents = info[first_agent]['active_agent']
            # print(active_agents)
            for ag in action.keys():
                for n in range(num_envs):
                    if ag != active_agents[n]:
                        action[ag][n] = 20

            # print("################# [CORRECTED ACTION] ###################")
            # print(action)

            # Clip to action space
            clipped_action = {}
            for agent_id, agent_action in action.items():
                shared_id = agent.get_homo_id(agent_id)
                actor_idx = agent.shared_agent_ids.index(shared_id)
                agent_space = agent.action_space[agent_id]
                if isinstance(agent_space, spaces.Box):
                    if agent.actors[actor_idx].squash_output:
                        clipped_agent_action = agent.actors[actor_idx].scale_action(agent_action)
                    else:
                        clipped_agent_action = np.clip(agent_action, agent_space.low, agent_space.high)
                else:
                    clipped_agent_action = agent_action

                clipped_action[agent_id] = clipped_agent_action

            # Act in environment
            next_obs, reward, termination, truncation, info = env.step(
                clipped_action
            )

            # Compute score increment (replace NaNs representing inactive agents with 0)
            agent_rewards = np.array(list(reward.values())).transpose()
            agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
            score_increment = np.sum(agent_rewards, axis=-1)[:, np.newaxis]

            scores += score_increment
            steps += num_envs


            # print(f"agents: {obs.keys()}")
            # Save transition
            for agent_id in obs:
                states[agent_id].append(obs[agent_id])
                rewards[agent_id].append(reward[agent_id])
                actions[agent_id].append(action[agent_id])
                log_probs[agent_id].append(log_prob[agent_id])
                entropies[agent_id].append(entropy[agent_id])
                values[agent_id].append(value[agent_id])
                dones[agent_id].append(done[agent_id])

            # Find which agents are "done" - i.e. terminated or truncated
            next_done = {}
            for agent_id in termination:
                terminated = termination[agent_id]
                truncated = truncation[agent_id]

                # Process asynchronous dones
                mask = ~(np.isnan(terminated) | np.isnan(truncated))
                result = np.full_like(mask, np.nan, dtype=float)
                result[mask] = np.logical_or(
                    terminated[mask], truncated[mask]
                )

                next_done[agent_id] = result

            obs = next_obs
            done = next_done
            for idx, agent_dones in enumerate(zip(*next_done.values())):
                if all(agent_dones):
                    completed_score = list(scores[idx])
                    completed_episode_scores.append(completed_score)
                    agent.scores.append(completed_score)
                    scores[idx].fill(0)

                    done = {
                        agent_id: np.zeros(num_envs)
                        for agent_id in agent.agent_ids
                    }

        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_obs,
            next_done,
        )

        # Learn according to agent's RL algorithm
        loss = agent.learn(experiences)
        pbar.update(-(agent.learn_step // -num_envs))
        pbar.set_description(f"Loss: {loss}")

        loss_mean = np.mean(list(loss.values()))
        print(loss_mean)
        if loss_mean < min_losses[0]:
            min_losses[0] = loss_mean
            # Save the trained algorithm
            path = "./models/IPPO"
            filename = "IPPO_min_avg_loss.pt"
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path, filename)
            agent.save_checkpoint(save_path)

        for i in range(1,5):
            agent_name = f"agentt{i}"
            if loss[agent_name] < min_losses[i]:
                min_losses[i] = loss[agent_name]
                # Save the trained algorithm
                path = "./models/IPPO"
                filename = f"IPPO_min_loss_agentt{i}.pt"
                os.makedirs(path, exist_ok=True)
                save_path = os.path.join(path, filename)
                agent.save_checkpoint(save_path)
        
        
    agent.steps[-1] += steps


# Save the trained algorithm
path = "./models/IPPO"
filename = "IPPO_final_trained_agent.pt"
os.makedirs(path, exist_ok=True)
save_path = os.path.join(path, filename)
agent.save_checkpoint(save_path)

pbar.close()
env.close()

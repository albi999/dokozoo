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

def calc_avg_IPPO_agent_reward_vs_randoms(ippo):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    # env = turn_based_aec_to_parallel(tb_doko_env_V4.env(render_mode = "ansi"))
    env = turn_based_aec_to_parallel(tb_doko_env_V5.env())
    env.reset()
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        discrete_actions = True
        max_action = None
        min_action = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        discrete_actions = False
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]


    # Extracting number of agents and agent IDs from env
    n_agents = env.num_agents
    agent_ids = env.agents

    # Define test loop parameters
    episodes = 10  # Number of episodes to test agent on
    max_steps = 40  # Max number of steps to take in the environment in each episode = 40

    ippo_rewards = [] # List to collect sum of episodic reward of IPPO agents

    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset(ep)

        termination = {agent_id: False for agent_id in agent_ids}
        truncation = {agent_id: False for agent_id in agent_ids}
        agent_reward = {agent_id: 0 for agent_id in agent_ids}


        # determine which agents are ippo_agents and which are random
        # let's say agentt1 is always an IPPO agent
        # print(env.aec_env.teams)
        agentt1_team = env.aec_env.teams[0]
        ippo_agents_indices = np.where(env.aec_env.teams == agentt1_team)[0]
        ippo_agents = []
        for i in ippo_agents_indices:
            ippo_agents.append(f"agentt{i+1}")

        
        for _ in range(max_steps):
            active_agent = env.aec_env.agent_selection
            # print(f"active_agent: {active_agent}")
            if active_agent in ippo_agents:
                # Get next action from IPPO-agent
                action, log_prob, entropy, value = ippo.get_action(
                    obs=state, infos=info
                )
                # print(f"ippo_action: {action}")
            else: 
                # get random action
                if termination[active_agent] or truncation[active_agent]:
                    print("term trunc")
                    rand_action = None  # Action is set to None if the episode is over
                else:
                    # Check if the environment provides an action mask
                    if "action_mask" in info[active_agent]:
                        mask = np.array(info[active_agent]["action_mask"])
                    else:
                        mask = None  # If no mask is provided, proceed without it
                    
                    if mask is not None:
                        # Ensure mask is applied to the valid actions (modify as necessary based on your environment)
                        valid_actions = np.where(mask == 1)[0]
                        if valid_actions.size > 0:
                            rand_action = np.random.choice(valid_actions)  # Randomly sample a valid action
                        else:
                            print("no valid actions")
                            rand_action = None  # If no valid actions, set to None or handle as needed
                    else:
                        rand_action = env.action_space(active_agent).sample()  # Sample a random action if no mask

                
                action = {}
                for agent in env.agents:
                    if agent == env.aec_env.agent_selection:
                        action[agent] = rand_action
                    else:
                        action[agent] = np.array([20])

                # print(f"random_action: {action}")
            
            # Take action in environment
            state, reward, termination, truncation, info = env.step(
                {agent: a.squeeze() for agent, a in action.items()}
            )   

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        
        # Record IPPO agents episodic reward
        ippo_reward = 0
        for agent_id in ippo_agents:
            ippo_reward += agent_reward[agent_id]

        ippo_rewards.append(ippo_reward)

    env.close()
    return np.mean(ippo_rewards)



if __name__ == "__main__":
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


    IPPO_agent = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        net_config = NET_CONFIG,
        device=device,
        batch_size=128, 
        learn_step=1280 # TODO change
    )

    # print(IPPO_agent.actors)
    # print(IPPO_agent.critics)

    # Initial values for deciding which model to save
    min_avg_loss = np.inf
    max_avg_ippo_reward = 0


    # Define training loop parameters
    max_steps = 10000  # Max steps
    pbar = tqdm(total=max_steps)    
    while IPPO_agent.steps[-1] < max_steps:
        obs, info  = env.reset() # Reset environment at start of episode
        # print("ippo_doko_loop")
        # print(f"obs: {obs}")
        scores = np.zeros((num_envs, len(IPPO_agent.shared_agent_ids)))
        completed_episode_scores = []
        steps = 0
        for _ in range(IPPO_agent.learn_step):
            states = {agent_id: [] for agent_id in IPPO_agent.agent_ids}
            actions = {agent_id: [] for agent_id in IPPO_agent.agent_ids}
            log_probs = {agent_id: [] for agent_id in IPPO_agent.agent_ids}
            entropies = {agent_id: [] for agent_id in IPPO_agent.agent_ids}
            rewards = {agent_id: [] for agent_id in IPPO_agent.agent_ids}
            dones = {agent_id: [] for agent_id in IPPO_agent.agent_ids}
            values = {agent_id: [] for agent_id in IPPO_agent.agent_ids}

            done = {agent_id: np.zeros(num_envs) for agent_id in IPPO_agent.agent_ids}

            for _ in range(-(IPPO_agent.learn_step // -num_envs)):
                # Get next action from agent
                action, log_prob, entropy, value = IPPO_agent.get_action(
                    obs=obs, infos=info
                )
                
                
                # print("################# [ACTION] ###################")
                # print(action)


                # ignore hallucinated actions
                # get active_agents from info_dict
                first_agent = IPPO_agent.agent_ids[0]
                active_agents = info[first_agent]['active_agent']
                # print(active_agents)
                for ag in action.keys():
                    for n in range(num_envs):
                        if ag != active_agents[n]:
                            action[ag][n] = 20

                # print("################# [CORRECTED ACTION] ###################")
                # print(action)

                # Act in environment
                next_obs, reward, termination, truncation, info = env.step(
                    action
                )

                # Compute score increment (replace NaNs representing inactive agents with 0)
                agent_rewards = np.array(list(reward.values())).transpose()
                agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
                score_increment = np.sum(agent_rewards, axis=-1)[:, np.newaxis]

                scores += score_increment
                steps += num_envs


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
                        IPPO_agent.scores.append(completed_score)
                        scores[idx].fill(0)

                        done = {
                            agent_id: np.zeros(num_envs)
                            for agent_id in IPPO_agent.agent_ids
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
            loss = IPPO_agent.learn(experiences)
            

            avg_ippo_reward = calc_avg_IPPO_agent_reward_vs_randoms(IPPO_agent)
            if avg_ippo_reward > max_avg_ippo_reward:
                max_avg_ippo_reward = avg_ippo_reward
                # Save the trained algorithm
                path = "./models/IPPO"
                filename = "max_avg_IPPO_reward.pt"
                os.makedirs(path, exist_ok=True)
                save_path = os.path.join(path, filename)
                IPPO_agent.save_checkpoint(save_path)

            pbar.update(-(IPPO_agent.learn_step // -num_envs))
            pbar.set_description(f"avg_IPPO_reward: {avg_ippo_reward} | max_avg_ippo_reward: {max_avg_ippo_reward}")


            """
            loss_mean = np.mean(list(loss.values()))
            # print(loss_mean) 
            if loss_mean < min_avg_loss:
                min_avg_loss = loss_mean
                # Save the trained algorithm
                path = "./models/IPPO"
                filename = "IPPO_min_avg_loss.pt"
                os.makedirs(path, exist_ok=True)
                save_path = os.path.join(path, filename)
                IPPO_agent.save_checkpoint(save_path)
            """
            
            
        IPPO_agent.steps[-1] += steps


    # Save the trained algorithm
    path = "./models/IPPO"
    filename = "IPPO_final_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    IPPO_agent.save_checkpoint(save_path)

    pbar.close()
    env.close()




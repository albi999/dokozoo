import os

import numpy as np
import torch
import tb_doko_env_V5
from pettingzoo.utils.conversions import turn_based_aec_to_parallel

# from agilerl.algorithms import IPPO
from ippo import IPPO


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = turn_based_aec_to_parallel(tb_doko_env_V5.env(render_mode = "ansi"))
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

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    # Load the saved agent
    path = "./models/IPPO/IPPO_max_avg_IPPO_reward.pt"
    ippo = IPPO.load(path, device)

    # Define test loop parameters
    episodes = 1  # Number of episodes to test agent on
    max_steps = 40  # Max number of steps to take in the environment in each episode TODO: probably 40, maybe 10

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            # Get next action from agent
            action, log_prob, entropy, value = ippo.get_action(
                obs=state, infos=info
            )

            # Save the frame for this step and append to frames list
            # print(env.render())
            # frames.append(_label_with_episode_number(frame, episode_num=ep))
            
            # Take action in environment
            state, reward, termination, truncation, info = env.step(
                {agent: a.squeeze() for agent, a in action.items()}
            )

            print("actions")
            print(action) 
            
            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    
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

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    # Load the saved agent
    path = "./models/IPPO/IPPO_min_avg_loss.pt"
    ippo = IPPO.load(path, device)

    # Define test loop parameters
    episodes = 10  # Number of episodes to test agent on
    max_steps = 40  # Max number of steps to take in the environment in each episode = 40

    rewards = []  # List to collect total episodic reward
    ippo_rewards = [] # List to collect sum of episodic reward of IPPO agents
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards







    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset(ep)
        env.aec_env.print_player_cards()

        termination = {agent_id: False for agent_id in agent_ids}
        truncation = {agent_id: False for agent_id in agent_ids}
        
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0


        # determine which agents are ippo_agents and which are random
        # let's say agentt1 is always an IPPO agent
        # print(env.aec_env.teams)
        agentt1_team = env.aec_env.teams[0]
        # print(agentt1_team)
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

        # Determine total score for the episode and then append to rewards list
        score = sum(agent_reward.values())
        rewards.append(score)

        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print(ippo_agents)
        print("IPPO agents rewards:")
        ippo_reward = 0
        for agent_id in ippo_agents:
            print(f"{agent_id} reward: {indi_agent_rewards[agent_id][-1]}")
            ippo_reward += indi_agent_rewards[agent_id][-1]
        print(f"IPPO agent sum: {ippo_reward}")
        ippo_rewards.append(ippo_reward)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")

    print(ippo_rewards)
    print(np.mean(ippo_rewards))
    env.close()

    
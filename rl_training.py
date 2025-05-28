import doko_environment_mdp_minimal
import numpy as np

env = doko_environment_mdp_minimal.env(render_mode = "ansi")
env.reset(seed=42)


for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None  # Action is set to None if the episode is over
    else:
        # Check if the environment provides an action mask
        if "action_mask" in info:
            mask = info["action_mask"]
        elif isinstance(observation, dict) and "action_mask" in observation:
            mask = observation["action_mask"]
        else:
            mask = None  # If no mask is provided, proceed without it
        
        if mask is not None:
            # Ensure mask is applied to the valid actions (modify as necessary based on your environment)
            valid_actions = np.where(mask == 1)[0]
            if valid_actions.size > 0:
                action = np.random.choice(valid_actions)  # Randomly sample a valid action
            else:
                action = None  # If no valid actions, set to None or handle as needed
        else:
            action = env.action_space(agent).sample()  # Sample a random action if no mask
            
    
    env.step(action)  # Step with the chosen action
    done = all(env.terminations.values())  # Check if all agents are terminated or truncated
    if done:
        print("Game Over!")
        break  # Exit the loop if the game is over
env.close()

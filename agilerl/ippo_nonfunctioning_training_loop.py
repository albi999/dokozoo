import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tqdm import trange

from agilerl.algorithms import IPPO
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.utils.algo_utils import obs_channels_to_first

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_envs = 8
env = AsyncPettingZooVecEnv(
    [
        lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=True)
        for _ in range(num_envs)
    ]
)
env.reset()

# Configure the multi-agent algo input arguments
observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
action_spaces = [env.single_action_space(agent) for agent in env.agents]
agent_ids = [agent_id for agent_id in env.agents]

channels_last = False  # Flag to swap image channels dimension from last to first [H, W, C] -> [C, H, W]

agent = IPPO(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    device=device,
)

# Define training loop parameters
max_steps = 100000  # Max steps
while agent.steps[-1] < max_steps:
    state, info  = env.reset() # Reset environment at start of episode
    scores = np.zeros((num_envs, len(agent.shared_agent_ids)))
    completed_episode_scores = []
    steps = 0

    if channels_last:
        state = {
            agent_id: obs_channels_to_first(s)
            for agent_id, s in state.items()
        }

    for _ in range(agent.learn_step):

        states = {agent_id: [] for agent_id in agent.agent_ids}
        actions = {agent_id: [] for agent_id in agent.agent_ids}
        log_probs = {agent_id: [] for agent_id in agent.agent_ids}
        rewards = {agent_id: [] for agent_id in agent.agent_ids}
        dones = {agent_id: [] for agent_id in agent.agent_ids}
        values = {agent_id: [] for agent_id in agent.agent_ids}

        done = {agent_id: np.zeros(num_envs) for agent_id in agent.agent_ids}

        for idx_step in range(-(agent.learn_step // -num_envs)):

            # Get next action from agent
            action, log_prob, _, value = agent.get_action(obs=state, infos=info)

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
            next_state, reward, termination, truncation, info = env.step(clipped_action)
            scores += np.array(list(reward.values())).transpose()

            steps += num_envs

            next_done = {}
            for agent_id in agent.agent_ids:
                states[agent_id].append(state[agent_id])
                actions[agent_id].append(action[agent_id])
                log_probs[agent_id].append(log_prob[agent_id])
                rewards[agent_id].append(reward[agent_id])
                dones[agent_id].append(done[agent_id])
                values[agent_id].append(value[agent_id])
                next_done[agent_id] = np.logical_or(termination[agent_id], truncation[agent_id]).astype(np.int8)

            if channels_last:
                next_state = {
                    agent_id: obs_channels_to_first(s)
                    for agent_id, s in next_state.items()
                }

            # Find which agents are "done" - i.e. terminated or truncated
            dones = {
                agent_id: termination[agent_id] | truncation[agent_id]
                for agent_id in agent.agent_ids
            }

            # Calculate scores for completed episodes
            for idx, agent_dones in enumerate(zip(*dones.values())):
                if all(agent_dones):
                    completed_score = list(scores[idx])
                    completed_episode_scores.append(completed_score)
                    agent.scores.append(completed_score)
                    scores[idx].fill(0)

            state = next_state
            done = next_done

        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
            next_done,
        )

        # Learn according to agent's RL algorithm
        loss = agent.learn(experiences)

    agent.steps[-1] += steps
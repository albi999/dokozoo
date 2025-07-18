from pettingzoo.test import parallel_api_test
from pettingzoo.utils.conversions import turn_based_aec_to_parallel
import tb_doko_rewards_env
import numpy as np


if __name__ == "__main__":
    env = tb_doko_rewards_env.env(render_mode = "ansi")
    parallel_env = turn_based_aec_to_parallel(env)
    parallel_api_test(parallel_env, num_cycles=1_000_000)
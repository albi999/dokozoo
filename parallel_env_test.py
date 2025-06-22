import doko_parallel_env

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = doko_parallel_env.env(render_mode = "ansi")
    parallel_api_test(env, num_cycles=1_000_000)

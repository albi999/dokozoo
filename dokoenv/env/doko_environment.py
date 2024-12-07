import functools
import random
from copy import copy
import numpy as np


from pettingzoo import AECEnv
import doko_cards.py as cards

class DokoEnvironment(AECEnv):
    metadata = {
        "name": "doko_environment_v0",
    }

    def __init__(self):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - round
        - cards of each player
        - possible_agents
        - Cards in the middle?????

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.round = None
        self.player1_cards = None
        self.player2_cards = None
        self.player3_cards = None
        self.player4_cards = None
        self.possible_agents = ["player1", "player2", "player3", "player4"]

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - round
        - player cards
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.round = 1
        deck = cards.create_deck()



    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
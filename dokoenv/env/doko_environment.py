import random
from copy import copy, deepcopy
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete, Dict as SpaceDict
# from gymnasium.utils import EzPickle
# TODO: Pickleeeee


from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
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
        self.deck = None
        self.player1_cards = None
        self.player2_cards = None
        self.player3_cards = None
        self.player4_cards = None
        self.possible_agents = ['player1', 'player2', 'player3', 'player4']
        self.starting_players = None
        self.played_cards = None
        # werde definitiv augen aufnehmen müssen

    
    
    
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

        self.deck = cards.create_deck()
        deck_copy = copy(self.deck)

        random.shuffle(deck_copy)
        self.player1_cards = self.cards2binary(deck_copy[:10])
        self.player2_cards = self.cards2binary(deck_copy[10:20])
        self.player3_cards = self.cards2binary(deck_copy[20:30])
        self.player4_cards = self.cards2binary(deck_copy[30:40])

        self.played_cards = np.zeros((10,4,40), dtype=bool)
        self.starting_player = np.zeros(10, dtype=bool)


        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.starting_players[0] = int(self.agent_selection[6])

        """
        ??????????????????????????????????????

        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}'
        """

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        


        

    def observe(self, agent):
        current_index = self.possible_agents.index(agent) # player1 has index 0 
        observation = {
            "round": self.round,
            "starting_players": self.starting_players
        }


        return {"observation": observation, "action_mask": action_mask}

    def render(self):
        pass

    def observation_space(self):
        observation_space = SpaceDict({
            "round": Discrete(10, start = 1),
            "starting_players": ([4] * 10),
            "cards_in_hand": MultiBinary(40),
            "cards_played": MultiBinary([10,4,40]),
            "action_mask": MultiBinary(10)
        })

        return observation_space

    def action_space(self):
        return Discrete(11-self.round)
    
    # TODO:I dont need the whole env, just the deck which could be static
    def cards2binary(self, cards): 
        binary_representation = np.zeros(40, dtype=bool)
        for card in cards:
            binary_representation[self.deck.index(card)] = True
        return binary_representation

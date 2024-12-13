import random
from copy import copy, deepcopy
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete, Dict as SpaceDict
# from gymnasium.utils import EzPickle
# TODO: Pickleeeee


from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector





"""
i   action   card  
--------------------
1   0        [♥10*]
2   1        [♣Q*] 
3   2        [♠Q*] 
4   3        [♥Q*] 
5   4        [♦Q*] 
6   5        [♣J*] 
7   6        [♠J*] 
8   7        [♥J*] 
9   8        [♦J*] 
10  9        [♦A*] 
11  10       [♦10*]
12  11       [♦K*] 
13  12       [♣A]  
14  13       [♣10] 
15  14       [♣K]  
16  15       [♠A]  
17  16       [♠10] 
18  17       [♠K]  
19  18       [♥A]  
20  19       [♥K]  
"""
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
        # self.deck = None # I don't think I need it anymore. If I know 0 -> Hearts 10 and so forth, is more efficient I think

        self.possible_agents = ['player1', 'player2', 'player3', 'player4']
        self.round = None
        self.player_cards = None
        self.starter = None # player starting at the beginning of the game
        self.cards_played = None
        self.tricks_won_by = None
        # TODO: werde definitiv augen aufnehmen müssen


    
    
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
        self.round = 0 # rounds 0 ... 9

        # dealing cards
        deck = np.repeat(np.arange(1,21), 2)
        random.Random(seed).shuffle(deck)
        player1_cards = deck[:10]
        player2_cards = deck[10:20]
        player3_cards = deck[20:30]
        player4_cards = deck[30:40]
        self.player_cards = {
            1: player1_cards,
            2: player2_cards,
            3: player3_cards,
            4: player4_cards
        }


        # TODO should be random who starts
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.starter = int(self.agent_selection[6]) # starter between 1 and 4 


        # no cards played so far, no tricks won by anyone
        self.cards_played = np.zeros((10,4), dtype=np.int64)
        self.tricks_won_by = np.zeros(10, dtype=np.int64)

        

        # copy paste shit
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        

    def step(self, action):
        # action number between 0 and 19 corresponding to cards between 1 and 20
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        agent_number = int(agent[6])
        cards_in_hand = self.player_cards[agent_number]
        action_card_indices = np.where(cards_in_hand == action+1)[0]
        if action_card_indices.size > 0:
            cards_in_hand[action_card_indices[0]] = 0
        else:
            print("Something went wrong and your code fucking sucks")
            print("@step() action_card_index is empty")
        
        self.player_cards[agent_number] = cards_in_hand
        self.cards_played[self.round][agent_number-1] = action+1

        # TODO Check if action was allowed
        # but assuming action masking works, unnecessary
        # but would be good practice I guess


        # check if trick is over
        if self._agent_selector.is_last():
            # compute trick winner 
            winner = self.trick_winner_calc(self) # TODO Implement trick_winner_calc
            self.tricks_won_by[self.round] = winner
            # TODO Check if game ended
            # if game ended: 
            #   do_shit()
            # else:
            agent_order = copy(self.possible_agents)
            winner_string = f'player{winner}'
            index = np.where(agent_order==winner_string)[0][0]
            new_agent_order = np.concatenate((agent_order[index:], agent_order[:index]))
            self._agent_selector.reinit(new_agent_order)




        

    def observe(self, agent):
        action_mask = self.action_mask_calc(self) if agent==self.agent_selection else []
        player_number = int(self.agent_selection[6])
        observation = {
            "round": self.round,
            "cards_in_hand": self.player_cards[player_number], 
            "cards_played": self.cards_played,
            "tricks_won_by": self.tricks_won_by,
            "action_mask": action_mask
        }


        return observation

    def render(self):
        pass

    def observation_space(self):
        observation_space = SpaceDict({
            "round": Discrete(10),
            "cards_in_hand": MultiDiscrete(10 * [21]),
            "cards_played": MultiDiscrete(10 * [4 * [21]]),
            "tricks_won_by": MultiDiscrete(10 * [5]),
            "action_mask": MultiBinary(20)
        })
        return observation_space

    # 40 cards but because of Duplicates only 20 possible actions
    # theoretically only max. 10 actions but this should work as well
    def action_space(self):
        return Discrete(20) 
    

    def action_mask_calc(self):
        # initializing the action mask
        action_mask = np.zeros(20, dtype=np.int8)

        # cards the selected agent could play
        # [values between 0 and 20] 
        # 0: nocard, 1: Hearts 10, 19: Hearts King 
        cards = self.player_cards[int(self.agent_selection[6])] # player cards i --> player i
        cards = cards[(1 <= cards)] # entry of 0 means there's no cards there
        
        # getting round and trick starter
        r = self.round
        trick_starter = self.starter if r == 0 else self.tricks_won_by[r-1] # P1-4
        
        # if agent starts a trick he can play any card he has
        if trick_starter == int(self.agent_selection[6]): # player 1 to 4
            return action_mask
        
        # determining which cards was played first
        # value in [1,...,20]
        # trickstarte - 1 cuz card_played indexing
        firstcard = self.cards_played[r][trick_starter-1] 

        # if first card Trump
        if firstcard in range(1,13):
            trumps = cards[(cards <= 12)]
            if np.any(trumps):
                action_mask[trumps-1] = 1 # trumps - 1 um korrespondierende action zu erhalten
                return action_mask
            else:
                action_mask[cards-1] = 1
                return action_mask
        
        # if first card non-Trump Clubs
        if firstcard in range(13,16):
            nont_clubs = cards[(13 <= cards) & (cards <= 15)]
            if np.any(nont_clubs):
                action_mask[nont_clubs-1] = 1
                return action_mask
            else: 
                action_mask[cards-1] = 1
                return action_mask
        
        # if first card non-Trump Spades
        if firstcard in range(16,19):
            nont_spades = cards[(16 <= cards) & (cards <= 18)]
            if np.any(nont_spades):
                action_mask[nont_spades-1] = 1
                return action_mask
            else: 
                action_mask[cards-1] = 1
                return action_mask
            
        # if first card non-Trump Hearts
        if firstcard in range(19,21):
            nont_hearts = cards[(19 <= cards)]
            if np.any(nont_hearts):
                action_mask[nont_hearts-1] = 1
                return action_mask 
            else:
                action_mask[cards-1] = 1
                return action_mask

        print("Something went wrong and your code fucking sucks")
        print("We're at the end of action_mask_calc()")  
        return action_mask



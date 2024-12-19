import random
from typing import Union
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete, Dict as SpaceDict
import gymnasium
# import doko_cards.py as cards
# from gymnasium.utils import EzPickle
# TODO: Pickleeeee


from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers





"""
i   action   card     tPower  
--------------------
1   0        [♥10*]   12      
2   1        [♣Q*]    11      
3   2        [♠Q*]    10      
4   3        [♥Q*]    9       
5   4        [♦Q*]    8       
6   5        [♣J*]    7       
7   6        [♠J*]    6       
8   7        [♥J*]    5       
9   8        [♦J*]    4       
10  9        [♦A*]    3       
11  10       [♦10*]   2       
12  11       [♦K*]    1       
13  12       [♣A]     0       
14  13       [♣10]    0       
15  14       [♣K]     0       
16  15       [♠A]     0       
17  16       [♠10]    0       
18  17       [♠K]     0       
19  18       [♥A]     0       
20  19       [♥K]     0       
"""


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {
        "name": "doko_environment",
        "render_modes": ["ansi"]
    }


    def __init__(self, render_mode: Union[str, None] = None):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - possible_agents                   ['player1', 'player2', 'player3', 'player4']
        - round                             0 ... 9 
        - player_cards                      {1: player1_cards, ..., 4: player4_cards}
        - cards_player                      [[12 1 2 3], [17 20 1 7], ...]
        - player_points                     [player1_points, ..., player4_points]
        - team_points                       [team1_points, team2_points]
        - current_card_index                0 ... 3
        - tricks_won_by                     [3 1 3 4 0 0 0 0 0 0]: player3 won first trick, player1 won second trick, ..., playing fifth trick atm
        - agent_selection                   'player1', ..., 'player4'
        - starter                           1, ..., 4: player who started the game
        - render_mode                       'ansi'

        These attributes should not be changed after initialization.
        """

        self.possible_agents = ['player1', 'player2', 'player3', 'player4']
        self.round = None

        self.deck = None 
        self.player_cards = None

        self.current_card_index = None 

        self.cards_played = None
        self.tricks_won_by = None

        self.player_points = None
        self.team_points = None

        # agent selection attributes
        self.agent_selection = None
        self.starter = None # player starting at the beginning of the game

        self.render_mode = render_mode

    
    
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


        self.current_card_index = 0 # indicates which card has to be played during a trick; 0:first - 3: fourth
        # no cards played so far, no tricks won by anyone
        self.cards_played = np.zeros((10,4), dtype=np.int64)
        self.tricks_won_by = np.zeros(10, dtype=np.int64)
       
        self.player_points = np.zeros(4, dtype=np.int64)
        self.team_points = np.zeros(2, dtype=np.int64)


        
        

        # TODO should be random who starts
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.starter = int(self.agent_selection[6]) # starter between 1 and 4

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
        
        r = self.round
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
        self.cards_played[r][self.current_card_index] = action+1


        # check if trick is over
        if self._agent_selector.is_last():
            # compute trick winner 
            trick_winner = self.trick_winner_calc() 
            self.tricks_won_by[r] = trick_winner
            trick_points = self.trick_points_calc()
            self.player_points[trick_winner-1] += trick_points
            
            # if game is over
            # Implemented as free for all
            # TODO: Change to team vs. team game
            if r == 9:
                game_winner = np.argmax(self.player_points) + 1
                self.set_game_result(game_winner)
                if self.render_mode == "ansi":
                    print(self.render())
                    print(self.player_points)
            
            agent_order = copy(self.possible_agents)
            # trick_winner_string = f'player{trick_winner}'
            index = trick_winner-1
            new_agent_order = np.concatenate((agent_order[index:], agent_order[:index]))
            self._agent_selector.reinit(new_agent_order)
            self.round += 1
            self.current_card_index = -1 # because we increment self.current_card_index at the end of step(), so the next would be 0 
        
        self.current_card_index += 1
        self.agent_selection = self._agent_selector.next()
        

        
        


        

    def observe(self, agent):
        # TODO add "knows_partner" bool to observations
        action_mask = self.action_mask_calc() if agent==self.agent_selection else np.zeros(20, dtype=np.int8)
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
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            render = str(self.tricks_won_by)+'\n'+str(self.cards_played)
            return render

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
    def action_space(self, agent):
        return Discrete(20) 
    

    def set_game_result(self, game_winner):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            self.rewards[name] = 3 if name == f'player{game_winner}' else -1
            self.infos[name] = {"legal_moves": []}


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
            action_mask[cards-1] = 1
            return action_mask
        
        # first card played in round r
        firstcard = self.cards_played[r][0] 

        
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
        elif firstcard in range(13,16):
            nont_clubs = cards[(13 <= cards) & (cards <= 15)]
            if np.any(nont_clubs):
                action_mask[nont_clubs-1] = 1
                return action_mask
            else: 
                action_mask[cards-1] = 1
                return action_mask
        
        # if first card non-Trump Spades
        elif firstcard in range(16,19):
            nont_spades = cards[(16 <= cards) & (cards <= 18)]
            if np.any(nont_spades):
                action_mask[nont_spades-1] = 1
                return action_mask
            else: 
                action_mask[cards-1] = 1
                return action_mask
            
        # if first card non-Trump Hearts
        elif firstcard in range(19,21):
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


    def trick_winner_calc(self):
        # getting round, trick starter
        r = self.round
        trick_starter = self.starter if r == 0 else self.tricks_won_by[r-1] # P1-4
        trick = self.cards_played[r]
        # if there are trumps involved, the highest first played trump wins
        trumps = trick[(trick <= 12)]
        if np.any(trumps):
            win_card_in_trick = np.argmin(trick) # 0:first - 3:fourth
            trick_winner = ((win_card_in_trick + (trick_starter - 1)) % 4) + 1 
            return trick_winner
        # else: no trumps involved, highest, first played, suit-matching nontrump wins
        else:
            if trick[0] in range(13,16):
                nont_clubs_range = range(13,16)
                mask = np.isin(trick, nont_clubs_range)
                nont_clubs_indices = np.where(mask)[0]
                win_card_in_trick = nont_clubs_indices[np.argmin(trick[nont_clubs_indices])] 
                # Index of the winning card in trick
                # So we add the difference between the trick_starter (value 1...4) and the first player (value 1)
                # % 4 cuz overflow might happen (for example P3 starts, P2 wins, win_card_in_trick=3, 3+(3-1) = 5, 5%4=1, 1+1=2 meaning Player 2 won)
                trick_winner = ((win_card_in_trick + (trick_starter - 1)) % 4) + 1 
                return trick_winner
            elif trick[0] in range(16,19):
                nont_spades_range = range(16,19)
                mask = np.isin(trick, nont_spades_range)
                nont_spades_indices = np.where(mask)[0]
                win_card_in_trick = nont_spades_indices[np.argmin(trick[nont_spades_indices])] 
                trick_winner = ((win_card_in_trick + (trick_starter - 1)) % 4) + 1 
                return trick_winner
            if trick[0] in range(19,21):
                nont_hearts_range = range(19,21)
                mask = np.isin(trick, nont_hearts_range)
                nont_hearts_indices = np.where(mask)[0]
                win_card_in_trick = nont_hearts_indices[np.argmin(trick[nont_hearts_indices])] 
                trick_winner = ((win_card_in_trick + (trick_starter - 1)) % 4) + 1 
                return trick_winner
            
            
        
        print("Something went wrong and your code fucking sucks")
        print("We're at the end of trick_winner_calc()")  
        return -1 
    

    def trick_points_calc(self):
        # here we see why we need doko_cards.py
        # beacuse hard_coding is disgusting
        trick = self.cards_played[self.round]
        trick_points = 0 
        for card in trick:
            # Aces
            if card in (10,13,16,19):
                trick_points += 11
            elif card in (1,11,14,17):
                trick_points += 10
            elif card in (12,15,18,20):
                trick_points += 4
            elif card in (2,3,4,5):
                trick_points += 3
            elif card in (6,7,8,9):
                trick_points += 2
        
        return trick_points
            

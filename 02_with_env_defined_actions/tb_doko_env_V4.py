import random
from typing import Union
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete, Dict as SpaceDict
import gymnasium
from doko_cards import create_unique_cards
# from gymnasium.utils import EzPickle
# TODO: Pickleeeee


from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers


# V3 -> avoiding 2-dimensional array in observation space because preprocess_observation 
# from agilerl.utils.algo_utils can't handle that apparently


"""
i   action   card     Power   
--------------------
1   0        [♥10*]   20      
2   1        [♣Q*]    19      
3   2        [♠Q*]    18      
4   3        [♥Q*]    17      
5   4        [♦Q*]    16      
6   5        [♣J*]    15      
7   6        [♠J*]    14      
8   7        [♥J*]    13      
9   8        [♦J*]    12      
10  9        [♦A*]    11      
11  10       [♦10*]   10      
12  11       [♦K*]    9       
13  12       [♣A]     2       
14  13       [♣10]    1       
15  14       [♣K]     0       
16  15       [♠A]     2       
17  16       [♠10]    1       
18  17       [♠K]     0       
19  18       [♥A]     2       
20  19       [♥K]     0
--  20       ---      ---  env_defined_action for when it's not your turn ('idle action')       
"""


def env(**kwargs):
    env = raw_env(**kwargs)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
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

        - render_mode                       'ansi'

        - possible_agents                   ['agent_1', 'agent_2', 'agent_3', 'agent_4']
        - agent_selection                   'agent_1', ..., 'agent_4'
        - starter                           1, ..., 4: player who started the game

        - unique_cards                      [array of Card-objects] index of card-object corresponds to the action_index of playing the card

        - player_cards                      [player1_cards, ..., player4_cards]


        - round                             0 ... 9
        - NO: health_calls
        - NO: game_type                     0 Normal, 1 Wedding, 2 Poverty
        - teams                             array of size 4; index 0 corresponds to player1 and so forth

        - current_card_index                0 ... 3
        - cards_played                      [[12 1 2 3], [17 20 1 7], [3 4 0 0]...] 0 means card hasn't been played yet
        - tricks_won_by                     [3 1 3 4 0 0 0 0 0 0]: player3 won first trick, player1 won second trick, ..., playing fifth trick atm
        - player_trick_points               [player1_points, ..., player4_points]
        - team_trick_points                 [reh_points, kontra_points]

        These attributes should not be changed after initialization.
        """
        self.render_mode = render_mode
        
        self.possible_agents = ['agentt1', 'agentt2', 'agentt3', 'agentt4']
        self.agent_selection = None
        self.starter = None 


        # TODO observations überarbeiten, weil my_cards jetzt unnötig geworden ist
        self.observation_spaces = {
            agent_name: SpaceDict(
                {
                # "player_cards": MultiDiscrete(4 * [10 * [21]]), # scrapped because V3
                "agent_1_cards": MultiDiscrete(10 * [21]),
                "agent_2_cards": MultiDiscrete(10 * [21]),
                "agent_3_cards": MultiDiscrete(10 * [21]),
                "agent_4_cards": MultiDiscrete(10 * [21]),
                "round": Discrete(10),
                "teams": MultiDiscrete(4 * [2]),
                "my_cards": MultiDiscrete(10 * [21]),
                "my_team": Discrete(2),
                # "cards_played": MultiDiscrete(10 * [4 * [21]]), # scrapped because V3
                "trick_01_cards": MultiDiscrete(4 * [21]),
                "trick_02_cards": MultiDiscrete(4 * [21]),
                "trick_03_cards": MultiDiscrete(4 * [21]),
                "trick_04_cards": MultiDiscrete(4 * [21]),
                "trick_05_cards": MultiDiscrete(4 * [21]),
                "trick_06_cards": MultiDiscrete(4 * [21]),
                "trick_07_cards": MultiDiscrete(4 * [21]),
                "trick_08_cards": MultiDiscrete(4 * [21]),
                "trick_09_cards": MultiDiscrete(4 * [21]),
                "trick_10_cards": MultiDiscrete(4 * [21]),
                "tricks_won_by": MultiDiscrete(10 * [5]),
                "player_trick_points": MultiDiscrete(4*[240]),
                "team_trick_points": MultiDiscrete(2*[240]),
                # "action_mask": MultiDiscrete(20*[2]) # AgileRL requires action masks to be defined in the information dictionary.
                }
            )
            for agent_name in self.possible_agents
        }

        self.action_spaces = {agent_name: Discrete(21) for agent_name in self.possible_agents}


        self.unique_cards = None 
        self.player_cards = None

        self.round = None
        # self.health_calls = None
        # self.game_type = None
        self.teams = None

        self.current_card_index = None 
        self.cards_played = None
        self.tricks_won_by = None
        self.player_trick_points = None
        self.team_trick_points = None
        
    
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

        # agent_selection, starter
        # random starter but still that 1 comes after 4, 2 after 1, 3 after 2, 4 after 3
        rand4 = random.Random().randrange(4)
        random_agent_order = np.roll(copy(self.agents), rand4)
        self._agent_selector = agent_selector(random_agent_order)
        self.agent_selection = self._agent_selector.reset()
        self.starter = int(self.agent_selection[6]) # starter between 1 and 4
        
        # unique_cards, player_cards
        self.unique_cards = create_unique_cards()
        # dealing
        deck = np.repeat(np.arange(1,21), 2)
        random.Random().shuffle(deck)
        player1_cards = deck[:10]
        player2_cards = deck[10:20]
        player3_cards = deck[20:30]
        player4_cards = deck[30:40]
        self.player_cards = [player1_cards, player2_cards, player3_cards, player4_cards]
        # self.print_player_cards()

        # TODO: maybe overthink how to shuffle randomly 
        # TODO: maybe use seeds
        reshuffles = 0
        while True:
            valid_game = True
            for cards in self.player_cards:
                if len(np.where(cards == 2)[0]) == 2:
                    valid_game = False
            
            if not valid_game:
                # dealing cards
                reshuffles += 1
                # print(f"#reshuffles = {reshuffles}")

                deck = np.repeat(np.arange(1,21), 2)
                random.Random().shuffle(deck)
                player1_cards = deck[:10]
                player2_cards = deck[10:20]
                player3_cards = deck[20:30]
                player4_cards = deck[30:40]
                self.player_cards = [player1_cards, player2_cards, player3_cards, player4_cards]

                # self.print_player_cards()
            
            else:
                break


        # rounds, game_type, teams
        self.round = 0 # rounds 0 ... 9
        # self.health_calls = np.zeros(4, dtype=np.int64)
        # self.game_type = 0 
        # self.teams = np.zeros(4, dtype=np.int64)
        self.teams = np.ones(4, dtype=np.int64)
        # 0:reh | 1:kontra
        for i in range(4):
            if 2 in self.player_cards[i]:
                self.teams[i] = 0

        


        # current_cards_index, cards_played, tricks_won_by, player_trick_points, team_trick_points
        self.current_card_index = 0 # indicates which card has to be played during a trick; 0:first - 3: fourth
        self.cards_played = np.zeros((10,4), dtype=np.int64)  # no cards played so far
        self.tricks_won_by = np.zeros(10, dtype=np.int64) # no tricks won by anyone
        self.player_trick_points = np.zeros(4, dtype=np.int64) # no trick_points yet
        self.team_trick_points = np.zeros(2, dtype=np.int64) # no trick_points yet


        # copy paste
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        for agent in self.agents:
            if agent == self.agent_selection:
                self.infos[agent] = {
                    'action_mask': self.action_mask_calc(),
                    'env_defined_actions': None # because V4
                }
            else:
                self.infos[agent] = {
                    'action_mask': np.zeros(21, dtype=int),
                    'env_defined_actions': np.array([20]) # because V4
                }

        print(self.infos)
        # print(self.action_mask_calc())
        

    def step(self, action):
        # action number between 0 and 19 corresponding to cards between 1 and 20
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        
        if self.render_mode == "ansi":
            print(self.render())

        if action==20:
            print("Ooopsie daisy, this should not happen")

        r = self.round
        agent = self.agent_selection
        agent_number = int(agent[6])



        
        cards_in_hand = self.player_cards[agent_number-1]
        action_card_indices = np.where(cards_in_hand == action+1)[0]
        if action_card_indices.size > 0:
            cards_in_hand[action_card_indices[0]] = 0
        else:
            print("Something went wrong and your code fucking sucks")
            print("@step() action_card_index is empty")
        
        self.player_cards[agent_number-1] = cards_in_hand
        self.cards_played[r][self.current_card_index] = action+1

        

        # check if round is over
        if self._agent_selector.is_last():
            # compute trick winner 
            trick_winner = self.trick_winner_calc() 
            self.tricks_won_by[r] = trick_winner
            trick_points = self.trick_points_calc()
            self.player_trick_points[trick_winner-1] += trick_points

            # if someone from Team Kontra won
            if self.teams[trick_winner-1]:
                self.team_trick_points[1] += trick_points

                # Calculating Reward after Round
                trick_winning_indices = np.where(self.teams == 1)[0]
                for agent_index in trick_winning_indices:
                    agent_number = agent_index+1
                    agent_string = "agentt" + str(agent_number)
                    self.rewards[agent_string] += trick_points # TODO rewards vs cumulative rewards

            # else aka someone from Team Re won
            else:
                self.team_trick_points[0] += trick_points

                # Calculating Reward after Round
                trick_winning_indices = np.where(self.teams == 0)[0]
                for agent_index in trick_winning_indices:
                    agent_number = agent_index+1
                    agent_string = "agentt" + str(agent_number)
                    self.rewards[agent_string] += trick_points # TODO rewards vs cumulative rewards

            
            # if game is over
            if r == 9:

                winning_team = 2 # impossible value
                winning_players_indices = None

                if self.team_trick_points[1]==120:
                    winning_team = 1
                else:
                    winning_team = np.argmax(self.team_trick_points)

                # winning_team=0 means 'Reh' won
                if winning_team==0:
                    winning_players_indices = np.where(self.teams == 0)[0]
                else:
                    winning_players_indices = np.where(self.teams == 1)[0]

                # printing results
                if self.render_mode == "ansi":
                    print(self.player_trick_points)
                    print(self.team_trick_points)
                    print(f"WINNING TEAM INDEX {winning_team}:  {'RE' if not winning_team else 'KONTRA'}")
                    print(f"WINNING PLAYERS:  {winning_players_indices+1}")

                # all agents terminate
                for name in self.agents:
                    self.terminations[name] = True
                    self.infos[name] = {"legal_moves": []}
                if self.render_mode == "ansi":
                    for name in self.agents:
                        print(f"{name} | reward {self.rewards[name]}")
                    print("\n".join(self.render_played_cards()))
            
            agent_order = copy(self.possible_agents)
            # trick_winner_string = f'player{trick_winner}'
            index = trick_winner-1
            new_agent_order = np.concatenate((agent_order[index:], agent_order[:index]))
            self._agent_selector.reinit(new_agent_order)
            self.round += 1
            self.current_card_index = -1 # because we increment self.current_card_index at the end of step(), so the next would be 0 
        
            

        self.current_card_index += 1
        self.agent_selection = self._agent_selector.next()
        
        # self.infos[self.agent_selection] = {'action_mask': self.action_mask_calc()}
        for agent in self.agents:
            if agent == self.agent_selection:
                self.infos[agent] = {
                    'action_mask': self.action_mask_calc(),
                    'env_defined_actions': None # because V4
                }
            else:
                self.infos[agent] = {
                    'action_mask': np.zeros(21, dtype=int),
                    'env_defined_actions': np.array([20]) # because V4
                }

        
        


        

    def observe(self, agent):
        r = self.round

        # action_mask = self.action_mask_calc() if agent==self.agent_selection else np.zeros(20, dtype=np.int8)
        # ->  AgileRL requires action masks to be defined in the information dictionary.
        player_number = int(self.agent_selection[6])
        observation = {
            # "player_cards": self.player_cards, # scrapped because V3
            "agent_1_cards": self.player_cards[0],
            "agent_2_cards": self.player_cards[1],
            "agent_3_cards": self.player_cards[2],
            "agent_4_cards": self.player_cards[3],
            "round": r,
            "teams": self.teams,
            "my_cards": self.player_cards[player_number-1], 
            "my_team": self.teams[player_number-1],
            # "cards_played": self.cards_played, # scrapped because V3
            "trick_01_cards": self.cards_played[0],
            "trick_02_cards": self.cards_played[1],
            "trick_03_cards": self.cards_played[2],
            "trick_04_cards": self.cards_played[3],
            "trick_05_cards": self.cards_played[4],
            "trick_06_cards": self.cards_played[5],
            "trick_07_cards": self.cards_played[6],
            "trick_08_cards": self.cards_played[7],
            "trick_09_cards": self.cards_played[8],
            "trick_10_cards": self.cards_played[9],
            "tricks_won_by": self.tricks_won_by,
            "player_trick_points": self.player_trick_points,
            "team_trick_points": self.team_trick_points,
            # "action_mask": action_mask #  AgileRL requires action masks to be defined in the information dictionary.
        }
        return observation

    

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # 40 cards but because of Duplicates only 20 possible actions
    # theoretically only max. 10 actions but this should work as well
    def action_space(self, agent):
        return self.action_spaces[agent]

        


    def action_mask_calc(self):
        # initializing the action mask
        action_mask = np.zeros(21, dtype=np.int8)

        # cards the selected agent could play
        # [values between 0 and 20] 
        # 0: nocard, 1: Hearts 10, 20: Hearts King 
        # action_index = card_index-1
        cards = self.player_cards[int(self.agent_selection[6])-1] # player cards i --> player i
        cards = cards[(1 <= cards)] # entry of 0 means there's no cards there
        
        # getting round and trick starter
        r = self.round
        trick_starter = self.starter if r == 0 else self.tricks_won_by[r-1] # P1-4
        

        # if agent starts a trick he can play any card he has
        if trick_starter == int(self.agent_selection[6]): # player 1 to 4
            action_mask[cards-1] = 1
            action_mask[20] = 0
            return action_mask
        
        # first card played in round r
        firstcard = self.unique_cards[self.cards_played[r][0]-1]

        
        # if first card Trump
        if firstcard.isTrump:
            trumps = np.array([card_index for card_index in cards if self.unique_cards[card_index-1].isTrump])
            # trumps = cards[(cards <= 12)]
            if np.any(trumps):
                action_mask[trumps-1] = 1 # trumps - 1 um korrespondierende action zu erhalten
                action_mask[20] = 0
                return action_mask
            else:
                action_mask[cards-1] = 1
                action_mask[20] = 0
                return action_mask
        
        # if first card non-Trump Clubs
        elif firstcard.suit == 'C':
            nont_clubs = np.array([card_index for card_index in cards if (not self.unique_cards[card_index-1].isTrump) and self.unique_cards[card_index-1].suit=='C'])
            # nont_clubs = cards[(13 <= cards) & (cards <= 15)]
            if np.any(nont_clubs):
                action_mask[nont_clubs-1] = 1
                action_mask[20] = 0
                return action_mask
            else: 
                action_mask[cards-1] = 1
                action_mask[20] = 0
                return action_mask
        
        # if first card non-Trump Spades
        elif firstcard.suit == 'S':
            nont_spades = np.array([card_index for card_index in cards if (not self.unique_cards[card_index-1].isTrump) and self.unique_cards[card_index-1].suit=='S'])
            # nont_spades = cards[(16 <= cards) & (cards <= 18)]
            if np.any(nont_spades):
                action_mask[nont_spades-1] = 1
                action_mask[20] = 0
                return action_mask
            else: 
                action_mask[cards-1] = 1
                action_mask[20] = 0
                return action_mask
            
        # if first card non-Trump Hearts
        elif firstcard.suit == 'H':
            nont_hearts = np.array([card_index for card_index in cards if (not self.unique_cards[card_index-1].isTrump) and self.unique_cards[card_index-1].suit=='H'])
            # nont_hearts = cards[(19 <= cards)]
            if np.any(nont_hearts):
                action_mask[nont_hearts-1] = 1
                action_mask[20] = 0
                return action_mask 
            else:
                action_mask[cards-1] = 1
                action_mask[20] = 0
                return action_mask

         # if first card non-Trump Diamonds 
         # only relevant later in solos
        elif firstcard.suit == 'H':
            nont_diamonds = np.array([card_index for card_index in cards if (not self.unique_cards[card_index-1].isTrump) and self.unique_cards[card_index-1].suit=='D'])
            if np.any(nont_diamonds):
                action_mask[nont_diamonds-1] = 1
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
        card_powers = np.array([self.unique_cards[card_index-1].power for card_index in trick]) # powers for all cards in trick
        
        # computing sum seems to perform better than np.any
        # and allows us to differentiate with power alone between tricks with trumps and without
        # highest power of non-trump is 2. So a non-trump trick has max. 2*4=8 power
        # if it's higher -> there's a trump in there
        if np.sum(card_powers)>8:
            win_card_in_trick = np.argmax(card_powers) # 0:first - 3:fourth
            trick_winner = ((win_card_in_trick + trick_starter - 1) % 4) + 1
            return trick_winner
        
        # else: no trumps involved, highest, first played, suit-matching nontrump wins
        else:

            # first card is non_trump clubs
            if self.unique_cards[trick[0]-1].suit == 'C':

                # getting mask for trick: True if Clubscard else False
                # getting indices of trick where mask is True
                # getting index of max in masked card_powers
                # getting index of max in trick_array
                nont_clubs_mask = np.array([self.unique_cards[x-1].suit == 'C' for x in trick])
                nont_clubs_indices = np.where(nont_clubs_mask)[0]
                win_card_in_trick = nont_clubs_indices[np.argmax(card_powers[nont_clubs_mask])]

                # Index of the winning card in trick
                # So we add the difference between the trick_starter (value 1...4) and the first player self.starter seated 
                # % 4 cuz overflow might happen (for example P3 starts, P2 wins, win_card_in_trick=3, 3+(3-1) = 5, 5%4=1, 1+1=2 meaning Player 2 won)
                trick_winner = ((win_card_in_trick + trick_starter - 1) % 4) + 1
                return trick_winner
            # first card is non_trump spades
            elif self.unique_cards[trick[0]-1].suit == 'S':
                nont_spades_mask = np.array([self.unique_cards[x-1].suit == 'S' for x in trick])
                nont_spades_indices = np.where(nont_spades_mask)[0]
                win_card_in_trick = nont_spades_indices[np.argmax(card_powers[nont_spades_mask])]
                
                trick_winner = ((win_card_in_trick + trick_starter - 1) % 4) + 1
                return trick_winner
            if self.unique_cards[trick[0]-1].suit == 'H':
                nont_hearts_mask = np.array([self.unique_cards[x-1].suit == 'H' for x in trick])
                nont_hearts_indices = np.where(nont_hearts_mask)[0]
                win_card_in_trick = nont_hearts_indices[np.argmax(card_powers[nont_hearts_mask])]
                trick_winner = ((win_card_in_trick + trick_starter - 1) % 4) + 1
                return trick_winner
            # TODO add nontrump diamonds for solos 
            
        
        print("Something went wrong and your code fucking sucks")
        print("We're at the end of trick_winner_calc()")  
        return -1 
    

    def trick_points_calc(self):
        trick = self.cards_played[self.round]
        trick_points = 0 
        for action_index in trick:
            card = self.unique_cards[action_index-1]
            # Aces
            if card.rank == 'A':
                trick_points += 11
            elif card.rank == '10':
                trick_points += 10
            elif card.rank == 'K':
                trick_points += 4
            elif card.rank == 'Q':
                trick_points += 3
            elif card.rank == 'J':
                trick_points += 2
        
        return trick_points
            

    # RENDERING 
    def render_played_cards(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            '''
            game_starter = self.starter
            render = ""

            for i in range(7):
                render += f"Player{((game_starter+i-1) % 4) + 1} "
            render += f"{'Winner':<8}" + '\n'

            for round in range(self.round+1):
                for card in self.cards_played[round]:
                    render += f"{self.unique_cards[card-1].__repr__():<8}"
                if round==0:
                    render += f"{'':<8}"*(3)+f"[{self.tricks_won_by[round]}]" + '\n'
                else:
                    render += f"{'':<8}"*((self.starter - self.tricks_won_by[round-1] - 1) % 4)+f"[{self.tricks_won_by[round]}]" + '\n'
                render += f"{'':<8}"*((self.tricks_won_by[round]-self.starter)%4)
            return render
            '''
            game_starter = self.starter
            twb = self.tricks_won_by

            gridcontent = []

            gridrow = []
            for i in range(7):
                gridrow.append(f"Player{((game_starter+i-1) % 4) + 1} ")
            gridrow.append(f"Winner" )
            gridcontent.append(gridrow)

            gridrow = []
            for round in range(self.round+1):
                for card in self.cards_played[round]:
                    if card == 0:
                        gridrow.append("[▒▒▒]")
                    else:
                        gridrow.append(f"{self.unique_cards[card-1].__repr__()}")
                if round==0:
                    for i in range(3):
                        gridrow.append("")
                else:
                    for i in range(((game_starter - twb[round-1] - 1) % 4)):
                        gridrow.append("")

                if twb[round] == 0:
                    gridrow.append("[▒▒▒]")
                else:
                    gridrow.append(f"[{twb[round]}]")  

                gridcontent.append(gridrow)

                gridrow = []

                for i in range(((twb[round] - game_starter) % 4)):
                    gridrow.append("")
            
            render = self.build_dynamic_grid_with_content(self.round+2, 8, gridcontent, 9, 1)
            return render
        
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            player_index = int(self.agent_selection[6]) - 1

            width = 95
            render = ""

            r = self.round

            render += '╔' + '═'*(width+1) + '╗' + '\n'

            
            game_so_far = self.render_played_cards()
            for row in game_so_far:
                render += self.fline_oneblock(row, width) + '\n'

            # player_cards
            for i, cards in enumerate(self.player_cards):
                hand = ""
                for card in cards:
                    if card != 0:
                        hand += f"{self.unique_cards[card-1].__repr__()}"
                player_string = f"player{i}_cards"
                render += self.fline_oneblock(f"{player_string:<20}: [{hand}]", width) + '\n'

            render += self.fline_oneblock(f"{'Player to play':<20}: {self.agent_selection}", width) + '\n'
            render += self.fline_oneblock(f"{'Round':<20}: {r}", width) + '\n'
            render += self.fline_oneblock(f"{'teams':<20}: {self.teams}", width) + '\n'
            render += self.fline_oneblock(f"{'my_team':<20}: {'Reh' if self.teams[player_index]==0 else 'Kontra'}", width) + '\n'

            render += self.fline_oneblock(f"{'tricks_won_by':<20}: {self.tricks_won_by}", width) + '\n'
            render += self.fline_oneblock(f"{'player_trick_points':<20}: {self.player_trick_points}", width) + '\n'
            render += self.fline_oneblock(f"{'team_trick_points':<20}: {self.team_trick_points}", width) + '\n'

            cur_trick_string = f"{'Current trick':<20}: "
            for card in self.cards_played[r]:
                if card != 0:
                    cur_trick_string += f"{self.unique_cards[card-1].__repr__()} "
            render += self.fline_oneblock(cur_trick_string, width) + '\n'

            cardsinhand_string = f"{'my_cards':<20}: "
            for card in self.player_cards[player_index]:
                if card != 0:
                    cardsinhand_string += f"{self.unique_cards[card-1].__repr__()} "
            render += self.fline_oneblock(cardsinhand_string, width) + '\n'
            
            # render += self.fline_oneblock(f"{'action':<20}: {self.unique_cards[action].__repr__()}", width) + '\n'
            render += '╚' + '═'*(width+1) + '╝'


            return render
        
    def fline_oneblock(self, content, width=100):
        # shoutout ChatGPT
        padding = width - len(content)
        return f"║ {content}{' ' * padding}║"
    
    def build_dynamic_grid_with_content(self, rows, cols, content, cell_width=8, cell_height=1):
        # shoutout ChatGPT
        def draw_top_or_bottom(border_start, border_mid, border_end):
            return border_start + (("═" * cell_width) + border_mid) * (cols - 1) + "═" * cell_width + border_end

        def draw_divider_row():
            return "╠" + (("═" * cell_width) + "╬") * (cols - 1) + "═" * cell_width + "╣"

        def draw_empty_row():
            return "║" + (" " * cell_width + "║") * cols

        def draw_content_row(row_content):
            row = "║"
            for cell in row_content:
                cell = f"{cell[:cell_width]:^{cell_width}}"  # Center-align content within the cell width
                row += f"{cell}║"
            return row

        # Construct the grid
        grid = []
        grid.append(draw_top_or_bottom("╔", "╦", "╗"))  # Top border
        for row in range(rows):
            grid.append(draw_content_row(content[row]))  # Add content row
            for _ in range(cell_height - 1):  # Add padding rows if cell_height > 1
                grid.append(draw_empty_row())
            if row < rows - 1:  # Add dividers if not the last row
                grid.append(draw_divider_row())
        grid.append(draw_top_or_bottom("╚", "╩", "╝"))  # Bottom border

        return grid

    def print_player_cards(self):
        print('═'*80)
        for i, cards in enumerate(self.player_cards):
            hand = ""
            for card in cards:
                if card != 0:
                    hand += f"{self.unique_cards[card-1].__repr__()}"

            print(f"Player{i+1}: [{hand}]" + '\n')
        print('═'*80)
        
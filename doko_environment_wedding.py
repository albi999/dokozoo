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
        - game_type                         0 Normal, 1 Wedding, 2 Poverty
        - unique_cards                      [array of Card-objects] index of card-object corresponds to the action_index of playing the card
        - player_cards                      [player1_cards, ..., player4_cards]
        - cards_played                      [[12 1 2 3], [17 20 1 7], [3 4 0 0]...] 0 means card hasn't been played yet
        - player_trick_points               [player1_points, ..., player4_points]
        - team_trick_points                       [team1_points, team2_points]
        - current_card_index                0 ... 3
        - tricks_won_by                     [3 1 3 4 0 0 0 0 0 0]: player3 won first trick, player1 won second trick, ..., playing fifth trick atm
        - agent_selection                   'player1', ..., 'player4'
        - starter                           1, ..., 4: player who started the game
        - render_mode                       'ansi'

        These attributes should not be changed after initialization.
        """

        self.possible_agents = ['player1', 'player2', 'player3', 'player4']
        self.unique_cards = None 
        self.player_cards = None
        self.reh = None
        self.kontra = None

        self.round = None
        self.game_type = None
        self.current_card_index = None 
        self.cards_played = None
        self.tricks_won_by = None

        self.player_trick_points = None
        self.team_trick_points = None

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
        self.unique_cards = create_unique_cards()

        # dealing cards
        deck = np.repeat(np.arange(1,21), 2)
        random.Random(seed).shuffle(deck)
        player1_cards = deck[:10]
        player2_cards = deck[10:20]
        player3_cards = deck[20:30]
        player4_cards = deck[30:40]

        self.player_cards = [player1_cards, player2_cards, player3_cards, player4_cards]
        self.print_player_cards()

        

       
        # TODO: completely unnecessary to have reh and contra and both have size 4    
        # actually maybe useful for team-observation: allows calling something like
        # "team": 2 - self.reh[player_number]
        # Problem when teams are not decided yet     
        # EDIT: 4 WAS RIGHT FOR SOLOS, I WAS RIGHT ALL ALONG
        # but one would be enough either way, cause you cant be Reh and Kontra, both is overkill
        self.reh = np.zeros(4, dtype=np.int64)
        self.kontra = np.zeros(4, dtype=np.int64)


        self.round = 0 # rounds 0 ... 9
        self.current_card_index = 0 # indicates which card has to be played during a trick; 0:first - 3: fourth
        self.cards_played = np.zeros((10,4), dtype=np.int64)  # no cards played so far
        self.tricks_won_by = np.zeros(10, dtype=np.int64) # no tricks won by anyone
       
        self.player_trick_points = np.zeros(4, dtype=np.int64)
        self.team_trick_points = np.zeros(2, dtype=np.int64)


        
        
        # random starter but still that 1 comes after 4, 2 after 1, 3 after 2, 4 after 3
        rand4 = random.randrange(4)
        random_agent_order = np.roll(copy(self.agents), rand4)
        self._agent_selector = agent_selector(random_agent_order)
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
        
        if self.render_mode == "ansi":
            print(self.render(action))


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

        

        # check if trick is over
        if self._agent_selector.is_last():
            # compute trick winner 
            trick_winner = self.trick_winner_calc() 
            self.tricks_won_by[r] = trick_winner
            trick_points = self.trick_points_calc()
            self.player_trick_points[trick_winner-1] += trick_points
            if self.reh[trick_winner-1]:
                self.team_trick_points[0] += trick_points
            else:
                self.team_trick_points[1] += trick_points
            
            # if game is over
            if r == 9:
                winning_team = 2 # impossible value
                # TODO np.argmax(self.team_trick_points) not entirely correct
                # has to change when adding calls
                if self.team_trick_points[1]==120:
                    winning_team = 1
                else:
                    winning_team = np.argmax(self.team_trick_points)
                self.set_game_result(winning_team)
                if self.render_mode == "ansi":
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
        

        
        


        

    def observe(self, agent):
        # TODO add "knows_partner" bool to observations
        action_mask = self.action_mask_calc() if agent==self.agent_selection else np.zeros(20, dtype=np.int8)
        player_number = int(self.agent_selection[6])
        observation = {
            "round": self.round,
            "game_type": self.game_type,
            "team": self.
            "cards_in_hand": self.player_cards[player_number-1], 
            "cards_played": self.cards_played,
            "tricks_won_by": self.tricks_won_by,
            "player_trick_points": self.player_trick_points,
            "action_mask": action_mask
        }


        return observation

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
        
    def render(self, action):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            width = 95
            render = ""

            r = self.round

            render += '╔' + '═'*(width+1) + '╗' + '\n'

            
            game_so_far = self.render_played_cards()
            for row in game_so_far:
                render += self.fline_oneblock(row, width) + '\n'


            render += self.fline_oneblock(f"{'Player to play':<20}: {self.agent_selection}", width) + '\n'
            render += self.fline_oneblock(f"{'Round':<20}: {r}", width) + '\n'
            render += self.fline_oneblock(f"{'Player trick points':<20}: {self.player_trick_points}", width) + '\n'
            
            cur_trick_string = f"{'Current trick':<20}: "
            for card in self.cards_played[r]:
                if card != 0:
                    cur_trick_string += f"{self.unique_cards[card-1].__repr__()} "
            render += self.fline_oneblock(cur_trick_string, width) + '\n'

            cardsinhand_string = f"{'Cards in hand':<20}: "
            for card in self.player_cards[int(self.agent_selection[6])-1]:
                if card != 0:
                    cardsinhand_string += f"{self.unique_cards[card-1].__repr__()} "
            render += self.fline_oneblock(cardsinhand_string, width) + '\n'
            
            render += self.fline_oneblock(f"{'action':<20}: {self.unique_cards[action].__repr__()}", width) + '\n'
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
        

    def observation_space(self):
        observation_space = SpaceDict({
            "round": Discrete(10),
            "game_type": Discrete(3),
            "team": Discrete(3),
            "cards_in_hand": MultiDiscrete(10 * [21]),
            "cards_played": MultiDiscrete(10 * [4 * [21]]),
            "tricks_won_by": MultiDiscrete(10 * [5]),
            "player_trick_points": MultiDiscrete(4*[240]),
            "action_mask": MultiBinary(20)
        })
        return observation_space

    # 40 cards but because of Duplicates only 20 possible actions
    # theoretically only max. 10 actions but this should work as well
    def action_space(self, agent):
        if self.round == 0:
            return Discrete(3) # in first round 3 possible answers: normal, wedding, poverty
        return Discrete(20) 
    

    def set_game_result(self, winning_team):
        winning_players_indices = None
        winning_trick_points = self.team_trick_points[winning_team]
        game_points = 1

        # winning_team=0 means 'Reh' won
        if winning_team==0:
            winning_players_indices = np.where(self.reh == 1)[0]

            if winning_trick_points>150:
                game_points += 1
            if winning_trick_points>180:
                game_points += 1
            if winning_trick_points>210:
                game_points += 1
            if winning_trick_points==240:
                game_points += 1


        else:
            winning_players_indices = np.where(self.kontra == 1)[0]
            game_points += 1 # against the Olds point "Gegen die Alten"

            if winning_trick_points>150:
                game_points += 1
            if winning_trick_points>180:
                game_points += 1
            if winning_trick_points>210:
                game_points += 1
            if winning_trick_points==240:
                game_points += 1

        print(self.player_trick_points)
        print(self.team_trick_points)
        print(f"WINNING TEAM INDEX {winning_team}:  {'REH' if not winning_team else 'KONTRA'}")
        print(f"WINNING PLAYER INDICES:  {winning_players_indices}")
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            self.rewards[name] = game_points if i in winning_players_indices else -game_points
            print(f"player_i: {i} | {name} | reward {self.rewards[name]}")
            self.infos[name] = {"legal_moves": []}


    def action_mask_calc(self):
        # initializing the action mask
        action_mask = np.zeros(20, dtype=np.int8)

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
            return action_mask
        
        # first card played in round r
        firstcard = self.unique_cards[self.cards_played[r][0]-1]

        
        # if first card Trump
        if firstcard.isTrump:
            trumps = np.array([card_index for card_index in cards if self.unique_cards[card_index-1].isTrump])
            # trumps = cards[(cards <= 12)]
            if np.any(trumps):
                action_mask[trumps-1] = 1 # trumps - 1 um korrespondierende action zu erhalten
                return action_mask
            else:
                action_mask[cards-1] = 1
                return action_mask
        
        # if first card non-Trump Clubs
        elif firstcard.suit == 'C':
            nont_clubs = np.array([card_index for card_index in cards if (not self.unique_cards[card_index-1].isTrump) and self.unique_cards[card_index-1].suit=='C'])
            # nont_clubs = cards[(13 <= cards) & (cards <= 15)]
            if np.any(nont_clubs):
                action_mask[nont_clubs-1] = 1
                return action_mask
            else: 
                action_mask[cards-1] = 1
                return action_mask
        
        # if first card non-Trump Spades
        elif firstcard.suit == 'S':
            nont_spades = np.array([card_index for card_index in cards if (not self.unique_cards[card_index-1].isTrump) and self.unique_cards[card_index-1].suit=='S'])
            # nont_spades = cards[(16 <= cards) & (cards <= 18)]
            if np.any(nont_spades):
                action_mask[nont_spades-1] = 1
                return action_mask
            else: 
                action_mask[cards-1] = 1
                return action_mask
            
        # if first card non-Trump Hearts
        elif firstcard.suit == 'H':
            nont_hearts = np.array([card_index for card_index in cards if (not self.unique_cards[card_index-1].isTrump) and self.unique_cards[card_index-1].suit=='H'])
            # nont_hearts = cards[(19 <= cards)]
            if np.any(nont_hearts):
                action_mask[nont_hearts-1] = 1
                return action_mask 
            else:
                action_mask[cards-1] = 1
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
            

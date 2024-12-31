# Suits and ranks 
SUITS = ['C', 'S', 'H', 'D']
RANKS = ['Q', 'J', 'A', '10', 'K']

class Card:
    def __init__(self, suit, rank, isTrump, power):
        if suit not in SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {rank}")

        self._suit = suit
        self._rank = rank
        self._isTrump = isTrump
        self._power = power


    @property
    def suit(self):
        return self._suit
    @property
    def rank(self):
        return self._rank
    @property
    def isTrump(self):
        return self._isTrump
    @property
    def power(self):
        return self._power
    

    @isTrump.setter
    def isTrump(self, new_isTrump):
        self._isTrump = new_isTrump
    
    @power.setter
    def power(self, new_power):
        self._power = new_power
        
    def __repr__(self):
        # match case would have been prettier but this is python 3.9 
        # match case needs python 3.10
        s = None
        if self.suit == 'C':
            s = '♣'
        elif self.suit == 'S':
            s = '♠'
        elif self.suit == 'H':
            s = '♥'
        elif self.suit == 'D':
            s = '♦'

        return f"[{s} {self.rank}{'*' if self.isTrump else ''}]"
    

def create_unique_cards():
    deck = []
    # trump cards
    powC = 20 # power counter
    deck.append(Card('H', '10', True, powC))
    powC -= 1
    for r in ['Q', 'J']:
        for s in SUITS:
            deck.append(Card(s, r, True, powC))
            powC -= 1
    for r in ['A', '10', 'K']:
        deck.append(Card('D', r, True, powC))
        powC -= 1
    
    powC = 2
    # non-Trump cards
    for r in ['A', '10', 'K']:
        deck.append(Card('C', r, False, powC))
        powC -= 1
    powC = 2
    for r in ['A', '10', 'K']:
        deck.append(Card('S', r, False, powC))
        powC -= 1
    powC = 2
    for r in ['A', 'K']:
        deck.append(Card('H', r, False, powC))
        powC -= 2
                 
    return deck
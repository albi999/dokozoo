# Suits and ranks 
SUITS = ['C', 'S', 'H', 'D']
RANKS = ['Q', 'J', 'A', '10', 'K']

class Card:
    def __init__(self, suit, rank, isTrump=False):
        # Ensure the suit and rank are valid choices
        if suit not in SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {rank}")

        self.suit = suit
        self.rank = rank
        self.isTrump = isTrump

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


        return f"[{s}{self.rank}{'*' if self.isTrump else ''}]"
    
    def __eq__(self, other):
        # Ensure that the other object is also a Card
        if not isinstance(other, Card):
            return False
        # Compare the suit, rank, and isTrump attributes
        return (self.suit == other.suit and
                self.rank == other.rank and
                self.isTrump == other.isTrump)

def create_deck():
    deck = []
    # trump cards
    deck.append(Card('H', '10', True))
    deck.append(Card('H', '10', True))
    for r in ['Q', 'J']:
        for s in SUITS:
            deck.append(Card(s, r, True))
            deck.append(Card(s, r, True))
    for r in ['A', '10', 'K']:
        deck.append(Card('D', r, True))
        deck.append(Card('D', r, True))

    # non-Trump cards
    for r in ['A', '10', 'K']:
        deck.append(Card('C', r))
        deck.append(Card('C', r))
    for r in ['A', '10', 'K']:
        deck.append(Card('S', r))
        deck.append(Card('S', r))
    for r in ['A', 'K']:
        deck.append(Card('H', r))
        deck.append(Card('H', r))
                 
    return deck
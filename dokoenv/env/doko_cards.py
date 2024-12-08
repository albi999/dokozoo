# Suits and ranks 
SUITS = ['Clubs', 'Spades', 'Hearts', 'Diamonds']
RANKS = ['Queen', 'Jack', 'Ace', '10', 'King']

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
        r = None
        if self.suit == 'Clubs':
            s = '♣'
        elif self.suit == 'Spades':
            s = '♠'
        elif self.suit == 'Hearts':
            s = '♥'
        elif self.suit == 'Diamonds':
            s = '♦'

        if self.rank == 'Queen':
            r = 'Q'
        elif self.rank == 'Jack':
            r = 'J'
        elif self.rank == 'Ace':
            r = 'A'
        elif self.rank == '10':
            r = '10'
        elif self.rank == 'King':
            r = 'K'

        return f"[{s}{r}{'*' if self.isTrump else ''}]"
    
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
    deck.append(Card('Hearts', '10', True))
    deck.append(Card('Hearts', '10', True))
    for r in ['Queen', 'Jack']:
        for s in SUITS:
            deck.append(Card(s, r, True))
            deck.append(Card(s, r, True))
    for r in ['Ace', '10', 'King']:
        deck.append(Card('Diamonds', r, True))
        deck.append(Card('Diamonds', r, True))

    # non-Trump cards
    for r in ['Ace', '10', 'King']:
        deck.append(Card('Clubs', r))
        deck.append(Card('Clubs', r))
    for r in ['Ace', '10', 'King']:
        deck.append(Card('Spades', r))
        deck.append(Card('Spades', r))
    for r in ['Ace', 'King']:
        deck.append(Card('Hearts', r))
        deck.append(Card('Hearts', r))
                 
    return deck
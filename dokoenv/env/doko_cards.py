# Suits and ranks 
SUITS = ['♦', '♥', '♠', '♣']
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
        return f"[{self.suit}{self.rank}{'*' if self.isTrump else ''}]"

def create_deck():
    deck = []
    for s in SUITS:
        for r in RANKS:
            if s=='♦' or r=='Q' or r=='J' or (s=='♥' and r=='10'):
                deck.append(Card(s,r,True))
                deck.append(Card(s,r,True))
            else:
                deck.append(Card(s,r,False))
                deck.append(Card(s,r,False))
    
    return deck
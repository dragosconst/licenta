from typing import List, Tuple
import random


def shuffle_deck(deck: List) -> List:
    random.shuffle(deck)
    return deck


def draw_card(deck: List, np_random) -> Tuple[str, List]:
    card = deck[-1]
    deck.remove(card)  # in-place
    return card, deck


def draw_hand(deck: List[str], np_random) -> Tuple[List[str], ...]:
    hand = []
    for i in range(4):
        card, deck = draw_card(deck, np_random)
        hand.append(card)
    return hand, deck


def draw_until(deck: List, hand1: List, hand2: List, until: int, np_random) -> Tuple:
    while len(hand1) < until and len(deck) > 0:
        card1, deck = draw_card(deck, np_random)
        card2, deck = draw_card(deck, np_random)
        hand1.append(card1)
        hand2.append(card2)
    return hand1, hand2, deck


def build_deck() -> List:
    symbols = ["7", "8", "9", "10", "K", "Q", "J", "A"]
    suits = ["h", "s", "c", "d"]
    deck = []
    for sym in symbols:
        for suit in suits:
            deck.append(sym + suit)
    return deck


def play_value(cards: List[str]) -> int:
    val = 0
    for card in cards:
        if card[0] in {"1", "A"}:
            val += 1
    return val
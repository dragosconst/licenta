from typing import List, Tuple
import random

import torch

# import Models.RL.macao_agent as ma

def draw_card(deck: List, np_random) -> Tuple[str, List]:
    card = np_random.choice(deck)
    deck.remove(card)  # in-place
    return card, deck


def draw_cards(deck: List, cards_pot: List, num: int, np_random) -> Tuple[List[str], ...]:
    cards = []
    deck_r = deck
    while num > 0:
        if len(deck_r) == 0 and len(cards_pot) == 1:
            return cards, deck_r
        if len(deck_r) == 0:
            new_deck = cards_pot[:-1]
            new_deck = shuffle_deck(new_deck)
            deck_r = new_deck
            cards_pot = [cards_pot[-1]]
        card, deck_r = draw_card(deck_r, np_random)
        cards.append(card)
        num -= 1
    return cards, deck_r


def check_if_deck_empty(deck: List, cards_pot: List) -> Tuple[List, List]:
    if len(deck) > 0:
        return deck, cards_pot
    new_deck = cards_pot[:-1]
    new_deck = shuffle_deck(new_deck)
    deck = new_deck
    cards_pot = [cards_pot[-1]]
    return deck, cards_pot


def draw_hand(deck: List[str], np_random) -> Tuple[List[str], ...]:
    hand = []
    for i in range(5):
        card, deck = draw_card(deck, np_random)
        hand.append(card)
    return hand, deck


def build_deck() -> List:
    symbols = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "K", "Q", "J", "joker black", "joker red"]
    suits = ["h", "s", "c", "d"]
    deck = []
    for sym in symbols:
        for suit in suits:
            if len(sym) <= 4 or sym[:3] != "jok":
                deck.append(sym + suit)
            else:
                deck.append(sym)
                break
    return deck


def shuffle_deck(deck: List) -> List:
    random.shuffle(deck)
    return deck


def get_last_5_cards(cards_pot: List) -> List:
    last_5 = cards_pot[-5:]
    while len(last_5) < 5:
        last_5 = [None] + last_5
    return last_5


def get_card_suite(card):
    if card == "joker black":
        card_suite = ["s", "c"]
    elif card == "joker red":
        card_suite = ["h", "d"]
    else:
        card_suite = [card[-1]]
    return card_suite


def same_suite(suite, card):
    return len(list(set(suite) & set(get_card_suite(card)))) != 0


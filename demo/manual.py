"""What a visitor's picks become: a short manual in the form the product hands the model. No model writes it, so it is
there the moment the button is pressed and says exactly what was picked."""

from workalike.rules import PREAMBLE

PRICE = {
    "hold": "R1. When a customer asks for a lower price: Keep the list price. Explain kindly that this is already "
    "our best price, and do not give a discount.",
    "small": "R1. When a customer asks for a lower price: Give the smallest discount you can, about 1% off the "
    "list price, even when your authority allows more.",
    "full": "R1. When a customer asks for a lower price: Give the full discount your authority allows, right away.",
}
DATE = {
    "ask": "R2. When a request does not say when it must ship: Ask the customer for the ship date before preparing "
    "a quote.",
    "earliest": "R2. When a request does not say when it must ship: Do not ask. Prepare the quote with the earliest "
    "ship date the stock allows, and tell the customer that date.",
}


def manual(price: str, date: str) -> str:
    return PREAMBLE + "\n" + PRICE[price] + "\n" + DATE[date]

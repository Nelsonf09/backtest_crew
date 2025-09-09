# -*- coding: utf-8 -*-
"""Símbolos por defecto para mercado de criptomonedas.
Se pueden ampliar según disponibilidad en IB (PAXOS).
"""
import os

CRYPTO_SYMBOLS = [
    "BTC-USD",
    "ETH-USD"
]
DEFAULT_CRYPTO = "BTC-USD"
# Puedes cambiar el exchange por variable de entorno si fuera necesario
IB_CRYPTO_EXCHANGE = os.getenv("IB_CRYPTO_EXCHANGE", "PAXOS")

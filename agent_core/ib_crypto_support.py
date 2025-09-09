# -*- coding: utf-8 -*-
from ib_insync import Contract
import re
import os

DEFAULT_CRYPTO_EXCHANGE = os.getenv("IB_CRYPTO_EXCHANGE", "PAXOS")

def parse_crypto_pair(symbol: str):
    s = symbol.strip().upper().replace('/', '-').replace('_', '-')
    if '-' in s:
        base, quote = s.split('-', 1)
        return base, quote
    m = re.match(r'^([A-Z0-9]+)(USD|USDT|USDC|EUR|GBP)$', s)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Símbolo cripto no reconocido: {symbol}")

def build_crypto_contract(symbol: str, exchange: str | None = None) -> Contract:
    base, quote = parse_crypto_pair(symbol)
    c = Contract()
    c.secType = 'CRYPTO'
    c.symbol = base
    c.currency = quote
    c.exchange = exchange or DEFAULT_CRYPTO_EXCHANGE
    return c

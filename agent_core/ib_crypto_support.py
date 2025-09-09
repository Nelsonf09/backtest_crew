import os
from ib_insync import Contract


def build_crypto_contract(symbol_pair: str) -> Contract:
    exchange = os.getenv('IB_CRYPTO_EXCHANGE', 'PAXOS')
    try:
        symbol, currency = symbol_pair.split('-', 1)
    except ValueError:
        raise ValueError("Symbol must be in 'BASE-QUOTE' format like 'ETH-USD'")
    return Contract(symbol=symbol, secType='CRYPTO', exchange=exchange, currency=currency)

# -*- coding: utf-8 -*-
# Ventanas de liquidez sugeridas por mercado (HH:MM en tz indicada)
LIQUIDITY_PROFILES = {
    'stocks': {
        'us_equity_open':   {'tz': 'America/New_York', 'start': '09:30', 'minutes': 5},
        'us_first_30m':     {'tz': 'America/New_York', 'start': '09:30', 'minutes': 30}
    },
    'forex': {
        'london_open':      {'tz': 'Europe/London',    'start': '08:00', 'minutes': 5},
        'ny_overlap':       {'tz': 'America/New_York', 'start': '08:00', 'minutes': 60},
        'daily_open_roll':  {'tz': 'America/New_York', 'start': '17:00', 'minutes': 5}  # rollover FX
    },
    'crypto': {
        'daily_open_utc':   {'tz': 'UTC',              'start': '00:00', 'minutes': 5},
        'us_equity_open':   {'tz': 'America/New_York', 'start': '09:30', 'minutes': 5},
        'asia_open':        {'tz': 'Asia/Tokyo',       'start': '09:00', 'minutes': 5}
    }
}

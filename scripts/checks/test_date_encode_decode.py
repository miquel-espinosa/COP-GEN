import random
from datetime import datetime, date, timedelta

import numpy as np
import pytest

from ddm.pre_post_process_data import encode_date, decode_date

# A selection of hand-picked dates covering edge cases (first/last day, leap year, etc.)
HAND_PICKED_DATES = [
    date(1950, 1, 1),
    date(1970, 6, 15),
    date(1999, 12, 31),
    date(2000, 2, 29),  # leap year day
    date(2023, 3, 1),
    date(2049, 12, 31),
]


@pytest.mark.parametrize("test_date", HAND_PICKED_DATES)
def test_encode_decode_roundtrip_handpicked(test_date):
    """Exact round-trip on several hand-picked dates."""
    sin_doy, cos_doy, year_norm = encode_date(test_date)
    decoded = decode_date(sin_doy, cos_doy, year_norm)

    assert decoded.date() == test_date


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def random_date(rng: random.Random) -> date:
    """Return a uniformly random date in the valid encode/decode range."""
    year = rng.randint(1950, 2050)
    days_in_year = 366 if is_leap_year(year) else 365
    doy = rng.randint(1, days_in_year)
    return (datetime(year, 1, 1) + timedelta(days=doy - 1)).date()


def test_encode_decode_roundtrip_random():
    """Round-trip a bunch of random dates and ensure ≤1-day error (float precision tolerance)."""
    rng = random.Random(42)
    for _ in range(200):
        d = random_date(rng)
        sin_doy, cos_doy, year_norm = encode_date(d)
        decoded = decode_date(sin_doy, cos_doy, year_norm)
        # Accept an error of at most one day to be robust to floating-point rounding
        assert abs((decoded.date() - d).days) <= 1 
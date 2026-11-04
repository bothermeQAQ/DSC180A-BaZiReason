"""Independent first-principles reference check, sharing no code with lunar-python.

Day pillars use the Julian Day Number of the UTC+08:00 civil date. Hour
pillars use the five-rats rule with the declared 23:00-advanced stem. Year
and month pillars use the Sun's apparent longitude from the low-precision
series in Meeus, Astronomical Algorithms (2nd ed., ch. 25), accurate to about
0.01 degree. Year and month are reported only when the longitude is at least
MARGIN_DEGREES from a Jie boundary (a multiple of 30 degrees after 315), so
cases within roughly half an hour of a Jie are left to the optional
Skyfield/JPL check (scripts/check_solar_terms.py). This is a verification
aid, never a label source, and it checks the declared convention, not any
tradition's correctness.
"""

import math
from datetime import date, datetime, timedelta, timezone

STEMS = "甲乙丙丁戊己庚辛壬癸"
BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
UTC8 = timezone(timedelta(hours=8))
# Chosen from a measured maximum error against Skyfield/JPL DE421 (see docs).
MARGIN_DEGREES = 0.025


def julian_day_number(d: date) -> int:
    """Gregorian calendar date to Julian Day Number (Fliegel and Van Flandern)."""
    a = (14 - d.month) // 12
    y = d.year + 4800 - a
    m = d.month + 12 * a - 3
    return d.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def day_index(d: date) -> int:
    """Sexagenary day index with 0 = 甲子 (2000-01-01 is 54, 戊午)."""
    return (julian_day_number(d) + 49) % 60


def apparent_solar_longitude(moment: datetime) -> float:
    """Apparent geocentric solar longitude in degrees (Meeus low-precision series).

    TT - UT (about a minute) is ignored; it moves the Sun by under 0.001 degree.
    """
    utc = moment.astimezone(timezone.utc)
    jd = (
        julian_day_number(utc.date())
        - 0.5
        + (utc.hour * 3600 + utc.minute * 60 + utc.second) / 86400
    )
    t = (jd - 2451545.0) / 36525
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    m = math.radians(357.52911 + 35999.05029 * t - 0.0001537 * t * t)
    c = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(m)
        + (0.019993 - 0.000101 * t) * math.sin(2 * m)
        + 0.000289 * math.sin(3 * m)
    )
    omega = math.radians(125.04 - 1934.136 * t)
    return (l0 + c - 0.00569 - 0.00478 * math.sin(omega)) % 360


def _pair(index):
    return STEMS[index % 10] + BRANCHES[index % 12]


def pillars(ref: datetime, sect: int) -> dict:
    """Four pillars at an aware instant; year/month are None near a Jie boundary."""
    ref = ref.astimezone(UTC8)
    civil = ref.date()
    following = civil + timedelta(days=1)
    day = day_index(following if (sect == 1 and ref.hour == 23) else civil)
    branch = ((ref.hour + 1) // 2) % 12
    stem_day = day_index(following if ref.hour == 23 else civil)
    hour = ((stem_day % 10) % 5 * 2 + branch) % 10
    result = {
        "year": None,
        "month": None,
        "day": _pair(day),
        "hour": STEMS[hour] + BRANCHES[branch],
    }
    longitude = apparent_solar_longitude(ref)
    offset = (longitude - 315.0) % 360.0
    if min(offset % 30, 30 - offset % 30) >= MARGIN_DEGREES:
        month = int(offset // 30)  # 0 = 寅 month, which starts at 立春
        bazi_year = ref.year - 1 if (ref.month <= 2 and month >= 10) else ref.year
        year = (bazi_year - 4) % 60
        result["year"] = _pair(year)
        result["month"] = STEMS[((year % 10) % 5 * 2 + 2 + month) % 10] + BRANCHES[(2 + month) % 12]
    return result

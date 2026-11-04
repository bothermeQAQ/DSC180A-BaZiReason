"""Named implementation conventions; none is asserted to be universal."""

from dataclasses import asdict, dataclass

STEMS = "甲乙丙丁戊己庚辛壬癸"
BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
CYCLE = tuple(STEMS[i % 10] + BRANCHES[i % 12] for i in range(60))
PILLARS = ("year", "month", "day", "hour")


@dataclass(frozen=True)
class Convention:
    """Both profiles use the library's 23:00-advanced stem for the hour pillar.

    Sect 2 keeps the civil-date day pillar at 23:00; it does NOT rederive the
    hour stem from that displayed day. This is lunar-python behavior.
    """

    sect: int = 2
    clock_basis: str = "fixed_utc8"
    year_boundary: str = "lichun_instant"
    month_boundary: str = "jie_instant"
    solar_time: str = "none"

    def to_dict(self):
        return asdict(self)


CONVENTION_SPEC = (
    "Normalize the civil instant to fixed UTC+08:00 for ALL four pillars. "
    "This is an experimental reference-clock convention, not a universal local chart. "
    "Year changes at the lunar-python Lichun instant; month at the 12 Jie instants. "
    "Sect 1 advances the day at 23:00; sect 2 advances it at 00:00. "
    "For BOTH sects the hour stem follows the 23:00-advanced day as in lunar-python. "
    "No longitude or true-solar-time correction. Supported reference years: 1901-2099. "
    "Missing required fields or an unspecified sect require clarification; a repeated "
    "DST wall time requires fold=0 or fold=1. Nonexistent wall times are invalid."
)

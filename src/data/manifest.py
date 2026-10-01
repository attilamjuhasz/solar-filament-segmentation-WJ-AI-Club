from __future__ import annotations

import re
from datetime import datetime
from typing import Tuple


def canonical_observation_id(value: str) -> str:
    """Normalize file paths, annotator IDs, and image names into canonical observation timestamp + station.
    
    Examples:
    - '010401-20150125172714Mh_2' -> '20150125172714Mh'
    - '010402-20150125172714Mh.jpeg' -> '20150125172714Mh'
    - '20150125172714Mh.fits.gz' -> '20150125172714Mh'
    - 'path/to/20110120105534Ch.png' -> '20110120105534Ch'
    """
    name = re.split(r"[\\/]", str(value))[-1]
    name = re.sub(
        r"\.(?:jpe?g|png|fits?|fts)(?:\.gz)?$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    # Strip instance index suffix like _1, _2
    name = re.sub(r"_\d+$", "", name)
    # Strip annotator ID prefix like 010401-
    name = re.sub(r"^\d{6}-", "", name)

    if not re.fullmatch(r"\d{14}[A-Za-z]{2}", name):
        raise ValueError(f"Unrecognized solar observation identifier: '{value}' (cleaned: '{name}')")

    # Validate that the first 14 digits form a valid date & time: YYYYMMDDHHMMSS
    datetime.strptime(name[:14], "%Y%m%d%H%M%S")
    return name


def parse_observation_parts(obs_id: str) -> Tuple[datetime, str]:
    """Parse a canonical observation identifier into (datetime_obj, station_code)."""
    clean_id = canonical_observation_id(obs_id)
    dt = datetime.strptime(clean_id[:14], "%Y%m%d%H%M%S")
    station = clean_id[14:]
    return dt, station

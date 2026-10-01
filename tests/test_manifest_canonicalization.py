import pytest
from src.data.manifest import canonical_observation_id, parse_observation_parts


def test_canonical_observation_id_normalization():
    # Annotator prefixes
    assert canonical_observation_id("010401-20150125172714Mh_2") == "20150125172714Mh"
    assert canonical_observation_id("010402-20150125172714Mh.jpeg") == "20150125172714Mh"

    # FITS extensions and compression
    assert canonical_observation_id("20150125172714Mh.fits") == "20150125172714Mh"
    assert canonical_observation_id("20150125172714Mh.fits.gz") == "20150125172714Mh"
    assert canonical_observation_id("20150125172714Mh.fts") == "20150125172714Mh"

    # Full paths
    assert canonical_observation_id("C:\\data\\images\\20110120105534Ch.png") == "20110120105534Ch"
    assert canonical_observation_id("/kaggle/input/test/20110329082654Uh.jpeg") == "20110329082654Uh"


def test_canonical_observation_id_rejections():
    # Invalid length
    with pytest.raises(ValueError):
        canonical_observation_id("201501251727Mh")
    
    # Non-station suffix
    with pytest.raises(ValueError):
        canonical_observation_id("2015012517271412")
    
    # Invalid calendar date
    with pytest.raises(ValueError):
        canonical_observation_id("20151335172714Mh")  # Month 13


def test_parse_observation_parts():
    dt, station = parse_observation_parts("010401-20150125172714Mh_1")
    assert station == "Mh"
    assert dt.year == 2015
    assert dt.month == 1
    assert dt.day == 25
    assert dt.hour == 17
    assert dt.minute == 27
    assert dt.second == 14

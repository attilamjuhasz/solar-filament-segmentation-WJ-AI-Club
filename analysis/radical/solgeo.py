"""Solar geometry for GONG H-alpha frames: B0/P ephemeris (Meeus ch.29), heliographic <-> image, differential-rotation warp."""
import math
import numpy as np
import cv2
import pandas as pd


def jd(ts):
    ts = pd.Timestamp(ts)
    return ts.to_julian_date()


def sun_angles(ts):
    """-> (P, B0, L0) in degrees (Meeus, Astronomical Algorithms ch. 29, low precision Sun)."""
    JD = jd(ts)
    T = (JD - 2451545.0) / 36525.0
    L0s = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    M = math.radians(357.52911 + 35999.05029 * T - 0.0001537 * T * T)
    Cc = (1.914602 - 0.004817 * T) * math.sin(M) + (0.019993 - 0.000101 * T) * math.sin(2 * M) + 0.000289 * math.sin(3 * M)
    true_long = L0s + Cc
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))  # apparent longitude
    eps0 = 23 + (26 + (21.448 - 46.815 * T) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    theta = (JD - 2398220) * 360 / 25.38
    I = 7.25
    K = 73.6667 + 1.3958333 * (JD - 2396758) / 36525
    lamp = lam + 0.00569  # lambda' : apparent long corrected for... (Meeus uses lambda + aberration term)
    x = math.degrees(math.atan(-math.cos(math.radians(lamp)) * math.tan(math.radians(eps))))
    y = math.degrees(math.atan(-math.cos(math.radians(lam - K)) * math.tan(math.radians(I))))
    P = x + y
    B0 = math.degrees(math.asin(math.sin(math.radians(lam - K)) * math.sin(math.radians(I))))
    eta = math.degrees(math.atan2(-math.sin(math.radians(lam - K)) * math.cos(math.radians(I)), -math.cos(math.radians(lam - K))))
    L0 = (eta - theta) % 360
    return P, B0, L0


def omega_syn(lat_rad, model="mag"):
    """synodic rotation rate deg/day for latitude (radians)."""
    s2 = np.sin(lat_rad) ** 2
    if model == "mag":  # Snodgrass 1983 magnetic
        w = 14.252 - 1.678 * s2 - 2.401 * s2 * s2
    else:  # Howard/Harvey spectroscopic
        w = 14.71 - 2.39 * s2 - 1.78 * s2 * s2
    return w - 0.9856


def img_to_helio(xs, ys, info, B0, P=0.0):
    """pixel coords -> (lat, Lcm, Z) radians; P = assumed rotation (deg, CCW) of solar north from image up."""
    X = (xs - info["cx"]) / info["r"]
    Y = -(ys - info["cy"]) / info["r"]
    if P:
        p = math.radians(P)
        X, Y = X * math.cos(p) + Y * math.sin(p), -X * math.sin(p) + Y * math.cos(p)
    rho2 = X * X + Y * Y
    Z = np.sqrt(np.clip(1 - rho2, 0, None))
    b = math.radians(B0)
    sinlat = np.clip(Y * math.cos(b) + Z * math.sin(b), -1, 1)
    lat = np.arcsin(sinlat)
    L = np.arctan2(X, Z * math.cos(b) - Y * math.sin(b))
    return lat, L, rho2 <= 1


def helio_to_img(lat, L, info, B0, P=0.0):
    b = math.radians(B0)
    X = np.cos(lat) * np.sin(L)
    Y = np.sin(lat) * math.cos(b) - np.cos(lat) * np.cos(L) * math.sin(b)
    Z = np.sin(lat) * math.sin(b) + np.cos(lat) * np.cos(L) * math.cos(b)
    if P:
        p = math.radians(P)
        X, Y = X * math.cos(p) - Y * math.sin(p), X * math.sin(p) + Y * math.cos(p)
    xs = info["cx"] + X * info["r"]
    ys = info["cy"] - Y * info["r"]
    return xs, ys, Z > 0


def warp_maps(info_t, ts_t, info_n, ts_n, shape=(2048, 2048), step=1.0, P=0.0, rate=1.0, model="mag", dlon=0.0):
    """Maps (mx, my) so that cv2.remap(neighbour_img, mx, my) lies in the target frame; plus validity mask.
    step: target grid pixel = step native pixels (coords given in native units of each image)."""
    h, w = shape
    xs = (np.arange(w, dtype=np.float64) + 0.5) * step - 0.5
    ys = (np.arange(h, dtype=np.float64) + 0.5) * step - 0.5
    X, Y = np.meshgrid(xs, ys)
    _, B0t, _ = sun_angles(ts_t)
    _, B0n, _ = sun_angles(ts_n)
    lat, L, on = img_to_helio(X, Y, info_t, B0t, P)
    dt = (pd.Timestamp(ts_t) - pd.Timestamp(ts_n)).total_seconds() / 86400.0
    Ln = L - np.radians(omega_syn(lat, model) * dt * rate + dlon)
    xn, yn, vis = helio_to_img(lat, Ln, info_n, B0n, P)
    valid = on & vis
    return ((xn + 0.5) / step - 0.5).astype(np.float32), ((yn + 0.5) / step - 0.5).astype(np.float32), valid, dt


def stem_ts(stem):
    return pd.Timestamp(pd.to_datetime(stem[:14], format="%Y%m%d%H%M%S"))

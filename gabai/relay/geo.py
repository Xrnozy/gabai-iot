"""Small geodesy helpers for the relay. Flat-earth maths is accurate to centimetres at route scale."""
import math

EARTH_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_M * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def distance_to_route_m(lat, lon, route):
    """Shortest distance from a point to a polyline of [lat, lon] pairs; None for an empty route."""
    if not route:
        return None
    k = math.cos(math.radians(lat))

    def xy(p):
        return (math.radians(p[1] - lon) * k * EARTH_M, math.radians(p[0] - lat) * EARTH_M)

    pts = [xy(p) for p in route]
    if len(pts) == 1:
        return math.hypot(*pts[0])
    best = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / seg2))
        best = min(best, math.hypot(ax + t * dx, ay + t * dy))
    return best

import numpy as np

from mapper.walkable import build, doorways, guide, to_cell


def _room_pair(hole=True):
    """Two 3x3 m rooms side by side, a wall at x = 3 with a 0.9 m opening at y 1.05..1.95, a 0.4 m box in the left room, and a 0.3 m hole in the floor of the right room."""
    g = np.arange(0, 6.0, 0.02)
    X, Y = np.meshgrid(g, np.arange(0, 3.0, 0.02))
    floor = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    if hole:
        floor = floor[~((floor[:, 0] > 4.4) & (floor[:, 0] < 4.7) & (floor[:, 1] > 1.2) & (floor[:, 1] < 1.5))]
    zs = np.arange(0.25, 2.4, 0.1)

    def wall(xs, ys):
        return np.array([[x, y, z] for x, y in zip(xs, ys) for z in zs])

    t = np.arange(0, 3.0, 0.03)
    pts = [floor,
           wall(np.full_like(t, 3.0), t)[~((np.repeat(t, len(zs)) > 1.05) & (np.repeat(t, len(zs)) < 1.95))],
           wall(np.arange(0, 6, 0.03), np.zeros(200)), wall(np.arange(0, 6, 0.03), np.full(200, 3.0)), wall(np.zeros(100), t), wall(np.full(100, 6.0), t)]
    bx = np.arange(0.8, 1.2, 0.02)
    pts.append(np.array([[x, 2.0 + dy, z] for x in bx for dy in (0.0, 0.4) for z in (0.25, 0.45, 0.65)]))
    return np.vstack(pts)


def test_walkable_area_inferred_floor_and_obstacle():
    g = build(_room_pair())
    assert abs(g["floor_z"]) < 0.03
    assert g["stats"]["walkable_m2"] > 0.75 * 18 * 0.95 - 1.0
    assert g["obstacle"][to_cell(g, 1.0, 2.0)]                           # the box
    assert g["walkable"][to_cell(g, 4.55, 1.35)] and g["inferred"][to_cell(g, 4.55, 1.35)]     # the 0.3 m hole in the floor: walkable, marked inferred
    assert 0.04 < g["stats"]["inferred_m2"] < 0.3


def test_big_sparse_gap_is_walkable_but_inferred():
    P = _room_pair(hole=False)
    P = P[~((P[:, 2] == 0) & (P[:, 0] > 4.0) & (P[:, 0] < 5.4) & (P[:, 1] > 0.4) & (P[:, 1] < 2.0))]      # 1.4 x 1.6 m with no floor points at all (a region the cameras never saw the floor of)
    g = build(P)
    assert g["walkable"][to_cell(g, 4.7, 1.2)] and g["inferred"][to_cell(g, 4.7, 1.2)]
    assert not g["inferred"][to_cell(g, 1.5, 0.6)]                       # measured floor stays measured


def test_few_points_do_not_make_an_obstacle():
    P = _room_pair(hole=False)
    speck = np.array([[5.0, 2.5, 0.5], [5.01, 2.5, 0.7], [5.02, 2.51, 0.9], [5.0, 2.52, 1.1]])             # four floating points
    g = build(np.vstack([P, speck]))
    assert not g["obstacle"][to_cell(g, 5.0, 2.5)] and g["walkable"][to_cell(g, 5.0, 2.5)]


def test_one_doorway_in_the_dividing_wall():
    d = doorways(build(_room_pair()))
    assert len(d) == 1
    assert abs(d[0]["center"][0] - 3.0) < 0.35 and abs(d[0]["center"][1] - 1.5) < 0.3
    assert 0.6 < d[0]["width"] < 1.1


def test_path_to_the_door_steers_towards_it():
    g = build(_room_pair())
    d = doorways(g)
    r = guide(g, np.array([0.7, 0.6]), np.array([1.0, 0.0]), d)         # user near the left room's corner, looking along +x
    assert r["target"] == "doorway" and r["dist"] > 2.0
    assert r["bearing_deg"] > 5                                          # the door is up and to the left (+y is left)
    p = r["path"]
    assert np.hypot(*(p[-1] - d[0]["center"])) < 0.8                    # ends at the door
    # no waypoint inside the box
    assert not np.any((p[:, 0] > 0.75) & (p[:, 0] < 1.25) & (p[:, 1] > 1.95) & (p[:, 1] < 2.45))


def test_frontier_when_there_is_no_door():
    g = build(_room_pair())
    r = guide(g, np.array([0.7, 0.6]), np.array([1.0, 0.0]), [])
    assert r is None or r["target"] == "frontier"


def test_a_gap_between_low_furniture_is_not_a_doorway():
    g = np.arange(0, 6.0, 0.02)
    X, Y = np.meshgrid(g, np.arange(0, 3.0, 0.02))
    floor = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    zs = np.arange(0.25, 0.9, 0.1)                                            # two 0.9 m high cabinets with a 0.9 m gap, no wall anywhere
    cab = np.array([[x, y, z] for x in np.arange(2.9, 3.1, 0.02) for y in list(np.arange(0, 1.05, 0.03)) + list(np.arange(1.95, 3.0, 0.03)) for z in zs])
    assert doorways(build(np.vstack([floor, cab]))) == []

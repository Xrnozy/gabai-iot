"""Numeric dictionary for server -> phone messages: a few small ints instead of a sentence. One source of truth, shared with the phone:
mobile/app/src/main/assets/lexicon.json (ids are permanent, 0 = unspecified, new words are only appended).

Detection packet  [obj, dist, dir]  (+ [rel, obj2])  (+ [state])      e.g. [1, 1, 2] = "Human near you on your right."
Message           {"d": [packets], "e": [env ids], "c": [cue ids]}      empty keys are left out
"""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mobile", "app", "src", "main", "assets", "lexicon.json")
FIELDS = ("obj", "dist", "dir", "rel", "obj", "state")                       # packet slot -> table
with open(PATH, encoding="utf-8") as f:
    LEX = json.load(f)
IDS = {t: {v: int(k) for k, v in LEX[t].items()} for t in ("obj", "dist", "dir", "rel", "state", "env", "cue")}


def _id(table, word):
    return 0 if word is None else IDS[table][word]


def packet(obj, dist=None, direction=None, rel=None, obj2=None, state=None):
    p = [_id("obj", obj), _id("dist", dist), _id("dir", direction)]
    if rel is not None:
        p += [_id("rel", rel), _id("obj", obj2)]
    if state is not None:
        p += [0, 0] * (rel is None) + [_id("state", state)]
    return p


def message(detections=(), env=(), cue=()):
    m = {"d": [packet(*d) if not isinstance(d, dict) else packet(**d) for d in detections], "e": [_id("env", e) for e in env], "c": [_id("cue", c) for c in cue]}
    return {k: v for k, v in m.items() if v}


def _word(table, i):
    return LEX[table].get(str(i), "") if i else ""


def say(p):
    """One detection packet -> sentence (the phone does the same in Lexicon.kt)."""
    s = " ".join(w for w in (_word("obj", p[0]), _word("dist", p[1]), _word("dir", p[2])) if w)
    if len(p) >= 5 and p[3]:
        s += ", " + _word("rel", p[3]) + " " + _word("obj", p[4])
    if len(p) >= 6 and p[5]:
        s += ", " + _word("state", p[5])
    return s[:1].upper() + s[1:] + "."


def decode(m):
    return [say(p) for p in m.get("d", [])] + [_word("env", e).capitalize() + "." for e in m.get("e", [])] + [_word("cue", c).capitalize() + "." for c in m.get("c", [])]

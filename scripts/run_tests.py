"""Run the test modules in tests/ one by one without pytest and print a pass/fail line for each.

    py scripts/run_tests.py                 all modules
    py scripts/run_tests.py diag_log pipeline   only modules whose name contains one of these words
    py scripts/run_tests.py -v              also print the traceback of every failure
    py scripts/run_tests.py --priority      only the focused set listed in tests/priority_modules.txt (map, depth, sync, safety)
    py scripts/run_tests.py --deferred      every module that is not in that list

A module runs in its own process, so a crash or a hang in one cannot hide the others. Exit code is 0 only if nothing failed.
Tests marked skip (pytest.mark.skipif) are counted as skipped, not passed.
"""
import glob
import os
import subprocess
import sys
import time

CHILD = r'''
import importlib, sys, traceback, inspect
verbose = sys.argv[2] == "1"
m = importlib.import_module("tests." + sys.argv[1])
n = f = s = 0
for k in sorted(dir(m)):
    if not k.startswith("test_"):
        continue
    fn = getattr(m, k)
    marks = getattr(fn, "pytestmark", [])
    if any(getattr(x, "name", "") == "skipif" and x.args and x.args[0] for x in marks):
        s += 1
        continue
    try:
        if inspect.signature(fn).parameters:
            print("SKIP-fixture", k); s += 1; continue
        fn(); n += 1
    except Exception as e:
        f += 1
        print("FAIL", k, repr(e)[:300])
        if verbose: traceback.print_exc()
print(f"RESULT {n} {f} {s}")
'''


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mods = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(root, "tests", "test_*.py")))
    prio_file = os.path.join(root, "tests", "priority_modules.txt")
    prio = [l.strip() for l in open(prio_file) if l.strip() and not l.startswith("#")] if os.path.exists(prio_file) else []
    unknown = [m for m in prio if m not in mods]
    if unknown:
        sys.exit(f"tests/priority_modules.txt names modules that do not exist: {unknown}")
    if "--priority" in sys.argv:
        mods = [m for m in mods if m in prio]
    elif "--deferred" in sys.argv:
        mods = [m for m in mods if m not in prio]
    if args:
        mods = [m for m in mods if any(a in m for a in args)]
    tot = [0, 0, 0]
    bad = []
    for m in mods:
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-c", CHILD, m, "1" if verbose else "0"], cwd=root, capture_output=True, text=True, timeout=600)
            out = r.stdout.strip().splitlines()
            res = [l for l in out if l.startswith("RESULT")]
            if not res:
                print(f"{m:34s} CRASH (exit {r.returncode}) {r.stderr.strip().splitlines()[-1:] }")
                bad.append(m)
                tot[1] += 1
                continue
            n, f, s = map(int, res[-1].split()[1:])
        except subprocess.TimeoutExpired:
            print(f"{m:34s} TIMEOUT")
            bad.append(m)
            tot[1] += 1
            continue
        tot[0] += n; tot[1] += f; tot[2] += s
        print(f"{m:34s} {n:3d} pass {f:3d} fail {s:3d} skip   {time.time() - t0:5.1f}s")
        for l in out:
            if l.startswith("FAIL") or l.startswith("SKIP-fixture") or (verbose and not l.startswith("RESULT")):
                print("    " + l)
        if f:
            bad.append(m)
    print(f"\nTOTAL {tot[0]} passed, {tot[1]} failed, {tot[2]} skipped in {len(mods)} modules")
    sys.exit(1 if tot[1] else 0)


if __name__ == "__main__":
    main()

"""Make Ctrl-C always end the program.

Python only reacts to Ctrl-C between two bytecodes, so a main thread inside a native call (Basalt, the GPU, a socket) or a wait that Windows cannot interrupt never
sees it, and the viewer, which is a child process in the same console, quits at the first press and leaves the run stuck. This installs a console control handler,
which Windows runs on its own thread at once: the first press starts a countdown (`grace_s`) in which the normal shutdown (map saved, files closed) may finish, and
if the process is still alive afterwards, or a second press comes, it is ended. A clean exit before the countdown is not affected.
"""
import os
import sys
import threading

_keep = []


def install(grace_s=8.0, exit_code=130):
    if os.name != "nt":
        return False
    import ctypes
    presses = [0]

    def die(why):
        print(f"\n{why}: ending the process now", file=sys.stderr, flush=True)
        os._exit(exit_code)

    def handler(ctrl_type):
        if ctrl_type in (0, 1):                           # CTRL_C_EVENT, CTRL_BREAK_EVENT
            presses[0] += 1
            if presses[0] >= 2:
                die("second Ctrl-C")
            print(f"\nstopping (Ctrl-C again to end it at once, or it ends itself in {grace_s:.0f} s) ...", file=sys.stderr, flush=True)
            t = threading.Timer(grace_s, die, args=("the shutdown did not finish",))
            t.daemon = True
            t.start()
        return 0                                          # not handled: Python's own handler still runs and raises KeyboardInterrupt

    proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
    cb = proto(handler)
    _keep.append(cb)                                      # the callback must stay alive as long as the process does
    return bool(ctypes.windll.kernel32.SetConsoleCtrlHandler(cb, True))

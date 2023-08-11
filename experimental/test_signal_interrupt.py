#!/usr/bin/env python3
import signal
import time
import sys
import os

class InterruptHandler(object):
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)
        def handler(signum, frame):
            print("Got SIGINT", flush=True)
            self.release()
            self.interrupted = True
        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True

def try_restart(i):
    try:
        with open("example.checkpoint", "r") as f:
            i = int(f.readline())
            print(f"Restarting from {i}")
            return i
    except:
        return i

def checkpoint(i):
    with open("example.checkpoint", "w") as f:
        print("checkpointing...", end="")
        f.write(str(i))
        print("done")
    
def main():
    with InterruptHandler() as h:    
        for i in range(try_restart(0), 300):
            print(f"iteration = {i}", flush=True)
            time.sleep(5)
            if h.interrupted:
                checkpoint(i)
                sys.exit(0)
        sys.exit(0)

if __name__ == "__main__":
    main()

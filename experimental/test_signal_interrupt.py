import signal
import time
import sys
import os

# Define a variable to track if a signal is received
signal_received = False

# Define a signal handler function
def handle_signal(signum, frame):
    print('Signal handler called with signal', signum)
    global signal_received
    signal_received = True

# Associate the signal handler function with the USR1 signal
signal.signal(signal.SIGUSR1, handle_signal)
# signal.signal(signal.SIGINT, handle_signal)

if __name__ == '__main__':
    i=0
    sleep_time = 1
    while True:
        print(f"Sleeping for {i} seconds")
        sys.stdout.flush()
        time.sleep(sleep_time)
        i+=1

        if signal_received:
            print("Signal received, can checkpoint here.")
            sys.stdout.flush()
            sys.exit(1)
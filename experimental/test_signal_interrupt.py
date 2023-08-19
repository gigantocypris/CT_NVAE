import signal
import time
import sys
import os

# Define a signal handler function
def handle_signal(signum, frame):
    print('Signal handler called with signal', signum)
    # Sleep for two minutes (double the checkpoint time)
    time.sleep(120)


# Associate the signal handler function with the USR1 signal
signal.signal(signal.SIGUSR1, handle_signal)

if __name__ == '__main__':
    print("Started script")
    sys.stdout.flush()
    i=0
    sleep_time = 1
    while True:
        print(f"Sleeping for {i} seconds")
        sys.stdout.flush()
        time.sleep(sleep_time)
        i+=1
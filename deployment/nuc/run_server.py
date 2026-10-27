"""ZeroRPC bridge inside the verified DROID/Polymetis image."""

import zerorpc
from robot import FrankaRobot

if __name__ == "__main__":
    server = zerorpc.Server(FrankaRobot(), heartbeat=None)
    server.bind("tcp://0.0.0.0:4242")
    server.run()

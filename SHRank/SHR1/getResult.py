import time
from .hotpotServer import Hotpots

NUM=20
class TestResult:
    def __init__(self, test):
        self.test = test

if __name__ == "__main__":
    MygetHot=Hotpots(1)
    MygetHot.hotpotIter(NUM)
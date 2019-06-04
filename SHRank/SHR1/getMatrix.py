import time
import os.path
import hotpotServer

__PATH__GRAPH = '../Graph/graphfile'
__PATH__RANK = './rankresult/'

L = 48
NUM = 20


class TestMatrix:
    def __init__(self, test):
        self.test = test

if __name__ == "__main__":
    MygetHot = hotpotServer.Hotpots(__PATH__GRAPH, __PATH__RANK, 24)
    starttime = time.time()
    funcNij = MygetHot.getFuncNij(NUM)
    getMatrixtime = time.time() - starttime
    print(getMatrixtime)
    f = os.file(os.path.join(__PATH__RANK, 'getMatrix' + str(NUM) + '.txt'), "a+")
    f.write(str(getMatrixtime) + '\n')
    f.close()
from enum import Enum
import librosa
import time
import config as cfg
import numpy as np
from toolkit import timeFreqStft, dict_inf
from collections import defaultdict
from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
from frame_reader import FrameReader
MAX_RUN = 3
DIAG_COST_FACTOR=1
SECS_TO_TICKS = 1 * 1000 * 1000 * 10

class Direction(Enum):
    NONE=0,
    ROW=1,
    COL=2,
    BOTH=3
    
class ODTW:
    def __init__(self):
        self.fr = FrameReader()
        self._t1 = np.array([], dtype=np.float32)
        self._t2 = np.array([], dtype=np.float32)
        # self.totalCostMatrix = defaultdict(lambda: (float('inf'),))
        # self.cellCostMatrix = defaultdict(lambda: (float('inf'),))
        self.totalCostMatrix = defaultdict(dict_inf)
        self.cellCostMatrix = defaultdict(dict_inf)
        self.searchWidth = 5 # sec
    
    def getPathLen(self, i, j):
        return 1+i+j
    
    def optimalWarpingPath(self, totalCostMatrix, i, j):
        path = []
        path.append((i,j))
        factor = 1.0
        while i>0 or j>0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                dmin = min(totalCostMatrix[i-1, j], totalCostMatrix[i, j-1], totalCostMatrix[i-1, j-1])
                if dmin == totalCostMatrix[i-1, j-1]:
                    i-=1
                    j-=1
                elif dmin == totalCostMatrix[i-1, j]:
                    i-=1
                    factor-=0.001
                else:
                    j-=1
                    factor+=0.001
            path.append((i,j))
        return path, factor
    
    def warpingPathTimes(self, path, optimize):
        if optimize:
            pairBuffer = []
            cleanedPath = []
            for i,j in path:
                if len(pairBuffer) == 0:
                    pairBuffer.append((i,j))
                else:
                    if i == pairBuffer[len(pairBuffer)-1][0] and j == pairBuffer[len(pairBuffer)-1][1]+1:
                        pairBuffer.append((i,j))
                    elif j == pairBuffer[len(pairBuffer)-1][1] and i == pairBuffer[len(pairBuffer)-1][0]+1:
                        pairBuffer.append((i,j))
                    else:
                        if len(pairBuffer) == 1:
                            cleanedPath.append(pairBuffer[0])
                        elif len(pairBuffer) > 1:
                            new_pb = list(zip(*pairBuffer))
                            cleanedPath.append((sum(new_pb[0])/len(new_pb[0]), sum(new_pb[1])/len(new_pb)))
                        pairBuffer.clear()
                        pairBuffer.append((i,j))
            cleanedPath+=pairBuffer
            path = cleanedPath
        pathTime = []
        for i,j in path:
            # timePair = (self.positionToTimeSpan(i*cfg.STFT['hop']),
            #             self.positionToTimeSpan(j*cfg.STFT['hop']))
            timePair = (i*cfg.STFT['hop'],
                        j*cfg.STFT['hop'])
            if timePair[0] >= 0 and timePair[1] >= 0:
                pathTime.append(timePair)
        return pathTime
    
    def positionToTimeSpan(self, pos):
        return round(pos/cfg.SAMPLE_RATE)
        # return round(pos/cfg.SAMPLE_RATE)
    
    def run(self, t1, t2):
        # prepare
        self._t1 = self.fr.readStream(timeFreqStft(t1)) # time FRAMESIZE(84)
        self._t2 = self.fr.readStream(timeFreqStft(t2))
        # self._t1 = timeFreqStft(t1)
        # self._t2 = timeFreqStft(t2)
        t1_len = len(self._t1)
        t2_len = len(self._t2)
        c = int(self.searchWidth*(1.0*cfg.SAMPLE_RATE/cfg.STFT['hop']))
        # c = min(c, min(t1_len, t2_len))
        i,j = 0,0
        minI,minJ = 0,0
        direction = Direction.NONE
        pre_direction = Direction.NONE
        run_count = 0
        path_len_row = np.zeros((c+MAX_RUN))
        path_len_col = np.zeros((c+MAX_RUN))
        pre_path_len_row = np.zeros((c+MAX_RUN))
        pre_path_len_col = np.zeros((c+MAX_RUN))
        total_frame = t1_len + t2_len
        
        self.totalCostMatrix[0, 0] = self.calculateCost(self._t1[0], self._t2[0])
        self.cellCostMatrix[0, 0] = self.totalCostMatrix[0, 0]
        print(c, self.cellCostMatrix, self.totalCostMatrix)
        while i < t1_len-1 or j < t2_len-1:
            if i < c: # build initial square matrix
                direction = Direction.BOTH
                minI, minJ = i, j
            else:
                xi = i
                xv = np.inf
                for x in range(i-c+1, i+1):
                    pc = self.totalCostMatrix[x, j]/path_len_row[x % len(path_len_row)] # norm
                    if pc <= xv:
                        xv = pc
                        xi = x
                yi = j
                yv = np.inf
                for y in range(j-c+1, j+1):
                    pc = self.totalCostMatrix[i, y]/path_len_col[y % len(path_len_row)]
                    if pc <= yv:
                        yv = pc
                        yi = y
                if xi == i and yi == j:
                    direction = Direction.BOTH
                    minI = i
                    minJ = j
                elif xv < yv:
                    direction = Direction.ROW
                    minI = xi
                    minJ = j
                else:
                    direction = Direction.COL
                    minI = i
                    minJ = yi
            if direction == pre_direction:
                run_count+=1
            if direction == Direction.BOTH:
                run_count=0
            elif run_count >= MAX_RUN:
                if direction == Direction.ROW:
                    direction = Direction.COL
                elif direction == Direction.COL:
                    direction = Direction.ROW
                run_count = 0
            
            # add row
            if j < t2_len-1 and (direction == Direction.ROW or direction == Direction.BOTH):
                path_len_row, pre_path_len_row = pre_path_len_row, path_len_row # swap
                j+=1
                for x in range(max(i-c+1, 0), i+1):
                    cell_cost = self.calculateCost(self._t1[x],self._t2[j])
                    # print(type(cell_cost))
                    if x == 0:
                        self.totalCostMatrix[x,j] = self.totalCostMatrix[x,j-1] + cell_cost # 角
                    else:
                        self.totalCostMatrix[x,j] = min(self.totalCostMatrix[x-1, j-1]+DIAG_COST_FACTOR*cell_cost,
                                                        self.totalCostMatrix[x-1,j]+cell_cost,
                                                        self.totalCostMatrix[x,j-1]+cell_cost)
                    # print("test:", self.totalCostMatrix[x,j], cell_cost)
                    self.cellCostMatrix[x,j] = cell_cost
                    path_len_row[x%len(path_len_row)] = self.getPathLen(x,j)
                path_len_col[j%len(path_len_col)] = path_len_row[i%len(path_len_row)]
            # add col
            if i < t1_len-1 and (direction == Direction.COL or direction == Direction.BOTH):
                path_len_col, pre_path_len_col = pre_path_len_col, path_len_col # swap
                i+=1
                for y in range(max(j-c+1, 0), j+1):
                    cell_cost = self.calculateCost(self._t1[i],self._t2[y])
                    if y == 0:
                        self.totalCostMatrix[i,y] = self.totalCostMatrix[i-1,y] + cell_cost
                    else:
                        self.totalCostMatrix[i,y] = min(self.totalCostMatrix[i-1, y-1]+DIAG_COST_FACTOR*cell_cost,
                                                        self.totalCostMatrix[i-1,y]+cell_cost,
                                                        self.totalCostMatrix[i,y-1]+cell_cost)
                    self.cellCostMatrix[i,y] = cell_cost
                    path_len_col[y%len(path_len_col)] = self.getPathLen(i,y)
                path_len_row[i%len(path_len_row)] = path_len_col[j%len(path_len_col)]
            
            pre_direction = direction
        path, factor = self.optimalWarpingPath(self.totalCostMatrix, t1_len-1, t2_len-1)
        path.reverse()
        # return path
        return factor
        # return self.warpingPathTimes(path, True), factor
    
    def calculateCost(self, f1, f2):
        d = np.sum(np.abs(f1-f2))
        s = np.sum(f1+f2)
        if s == 0:
            return np.float64(0.0)
        
        weight = 8+np.log(s)/10.0
        if weight < 0:
            weight = 0
        elif weight > 1:
            weight = 1
        cost = 90.0*d/s*weight # weight = 0~1
        return cost
        # return np.sum(np.abs(f1-f2)**2)**0.5
# if __name__ == "__main__":        
#     odtw = ODTW()
#     t1, _ = librosa.load('./music/test/summer3rd_fix_1.wav', sr=44100)
#     t2, _ = librosa.load('./music/test/summer3rd_violin_15s.wav', sr=44100)
#     t1 = t1[:44100*6]
#     t2 = t2[:44100*6]
#     test1 = np.arange(0, 44100*10, dtype=np.float32)
#     test2 = np.arange(44100*10, 44100*20, dtype=np.float32)
#     start = time.time()
#     # params = [(test1,test2), (test1*2,test2*2), (test1*3,test2*3), (test1*4,test2*4)]
#     # # params = [(test1,test2)]
#     print("start")
#     # pool = Pool(4)
#     # res = pool.starmap(odtw.run, params)
#     # pool.close() # close只是關閉pool, 但已開啟的process還是會繼續執行
#     # pool.join()
#     res = odtw.run(t1, t2)
#     # print(res)
#     end = time.time()
#     print("total time", end-start)
#     print(len(t1), len(t2))
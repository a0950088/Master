from source import config as cfg
from source.toolkit import timeFreqStft, dict_inf, extractHighAndLowFeature

import time
import librosa
import numpy as np
from enum import Enum
from collections import defaultdict
from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process
from itertools import repeat
import matplotlib.pyplot as plt
from scipy import spatial
import numba as nb

MAX_RUN = 3 # +10/-10 bpm delay
DIAG_COST_FACTOR=1
SECS_TO_TICKS = 1 * 1000 * 1000 * 10

class Direction(Enum):
    I = 1
    J = 2
    
DIR_I = set([Direction.I])
DIR_J = set([Direction.J])
DIR_IJ = set([Direction.I, Direction.J])

VALID_RANGE = 225

class ODTW:
    def __init__(self, raw_ref, live_queue, rpe_queue, acc_queue, share_acc_record):
        self.live_queue = live_queue
        self.rpe_queue = rpe_queue
        self.rpe_possible_list = []
        self.rpe_live_frame = np.inf
        
        
        self.ref, _ = extractHighAndLowFeature(raw_ref)
        self.ref_len = self.ref.shape[0]
        self.live_feature_record = np.zeros((0, self.ref.shape[1]), dtype=np.float32)
        self.live = np.zeros((self.ref.shape[0],self.ref.shape[1]), dtype=np.float32)
        self.acc_queue = acc_queue # deque
        self.c = cfg.SEARCH_WINDOW
        self.run_count = 0
        
        self.totalCostMatrix = defaultdict(dict_inf)
        self.cellCostMatrix = defaultdict(dict_inf)
        self.path_len_row = np.zeros((self.c+MAX_RUN))
        self.path_len_col = np.zeros((self.c+MAX_RUN))
        self.pre_path_len_row = np.zeros((self.c+MAX_RUN))
        self.pre_path_len_col = np.zeros((self.c+MAX_RUN))
        
        self.prime_path = []
        self.offline_path = []
        
        self.share_acc_record = share_acc_record
        # self.acc_record = share_acc_record['acc_record']
        # self.acc = share_acc_record['acc']
        self.rpe_points = []
        self.others_rpe_points = []
        self.use_rpe = False
        
        self.prev_j_prime = [0,0] # record prev output j
        
    def run(self, start_raw_frame=0):
        i = 0
        j = 0
        # live_len = i
        
        curr = set()
        prev = set()
        
        i_prime = 0
        j_prime = 0
        # self.acc_queue.put((i_prime, j_prime))
        self.acc_queue.append((i_prime, j_prime))
        # self.test_acc_queue.put(j_prime*cfg.HOP_SIZE)
        self.share_acc_record['acc_record'] = np.concatenate((self.share_acc_record['acc_record'], self.share_acc_record['acc'][j_prime*cfg.HOP_SIZE:(j_prime*cfg.HOP_SIZE)+cfg.HOP_SIZE]))
        # l_i = self.live_queue.get()
        # # print(self.live_feature_record.shape, l_i.shape)
        # if l_i is None:
        #     # self.acc_queue.put(None)
        #     self.acc_queue.append(None)
        #     return
        # self.live_feature_record = np.concatenate((self.live_feature_record, l_i[np.newaxis,:]), dtype=np.float32)
        # self.receiveLive(i, l_i)
        # r_j = self.ref[j]
        
        data = self.live_queue.get()
        if data is None:
            self.acc_queue.append(None)
            return
        live_seg = data[0]
        now_live_len = data[1]
        self.live_feature_record = np.concatenate((self.live_feature_record, live_seg), dtype=np.float32)
        # for l in live_seg:
        #     # self.live_feature_record = np.concatenate((self.live_feature_record, l[np.newaxis,:]), dtype=np.float32)
        #     self.receiveLive(live_len, l)
        #     live_len+=1
        print("get new live", self.live_feature_record.shape)
        l_i = self.live_feature_record[i]
        r_j = self.ref[j]
            
        
        
        # live_seq, _ = extractHighAndLowFeature(live_seq)
        # seq_len = live_seq.shape[0]
        # first frame
        d = self.calculateCost(l_i, r_j)
        self.setCostMatrix(i, j, d)
        # self.cellCostMatrix[0, 0] = self.totalCostMatrix[0, 0]
        start_time = time.time()
        while True:
            # print("run ",i, j)
            inner_time = time.time()
            if j == len(self.ref)-1:
                # end tracking
                # self.acc_queue.put(None)
                self.acc_queue.append(None)
                print("odtw end: ", time.time()-start_time, i_prime, j_prime, self.totalCostMatrix[i_prime, j_prime], self.live_feature_record.shape)
                self.optimalWarpingPath()
                self.draw()
                return
            
            # add live
            if not self.live_queue.empty() or i>=self.live_feature_record.shape[0]-1:
                
                # get rpe
                # if not self.rpe_queue.empty():
                # if len(self.rpe_queue) > 0 and (self.rpe_live_frame == np.inf or self.rpe_live_frame<i):
                #     # self.rpe_possible_list, self.rpe_live_frame = self.rpe_queue.get() # rpe live frame
                #     self.rpe_possible_list, self.rpe_live_frame = self.rpe_queue.peek_last() # rpe live frame
                #     self.rpe_live_frame = self.rpe_live_frame//cfg.HOP_SIZE
                #     print("rpe possible_list: ",i,self.live_feature_record.shape,self.rpe_possible_list, self.rpe_live_frame)
                
                data = self.live_queue.get()
                if data is None:
                    self.acc_queue.append(None)
                    return
                live_seg = data[0]
                now_live_len = data[1]
                self.live_feature_record = np.concatenate((self.live_feature_record, live_seg), dtype=np.float32)
                # for l in live_seg:
                #     # self.live_feature_record = np.concatenate((self.live_feature_record, l[np.newaxis,:]), dtype=np.float32)
                #     self.receiveLive(live_len, l)
                #     live_len+=1
                print("get new live", self.live_feature_record.shape)
            
            if len(self.rpe_queue)>0 and (self.rpe_live_frame==np.inf or self.rpe_live_frame<i): # rpe_live_frame是否已計算完畢或目前live已經超過rpe取得的live
                # self.rpe_possible_list, self.rpe_live_frame = self.rpe_queue.get() # rpe live frame
                # self.use_rpe = True
                self.rpe_possible_list, self.rpe_live_frame = self.rpe_queue.peek_last() # rpe live frame
                self.rpe_live_frame = self.rpe_live_frame//cfg.HOP_SIZE
                print("rpe possible_list: ",i,self.live_feature_record.shape,self.rpe_possible_list, self.rpe_live_frame)

            if self.rpe_live_frame == i:
                # backtrack totalCostMatrix & compare
                self.rpe_possible_list.append(j_prime) # 加入現在位置計算
                print("dealsPossiblePosition: ", self.rpe_live_frame, self.rpe_possible_list)
                self.use_rpe = True
                start = time.time()
                cost, new_ref_pos = self.backtracking()
                print("back end: ", time.time()-start, cost, int(new_ref_pos))
                if int(new_ref_pos) != j_prime:
                    print("new position use!")
                j_prime = int(new_ref_pos)
                self.rpe_points.append((i_prime, j_prime))
                
                self.rpe_live_frame = np.inf
                self.run_count = 1
                curr = set()
                prev = set()
                j = j_prime
                # continue
            
            curr = self.get_next_direction(i, j, i_prime, j_prime, prev)
            # i_prime == live speed 讓i_prime一直累加
            i_prime+=1
            i+=1
            l_i = self.live_feature_record[i]
            # 計算+c/-c範圍的ref cost
            for J in range(max(0, j-self.c+1), min(j+1+self.c, self.ref_len)):
                r_J = self.ref[J]
                d = self.calculateCost(l_i, r_J)
                self.setCostMatrix(i, J, d)
            #     self.path_len_col[J%len(self.path_len_col)] = self.getPathLength(i, J)
            # self.path_len_row[i%len(self.path_len_row)] = self.path_len_col[j%len(self.path_len_col)]
            
            # add ref  
            if Direction.J in curr:
                # self.path_len_row, self.pre_path_len_row = self.pre_path_len_row, self.path_len_row # swap
                if j+1 < self.ref_len:
                    j+=1
                # print("now j:", j)
                # j=j_prime+1
                r_j = self.ref[j]
                for I in range(max(0, i-self.c+1), i+1):
                    l_I = self.live_feature_record[I]
                    d = self.calculateCost(l_I, r_j)
                    self.setCostMatrix(I, j, d)
                #     self.path_len_row[I%len(self.path_len_row)] = self.getPathLength(I, j)
                # self.path_len_col[j%len(self.path_len_col)] = self.path_len_row[i%len(self.path_len_row)]
            
            if curr == prev and prev != DIR_IJ:
                self.run_count += 1
            else:
                self.run_count = 1
            prev = curr
            
            # i_prime, j_prime = self.get_ij_prime(i, j, use_rpe=self.use_rpe)
            if self.use_rpe == True:
                _, j_prime = self.get_ij_prime(i_prime, j_prime)
            else:
                # search c
                _, j_prime = self.get_ij_prime(i_prime, j)
            
            print("odtw res: ", curr==DIR_I,curr==DIR_J,curr==DIR_IJ, i, j, i_prime, j_prime, self.totalCostMatrix[i_prime, j_prime], time.time()-inner_time)
            
            # self.acc_queue.append((i_prime*cfg.HOP_SIZE, j_prime*cfg.HOP_SIZE))
            self.acc_queue.append((i_prime*cfg.HOP_SIZE, j_prime*cfg.HOP_SIZE))
            self.prime_path.append((i_prime, j_prime))
            self.share_acc_record['acc_record'] = np.concatenate((self.share_acc_record['acc_record'], self.share_acc_record['acc'][j_prime*cfg.HOP_SIZE:(j_prime*cfg.HOP_SIZE)+cfg.HOP_SIZE]))
            # self.test_acc_queue.put(j_prime*cfg.HOP_SIZE)
            # if self.totalCostMatrix[i_prime, j_prime] > 500: # threshold
            #     # use possible pos
            #     # use pool to compute 4 pos
            #     # 紀錄live frame
            #     # 更新開始位置,ij從開始位置計算
            #     possible_list = self.rpe_queue.get()
        
    
    def optimalWarpingPath(self):
        # path.append((i*cfg.STFT['hop'],j*cfg.STFT['hop']))
        i,j = self.prime_path[-1][0], self.ref_len-1
        # i,j = 1081, self.ref_len-1
        self.offline_path.append((i,j))
        # factor = 1.0
        while i>0 or j>0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                dmin = min(self.totalCostMatrix[i-1, j], self.totalCostMatrix[i, j-1], self.totalCostMatrix[i-1, j-1])
                # print(self.totalCostMatrix[i-1, j], self.totalCostMatrix[i, j-1], self.totalCostMatrix[i-1, j-1])
                if dmin == self.totalCostMatrix[i-1, j-1]:
                    i-=1
                    j-=1
                elif dmin == self.totalCostMatrix[i-1, j]:
                    i-=1
                    # factor-=0.001
                else:
                    j-=1
                    # factor+=0.001
            # path.append((i*cfg.STFT['hop'],j*cfg.STFT['hop']))
            self.offline_path.append((i,j))
        self.offline_path.reverse()
    
    def draw(self):
        d = dict(self.totalCostMatrix)
        fig = plt.figure(figsize=(8, 6))
        cost_matrix = []
        for j in range(self.ref.shape[0]):
            x = []
            for i in range(self.live_feature_record.shape[0]):
                if (i,j) in d.keys():
                    try:
                        x.append(int(d[(i,j)]))
                    except:
                        x.append(np.inf)
                else:
                    x.append(np.inf)
            cost_matrix.append(x)
        # while not self.rpe_queue.empty():
        #     points, frame = self.rpe_queue.get()
        #     flag = True
        #     for p in points:
        #         plt.scatter(frame//882, int(p), c='black', s=2) if flag else plt.scatter(frame//882, int(p), c='gray', s=2)
        #         flag=False
        
        # pre_j = -1
        for i,j in self.offline_path:
            # if pre_j != j:
            #     print("i:", i, j)
            #     pre_j = j
            plt.scatter(i, j, c='b', s=1.5) # x: live y: ref
        for i,j in self.prime_path:
            # if pre_j != j:
            #     print("i:", i, j)
            #     pre_j = j
            plt.scatter(i, j, c='g', s=1) # x: live y: ref
        for i,j in self.others_rpe_points:
            plt.scatter(i, j, c='yellow', s=1) # x: live y: ref
            
        for i,j in self.rpe_points:
            plt.scatter(i, j, c='cyan', s=1) # x: live y: ref
            
        plt.title("Plot 2D array")
        plt.imshow(cost_matrix, cmap="inferno")
        plt.colorbar()
        plt.xlabel('live')
        plt.ylabel('ref')
        plt.gca().invert_yaxis()
        plt.show()
    
    def __backtracking(self, pos):
        SEC = int(9*cfg.SAMPLE_RATE/cfg.HOP_SIZE)
        row_maxcount = 0
        col_maxcount = 0
        row_flag = True
        col_flag = True
        step = 0
        count = 0
        i = self.rpe_live_frame
        j = int(pos)
        accumulate_cost = self.totalCostMatrix[i,j]
        self.others_rpe_points.append((i,pos))
        while (i>0 or j>0) and step < SEC:
            if i == 0:
                # break
                # minc = self.totalCostMatrix[i, j-1]
                j-=1
                # count+=1
            elif j == 0:
                # break
                # minc = self.totalCostMatrix[i-1, j]
                i-=1
                # count+=1
            else:
                if row_flag == False:
                    minc = min(self.totalCostMatrix[i, j-1], self.totalCostMatrix[i-1, j-1])
                elif col_flag == False:
                    minc = min(self.totalCostMatrix[i-1, j], self.totalCostMatrix[i-1, j-1])
                else:
                    minc = min(self.totalCostMatrix[i-1, j], self.totalCostMatrix[i, j-1], self.totalCostMatrix[i-1, j-1])
                    
                if minc == self.totalCostMatrix[i-1, j-1]:
                    i-=1
                    j-=1
                    count+=2
                    row_maxcount, col_maxcount = 0, 0
                    row_flag, col_flag = True, True
                elif minc == self.totalCostMatrix[i, j-1]:
                    j-=1
                    count+=1
                    row_maxcount = 0
                    row_flag = True
                    if col_maxcount+1 == MAX_RUN:
                        col_flag = False
                    col_maxcount+=1
                else:
                    i-=1
                    count+=1
                    col_maxcount = 0
                    col_flag = True
                    if row_maxcount+1 == MAX_RUN:
                        row_flag = False
                    row_maxcount+=1
            step+=1
            # accumulate_cost+=self.cellCostMatrix[i,j]
            accumulate_cost+=self.totalCostMatrix[i,j]
            
        print("backtrack res:",(accumulate_cost/count, accumulate_cost/(count+pos), accumulate_cost, pos), count, step)
        # return accumulate_cost/count
        return accumulate_cost/(count+pos)

    def backtracking(self):
        # SEC = int(9*cfg.SAMPLE_RATE/cfg.HOP_SIZE) # 9*44100/HOPSIZE
        min_pos_ret = (np.inf, None)
        min_prime_ret = (np.inf, None)
        cost_threshold = 500
        
        for ref_position in self.rpe_possible_list[:-1]:
        # for ref_position in self.rpe_possible_list:
            pos_cost = self.__backtracking(ref_position)
            min_pos_ret = min(min_pos_ret, (pos_cost, ref_position), key=lambda a: a[0])
            # print("backtrack res:",(accumulate_cost, ref_position))
            # min_ret = min(min_ret, (accumulate_cost, ref_position), key=lambda a: a[0])
            if min_pos_ret[0] == np.inf:
                min_pos_ret = (np.inf, self.rpe_possible_list[-1])
        
        # for ref_position in self.rpe_possible_list:
        #     for rp in range(max(0, int(ref_position)-9), min(int(ref_position)+9+1, self.ref_len)):
        #         pos_cost = self.__backtracking(rp)
        #         min_pos_ret = min(min_pos_ret, (pos_cost, rp), key=lambda a: a[0])
        #         # print("backtrack res:",(accumulate_cost, ref_position))
        #         # min_ret = min(min_ret, (accumulate_cost, ref_position), key=lambda a: a[0])
        #         if min_pos_ret[0] == np.inf:
        #             min_pos_ret = (np.inf, self.rpe_possible_list[-1])
        
        prime_key_point = self.rpe_possible_list[-1]
        for prime_position in range(prime_key_point-MAX_RUN, min(prime_key_point+(MAX_RUN)+1, self.ref_len)):
            pos_cost = self.__backtracking(prime_position)
            min_prime_ret = min(min_prime_ret, (pos_cost, prime_position), key=lambda a: a[0])
            # print("backtrack res:",(accumulate_cost, ref_position))
            # min_ret = min(min_ret, (accumulate_cost, ref_position), key=lambda a: a[0])
            if min_prime_ret[0] == np.inf:
                min_prime_ret = (np.inf, self.rpe_possible_list[-1])
        
        print("diff cost:", min_pos_ret,min_prime_ret,min_pos_ret[0]-min_prime_ret[0])
        
        # if min_pos_ret[0]>min_prime_ret[0] and (min_pos_ret[0]-min_prime_ret[0])<cost_threshold:
        #     return min_pos_ret
        
        # if min_pos_ret[0]<min_prime_ret[0] and (min_prime_ret[0]-min_pos_ret[0])<cost_threshold:
        #     return min_prime_ret
        
        return min(min_pos_ret, min_prime_ret, key=lambda a: a[0])
            
        # return min_pos_ret
    
    def receiveLive(self, i, l_i):
        if i >= self.live.shape[0]:
            required_len = int(i * 1.5)  # 1.5x leeway
            to_add = required_len - self.live.shape[0]
            print("to add: ", to_add)
            self.live = np.append(
                self.live, np.zeros((to_add, self.ref.shape[1]), dtype=np.float32), axis=0
            )
        self.live[i] = l_i
    
    def get_ij_prime(self, i, j):
        i_prime, j_prime = (i, j)
        min_cost = np.inf
        # curr_i = i
        # 不改變live的位置，因為live不會跳動
        # curr_i = i-1
        # while curr_i >= 0 and curr_i > (i-self.c):
        #     if self.totalCostMatrix[curr_i, j] < min_cost:
        #         i_prime, j_prime = (curr_i, j)
        #         min_cost = self.totalCostMatrix[i_prime, j_prime]
        #     curr_i -= 1
        # curr_j = j-1
        # curr_j = j+MAX_RUN-1
        # upper_limit = j+self.c-1 if not use_rpe else j+MAX_RUN-1
        
        # TODO: 加入斜率資訊當weight
        upper_limit = min(j+(MAX_RUN*2)-1, self.ref_len-1)
        # lower_limit = j-(MAX_RUN*2)
        lower_limit = j-1
        # if curr == DIR_IJ:
        #     upper_limit = j+self.c-1
        #     lower_limit = j-self.c
        # elif curr == DIR_J:
        #     upper_limit = j+self.c-1
        #     lower_limit = j
        # else:
        #     upper_limit = j-1
        #     lower_limit = j-self.c
        curr_j = upper_limit
        
        # while curr_j >= 0 and curr_j > (j-MAX_RUN):
        while curr_j >= 0 and curr_j > lower_limit:
            if self.totalCostMatrix[i, curr_j]+self.cellCostMatrix[i,curr_j] < min_cost:
                i_prime, j_prime = (i, curr_j)
                min_cost = self.totalCostMatrix[i_prime, j_prime]+self.cellCostMatrix[i,curr_j]
            curr_j -= 1
        # if use_rpe:
        #     print("limit upper & lower range")
        #     self.use_rpe = False
        jump_threshold = 2
        if self.prev_j_prime[0] == j_prime:
            if self.prev_j_prime[1] > MAX_RUN//2:
                print("jump_threshold")
                j_prime+=jump_threshold
                self.prev_j_prime = [j_prime, 0]
            else:
                self.prev_j_prime[1] += 1 # max run
        else:
            self.prev_j_prime = [j_prime, 0]
            
        return i_prime, j_prime
    
    def setCostMatrix(self, i, j, d):
        if (i, j) == (0, 0):
            self.totalCostMatrix[i, j] = d
        else:
            self.totalCostMatrix[i, j] = min(
                d+self.totalCostMatrix[i-1, j-1],
                d+self.totalCostMatrix[i-1, j],
                d+self.totalCostMatrix[i, j-1]
            )
        self.cellCostMatrix[i, j] = d
    
    def getPathLength(self, i, j):
        return 1+i+j
        
            
    def get_next_direction(self, i, j, i_prime, j_prime, prev):
        if i < self.c:
            return DIR_IJ # both
        elif self.run_count > MAX_RUN:
            if prev == DIR_I:
                return DIR_J
            return DIR_I
        
        if i_prime < i:
            return DIR_J
        elif j_prime < j:
            return DIR_I
        return DIR_IJ
    
    def calculateCost(self, f1, f2):
        # cos_sim = 1-(np.dot(f1, f2)/((np.linalg.norm(f1)*np.linalg.norm(f2))))
        # cos_sim = 1 - spatial.distance.cosine(f1, f2)
        # print("c", cos_sim)
        return np.sum(np.abs(f1-f2)**2)**0.5
        # return cos_sim
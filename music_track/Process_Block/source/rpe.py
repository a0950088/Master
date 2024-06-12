from enum import Enum
import librosa
import time
from source import config as cfg
import numpy as np
from source.toolkit import timeFreqStft, dict_inf, extractHighAndLowFeature
from collections import defaultdict
from multiprocessing import Process, Pipe, Queue, Pool, Manager, set_start_method, current_process

class Direction(Enum):
    NONE=0,
    ROW=1,
    COL=2,
    BOTH=3

class RPE:
    def __init__(self, ref):
        sec = 9
        # self.n = int((sec*cfg.SAMPLE_RATE)/(0.3*cfg.SAMPLE_RATE)) # backward alignments last n sec
        self.n = 150
        self.threshold = 0.95 # similarity
        self.max_dir_count = 3 # max count on search algorithm
        
        self.ref_high, self.ref_low = extractHighAndLowFeature(ref) # t-f
        self.ref_low = self.ref_high
        self.live = np.zeros(0, dtype=np.float32)
        self.cost_matrix = np.zeros((self.live.shape[0], self.ref_low.shape[0]))
        self.possible_ref = set()
        
    # def getLowResulotionFeature(self, data, frame_len=0.6, hop_len=0.3, sr=44100):
    #     return librosa.util.frame(data, frame_length=int(frame_len*sr), hop_length=int(hop_len*sr), axis=0) # (time, amplitude)
    
    def dist(self, f1):
        return np.sum(np.abs(self.ref_low-f1)**2, axis=1)**0.5

    def backtrack_align(self, start_ref_point): # n = 9*44100/0.3*44100
        # i: live_idx j: ref_idx
        direction = Direction.NONE
        prev_direction = Direction.NONE
        # similarity = np.zeros(min(start_ref_point+self.n, self.cost_matrix.shape[1])-start_ref_point, dtype=np.float32)
        now_live_len = self.cost_matrix.shape[0]
        # start_align = self.cost_matrix.shape[1]-n if self.cost_matrix.shape[1]>=n else 0 # live time
        search_range = (max(0, start_ref_point-self.n), min(start_ref_point+self.n-1, self.cost_matrix.shape[1]-1))
        similarity = np.zeros(search_range[1]-search_range[0], dtype=np.float32)
        print("backtrack_align: ", search_range, similarity.shape)
        plot_path = []
        for each_ref in range(search_range[0],search_range[1]): # 0 ~ len(ref) cost 10-100
            # i, j = each_ref, self.cost_matrix.shape[1]-1 # 0 ~ len(ref) 1
            i, j = now_live_len-1, each_ref # 0 ~ len(ref) 1
            row, col = 0, 0
            step = 1
            cost = 0
            temp_path = [(i,j)]
            # print("--------now--------: ", i, j)
            # while j >= start_align:
            while i >= now_live_len-self.n:
                # 30 0
                # print(i,j)
                # if i < 0:
                #     break
                # print("rpe cost:", cost, i, j, self.cost_matrix[i, j])
                cost+=self.cost_matrix[i, j]
                # print("start")
                if j == 0:
                    i, j = i-1, j
                    direction = Direction.ROW
                else:
                    c1 = self.cost_matrix[i-1, j-1]
                    c2 = self.cost_matrix[i-1, j]
                    c3 = self.cost_matrix[i, j-1]
                    min_c = min(c1,c2,c3)
                    if c1 == min_c:
                        i, j = i-1, j-1
                        direction = Direction.BOTH
                    elif c2 == min_c:
                        i, j = i-1, j
                        direction = Direction.ROW
                    else:
                        i, j = i, j-1
                        direction = Direction.COL
                temp_path.append((i, j))
                step+=1
            # print(each_ref, start_ref_point, each_ref-start_ref_point)
            plot_path.append(temp_path)
            # similarity[each_ref-start_ref_point] = cost/step
            similarity[each_ref-search_range[0]] = cost/step
        similarity = 1-(similarity-np.min(similarity))/(np.max(similarity)-np.min(similarity))
        possible_index = np.atleast_1d(np.squeeze(np.where(similarity>0.95)))
        test_sim = []
        for i in range(len(similarity)):
            # test_sim.append((i+start_ref_point, similarity[i]))
            test_sim.append((i+search_range[0], similarity[i]))
        print("similarity: ", test_sim)
            
        # print("possible_index: ", possible_index, possible_index+start_ref_point, type(possible_index))
        ret = []
        if len(possible_index) > 0:
            # similarity_list = list(zip(possible_index+start_ref_point, similarity[possible_index]))
            similarity_list = list(zip(possible_index+search_range[0], similarity[possible_index]))
            similarity_list = sorted(similarity_list, reverse=True, key=lambda a: a[1])
            print("similarity_list: ",similarity_list) # (frame, similarity)
            # sorter = np.argsort(similarity)
            # ret = sorter[np.searchsorted(similarity, ret, sorter=sorter)]
            ret = list(list(zip(*similarity_list))[0]) # 拿出 frame
        # print(ret)
        return np.array(ret, dtype=np.float32)
        
    def run(self, ref_point, live_frame=None):
        # live_frame = self.getLowResulotionFeature(live_frame) 
        high, low = extractHighAndLowFeature(live_frame)
        low = high
        # start = time.time()
        # print(self.cost_matrix.shape)
        for t in range(low.shape[0]):
            new_cost = self.dist(low[t])
            # print(new_cost)
            self.cost_matrix = np.append(self.cost_matrix, new_cost[np.newaxis,:], axis=0)
        
        print("cost matrix: ", self.cost_matrix.shape)
        if self.cost_matrix.shape[0] < self.n:
            return None
        possible_frame = self.backtrack_align(ref_point)
        # print("end: ", time.time()-start)
        
        # possible_frame = (possible_frame*0.3*cfg.SAMPLE_RATE)/cfg.HOP_SIZE # high feature frame
        # print(possible_frame)
        
        return list(possible_frame)

class RPELOW:
    def __init__(self, ref):
        sec = 3
        self.n = int((sec*cfg.SAMPLE_RATE)/(0.3*cfg.SAMPLE_RATE)) # backward alignments last n sec
        # self.n = 30
        self.threshold = 0.95 # similarity
        self.max_dir_count = 3 # max count on search algorithm
        
        _, self.ref_low = extractHighAndLowFeature(ref) # t-f
        
        self.live = np.zeros(0, dtype=np.float32)
        self.cost_matrix = np.zeros((self.live.shape[0], self.ref_low.shape[0]))
        self.possible_ref = set()
        
    # def getLowResulotionFeature(self, data, frame_len=0.6, hop_len=0.3, sr=44100):
    #     return librosa.util.frame(data, frame_length=int(frame_len*sr), hop_length=int(hop_len*sr), axis=0) # (time, amplitude)
    
    def dist(self, f1):
        return np.sum(np.abs(self.ref_low-f1)**2, axis=1)**0.5

    def backtrack_align(self, start_ref_point): # n = 9*44100/0.3*44100
        # i: live_idx j: ref_idx
        start_ref_point = int((start_ref_point*cfg.HOP_SIZE)/(0.3*cfg.SAMPLE_RATE))
        direction = Direction.NONE
        prev_direction = Direction.NONE
        # similarity = np.zeros(min(start_ref_point+self.n, self.cost_matrix.shape[1])-start_ref_point, dtype=np.float32)
        now_live_len = self.cost_matrix.shape[0]
        # start_align = self.cost_matrix.shape[1]-n if self.cost_matrix.shape[1]>=n else 0 # live time
        search_range = (max(0, start_ref_point-self.n), min(start_ref_point+self.n-1, self.cost_matrix.shape[1]-1))
        similarity = np.zeros(search_range[1]-search_range[0], dtype=np.float32)
        print("backtrack_align: ", search_range, similarity.shape)
        plot_path = []
        for each_ref in range(search_range[0],search_range[1]): # 0 ~ len(ref) cost 10-100
            # i, j = each_ref, self.cost_matrix.shape[1]-1 # 0 ~ len(ref) 1
            i, j = now_live_len-1, each_ref # 0 ~ len(ref) 1
            row, col = 0, 0
            step = 1
            cost = 0
            temp_path = [(i,j)]
            # print("--------now--------: ", i, j)
            # while j >= start_align:
            while i >= now_live_len-self.n:
                # 30 0
                # print(i,j)
                # if i < 0:
                #     break
                # print("rpe cost:", cost, i, j, self.cost_matrix[i, j])
                cost+=self.cost_matrix[i, j]
                # print("start")
                if j == 0:
                    i, j = i-1, j
                    direction = Direction.ROW
                else:
                    c1 = self.cost_matrix[i-1, j-1]
                    c2 = self.cost_matrix[i-1, j]
                    c3 = self.cost_matrix[i, j-1]
                    min_c = min(c1,c2,c3)
                    if c1 == min_c:
                        i, j = i-1, j-1
                        direction = Direction.BOTH
                    elif c2 == min_c:
                        i, j = i-1, j
                        direction = Direction.ROW
                    else:
                        i, j = i, j-1
                        direction = Direction.COL
                temp_path.append((i, j))
                step+=1
            # print(each_ref, start_ref_point, each_ref-start_ref_point)
            plot_path.append(temp_path)
            # similarity[each_ref-start_ref_point] = cost/step
            similarity[each_ref-search_range[0]] = cost/step
        similarity = 1-(similarity-np.min(similarity))/(np.max(similarity)-np.min(similarity))
        possible_index = np.atleast_1d(np.squeeze(np.where(similarity>self.threshold)))
        test_sim = []
        for i in range(len(similarity)):
            # test_sim.append((i+start_ref_point, similarity[i]))
            test_sim.append((i+search_range[0], similarity[i]))
        print("similarity: ", test_sim)
            
        # print("possible_index: ", possible_index, possible_index+start_ref_point, type(possible_index))
        ret = []
        if len(possible_index) > 0:
            # similarity_list = list(zip(possible_index+start_ref_point, similarity[possible_index]))
            similarity_list = list(zip(possible_index+search_range[0], similarity[possible_index]))
            similarity_list = sorted(similarity_list, reverse=True, key=lambda a: a[1])
            print("similarity_list: ",similarity_list) # (frame, similarity)
            # sorter = np.argsort(similarity)
            # ret = sorter[np.searchsorted(similarity, ret, sorter=sorter)]
            ret = list(list(zip(*similarity_list))[0]) # 拿出 frame
        # print(ret)
        return np.array(ret, dtype=np.float32)
        
    def run(self, ref_point, live_frame=None):
        # live_frame = self.getLowResulotionFeature(live_frame) 
        _, low = extractHighAndLowFeature(live_frame)
        # low = high
        # start = time.time()
        # print(self.cost_matrix.shape)
        for t in range(low.shape[0]):
            new_cost = self.dist(low[t])
            # print(new_cost)
            self.cost_matrix = np.append(self.cost_matrix, new_cost[np.newaxis,:], axis=0)
        
        print("cost matrix: ", self.cost_matrix.shape)
        if self.cost_matrix.shape[0] < self.n:
            return None
        possible_frame = self.backtrack_align(ref_point)
        # print("end: ", time.time()-start)
        
        possible_frame = (possible_frame*0.3*cfg.SAMPLE_RATE)/cfg.HOP_SIZE # high feature frame
        # print(possible_frame)
        
        return list(possible_frame)
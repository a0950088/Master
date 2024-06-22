import numpy as np
from enum import Enum
from collections import defaultdict
from librosa.sequence import dtw
from librosa import power_to_db
import matplotlib.pyplot as plt

import config as cfg
from logger import getmylogger
log = getmylogger(__name__)

class DTW:
    def __init__(self, ref_feature, silence_feature) -> None:
        # self.silence_feature = np.abs(silence_feature)**2
        # self.ref_feature = np.abs(ref_feature)**2
        self.silence_feature = silence_feature
        self.ref_feature = np.concatenate((self.silence_feature, ref_feature), axis=1) # silence + ref
        # self.mute_cost = self.getMuteCost()
        # self.cost_threshold = self.mute_cost-self.mute_cost*0.55
        # self.cost_threshold = self.mute_cost*0.5
        # self.cost_threshold = 1350
        # self.db_threshold = 6
        # self.counter = 3
        # self.min_cost = np.inf
        log.info(f"DTW Ref data shape: {self.ref_feature.shape}")
        
    def run(self, live_feature):
        combined_high = np.concatenate((live_feature, self.ref_feature), axis=1) # live + silence + ref
        combined_high_db = power_to_db(np.abs(combined_high)**2, ref=np.max)
        l_db = combined_high_db[:, :self.silence_feature.shape[1]]
        r_db = combined_high_db[:, -self.silence_feature.shape[1]:]
        # r_db_avg = np.max(np.sum(r_db, axis=1)/r_db.shape[1])
        # l_db_avg = np.max(np.sum(l_db, axis=1)/l_db.shape[1])
        
        cost, _ = dtw(X=l_db, Y=r_db, metric='euclidean')
        # self.min_cost = min(self.min_cost, cost[-1, -1])
        # log.info(f"cost: {cost[-1, -1], self.min_cost}")
        log.info(f"cost: {cost[-1, -1]}")
        
        if cost[-1, -1] < cfg.DTW_COST_THRESHOLD:
            return True
        else:
            return False
        # return False

class RPE:
    def __init__(self, ref_low_feature):
        self.ref_low_feature = ref_low_feature
        self.cost_matrix = np.zeros((0, self.ref_low_feature.shape[0]))
        self.rpe_point = []
        
    def __dist(self, live_low_feature):
        """_summary_

        Args:
            live_low_feature (nparray): shape = (time_frame, freq_band)

        Returns:
            _type_: _description_
        """
        return np.sum(np.abs(self.ref_low_feature-live_low_feature)**2, axis=1)**0.5

    def run(self, live_low_feature, now_acc_point):
        for t in range(live_low_feature.shape[0]):
            new_cost = self.__dist(live_low_feature[t])
            self.cost_matrix = np.append(self.cost_matrix, new_cost[np.newaxis,:], axis=0)
        log.info(f"RPE cost parameters: {self.cost_matrix.shape}")
            
        if self.cost_matrix.shape[0] >= cfg.RPE_SEARCH_N:
            if self.cost_matrix.shape[0] % 5 == 0:
                possible_frame = self.__backtracking(now_acc_point)
                possible_frame = possible_frame*cfg.LOW_HOP_SIZE # high feature frame
                return list(possible_frame)
        return None
            
    def __backtracking(self, now_acc_point):
        now_live_len = self.cost_matrix.shape[0]-1
        backtracking_range = now_live_len - cfg.RPE_SEARCH_N
        search_range = (0, self.cost_matrix.shape[1]) # all ref
        similarity = np.zeros(search_range[1]-search_range[0], dtype=np.float32)
        max_sim = -np.inf # record max similarity cost
        min_sim = np.inf # record min similarity cost
        log.info(f"RPE backtracking parameters: {search_range} | {similarity.shape}")
        
        for ref_idx in range(search_range[0], search_range[1]): # each ref position in search range
            i, j = now_live_len, ref_idx
            cost = 0 # accumulated cost
            step = 0 # normalized step
            row_count = 0 # compute row max count
            col_count = 0 # compute col max count
            
            while i >= backtracking_range:
                if j-1 < 0:
                    row_count += 1
                    col_count = 0
                    if row_count == cfg.RPE_MAX_RUN:
                        row_count = 0
                        cost += (self.cost_matrix[i, j])*5 # penalty?
                    else:
                        cost += self.cost_matrix[i, j]
                    i, j = i-1, j
                    step += 1
                else:
                    cost += self.cost_matrix[i, j]
                    if row_count == cfg.RPE_MAX_RUN:
                        # c1 = self.cost_matrix[i-1, j-1]
                        # c3 = self.cost_matrix[i, j-1]
                        # if c1 <= c3:
                        #     i, j = i-1, j-1
                        # else:
                        #     i, j = i, j-1
                        i, j = i, j-1
                        row_count = 0
                        step += 1
                    elif col_count == cfg.RPE_MAX_RUN:
                        # c1 = self.cost_matrix[i-1, j-1]
                        # c2 = self.cost_matrix[i-1, j]
                        # if c1 <= c2:
                        #     i, j = i-1, j-1
                        # else:
                        #     i, j = i-1, j
                        i, j = i-1, j
                        col_count = 0
                        step += 1
                    else:
                        c1 = self.cost_matrix[i-1, j-1]
                        c2 = self.cost_matrix[i-1, j]
                        c3 = self.cost_matrix[i, j-1]
                        if c1 <= c2 and c1 <= c3:
                            i, j = i-1, j-1
                            step += 2
                        elif c2 < c1 and c2 <= c3:
                            i, j = i-1, j
                            row_count += 1
                            col_count = 0
                            step += 1
                        elif c3 < c1 and c3 < c2:
                            i, j = i, j-1
                            row_count = 0
                            col_count += 1
                            step += 1
                        else:
                            log.error(f"RPE cost compute failed !")
                # step += 1
            res = cost/step
            # print(res, max_sim, min_sim)
            max_sim = res if res > max_sim else max_sim
            min_sim = res if res < min_sim else min_sim
            similarity[ref_idx-search_range[0]] = res
        
        # TODO:目前測試使用所有的possible frame，若要規定取多少frame需要排序cost值
        
        # similarity = 1-(similarity-np.min(similarity))/(np.max(similarity)-np.min(similarity))
        similarity = 1-(similarity-min_sim)/(max_sim-min_sim)
        possible_index = np.atleast_1d(np.squeeze(np.where(similarity > cfg.RPE_SIMILARITY_THRESHOLD))) # np.where return index
        # similarity_list = possible_index+search_range[0]
        similarity_list = list(zip(possible_index+search_range[0], similarity[possible_index]))
        similarity_list = sorted(similarity_list, reverse=True, key=lambda a: a[1])
        ret = list(list(zip(*similarity_list))[0])
        similarity_list = np.array(ret)
        log.info(f"RPE backtracking res: {similarity_list}")
        
        return similarity_list

    def draw(self):
        fig = plt.figure(figsize=(10, 8))
        # for i,j in self.rpe_point:
        #     plt.scatter(i, j, c='white', s=1)
        
        plt.title("RPE cost matrix")
        plt.imshow(self.cost_matrix.T, cmap="inferno")
        plt.colorbar()
        plt.xlabel('live')
        plt.ylabel('ref')
        plt.gca().invert_yaxis()
        plt.savefig(f"{cfg.FOLDER}/RPE.png")
        # plt.show()

class Direction(Enum):
    I = 1
    J = 2
    
DIR_I = set([Direction.I])
DIR_J = set([Direction.J])
DIR_IJ = set([Direction.I, Direction.J])

def dict_inf():
    return np.inf

class ODTW:
    def __init__(self, ref_feature, live_queue, acc_queue, res_path_queue):
        self.ref_len = ref_feature.shape[0]
        self.ref_feature = ref_feature
        self.live_queue = live_queue
        self.acc_queue = acc_queue # deque
        self.res_path_queue = res_path_queue
        
        self.live_feature_record = np.zeros((0, cfg.FRAME_SIZE), dtype=np.float32)
        self.totalCostMatrix = defaultdict(dict_inf)
        self.cellCostMatrix = defaultdict(dict_inf)
        # self.c = cfg.ODTW_SEARCH_C
        self.run_count = 0
        self.prev_j_prime = [0,0]
        
        self.min_rpe_ret = (np.inf, None)
        self.rpe_reset = False
        
        self.output_points = []
        # use to draw
        self.acc_path = []
        self.offline_path = []
        self.others_rpe_points = []
        self.others_back_rpe_points = []
        self.rpe_points = []
    
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
    
    def calculateCost(self, f1, f2):
        return np.sum(np.abs(f1-f2)**2)**0.5
    
    def get_next_direction(self, i, j, i_prime, j_prime, prev):
        if i < cfg.ODTW_SEARCH_C:
            return DIR_IJ # both
        if self.run_count > cfg.MAX_RUN:
            if prev == DIR_I:
                log.warning(f"return DIR_J!")
                return DIR_J
            else:
                log.warning(f"return DIR_I!")
                return DIR_I
            
        # if i_prime < i:
        #     return DIR_I
        # elif j_prime < j:
        #     return DIR_J
        # else:
        #     return DIR_IJ
        return DIR_IJ
    
    def run(self, music_tracker_event):
        # i: live pos j: ref pos
        i = 0
        j = 0
        # output position
        i_prime = 0
        j_prime = 0
        # direction
        # curr = set()
        # prev = set()
        self.acc_queue.append((i_prime, j_prime))
        self.acc_path.append((i_prime, j_prime))
        self.output_points.append(j_prime)
        self.res_path_queue.put((i_prime, j_prime))
        log.info(f"initial output acc ({i_prime}, {j_prime})")
        
        data = self.live_queue.get()

        live_seg = data
        self.live_feature_record = np.concatenate((self.live_feature_record, live_seg), dtype=np.float32)
        l_i = self.live_feature_record[i]
        r_j = self.ref_feature[j]
        d = self.calculateCost(l_i, r_j)
        self.setCostMatrix(i, j, d)
        
        while True:
            # inner_time = time.time()
            # add live
            # if not self.live_queue.empty() or i>=self.live_feature_record.shape[0]-1:
            #     data = self.live_queue.get()
            #     # if data is None:
            #     #     break
            #     live_seg = data
            #     self.live_feature_record = np.concatenate((self.live_feature_record, live_seg), dtype=np.float32)
            #     logging.info(f"get new live {self.live_feature_record.shape}")
            # if j >= self.ref_len-1 and j_prime >= self.ref_len-1:
            #     # end tracking
            #     log.info(f"Ref end !")
            #     break
            try:
                data = self.live_queue.get(timeout=1)
            except:
                log.info(f"ODTW live queue time out !")
                break
            live_seg = data
            self.live_feature_record = np.concatenate((self.live_feature_record, live_seg), dtype=np.float32)
            log.info(f"get new live: {live_seg.shape} {self.live_feature_record.shape}")
            
            # curr = self.get_next_direction(i, j, i_prime, j_prime, prev)
            i_prime+=1
            i+=1
            l_i = self.live_feature_record[i]
            # 計算+c/-c範圍的ref cost
            for J in range(max(0, j+1-cfg.ODTW_SEARCH_C), min(j+1+cfg.ODTW_SEARCH_C, self.ref_len)):
                r_J = self.ref_feature[J]
                d = self.calculateCost(l_i, r_J)
                self.setCostMatrix(i, J, d)

            # add ref  
            # if Direction.J in curr:
            #     j+=1
            #     r_j = self.ref_feature[min(j, self.ref_len-1)]
            #     for I in range(max(0, i-cfg.ODTW_SEARCH_C+1), i+1):
            #         l_I = self.live_feature_record[I]
            #         d = self.calculateCost(l_I, r_j)
            #         self.setCostMatrix(I, j, d)
            # if Direction.J in curr:
            j+=1
            r_j = self.ref_feature[min(j, self.ref_len-1)]
            for I in range(max(0, i-cfg.ODTW_SEARCH_C+1), i+1):
                l_I = self.live_feature_record[I]
                d = self.calculateCost(l_I, r_j)
                self.setCostMatrix(I, j, d)
            
            # if curr == prev and prev != DIR_IJ:
            #     self.run_count += 1
            # else:
            #     self.run_count = 1
            # prev = curr
            
            if self.rpe_reset:
                j_prime = self.min_rpe_ret[1] if self.min_rpe_ret[1] is not None else j_prime
                log.info(f"use rpe: {i_prime, j_prime}")
                # _, j_prime = self.get_ij_prime(i_prime, j_prime)
                j = j_prime
                
                # curr = set()
                # prev = set()
                self.run_count = 1
                self.rpe_reset = False
                self.min_rpe_ret = (np.inf, None)
            else:
                # _, j_prime = self.get_ij_prime(i_prime, j_prime) # TODO: 確認合理性
                if i_prime < cfg.ODTW_SEARCH_C:
                    _, j_prime = self.get_ij_prime(i_prime, j)
                else:
                    _, j_prime = self.get_ij_prime(i_prime, j_prime)
                    j = j_prime
            # if self.use_rpe == True:
            #     _, j_prime = self.get_ij_prime(i_prime, j_prime)
            # else:
            #     # search c
            #     _, j_prime = self.get_ij_prime(i_prime, j)
            log.info(f"odtw res: {i, j, i_prime, j_prime, self.totalCostMatrix[i_prime, j_prime]}")
            # self.acc_queue.append((i_prime*cfg.HOP_SIZE, j_prime*cfg.HOP_SIZE))
            j_prime = self.ref_len-1 if j_prime >= self.ref_len else j_prime
            self.acc_queue.append((i_prime, j_prime))
            self.acc_path.append((i_prime, j_prime))
            self.output_points.append(j_prime)
            self.res_path_queue.put((i_prime, j_prime))
        music_tracker_event.clear()
        log.info(f"odtw end")
        
    def get_ij_prime(self, i, j):
        i_prime, j_prime = (i, j)
        min_cost = np.inf
        
        upper_limit = min(j+(cfg.MAX_RUN*10)-1, self.ref_len-1)
        lower_limit = j-1

        curr_j = upper_limit
        
        while curr_j >= 0 and curr_j > lower_limit:
            if self.totalCostMatrix[i, curr_j]+self.cellCostMatrix[i,curr_j] < min_cost:
                i_prime, j_prime = (i, curr_j)
                min_cost = self.totalCostMatrix[i_prime, j_prime]+self.cellCostMatrix[i,curr_j]
            curr_j -= 1

        jump_threshold = 1
        if self.prev_j_prime[0] == j_prime:
            if self.prev_j_prime[1] > cfg.MAX_RUN*3:
                j_prime+=jump_threshold
                self.prev_j_prime = [j_prime, 0]
            else:
                self.prev_j_prime[1] += 1 # max run
        else:
            self.prev_j_prime = [j_prime, 0]
                   
        return i_prime, j_prime
    
    def deals_thread(self, q, live_pos):
        # deals rpe position
        while not q.empty():
            ref_pos = q.get()
            cost = self.__backtracking(ref_pos, live_pos)
            self.min_rpe_ret = min(self.min_rpe_ret, (cost, ref_pos), key=lambda a: a[0])
            q.task_done()
    
    def __backtracking(self, j, i):
        bt_i = i-cfg.ODTW_BT_RANGE if i-cfg.ODTW_BT_RANGE > 0 else 0
        bt_j = self.output_points[bt_i]
        live_pos = i
        ref_pos = j
        row_count = 0
        col_count = 0
        step = 0
        accumulate_cost = 0
        # log.info(f"bt_i bt_j and i j: {bt_i, bt_j}, {i, j}")
        while i-1 > bt_i or j-1 > bt_j:
            if i-1 == bt_i:
                col_count += 1
                row_count = 0
                if col_count == cfg.MAX_RUN:
                    col_count = 0
                    accumulate_cost += (self.totalCostMatrix[i, j])*5 # penalty?
                else:
                    accumulate_cost += self.totalCostMatrix[i, j]
                i, j = i, j-1
                step += 1
            elif j-1 == bt_j:
                row_count += 1
                col_count = 0
                if row_count == cfg.MAX_RUN:
                    row_count = 0
                    accumulate_cost += (self.totalCostMatrix[i, j])*5 # penalty?
                else:
                    accumulate_cost += self.totalCostMatrix[i, j]
                i, j = i-1, j
                step += 1
            else:
                accumulate_cost += self.totalCostMatrix[i,j]
                if row_count == cfg.MAX_RUN:
                    i, j = i, j-1
                    row_count = 0
                    step += 1
                elif col_count == cfg.MAX_RUN:
                    i, j = i-1, j
                    col_count = 0
                    step += 1
                else:
                    c1 = self.totalCostMatrix[i-1, j-1]
                    c2 = self.totalCostMatrix[i-1, j]
                    c3 = self.totalCostMatrix[i, j-1]
                    if c1 <= c2 and c1 <= c3:
                        i, j = i-1, j-1
                        row_count = 0
                        col_count = 0
                        step += 2
                    elif c2 < c1 and c2 <= c3:
                        i, j = i-1, j
                        row_count += 1
                        col_count = 0
                        step += 1
                    elif c3 < c1 and c3 < c2:
                        i, j = i, j-1
                        row_count = 0
                        col_count += 1
                        step += 1
                    else:
                        log.error(f"ODTW cost compute failed !")
            if accumulate_cost == np.inf:
                # log.info(f"accumulate_cost np.inf")
                break
            # step += 1
        # log.info(f"accumulate_cost {accumulate_cost}")
        if step:
            cost = accumulate_cost/step
            # cost = (accumulate_cost/step)*self.cellCostMatrix[live_pos, ref_pos]
        else:
            cost = np.inf
        log.info(f"backtrack res: ({ref_pos}, {cost}, {self.cellCostMatrix[live_pos, ref_pos]})")
        return cost
        
    # def __backtracking(self, ref_pos, live_pos):
    #     row_maxcount = 0
    #     col_maxcount = 0
    #     row_flag = True
    #     col_flag = True
    #     step = 0
    #     count = 0
    #     i = live_pos
    #     j = int(ref_pos)
    #     accumulate_cost = self.totalCostMatrix[i,j]
    #     while (i>0 or j>0) and step < cfg.ODTW_SEARCH_C:
    #         if i == 0:
    #             j-=1
    #         elif j == 0:
    #             i-=1
    #         else:
    #             if row_flag == False:
    #                 # minc = min(self.totalCostMatrix[i, j-1], self.totalCostMatrix[i-1, j-1])
    #                 minc = self.totalCostMatrix[i, j-1]
    #             elif col_flag == False:
    #                 # minc = min(self.totalCostMatrix[i-1, j], self.totalCostMatrix[i-1, j-1])
    #                 minc = self.totalCostMatrix[i-1, j]
    #             else:
    #                 minc = min(self.totalCostMatrix[i-1, j], self.totalCostMatrix[i, j-1], self.totalCostMatrix[i-1, j-1])
                
    #             if minc == np.inf:
    #                 count = 0
    #                 break
    #             if minc == self.totalCostMatrix[i-1, j-1]:
    #                 i-=1
    #                 j-=1
    #                 count+=2
    #                 row_maxcount, col_maxcount = 0, 0
    #                 row_flag, col_flag = True, True
    #             elif minc == self.totalCostMatrix[i, j-1]:
    #                 j-=1
    #                 count+=1
    #                 row_maxcount = 0
    #                 row_flag = True
    #                 if col_maxcount+1 == cfg.MAX_RUN:
    #                     col_flag = False
    #                 col_maxcount+=1
    #             else:
    #                 i-=1
    #                 count+=1
    #                 col_maxcount = 0
    #                 col_flag = True
    #                 if row_maxcount+1 == cfg.MAX_RUN:
    #                     row_flag = False
    #                 row_maxcount+=1
    #         step+=1
    #         accumulate_cost+=self.totalCostMatrix[i,j]
    #     if count:
    #         cost = accumulate_cost/count
    #     else:
    #         cost = np.inf
    #     # log.info(f"backtrack res: ({ref_pos}, {cost})")
    #     return cost
    
    def optimalWarpingPath(self):
        # path.append((i*cfg.STFT['hop'],j*cfg.STFT['hop']))
        i,j = self.acc_path[-1][0], self.ref_len-1
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
        cost_matrix = []
        for j in range(self.ref_feature.shape[0]):
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
        # for i,j in self.offline_path:
        #     # if pre_j != j:
        #     #     print("i:", i, j)
        #     #     pre_j = j
        #     plt.scatter(i, j, c='b', s=1.5) # x: live y: ref
        fig = plt.figure(figsize=(10, 8))
        for i,j in self.acc_path:
            # if pre_j != j:
            #     print("i:", i, j)
            #     pre_j = j
            plt.scatter(i, j, c='g', s=1) # x: live y: ref
        for i,j in self.others_rpe_points:
            plt.scatter(i, j, c='yellow', s=1) # x: live y: ref
        for i,j in self.others_back_rpe_points:
            plt.scatter(i, j, c='fuchsia', s=1) # x: live y: ref
            
        for i,j in self.rpe_points:
            plt.scatter(i, j, c='cyan', s=1) # x: live y: ref
        
        plt.title("DMA online cost matrix")
        plt.imshow(cost_matrix, cmap="inferno")
        plt.colorbar()
        plt.xlabel('live')
        plt.ylabel('ref')
        plt.gca().invert_yaxis()
        plt.savefig(f"{cfg.FOLDER}/DMA.png")
        # plt.show()
        

import numpy as np

#use class to avoid global stuff
class OTW:
    """
    score: 1D numpy array input reference score
    c: search width parameter (from paper)
    max_run_count: paper
    """
    def __init__(self, score, max_run_count=3, c=100, verbose=False):
        self.verbose = verbose
        self.score = np.asarray(score)
        self.lenR = len(self.score)
        self.c = int(c)
        self.max_run_ct = int(max_run_count)

        #current time index
        self.t = 0
        #current reference score index
        self.j = 0

        #initial t and j for current window
        self.t0 = 0
        self.j0 = 0

        #Keep only D cols in current window
        self.D = np.full((self.c, self.c), np.inf, dtype=float)
        #add path length for normalization
        self.L = np.zeros((self.c, self.c), dtype = float)

        self.initialized = False
        self.run_ct = 0
        self.prev_inc = None
        self.step = 0
        self.path = []
        self.live = []
        self.pending_live = []
        self.used_live = []

    def cost(self, u, v):
        eps = 1e-8
        if(not np.isfinite(u) or not np.isfinite(v)):
            return 0.2
        if(u<=eps and v<=eps):
            return 0.0
        if (u<=eps or v<=eps):
            # penalty when one seq silence
            return 0.2
        return abs(np.log2(u+1e-6) - np.log2(v+1e-6))
        # return abs(u - v)


    #shift window + D and L to local indexes
    def shift_window(self, new_t0, new_j0):
        if(new_t0<self.t0 or new_j0<self.j0):
            raise ValueError("window cant move backwards")
        if(new_t0==self.t0 and new_j0==self.j0):
            return

        dt = new_t0-self.t0
        dj = new_j0-self.j0
        old_D = self.D
        old_L = self.L

        self.D = np.full((self.c, self.c), np.inf, dtype=float)
        self.L = np.zeros((self.c, self.c), dtype=float)

        #copy overlapping data
        if(dt<self.c and dj<self.c):
            self.D[:self.c-dt, :self.c-dj] = old_D[dt:, dj:]
            self.L[:self.c-dt, :self.c-dj] = old_L[dt:, dj:]
        self.t0 = new_t0
        self.j0 = new_j0

    #check if reached window edge + shift
    def move_window(self):
        new_t0 = max(self.t0, self.t-self.c+1)
        new_j0 = max(self.j0, self.j-self.c+1)
        self.shift_window(new_t0, new_j0)


    #accum cost of path ending at t,j, normalized
    def path_cost(self, t, j):
        ti = t - self.t0
        ji = j - self.j0
        if not(0<=ti<self.c and 0<=ji<self.c):
            return np.inf
        D = self.D[ti, ji]
        L = self.L[ti, ji]

        if((not np.isfinite(D)) or L<=0):
            return np.inf
        return D/L

    def evaluate_path_cost(self, t, j, u):
        if(j<0  or j>=self.lenR):
            return

        ti = t - self.t0
        ji = j - self.j0
        if not(0<=ti<self.c and 0<=ji<self.c):
            return
        
        C_tj = self.cost(u, self.score[j])

        if(t==0 and j==0): #first cell
            self.D[ti, ji] = C_tj
            self.L[ti, ji] = 1
            return
        
        candidates = []
        if(ji>0): #left
            D_left = self.D[ti, ji-1]
            L_left = self.L[ti, ji-1]
            if(np.isfinite(D_left) and L_left>0):
                candidates.append((D_left, L_left))
        if(ti>0): #up
            D_up = self.D[ti-1, ji]
            L_up = self.L[ti-1, ji]
            if(np.isfinite(D_up) and L_up>0):
                candidates.append((D_up, L_up)) 
        if(ti>0 and ji>0): #diagonal
            D_diag = self.D[ti-1, ji-1]
            L_diag = self.L[ti-1, ji-1]
            if(np.isfinite(D_diag) and L_diag>0):
                candidates.append((D_diag, L_diag)) 

        if(not candidates):
            self.D[ti, ji] = np.inf
            self.L[ti, ji] = 0
            return
        
        best_D, best_L = min(candidates, key=lambda x:x[0]/x[1]) #look 1st elem

        self.D[ti, ji] = C_tj + best_D
        self.L[ti, ji] = best_L + 1
        return


    #path movement decision (not backpass)
    def get_inc(self):
        if(self.j >= self.lenR-1):
            return "row"
        if(self.t==0): #startup
            return "both"
        
        if(self.run_ct>self.max_run_ct and self.prev_inc is not None): #enforce max run ct
            if self.prev_inc == "row":
                return "column"
            if self.prev_inc =="column":
                return "row"

        row_start = max(self.j0, self.j-self.c+1)
        row_min = np.inf
        row_min_j = self.j
        for k in range(row_start, self.j+1):
            val = self.path_cost(self.t, k) # getaccum cost across t, and all j in window
            if(val < row_min):
                row_min = val
                row_min_j = k

        col_start = max(self.t0, self.t-self.c+1)
        col_min = np.inf
        col_min_t = self.t
        for k in range(col_start, self.t+1):
            val = self.path_cost(k, self.j) # getaccum cost across all t in wind, and j
            if(val < col_min):                    
                col_min = val
                col_min_t = k
        # print(
        #     f"current: ({self.t}, {self.j}), "
        #     f"row_min_j={row_min_j}, "
        #     f"col_min_t={col_min_t}"
        # )
        # print(
        #     f"t={self.t}, j={self.j}, "
        #     f"row_min={row_min:.4f} at j={row_min_j}, "
        #     f"col_min={col_min:.4f} at t={col_min_t}, "
        #     f"run={self.run_ct}, prev={self.prev_inc}"
        # )
        if(row_min<=col_min):
            if(row_min_j<self.j):
                return "row"
            return "both"
        else:
            if(col_min_t<self.t):
                return "column"
            return "both"

    def update(self, u):
        self.live.append(u)
        self.pending_live.append(u)
        if(not self.initialized):
            self.t = 0
            self.j = 0
            self.t0 = 0
            self.j0 = 0
            self.D.fill(np.inf)
            self.L.fill(0)
            first_u = self.pending_live.pop(0)
            self.used_live.append(first_u)
            self.evaluate_path_cost(0, 0, first_u)
            self.path.append((0, 0))
            self.initialized = True
            self.step=0
            self.prev_inc = None
            self.run_ct = 0
            return self.j

        if(self.j >= self.lenR-1):
            self.pending_live.clear()
            self.path.append((self.t, self.j))
            return self.j

        inc = self.get_inc()
        advanced_t = False
        advanced_j = False
        if(inc in ("row", "both") and self.pending_live):
            self.t += 1
            u_t = self.pending_live.pop(0) #if advance t then consume pending frame
            self.used_live.append(u_t)
            self.move_window()
            start_j = max(self.j0, self.j-self.c+1)
            for k in range(start_j, self.j+1):
                self.evaluate_path_cost(self.t, k, u_t)
            advanced_t = True
        if(inc in ("column", "both") and self.j<self.lenR-1):
            self.j += 1
            self.move_window()
            start_t = max(self.t0, self.t-self.c+1)
            for k in range(start_t, self.t+1):
                self.evaluate_path_cost(k, self.j, self.used_live[k])
            advanced_j = True

        if(advanced_t and advanced_j):
            inc = "both"
        elif(advanced_t):
            inc = "row"
        elif(advanced_j):
            inc = "column"
        else:
            inc = None

        self.path.append((self.t, self.j))

        #run ct logic:
        if(inc=="both"):
            self.run_ct = 0
            self.prev_inc = None
        elif(inc is not None):
            if(inc == self.prev_inc):
                self.run_ct += 1
            else:
                self.run_ct = 1
            self.prev_inc = inc

        self.step += 1

        # print(
        #     f"incoming={len(self.live)-1}, "
        #     f"t={self.t}, j={self.j}, "
        #     f"inc={inc},"
        #     f"pending={len(self.pending_live)}, "
        #     f"prev={self.prev_inc}, run={self.run_ct}"
        # )
    
        return self.j

    def get_path(self):
        return(np.asarray(self.path, dtype=int))
    def reset_path(self):
        self.path = []

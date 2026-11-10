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
        # u, v: scalars or arrays (broadcast); returns same shape
        eps = 1e-8
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u_on = np.isfinite(u) & (u>eps)
        v_on = np.isfinite(v) & (v>eps)
        # semitone distance saturating at 6, scaled to [0,1] (score-follow dixon_pitch)
        semis = 12*np.abs(np.log2(np.where(u_on, u, 1.0)) - np.log2(np.where(v_on, v, 1.0)))
        pitched = np.minimum(semis, 6.0)/6.0
        return np.where(u_on & v_on, pitched, np.where(u_on != v_on, 1.0, 0.0))


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

    #vectorized path_cost over slices of D and L
    def norm_cost(self, D, L):
        ok = np.isfinite(D) & (L>0)
        return np.where(ok, D/np.where(ok, L, 1.0), np.inf)

    def evaluate_path_cost(self, t, j, C_tj):
        # C_tj: precomputed local cost of cell (t, j)
        if(j<0  or j>=self.lenR):
            return

        ti = t - self.t0
        ji = j - self.j0
        if not(0<=ti<self.c and 0<=ji<self.c):
            return

        if(t==0 and j==0): #first cell
            self.D[ti, ji] = C_tj
            self.L[ti, ji] = 1
            return
        
        # best predecessor by normalized cost D/L; ties keep first of left, up, diagonal.
        # .item() + scalar compares: this runs ~c times per frame, numpy scalars are slow here
        best_D = best_L = None
        best_r = np.inf
        D, L = self.D, self.L
        if(ji>0): #left
            d, l = D.item(ti, ji-1), L.item(ti, ji-1)
            if(l>0 and d<np.inf and d/l<best_r):
                best_D, best_L, best_r = d, l, d/l
        if(ti>0): #up
            d, l = D.item(ti-1, ji), L.item(ti-1, ji)
            if(l>0 and d<np.inf and d/l<best_r):
                best_D, best_L, best_r = d, l, d/l
        if(ti>0 and ji>0): #diagonal
            d, l = D.item(ti-1, ji-1), L.item(ti-1, ji-1)
            if(l>0 and d<np.inf and d/l<best_r):
                best_D, best_L, best_r = d, l, d/l

        if(best_L is None):
            D[ti, ji] = np.inf
            L[ti, ji] = 0
            return

        D[ti, ji] = C_tj + best_D
        L[ti, ji] = best_L + 1


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

        # accum cost across t and all j in window; first (lowest-index) minimum wins ties
        row_start = max(self.j0, self.j-self.c+1)
        row_vals = self.norm_cost(self.D[self.t-self.t0, row_start-self.j0:self.j-self.j0+1],
                                  self.L[self.t-self.t0, row_start-self.j0:self.j-self.j0+1])
        row_min_j = self.j
        row_min = np.inf
        if(np.isfinite(row_vals).any()):
            a = int(np.argmin(row_vals))
            row_min, row_min_j = row_vals[a], row_start+a

        # accum cost across all t in window and j
        col_start = max(self.t0, self.t-self.c+1)
        col_vals = self.norm_cost(self.D[col_start-self.t0:self.t-self.t0+1, self.j-self.j0],
                                  self.L[col_start-self.t0:self.t-self.t0+1, self.j-self.j0])
        col_min_t = self.t
        col_min = np.inf
        if(np.isfinite(col_vals).any()):
            a = int(np.argmin(col_vals))
            col_min, col_min_t = col_vals[a], col_start+a
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

    #advance j and fill the new column over the window
    def add_column(self):
        self.j += 1
        self.move_window()
        start_t = max(self.t0, self.t-self.c+1)
        costs = self.cost(np.asarray(self.used_live[start_t:self.t+1], dtype=float), self.score[self.j])
        for n, k in enumerate(range(start_t, self.t+1)):
            self.evaluate_path_cost(k, self.j, costs[n])

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
            self.evaluate_path_cost(0, 0, self.cost(first_u, self.score[0]))
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

        decided = self.get_inc()
        # every arriving live frame must be consumed, otherwise pending_live grows
        # and t falls further behind the live stream on each "column" decision
        inc = "both" if decided == "column" else decided
        advanced_t = False
        advanced_j = False
        if(inc in ("row", "both") and self.pending_live):
            self.t += 1
            u_t = self.pending_live.pop(0) #if advance t then consume pending frame
            self.used_live.append(u_t)
            self.move_window()
            start_j = max(self.j0, self.j-self.c+1)
            costs = self.cost(u_t, self.score[start_j:self.j+1])
            for n, k in enumerate(range(start_j, self.j+1)):
                self.evaluate_path_cost(self.t, k, costs[n])
            advanced_t = True
        if(inc in ("column", "both") and self.j<self.lenR-1):
            self.add_column()
            advanced_j = True

        if(decided == "column" and advanced_j):
            inc = "column" # count the run by what was decided, not by the forced row
        elif(advanced_t and advanced_j):
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

        # let the reference catch up without waiting for new input (column-only moves)
        while(self.j<self.lenR-1 and self.get_inc()=="column"):
            self.add_column()
            self.path.append((self.t, self.j))
            self.run_ct = self.run_ct+1 if self.prev_inc=="column" else 1
            self.prev_inc = "column"

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

import numpy as np

#use class to avoid global stuff
class OTW:
    """
    score: 1D numpy array input reference score
    c: search width parameter (from paper)
    max_run_count: paper
    """
    def __init__(self, score, max_run_count=3, c=100):
        self.score = np.asarray(score)
        self.lenR = len(self.score)
        self.c = None if c is None else int(c)
        self.max_run_ct = int(max_run_count)

        #current time index
        self.t = 0
        #current reference score index
        self.j = 0

        #Keep only current & previous D columns
        self.D_prev = np.full(self.lenR, np.inf, dtype=float)
        self.D_curr = np.full(self.lenR, np.inf, dtype=float)

        self.initialized = False
        self.run_ct = 0
        self.prev_inc = None
        self.step = 0
        self.path = []
        self.live = []

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

    #accum cost of path ending at k,l
    def path_cost(self, k, l):
        if(l<0 or l>=self.lenR):
            return np.inf
        if(k==self.t):
            return self.D_curr[l]
        if(k==self.t-1):
            return self.D_prev[l]
        return np.inf

    def evaluate_path_cost(self, t, j, u):
        if(j<0  or j>=self.lenR):
            return
        C_tj = self.cost(u, self.score[j])
        if(t==0 and j==0):
            self.D_curr[j] = C_tj
            return
        left = self.D_curr[j-1] if j>0 else np.inf #D[t, j-1]
        up = self.D_prev[j]  if t>0 else np.inf #D[t-1, j] 
        diag = self.D_prev[j-1]  if (t>0 and j>0) else np.inf #D[t-1, j-1]
        best = min(left, up, diag)
        if np.isfinite(best):
            self.D_curr[j] = C_tj + best
        else:
            self.D_curr[j] = np.inf


    #path movement decision (not backpass)
    def get_inc(self, t, j):
        if((self.c is not None) and (t<self.c)):
            return "both"
        if(self.run_ct > self.max_run_ct and self.prev_inc is not None):
            if self.prev_inc == "row":
                return "column"
            else:
                return "row"

        start = 0 if self.c is None else max(0, j - self.c + 1)

        lookback=8
        start=max(start, j-lookback)
        if(start<=j):
            row_vals = [self.path_cost(t,l) for l in range (start, j+1)]
            row_min=min(row_vals) if row_vals else np.inf
        else:
            row_min=np.inf
        col_min = self.path_cost(t-1,j) if t>0 else np.inf

        if(col_min<row_min):
            inc="column"
        elif(row_min<col_min):
            inc="row"
        else:
            inc="both"
        print(f"step={self.step} t={t} j={j} inc={inc} row_min={row_min:.6f} col_min={col_min:.6f} prev_inc={self.prev_inc} run_ct={self.run_ct}")

        return inc

    def update(self, u):
        if(not self.initialized):
            self.t = 0
            self.j = 0
            self.D_prev.fill(np.inf)
            self.D_curr.fill(np.inf)
            self.live = [u]
            self.evaluate_path_cost(self.t, self.j, u)
            self.path.append((self.t, self.j))
            self.initialized = True
            self.step=0
            self.prev_inc = None
            self.run_ct = 1
            return self.j

        self.live.append(u)
        start = 0 if self.c is None else max(0, self.j - self.c + 1)
        for k in range(start, self.j + 1):
            self.evaluate_path_cost(self.t, k, u)

        inc = self.get_inc(self.t, self.j)

        did_col = False
        if (inc=="column" or inc=="both") and (self.j <self.lenR-1):
            self.j+=1
            self.evaluate_path_cost(self.t, self.j, u) #dont recompute past times
            self.path.append((self.t, self.j))
            did_col=True

        did_row = False
        if(inc=="row" or inc=="both"):
            self.t+=1
            self.D_prev, self.D_curr = self.D_curr, self.D_prev
            self.D_curr.fill(np.inf)
            start = 0 if self.c is None else max(0, self.j - self.c + 1)
            for k in range(start, self.j + 1):
                self.evaluate_path_cost(self.t, k, u)
            self.path.append((self.t, self.j))
            did_row = True

        if(not did_row):
            self.t+=1
            self.D_prev, self.D_curr = self.D_curr, self.D_prev
            self.D_curr.fill(np.inf)
            start = 0 if self.c is None else max(0, self.j - self.c + 1)
            for k in range(start, self.j + 1):
                self.evaluate_path_cost(self.t, k, u)
            self.path.append((self.t, self.j))

        if did_col and did_row:
            final_inc="both"
        elif did_col:
            final_inc="column"
        elif did_row:
            final_inc="row"
        else:
            final_inc=inc

        if(final_inc==self.prev_inc):
            self.run_ct += 1
        else:
            self.run_ct = 1
            self.prev_inc = final_inc
        self.step +=1
        return(self.j)

    def get_path(self):
        return(np.asarray(self.path, dtype=int))
    def reset_path(self):
        self.path = []

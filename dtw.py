import numpy as np

def dtw(seqA, seqB):
    seqA = np.asarray(seqA)
    seqB = np.asarray(seqB)


    C = np.abs(seqA[:, None] - seqB[None, :])

    D = np.zeros_like(C)
    D[0,0] = C[0,0]
    for i in range (C.shape[0]):
        for j in range(C.shape[1]):
            if(i==0 and j==0):
                continue
            left = D[i, j-1] if j>0 else np.inf
            up = D[i-1, j] if i>0 else np.inf
            diag = D[i-1, j-1] if (i>0 and j>0) else np.inf
            D[i, j] = C[i, j] + min(left, up, diag)

    i = C.shape[0] - 1
    j = C.shape[1] - 1
    path = [(i, j)]

    while(i>0 or j>0):
        left = D[i, j-1] if j>0 else np.inf
        up = D[i-1, j] if i>0 else np.inf
        diag = D[i-1, j-1] if (i>0 and j>0) else np.inf

        if(diag<=left and diag<=up):
            i -= 1
            j -= 1
        elif(left<=up):
            j -= 1    
        else:
            i -= 1

        path.append((i,j))
    path.reverse()
    return C, D, path


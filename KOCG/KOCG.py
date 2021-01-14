"""
Related work: "Finding Gangs in War from Signed Networks", KDD'16
This is a Python wrapper to run the Matlab code provided by the authors (repo link: https://github.com/lingyangchu/KOCG.SIGKDD2016)

Requirement:
 * Install python package `matlab_engine` following the link: https://se.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

Run the code for signedNucleus format: python KOCG.py
Update DATASET_LIST below and in runKOCG_datasets.m with your chosen dataset
"""
import os
import glob
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import matlab.engine
import sys
import shutil
sys.path.append('../')
from utility import *

DATASET_LIST = ['congress']

def init_graph(name):
    """ process graphs to be matlab to compare with baselines """
    if not os.path.exists('datasets/{}.mat'.format(name)):
        # read graph in scipy.sparse format
        with open("../datasets/{}.txt".format(name), "r") as f:
            data = f.read()
        N = int(data.split('\n')[0].split(' ')[1])
        A = sp.lil_matrix((N,N), dtype='d')
        for line in data.split('\n')[1:]:
            es = line.split('\t')
            if len(es)!=3: continue
            A[int(es[0]),int(es[1])] = int(es[2])
            A[int(es[1]),int(es[0])] = int(es[2])
        # convert to .mat
        sio.savemat('../datasets/{}.mat'.format(name), {'A':A})

def run(subG = False):
    if not os.path.exists('K2'): os.makedirs('K2')
    # run .m script
    eng = matlab.engine.start_matlab()
    if not subG:
        eng.runKOCG_datasets(nargout=0)
    else:
        eng.runKOCG_datasets_subG(nargout=0)

def eval(name, N=None, A=None):
    if N is None:
        N, A = read_graph("../datasets/{}.txt".format(name))
    fs = glob.glob('K2/result_{}_p*.mat'.format(name))
    V = N
    maxP = -1
    bestC = None
    while V > 1:
        C = [-1 for i in range(N)]
        cnt = 0
        for i in range(len(fs)):
            if cnt >= V: break
            cX = sio.loadmat('K2/result_{}_p{}.mat'.format(name, i+1))
            X = cX['X'].tocoo()

            for r,c,v in zip(X.row,X.col,X.data):
                if C[r]==-1:
                    C[r] = c+1
                    cnt += 1
        row = []
        col = []
        data = []
        for i, c in enumerate(C):
            if c == 1:
                row.append(0)
                col.append(i)
                data.append(1)
            elif c == 2:
                row.append(0)
                col.append(i)
                data.append(-1)
        V -= 1
        if len(data) == 0:
            continue
        x = sp.csr_matrix((data, (row, col)), shape=(1, N), dtype=np.int8)
        xT = x.transpose()
        polarity = ((x * A * xT) / (x * xT))[0,0]
        if polarity > maxP:
            maxP = polarity
            bestC = C
    return N, A, bestC

# prepare input in .mat for .m script
for name in DATASET_LIST: init_graph(name)
run()
for name in DATASET_LIST:
    f = open(name + '_' + 'KOCG_subgraphs', 'w')
    N, A, bestC = eval(name)
    S_1 = []
    S_2 = []
    for i, c in enumerate(bestC):
        if c == 1:
            S_1.append(i)
        elif c == 2:
            S_2.append(i)
    
    if len(S_2) > len(S_1):
        S_1, S_2 = S_2, S_1
    if len(S_2) == 0:
        for node in S_1:
            f.write(str(node) + ' ')
        f.write('-1 -1\n')
    else:
        for node in S_1:
            f.write(str(node) + ' ')
        f.write('-1 ')
        for node in S_2:
            f.write(str(node) + ' ')
        f.write('-1\n')
    queue = [S_1 + S_2]

    while len(queue) > 0:
        subG = queue.pop()
        subGSet = set(subG)

        cx = sp.coo_matrix(A)
        A = sp.lil_matrix((N,N), dtype='d')

        for i,j,v in zip(cx.row, cx.col, cx.data):
            if i not in subGSet or j not in subGSet:
                A[i, j] = v

        # convert to .mat
        sio.savemat('../datasets/{}.mat'.format("temp"), {'A':A})

        run(True)
        N, A, bestC = eval("temp", N, A.tocsr())

        S_1 = []
        S_2 = []
        for i, c in enumerate(bestC):
            if c == 1:
                S_1.append(i)
            elif c == 2:
                S_2.append(i)
        
        S = S_1 + S_2
        if len(S) >= 10:
            if len(S_2) > len(S_1):
                S_1, S_2 = S_2, S_1
            if len(S_2) == 0:
                for node in S_1:
                    f.write(str(node) + ' ')
                f.write('-1 -1\n')
            else:
                for node in S_1:
                    f.write(str(node) + ' ')
                f.write('-1 ')
                for node in S_2:
                    f.write(str(node) + ' ')
                f.write('-1\n')
            queue.append(S)
    
shutil.rmtree("K2")
os.remove('../datasets/temp.mat')

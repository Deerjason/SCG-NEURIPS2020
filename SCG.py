from scipy.sparse.linalg import eigsh
from utility import *

DATASET_LIST = ['bitcoin', 'congress', 'wikielections']
ROUNDING_LIST = ['min_angle', 'randomized', 'max_obj', 'bansal']

def SCG(dataset, rounding_strategy, subG=None, N=None):
    """ ADDED: designed for get_subgraphs in utility.py (K=2) """
    if subG is None:
        N, A = read_graph("datasets/{}.txt".format(dataset))
    else:
        N, A = read_subG(dataset, set(subG), N)

    # eigendecompose the core KI-1 matrix
    D, U = EigenDecompose_Core(2)
    U = U[:, D.argsort()]
    # initialization
    Y = np.zeros((N,2)) # to be determined, it must satisfy (1) Y_{i,1}=1/sqrt(K) if not neutral; 0 otherwise (2) Y_{i,2:} in {0, U_{1,2:}, ..., U_{K,2:}}
    C = np.array([-1 for i in range(N)]) # cluster assignment, C_i in {-1, 1, ..., K}, where -1 represent neutral
    mask = np.ones((N)) # list of nodes to be assigned
    maskA = A.copy() # adjacency matrix of the remaining graph

    _, lU = eigsh(maskA, k=1, which='LA') # the eigenvector of the largest eigenvalue
    v = lU[:,0].reshape((-1))
    # Round v to {-1,0,z}^n
    if rounding_strategy=='min_angle':
        v_round = round_by_min_angle(v, 1, -1, mask, N)
    elif rounding_strategy=='randomized':
        v_round = round_by_randomized_vector(v, 1, -1, mask, maskA, N)
    elif rounding_strategy=='max_obj':
        v_round = round_by_max_obj_one_threshold(v, 1, -1, mask, maskA, N)
    elif rounding_strategy=='bansal':
        v_round = round_by_cc_bansal(1, -1, mask, maskA, N)
    # assign to the new cluster(s)
    for i in range(N):
        if v_round[i]==0: continue
        if v_round[i]>0: C[i], Y[i,:] = 1, U[0,:].copy() # assign to the (K-1)-th cluster
        else: C[i], Y[i,:] = 2, U[1,:].copy() # assign to the K-th cluster
    return C, N, A

opt = parse_arg()

if opt.dataset == 'all': # experiment: real-world datasets
    for dataset in DATASET_LIST:
        if opt.rounding_strategy == 'min_angle':
            f = open(dataset + '_' + 'SCG-MA_subgraphs', 'w')
        elif opt.rounding_strategy == 'randomized':
            f = open(dataset + '_' + 'SCG-R_subgraphs', 'w')
        elif opt.rounding_strategy == 'max_obj':
            f = open(dataset + '_' + 'SCG-MO_subgraphs', 'w')
        elif opt.rounding_strategy == 'bansal':
            f = open(dataset + '_' + 'SCG-B_subgraphs', 'w')

        C, N, A = SCG(dataset, opt.rounding_strategy)
        S_1, S_2 = get_subgraphs(C, N, A)
        S_1 = list(S_1)
        S_2 = list(S_2)

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
        subG = S_1 + S_2
        while True:
            C, N, A = SCG(A, opt.rounding_strategy, set(subG), N)
            S_1, S_2 = get_subgraphs(C, N, A)
            S_1 = list(S_1)
            S_2 = list(S_2)
            S = S_1 + S_2
            if len(S) < 10:
                break
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
            subG = S
        f.close()
elif opt.dataset in DATASET_LIST:
    dataset = opt.dataset

    if opt.rounding_strategy == 'min_angle':
        f = open(dataset + '_' + 'SCG-MA_subgraphs', 'w')
    elif opt.rounding_strategy == 'randomized':
        f = open(dataset + '_' + 'SCG-R_subgraphs', 'w')
    elif opt.rounding_strategy == 'max_obj':
        f = open(dataset + '_' + 'SCG-MO_subgraphs', 'w')
    elif opt.rounding_strategy == 'bansal':
        f = open(dataset + '_' + 'SCG-B_subgraphs', 'w')

    C, N, A = SCG(dataset, opt.rounding_strategy)
    S_1, S_2 = get_subgraphs(C, N, A)
    S_1 = list(S_1)
    S_2 = list(S_2)
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
    subG = S_1 + S_2
    while True:
        C, N, A = SCG(A, opt.rounding_strategy, set(subG), N)
        S_1, S_2 = get_subgraphs(C, N, A)
        S_1 = list(S_1)
        S_2 = list(S_2)
        S = S_1 + S_2
        if len(S) < 10:
            break
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
        subG = S
    f.close()
else:
    raise Exception('Error: please specify dataset name in {} or just leave it blank to run ALL'.format(DATASET_LIST))

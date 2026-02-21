import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import pickle
import numpy as np

from const import PPI_DIST_MAT_FILE, PPI_DIST_MAT_NODE2ID_FILE

PPI_DIST_MAT = np.load(PPI_DIST_MAT_FILE)
with open(PPI_DIST_MAT_NODE2ID_FILE, 'rb') as f:
    PPI_DIST_MAT_NODE2ID = pickle.load(f)


if __name__ == '__main__':
    print(PPI_DIST_MAT.shape)
    print(PPI_DIST_MAT_NODE2ID)
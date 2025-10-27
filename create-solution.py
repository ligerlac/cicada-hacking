import h5py
import numpy as np


if __name__ == "__main__":
    with h5py.File("data/ZB_test.h5", "r") as f:
        ZB_test = f["et"][:]
    with h5py.File("data/tt_lep_test.h5", "r") as f:
        tt_lep_test = f["et"][:]

    test_labels = np.concatenate([np.zeros(len(ZB_test)), np.ones(len(tt_lep_test))], axis=0).astype(int)
    np.savetxt("data/solution.csv", test_labels, fmt='%d')

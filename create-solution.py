import h5py
import numpy as np
import pandas as pd



if __name__ == "__main__":

    with h5py.File("data/ZB_test.h5", "r") as f:
        ZB_test = f["et"][:]
    with h5py.File("data/tt_lep_test.h5", "r") as f:
        tt_lep_test = f["et"][:]

    # Labels: 0 for background, 1 for signal
    test_labels = np.concatenate([
        np.zeros(len(ZB_test)),
        np.ones(len(tt_lep_test))
    ]).astype(int)

    ids = np.arange(len(test_labels))

    # ✅ Randomized public/private split
    rng = np.random.default_rng(seed=42)
    mask = rng.random(len(test_labels)) < 0.5
    usage = np.where(mask, "Public", "Private")

    solution = pd.DataFrame({
        "Id": ids,
        "Label": test_labels,
        "Usage": usage
    })

    solution.to_csv("data/solution.csv", index=False)

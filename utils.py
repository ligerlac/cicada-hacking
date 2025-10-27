import numpy as np

def load_data_old():
    X_train = np.load('data/ZB_train.npy')
    X_val = np.load('data/ZB_val.npy')
    X_test = np.load('data/ZB_test.npy')
    return X_train, X_val, X_test


def load_data(data_dir="data"):
    import h5py
    data = {}
    for proc in ["ZB_train", "ZB_val", "ZB_test", "tt_lep_train", "tt_lep_val", "tt_lep_test", "tt_had"]:
        with h5py.File(f"{data_dir}/{proc}.h5", "r") as f:
            d = {
                "et": f["et"][:],
                "taubit": f["taubit"][:],
                "egbit": f["egbit"][:]
            }
            # stack them to form (18, 14, 3) shape
            d_stacked = np.stack([d["et"].astype(float), d["taubit"].astype(float), d["egbit"].astype(float)], axis=-1)
            data[proc] = d_stacked
    return data




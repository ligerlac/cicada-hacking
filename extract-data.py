import h5py
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import data


if __name__ == "__main__":

    file_names = {
        "ZB": "having_eg_tau_veto_bits/all_zb_train_files_open_data_calotree_input_ntuplizer.h5",
        "tt_had": "having_eg_tau_veto_bits/tt_hadronic_calotree_input_ntuplizer.h5",
        "tt_lep": "having_eg_tau_veto_bits/tt_2l2nu_calotree_input_ntuplizer.h5"
    }
    n_max = 160_000
    for proc, path in file_names.items():
        print(f"Processing {proc} from file {path}...")
        f = h5py.File(path, "r")
        data = f["CaloRegions"][:]
        X_et = data['et']
        X_taubit = data['taubit']
        X_egbit = data['egbit']

        if len(X_et) > n_max:
            # pick random n_max entries
            indices = np.random.choice(len(X_et), size=n_max, replace=False)
            X_et = X_et[indices]
            X_taubit = X_taubit[indices]
            X_egbit = X_egbit[indices]

        if proc == "tt_had":
            with h5py.File(f"data/{proc}.h5", "w") as f_train:
                f_train.create_dataset("et", data=X_et, dtype=np.int16)
                f_train.create_dataset("taubit", data=X_taubit, dtype=bool)
                f_train.create_dataset("egbit", data=X_egbit, dtype=bool)

        else:
            # split into train, val, test
            X_et, X_test_et, X_taubit, X_test_taubit, X_egbit, X_test_egbit = train_test_split(
                X_et, X_taubit, X_egbit, test_size=0.2, random_state=42)
            X_train_et, X_val_et, X_train_taubit, X_val_taubit, X_train_egbit, X_val_egbit = train_test_split(
                X_et, X_taubit, X_egbit, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
            
            # save as h5py files
            with h5py.File(f"data/{proc}_train.h5", "w") as f_train:
                f_train.create_dataset("et", data=X_train_et, dtype=np.int16)
                f_train.create_dataset("taubit", data=X_train_taubit, dtype=bool)
                f_train.create_dataset("egbit", data=X_train_egbit, dtype=bool)
            with h5py.File(f"data/{proc}_val.h5", "w") as f_val:
                f_val.create_dataset("et", data=X_val_et, dtype=np.int16)
                f_val.create_dataset("taubit", data=X_val_taubit, dtype=bool)
                f_val.create_dataset("egbit", data=X_val_egbit, dtype=bool)
            with h5py.File(f"data/{proc}_test.h5", "w") as f_test:
                f_test.create_dataset("et", data=X_test_et, dtype=np.int16)
                f_test.create_dataset("taubit", data=X_test_taubit, dtype=bool)
                f_test.create_dataset("egbit", data=X_test_egbit, dtype=bool)

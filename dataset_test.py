import numpy as np


path = "/home/chenyinuo/data/dataset/data_for_success/green_bell_pepper_plate_wooden/success/success_green_bell_pepper_plate_wooden_proc_0_num_175_seed_181_epsid_181.npz"

data = np.load(path, allow_pickle=True)["arr_0"].tolist()

print(data.keys())

breakpoint()
import numpy as np

def check_auc_X(X_first, X_new):
    if X_first.shape != X_new.shape:
        # print("wrong shape")
        return False
    return np.array_equal(X_first, X_new) 

def auc_combo_to_X_y(auc_combo_all):
    X, y = [], []

    # ------------------------------------------------------------------
    # reference stimulus
    first_key = next(iter(auc_combo_all))
    X_first   = np.array(list(auc_combo_all[first_key].keys()),
                         dtype=np.float64)

    # ------------------------------------------------------------------
    # 1️⃣  do we have identical stimuli everywhere?
    all_identical = True
    for key in auc_combo_all:
        X_ls = list(auc_combo_all[key].keys())
        if len(X_ls) > 0:
            X_new = np.array(X_ls, dtype=np.float64)
            if not check_auc_X(X_first, X_new):
                all_identical = False
                break
        else:
            print(f"Neuron {key} have no recorded responses; will be skipped")
        

    # If identical → keep exactly one copy now
    if all_identical:
        X.append(X_first)

    # ------------------------------------------------------------------
    # 2️⃣  main loop
    for key in auc_combo_all:
        X_ls = list(auc_combo_all[key].keys())
        if len(X_ls) > 0:
            X_new = np.array(X_ls, dtype=np.float64)
        # X_new = np.array(list(auc_combo_all[key].keys()), dtype=np.float64)
            if not all_identical:          # heterogeneous case → keep every X_new
                X.append(X_new)

            y.append(np.array(list(auc_combo_all[key].values())))
        else:
            print(f"Neuron {key} have no recorded responses; will be skipped")

    return X, y

def auc_clean_to_X_y(auc_clean, stimX):
    X, y = [], []
    all_identical = True
    for i in range(auc_clean.shape[0]):  # for each neuron
        leading_nan = np.argmax(~np.isnan(auc_clean[i]))
        # X_neuron_all = stimX[leading_nan:]
        # y_neuron_all = auc_clean[i][leading_nan:]
        if leading_nan != 0:
            all_identical = False
            break
    if all_identical:
        X.append(stimX)
    for i in range(auc_clean.shape[0]):  # for each neuron
        leading_nan = np.argmax(~np.isnan(auc_clean[i]))
        X_neuron_all = stimX[leading_nan:]
        y_neuron_all = auc_clean[i][leading_nan:]
        y.append(y_neuron_all)
        if not all_identical:
            X.append(X_neuron_all)
    if all_identical:
        X = X[0]
    return X, y

def count_leading_nan(arr):
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return len(arr)
    else:
        return np.argmax(~np.isnan(arr))
    
def gen_X_y_online_fits(total_stimulus_indices, auc_clean):
    X, y = [], []
    # get number of leading np.nan
    for i in range(len(auc_clean)):  # len(auc_clean) iterate through every neuron
        leading_nan = count_leading_nan(auc_clean[i])
        # print(i, leading_nan)
        X_i = total_stimulus_indices[leading_nan:]
        y_i = auc_clean[i][leading_nan:]
        # print(np.asarray(X_i).shape, len(y_i)) 
        # print(X_i)
        assert len(X_i) == len(y_i), "X_i and y_i does not have the same shape"
        if len(X_i) > 0:
            X.append(X_i)
            y.append(y_i)
        else:
            print(f"Neuron {i} have no recorded responses; will be skipped")
    return X, y

# def auc_combo_to_X_y(auc_combo_all):
#     X = []
#     # X = np.array(list(auc_combo_all[0].keys())) # (n, T, d); depend on the situation..?
#     y = []  # (n, T)
#     X_first = np.array(list(auc_combo_all[list(auc_combo_all.keys())[0]].keys()), dtype=np.float64)
#     X_append_done = False  
#     for i in auc_combo_all.keys():
#         # X_append_done = False
#         X_new = np.array(list(auc_combo_all[i].keys()), dtype=np.float64)
#         if check_auc_X(X_first, X_new):               # arrays are identical
#             if not X_append_done:                     # ⬅ ② append exactly once
#                 # print("okpass")
#                 X.append(X_first)
#                 X_append_done = True
#             # else: do nothing (skip duplicates)
#         else:                                         # arrays differ
#             # print(i, "appending!")
#             X.append(X_new) 
#         # if not check_auc_X(X_first, X_new) and not X_append_done:  # if X is different
#         #     # print("appending!!!")
#         #     # print(type(X))
#         #     X.append(X_new)
#         #     X_append_done = False
#         # else:
#         #     X.append(X_first)
#         #     X_append_done = True
#         # X.append(np.array(list(auc_combo_all[i].keys())))
#         y.append(np.array(list(auc_combo_all[i].values())))
#     return X, y


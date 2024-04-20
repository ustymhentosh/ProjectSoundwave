import json
from classifier import UsLeVoModel
from moses import feeling_lucky
from tqdm import tqdm

K = [20, 40, 70, 150, 200, 300, 500]
win_size = [50, 100, 150]
sex = ["male", "female", None]


def run_one_test(dst_fl, k, win_size, step, mtype):
    m = UsLeVoModel(k, win_size, step, mtype)
    m.train(dst_fl, "dataset/clips")
    return m.test(dst_fl, "dataset/clips")


train_n = 8
test_n = 2
users_n = 50

dct = {}
for s in sex:
    # feeling_lucky(
    #     f"results/sex_{s}_users_{users_n}_{train_n}-{test_n}.json",
    #     s,
    #     users_n,
    #     train_n,
    #     test_n,
    # )
    file_name = f"results/sex_{s}_users_{users_n}_results_lib.json"
    for k in tqdm(K):
        for win_s in win_size:
            r = run_one_test(
                f"results/sex_{s}_users_{users_n}_{train_n}-{test_n}.json",
                k,
                win_s,
                step=int(0.65 * win_s),
                mtype="lib",
            )
            dct[f"k={k}, win_size={win_s}, train={train_n}, test={test_n}"] = r
            # except:
            #     print("Ex...")

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(dct, f)
    dct = {}

from arpreprocessing.kemophone import KEmoPhone
from experiment.experiment import Experiment, prepare_experimental_setups_n_iterations, n_fold_split

SIGNALS_LEN = 6


class KEmoPhoneExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        folds = n_fold_split(KEmoPhone.SUBJECTS_IDS, n, seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "KEmoPhone", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)
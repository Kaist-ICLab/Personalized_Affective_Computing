from arpreprocessing.kemowork import KEmoWork
from clustering.kemoworkclustering import n_fold_split_cluster_trait, n_fold_split_cluster_trait_experiment
from experiment.experiment_cluster import Experiment, prepare_experimental_setups_n_iterations

SIGNALS_LEN = 11

class KEmoWorkExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        folds = n_fold_split_cluster_trait(KEmoWork.SUBJECTS_IDS, n, "KEmoWork", seed=seed)
        # folds = n_fold_split_cluster_trait_experiment(KEmoWork.SUBJECTS_IDS, n, "KEmoWork", seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "KEmoWork", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)

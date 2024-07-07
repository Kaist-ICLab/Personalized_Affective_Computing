from arpreprocessing.amigos import AMIGOS
from clustering.amigosclustering import n_fold_split_cluster_trait_mtl
from experiment.experiment import Experiment, prepare_experimental_setups_n_iterations, n_fold_split

SIGNALS_LEN = 20


class AMIGOSExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        folds = n_fold_split(AMIGOS.SUBJECTS_IDS, n, seed=seed)

        clusters = n_fold_split_cluster_trait_mtl(AMIGOS.SUBJECTS_IDS, n, "AMIGOS", seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]
        self.cluster_ids = clusters[i]["cluster"]

        Experiment.__init__(self, "AMIGOS", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids, self.cluster_ids)

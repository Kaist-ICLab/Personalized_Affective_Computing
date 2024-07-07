import sys

from experiment.hyperparametertuning import HyperparameterTuning
from experiment.wesadexperiment_cluster import WesadExperimentNFold
from experiment.caseexperiment_cluster import CaseExperimentNFold
from experiment.kemoconexperiment_cluster import KEmoConExperimentNFold
from experiment.ascertainexperiment_cluster import ASCERTAINExperimentNFold
from experiment.amigosexperiment_cluster import AMIGOSexperimentNFold
from utils.loggerwrapper import GLOBAL_LOGGER
from utils.utils import set_available_gpus


def get_dataset(name):
    if name.startswith("wesad_fold_"):
        return WesadExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    if name.startswith("case_fold_"):
        return CaseExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    if name.startswith("kemocon_fold_"):
        return KEmoConExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    if name.startswith("ascertain_fold_"):
        return ASCERTAINExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    if name.startswith("amigos_fold_"):
        return AMIGOSexperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    raise Exception(f"No such dataset/experiment as {name}")


if __name__ == '__main__':
    try:
        _, gpu_id, dataset_name, clas, max_eval = sys.argv

        set_available_gpus(gpu_id)
        dataset = get_dataset(dataset_name)

        HyperparameterTuning(dataset, [gpu_id]).tune_one(clas, int(max_eval))

    except Exception as e:
        GLOBAL_LOGGER.exception(e)
        raise e
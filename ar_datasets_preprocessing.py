import configparser
import pathlib

from arpreprocessing.wesad import Wesad
from arpreprocessing.case import Case
from arpreprocessing.kemocon import KEmoCon
from arpreprocessing.ascertain import ASCERTAIN
from arpreprocessing.amigos import AMIGOS
from arpreprocessing.kemophone import KEmoPhone
from utils.loggerwrapper import GLOBAL_LOGGER

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    #dataset = Wesad(GLOBAL_LOGGER, config['Paths']['wesad_dir']).get_dataset()
    #dataset.save(config['Paths']['mts_out_dir'])

    #dataset = Case(GLOBAL_LOGGER, config['Paths']['case_dir'], 'AROUSAL').get_dataset()
    #dataset = Case(GLOBAL_LOGGER, config['Paths']['case_dir'], 'VALENCE').get_dataset()
    #dataset.save(config['Paths']['mts_out_dir'])

    #dataset = KEmoCon(GLOBAL_LOGGER, config['Paths']['kemocon_dir'], 'arousal').get_dataset()
    #dataset = KEmoCon(GLOBAL_LOGGER, config['Paths']['kemocon_dir'], 'valence').get_dataset()
    #dataset.save(config['Paths']['mts_out_dir'])

    #dataset = ASCERTAIN(GLOBAL_LOGGER, config['Paths']['ascertain_dir'], 'AROUSAL').get_dataset()
    #dataset = ASCERTAIN(GLOBAL_LOGGER, config['Paths']['ascertain_dir'], 'VALENCE').get_dataset()
    #dataset.save(config['Paths']['mts_out_dir'])

    #dataset = AMIGOS(GLOBAL_LOGGER, config['Paths']['amigos_dir'], 'AROUSAL').get_dataset()
    # dataset = AMIGOS(GLOBAL_LOGGER, config['Paths']['amigos_dir'], 'VALENCE').get_dataset()
    # dataset.save(config['Paths']['mts_out_dir'])

    dataset = KEmoPhone(GLOBAL_LOGGER, config['Paths']['kemophone_dir'], 'STRESS').get_dataset()
    dataset.save(config['Paths']['mts_out_dir'])

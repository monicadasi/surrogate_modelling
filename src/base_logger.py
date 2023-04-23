'''
Base logger class for console and text file logging

Implemented by Monica Dasi
Developed for Master's Thesis
Topic : Surrogate Modeling For Freequency Domain Simulation Data - SoSe 2023
Department of Computer Science (Fb2), Frankfurt University of Applied Sciences
'''
import os
import logging
import utils
import time

FORMAT_STR = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FMT = '%Y-%m-%d %H:%M:%S'


class BaseLogger:
    def __init__(self, log_level) -> None:
        root = logging.getLogger()
        root.setLevel(logging.NOTSET)

        # setup logging to console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        # set console format
        formatter = logging.Formatter(FORMAT_STR, datefmt=DATE_FMT)
        ch.setFormatter(formatter)
        # add the console handler to the root logger
        root.addHandler(ch)

        _log_dir = utils.Utils().get_log_dir_path()
        _file = '{0}/' + f'{time.strftime("%Y%m%d_%H%M%S")}_log.txt'
        f_name = os.path.realpath(_file.format(_log_dir))
        # setup file handler logger
        fh = logging.FileHandler(f_name, mode='w', delay=True)
        fh.setFormatter(formatter)
        fh.setLevel(log_level)
        root.addHandler(fh)

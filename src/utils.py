'''
Utils which create folder for saving the plots and csv files.

Implemented by Monica Dasi
Developed for Master's Thesis
Topic : Surrogate Modeling For Freequency Domain Simulation Data
Semester : SoSe 2023
Department of Computer Science (Fb2), Frankfurt University of Applied Sciences
'''

import os
import pathlib

import json
from singleton import Singleton


class Utils(metaclass=Singleton):
    def __init__(self) -> None:
        pass

    def get_dir_path(self) -> str:
        # self._dir_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        self._dir_path = r'D:\MasterThesis\surrogate_modelling_repo\data'
        return self._dir_path

    def get_data_dir_path(self) -> str:
        _data_dir = self.get_dir_path() + '\orig_data'
        try:
            os.makedirs(_data_dir)
        except FileExistsError:
            # data directory already exists
            pass
        return _data_dir

    def get_results_dir_path(self) -> str:
        _res_dir = self.get_dir_path() + '\\results'
        try:
            os.makedirs(_res_dir)
        except FileExistsError:
            # results directory already exists
            pass
        return _res_dir

    def get_plots_dir_path(self) -> str:
        _plt_dir = self.get_dir_path() + '\\ref_plots'
        try:
            os.makedirs(_plt_dir)
        except FileExistsError:
            # plots directory already exists
            pass
        return _plt_dir
    
    def get_circle_plots_dir_path(self) -> str:
        _plt_dir = self.get_dir_path() + '\\circle_plots'
        try:
            os.makedirs(_plt_dir)
        except FileExistsError:
            # plots directory already exists
            pass
        return _plt_dir

    def get_models_dir_path(self) -> str:
        _model_dir = self.get_dir_path() + '\\models'
        try:
            os.makedirs(_model_dir)
        except FileExistsError:
            # model directory already exists
            pass
        return _model_dir

    def get_log_dir_path(self) -> str:
        _log_dir = self.get_dir_path() + '\\logs'
        try:
            os.makedirs(_log_dir)
        except FileExistsError:
            # model directory already exists
            pass
        return _log_dir
    
    def create_freq_dir(self, frq_name)->str:
        _fq_dir = f'frq_{frq_name}'
        _model_dir = self.get_circle_plots_dir_path() + '\\' + _fq_dir
        try:
            os.makedirs(_model_dir)
        except FileExistsError:
            # model directory already exists
            pass
        return _model_dir

    def _read_config(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config
    """
    ref: https://stackoverflow.com/questions/21035762/python-read-json-file-and-modify
    Parameter
    ---------
    key : str
        Key value in the json config
    val : str
        Corresponding value of the key in json config
    """

    def _write_to_config(self, key: str, val: str):
        with open('config.json', 'r+') as _file:
            _cfg = json.load(_file)
            _cfg[key] = val  # add the value to config
            _file.seek(0)  # reset the file position to the beginning
            json.dump(_cfg, _file, indent=4)
            _file.truncate()  # remove the remaining part

    def _draw_plots(self) -> bool:
        _config = self._read_config()
        if _config['draw_plots'] == "True":
            return True
        else:
            return False

    def _parse_data(self) -> bool:
        _config = self._read_config()
        if _config['parse_data'] == "True":
            return True
        else:
            return False

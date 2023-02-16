import os
import pathlib
from singleton import Singleton


class Utils(metaclass=Singleton):
    def __init__(self) -> None:
        pass

    def get_dir_path(self) -> str:
        #self._dir_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
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


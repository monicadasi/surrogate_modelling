'''
main file for running the prediction for the surrogate model.
Run help for different configuration that can be used as arguments.

Implemented by Monica Dasi
Developed for Master's Thesis
Topic : Surrogate Modeling For Freequency Domain Simulation Data
Semester : SoSe 2023
Department of Computer Science (Fb2), Frankfurt University of Applied Sciences
'''

import argparse
import logging
from least_squares_circle_classic import LeastSquaresCircleClassic

from utils import Utils
from base_logger import BaseLogger

from data_parser import DataParser
from data_visualizer import DataVisualizer
from least_squares_circle import LeastSquaresCircle
from three_point_circle import ThreePointCircle
from least_squares_circle_jacobian import LeastSquaresCircleJacobian

from data_modeling import DataModeling
from magnitude_plot import MagnitudePlot

log = logging.getLogger(__name__)


def main(log_level: str, draw_plot: str):
    BaseLogger(log_level)

    k = 'draw_plots'
    Utils()._write_to_config(k, draw_plot)

    parse_freq_data()  # parse the freqency data.. create dataframe..
    visualize_data()
    create_model_predict()


def parse_freq_data():
    dataparser = DataParser()
    if Utils()._parse_data() == True:
        log.info("Processing the frequency data ...")
        dataparser.parse_freq_data()
        dataparser.process_data()
        dataparser.create_data_frame()
        # data is parsed and processed, one can reuse the created dataframe in the next
        # subsequent runs , care should be taken to make the changes in config.json file to
        # process the data to "TRUE", if the data has been changed / deleted.
        Utils()._write_to_config('parse_data', 'False')
        Utils()._write_to_config('process_data', 'False')
    else:
        log.info("data is available...fast forwarding...")


def visualize_data():
    # draw the plot only the config value to draw is TRUE
    if Utils()._draw_plots() == True:
        _dv = DataVisualizer()
        _dv.plot_frf_data()
        _dv.draw_mag_parameter_plot()
        _dv.draw_polar_plot()


def create_model_predict():
    _lsc = LeastSquaresCircle()
    _lsc.process_circle_extraction()
    _df_list = _lsc._get_df_list()

    # _lscc = LeastSquaresCircleClassic()
    # _lscc.process_circle_extraction()
    # _dfc_list = _lscc._get_df_list()

    # process circle extraction using least squares jacobian method
    # _lscj = LeastSquaresCircleJacobian()
    # _lscj.process_circle_extraction()
    # _df_list = _lscj._get_df_list()

    # _tpc = ThreePointCircle()
    # _tpc.process_circle_extraction()
    # _df_list = _tpc._get_df_list()

    #log.info(f"Df list in main : {_df_list}")
    _dm = DataModeling()
    _dm.set_df_list(_df_list)
    _dm.model_data()
    _final_df = _dm.get_final_df()
    MagnitudePlot(_final_df)
    #CircleParamPlot(_df_list)


    # if Utils()._draw_plots() == 'True':
    #     _final_df = _dm.get_final_df()
    #     MagnitudePlot(_final_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Surrogate Modeling Using Complex Valued Frequency Domain Simulation Data')
    parser.add_argument('-l', '--log_level', type=str, default='info',
                        choices=['debug', 'info'], help='Change the log level to Debug or Info')
    parser.add_argument('-dp', '--draw_plot', type=str, default= 'False',
                        choices=['True', 'False'],
                        help='True if the plots (models, relationship plots) as to be\
                              saved into the local directory, False otherwise')
    args = parser.parse_args()
    main(log_level=args.log_level.upper(), draw_plot=args.draw_plot)

import argparse
import logging

from base_logger import BaseLogger

from data_visualizer import DataVisualizer
from least_squares_circle import LeastSquaresCircle
from data_modeling import DataModeling


def main(log_level):
    BaseLogger(log_level)
    _dv = DataVisualizer()
    _dv.plot_frf_data()
    _dv.draw_mag_parameter_plot()
    _dv.draw_polar_plot()

    _lsc = LeastSquaresCircle()
    _lsc.process_circle_extraction()

    _df = _lsc._get_dataframe()
    _df_list = _lsc._get_df_list()
    _dm = DataModeling(_df_list)

    _dm.model_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Surrogate Modeling Using Complex Valued Frequency Domain Simulation Data')
    parser.add_argument('-l', '--log_level', type=str, default='info',
                        choices=['debug', 'info'], help='Change the log level to Debug or Info')
    args = parser.parse_args()
    main(log_level=args.log_level.upper())

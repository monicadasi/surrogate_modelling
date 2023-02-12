from data_visualizer import DataVisualizer
from least_squares_circle import LeastSquaresCircle
from data_modeling import DataModeling


def main():
    _dv = DataVisualizer()
    _dv.plot_frf_data()
    _dv.draw_mag_parameter_plot()
    _dv.draw_polar_plot()

    _lsc = LeastSquaresCircle()
    _lsc.process_circle_extraction()

    _df = _lsc._get_dataframe()
    _dm = DataModeling(_df)
    _dm.model_data()


if __name__ == "__main__":
    main()

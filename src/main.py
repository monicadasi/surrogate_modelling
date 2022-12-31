from data_visualizer import DataVisualizer
from least_squares_circle import LeastSquaresCircle


def main():
    dv = DataVisualizer()
    dv.plot_frf_data()
    dv.draw_mag_parameter_plot()
    dv.draw_polar_plot()
    lsc = LeastSquaresCircle()
    lsc.process_circle_extraction()

if __name__ == "__main__":
    main()
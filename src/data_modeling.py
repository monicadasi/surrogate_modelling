import os
import utils
import math
import logging
import pandas as pd
import numpy as np

import matplotlib
#matplotlib.use('GTK3Agg') 

import matplotlib.pyplot as plt

from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

RADIUS = 'Radius'
ANGLE = 'Angle'
X_CENTER = 'x_center'
Y_CENTER = 'y_center'

log = logging.getLogger(__name__)

class DataModeling():

    def __init__(self, df) -> None:
        self.res_freqs = []
        self._model_dir = utils.Utils().get_models_dir_path()
        self._df_list = df

    def extract_dataframe_info(self):
        for _df in self._df_list:
            # print(_df)
            fq_list = _df['Frequency'].to_list()
            fq = list(dict.fromkeys(fq_list))
            fq = fq[-1]
            # self.res_freqs.append(temp[-1])
            res_df = _df[_df['Frequency'].isin([fq])]
            lambda_val = res_df['Lambda']

            radi_val = res_df[RADIUS]
            self.model_radius(lambda_val, radi_val, fq)
            ph_val = res_df[ANGLE]
            self.model_phase(lambda_val, ph_val, fq)
            x_coord = res_df[X_CENTER]
            self.model_xcoord(lambda_val, x_coord, fq)
            y_coord = res_df[Y_CENTER]
            self.model_ycoord(lambda_val, y_coord, fq)

    # def process_freqs(self) -> None:
    #     for fq in self.res_freqs:
    #         res_df = self._df_list[self._df_list['Frequency'].isin([fq])]
    #         lambda_val = res_df['Lambda']
    #         radi_val = res_df['Radius']
    #         self.model_radius(lambda_val, radi_val, fq)
    #         ph_val = res_df['Angle']
    #         self.model_phase(lambda_val, ph_val, fq)

    def model_data(self) -> None:
        self.extract_dataframe_info()
        # self.process_freqs()
        # self.model_radius()
        # self.model_phase()
        # self.model_xcoord()
        # self.model_ycoord()

    def model_radius(self, lmbda, radii, freq) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(radii)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False

        _, ax1 = plt.subplots(1, 1)

        ax1.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax1.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax1.legend(loc="upper left")
        # plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=9)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        # _test = x_test.reshape(-1,1)
        xtest_poly = poly.fit_transform(x_test)
        y_pred = poly_reg.predict(xtest_poly)

        log.debug(f'Score : {res.score(x_poly, y_train)}')

        pred_r = poly_reg.predict(poly.transform([[7]]))

        pred_r = list(pred_r.flatten())
        R = pred_r[-1]

        log.info(f'predicted radius : {R}')

        ax1.set_title('Regression w.r.t Radius')
        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('Radius')
        ax1.plot(x_train, poly_reg.predict(x_poly),
                 c='blue', label='Predicted Line')
        ax1.legend(loc="best")
        f_name = '{0}/' + f'Radius[Model]_{freq}.png'
        # plt.rcParams['savefig.compression_level'] = 9
        plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
        plt.close()

    def model_phase(self, lmbda, angle, freq) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(angle)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False

        _, ax2 = plt.subplots(1, 1)

        ax2.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax2.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax2.legend(loc="upper left")
        # plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=4)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        log.debug(f'Score : {res.score(x_poly, y_train)}')
        pred_an = poly_reg.predict(poly.transform([[11]]))

        pred_an = list(pred_an.flatten())
        Ph = pred_an[-1]

        log.info(f'Predicted Angle : {Ph}')

        # pred_ph_deg = math.degrees(Ph)
        # v2  = np.rad2deg(Ph)
        # print('predicted angle in degrees : {}, {}'.format(pred_ph_deg, v2))

        ax2.set_title('Regression w.r.t Angle')
        ax2.set_xlabel('Lambda')
        ax2.set_ylabel('Angle')
        ax2.plot(x_train, poly_reg.predict(x_poly),
                 c='blue', label='Predicted Line')
        ax2.legend(loc="upper right")

        f_name = '{0}/' + f'Phase[Model]_{freq}.png'
        plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
        plt.close()

    def model_xcoord(self, lmbda, x_center, freq) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(x_center)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False

        _, ax3 = plt.subplots(1, 1)

        ax3.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax3.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax3.legend(loc="upper left")
        # plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=9)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        log.debug(f'Score : {res.score(x_poly, y_train)}')

        pred_Xc = poly_reg.predict(poly.transform([[11]]))
        pred_Xc = list(pred_Xc.flatten())
        Xc = pred_Xc[-1]

        log.info(f'Predicted X-Center : {Xc}')

        ax3.set_title('Regression w.r.t X_Center')
        ax3.set_xlabel('Lambda')
        ax3.set_ylabel('X_Center')
        ax3.plot(x_train, poly_reg.predict(x_poly),
                 c='blue', label='Predicted Line')
        ax3.legend(loc="upper right")

        f_name = '{0}/' + f'X-Coordinate[Model]_{freq}.png'
        plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
        plt.close()

    def model_ycoord(self, lmbda, y_center, freq) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(y_center)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False

        _, ax4 = plt.subplots(1, 1)

        ax4.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax4.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax4.legend(loc="upper left")
        # plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=9)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        log.debug(f'Score :  {res.score(x_poly, y_train)}')
        pred_Yc = poly_reg.predict(poly.transform([[11]]))

        pred_Yc = list(pred_Yc.flatten())
        Yc = pred_Yc[-1]

        log.info(f'Predicted Y-Center : {Yc}')
        log.info('----------------------------------')

        ax4.set_title('Regression w.r.t y_center')
        ax4.set_xlabel('Lambda')
        ax4.set_ylabel('Y_Center')
        ax4.plot(x_train, poly_reg.predict(x_poly),
                 c='blue', label='Predicted Line')
        ax4.legend(loc="upper left")

        f_name = '{0}/' + f'Y-coordinate[Model]_{freq}.png'
        plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
        plt.close()

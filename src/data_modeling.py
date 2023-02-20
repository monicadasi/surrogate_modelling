import os
import utils
import math
import logging
import pandas as pd
import numpy as np

import time
#from datetime import timedelta
from humanfriendly import format_timespan

#import matplotlib
# matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt

from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

RADIUS = 'Radius'
ANGLE = 'Angle'
X_CENTER = 'x_center'
Y_CENTER = 'y_center'
LIST_SIZE = 1

log = logging.getLogger(__name__)


class DataModeling():

    def __init__(self, df) -> None:
        self.res_freqs = []
        self._model_dir = utils.Utils().get_models_dir_path()
        self._plot = utils.Utils()._draw_plots()
        self._df_list = df

        self.freqncy_list = []
        self.lambda_list = []
        #original data attributes
        self.orig_x_list = []
        self.orig_y_list = []
        self.orig_mag_list =[]
        #predicted data attributes
        self.pred_x_list = []
        self.pred_y_list = []
        self.pred_mag_list = []
        #newly created dataframe with original and predicted values
        self.new_df_list = []

    def model_data(self) -> None:
        log.info('Modeling the data...')
        self.extract_dataframe_info()

    def extract_dataframe_info(self):
        start_time = time.monotonic()
        for _df in self._df_list:
            fq_list = _df['Frequency'].to_list()
            _fq = list(dict.fromkeys(fq_list))

            self.fq = _fq[-1]
            res_df = _df[_df['Frequency'].isin([self.fq])]
            lambda_vals = res_df['Lambda'] #length here is 200

            self.model_radius(lambda_vals, res_df[RADIUS])
            self.model_phase(lambda_vals, res_df[ANGLE])
            self.model_xcenter(lambda_vals, res_df[X_CENTER])
            self.model_ycenter(lambda_vals, res_df[Y_CENTER])

            # models are prepared.. it's time for some predictions ;)
            log.info(f'Predictions for the frequency {self.fq} is,')
            for l_val in lambda_vals:
                self.predict_radius(l_val)
                self.predict_phase(l_val)
                self.predict_Xc(l_val)
                self.predict_Yc(l_val)
                self.extract_predicted_xy_coord(l_val, res_df)
            # create dataframe with the desired for regression modeling

            _new_df = pd.DataFrame(list(zip(self.freqncy_list, self.lambda_list, 
                                    self.orig_x_list, self.orig_y_list, self.orig_mag_list, 
                                    self.pred_x_list, self.pred_y_list, self.pred_mag_list)),
                                    columns=['Frequency', 'Lambda', 'Org_X', 'Org_Y', 'Org_Mag', 'Pred_X', 'Pred_Y', 'Pred_Mag'])
            self.clear_lists()
            self.new_df_list.append(_new_df)
        #endfor
        end_time = time.monotonic()
        log.info(f'Predictions finished in {format_timespan(end_time - start_time)}')
        # concat the dataframes in the list and create a single dataframe with the final data for further plotting
        self.final_df = pd.concat(self.new_df_list)
        self.final_df = self.final_df.sort_values(['Lambda', 'Frequency'], ascending = [True, True])
        self.final_df.reset_index(drop=True, inplace=True)
        df_name = '{0}/' + f'final_df.csv'
        self.final_df.to_csv(os.path.realpath(
                df_name.format(utils.Utils().get_results_dir_path())))

    def clear_lists(self) -> None:
        self.orig_x_list.clear()
        self.orig_y_list.clear()
        self.orig_mag_list.clear()
        self.pred_x_list.clear()
        self.pred_y_list.clear()
        self.pred_mag_list.clear()
        self.freqncy_list.clear()
        self.lambda_list.clear()

# --------------------------------------------------------------------------------------
    """
    Modeling Radius using polynomial regression fit.
    """

    def model_radius(self, lmbda, radii) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(radii)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)

        if self._plot == 'True':
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax1 = plt.subplots(1, 1)

            ax1.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax1.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax1.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.radii_poly = PolynomialFeatures(degree=9)
        x_poly = self.radii_poly.fit_transform(x_train)

        self.reg_radius = LinearRegression()
        res = self.reg_radius.fit(x_poly, y_train)

        xtest_poly = self.radii_poly.fit_transform(x_test)
        y_pred = self.reg_radius.predict(xtest_poly)

        log.debug(f'Model Score (Radius) : {res.score(x_poly, y_train)}')
        if self._plot == 'True':
            ax1.set_title('Regression w.r.t Radius')
            ax1.set_xlabel('Lambda')
            ax1.set_ylabel('Radius')
            ax1.plot(x_train, self.reg_radius.predict(x_poly),
                    c='blue', label='Predicted Line')
            ax1.legend(loc="best")
            f_name = '{0}/' + f'Radius[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
            plt.close()


#---------------------------------------------------------------------------
    def predict_radius(self, lambda_val):
        pred_r = self.reg_radius.predict(self.radii_poly.transform([[lambda_val]]))

        pred_r = list(pred_r.flatten())
        self.R = pred_r[-1]
        log.debug(f'Predicted Radius : {self.R}')

# --------------------------------------------------------------------------------------
    """
    Modeling Phase using polynomial regression fit.
    """

    def model_phase(self, lmbda, angle) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(angle)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)
        
        if self._plot == 'True':
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax2 = plt.subplots(1, 1)

            ax2.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax2.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax2.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.ph_poly = PolynomialFeatures(degree=4)
        x_poly = self.ph_poly.fit_transform(x_train)

        self.reg_phase = LinearRegression()
        res = self.reg_phase.fit(x_poly, y_train)

        log.debug(f'Model Score (Angle) : {res.score(x_poly, y_train)}')
        # pred_ph_deg = math.degrees(Ph)
        # v2  = np.rad2deg(Ph)
        # print('predicted angle in degrees : {}, {}'.format(pred_ph_deg, v2))
        if self._plot == 'True':
            ax2.set_title('Regression w.r.t Angle')
            ax2.set_xlabel('Lambda')
            ax2.set_ylabel('Angle')
            ax2.plot(x_train, self.reg_phase.predict(x_poly),
                    c='blue', label='Predicted Line')
            ax2.legend(loc="upper right")

            f_name = '{0}/' + f'Phase[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
            plt.close()

#---------------------------------------------------------------------------
    def predict_phase(self, lambda_val):
        pred_an = self.reg_phase.predict(self.ph_poly.transform([[lambda_val]]))

        pred_an = list(pred_an.flatten())
        self.Ph = pred_an[-1]
        log.debug(f'Predicted Angle : {self.Ph}')

# --------------------------------------------------------------------------------------
    """
    Modeling X- Center using polynomial regression fit.
    """

    def model_xcenter(self, lmbda, x_center) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(x_center)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)
        
        if self._plot == 'True':
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax3 = plt.subplots(1, 1)

            ax3.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax3.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax3.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.poly_xc = PolynomialFeatures(degree=9)
        x_poly = self.poly_xc.fit_transform(x_train)

        self.reg_xc = LinearRegression()
        res = self.reg_xc.fit(x_poly, y_train)

        log.debug(f'Model Score (X-Center) : {res.score(x_poly, y_train)}')
        if self._plot == 'True':
            ax3.set_title('Regression w.r.t X_Center')
            ax3.set_xlabel('Lambda')
            ax3.set_ylabel('X_Center')
            ax3.plot(x_train, self.reg_xc.predict(x_poly),
                    c='blue', label='Predicted Line')
            ax3.legend(loc="upper right")

            f_name = '{0}/' + f'X-Coordinate[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
            plt.close()


#---------------------------------------------------------------------------
    def predict_Xc(self, lambda_val):
        pred_Xc = self.reg_xc.predict(self.poly_xc.transform([[lambda_val]]))
        pred_Xc = list(pred_Xc.flatten())
        self.Xc = pred_Xc[-1]
        log.debug(f'Predicted X-Center : {self.Xc}')


# --------------------------------------------------------------------------------------
    """
    Modeling Y-Center using polynomial regression fit.
    """

    def model_ycenter(self, lmbda, y_center) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(y_center)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)
        if self._plot == 'True':
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax4 = plt.subplots(1, 1)

            ax4.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax4.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax4.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.poly_yc = PolynomialFeatures(degree=9)
        x_poly = self.poly_yc.fit_transform(x_train)

        self.reg_yc = LinearRegression()
        res = self.reg_yc.fit(x_poly, y_train)

        log.debug(f'Model Score (Y-Center) :  {res.score(x_poly, y_train)}')
        if self._plot == 'True':
            ax4.set_title('Regression w.r.t y_center')
            ax4.set_xlabel('Lambda')
            ax4.set_ylabel('Y_Center')
            ax4.plot(x_train, self.reg_yc.predict(x_poly),
                    c='blue', label='Predicted Line')
            ax4.legend(loc="upper left")

            f_name = '{0}/' + f'Y-coordinate[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name.format(self._model_dir)))
            plt.close()

#---------------------------------------------------------------------------
    def predict_Yc(self, lambda_val):
        pred_Yc = self.reg_yc.predict(self.poly_yc.transform([[lambda_val]]))
        pred_Yc = list(pred_Yc.flatten())
        self.Yc = pred_Yc[-1]
        log.debug(f'Predicted Y-Center : {self.Yc}')

#---------------------------------------------------------------------------
    def extract_predicted_xy_coord(self, l_val, _df):
        a = math.cos(self.Ph)
        b = math.sin(self.Ph)
        X_coord =  self.Xc + (self.R * a)
        Y_coord = self.Yc + (self.R * b)
        # for now the list size is 1
        #_df = self._df_list[-1]
        _dframe = _df[_df.Lambda == l_val]
        log.info(f'Original Data : \n {_dframe}')
        org_crd = _dframe['coordinates'].to_list()
        self.freqncy_list.append(self.fq)
        self.lambda_list.append(l_val)
        if len(org_crd) == LIST_SIZE:
            coord_tup = org_crd[-1]
            org_x = coord_tup[0]
            org_y = coord_tup[1]
            self.orig_x_list.append(org_x)
            self.orig_y_list.append(org_y)
            org_cn = complex(org_x, org_y)
            org_mag = abs(org_cn)
            self.orig_mag_list.append(org_mag)
        else:
            log.error("Please check the coordinates extraction")

        log.info(f'Predicted Coordinate : ({X_coord}, {Y_coord})')
        logging.info('-------------------------------------------------------')
        self.pred_x_list.append(X_coord)
        self.pred_y_list.append(Y_coord)
        cn = complex(X_coord, Y_coord)
        mag = abs(cn)
        self.pred_mag_list.append(mag)

    #----------------------------------------------------------------------------------
    def get_final_df(self):
        return self.final_df

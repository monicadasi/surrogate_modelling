import os
import utils
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class DataModeling():

    def __init__(self, df) -> None:
        self._df =  df
        print(self._df.head())

    def model_data(self) -> None:
        self.model_radius()
        self.model_phase()
        self.model_xcoord()
        self.model_ycoord()

    def model_radius(self) -> None:
        X = np.array(self._df['Lambda']).reshape(-1, 1)
        y = np.array(self._df['Radius'])

        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False

        _, ax1 = plt.subplots(1,1)

        ax1.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax1.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax1.legend(loc="upper left")
        #plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:,0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=7)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        #_test = x_test.reshape(-1,1)
        xtest_poly = poly.fit_transform(x_test)
        y_pred = poly_reg.predict(xtest_poly)

        print('Score :', res.score(x_poly, y_train))

        pred_r = poly_reg.predict(poly.transform([[7]]))

        pred_r = list(pred_r.flatten())
        R = pred_r[-1]

        print('predicted radius : ', R)

        ax1.set_title('Regression w.r.t Radius')
        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('Radius')
        ax1.plot(x_train, poly_reg.predict(x_poly), c='blue', label='Predicted Line')
        ax1.legend(loc="best")

        path = '{0}/results/' + f'Radius[Model].png'
        plt.savefig(os.path.realpath(path.format(utils.get_dir_path())))

    def model_phase(self) -> None:
        X = np.array(self._df['Lambda']).reshape(-1, 1)
        y = np.array(self._df['Angle'])

        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False
        
        _, ax2 = plt.subplots(1,1)

        ax2.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax2.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax2.legend(loc="upper left")
        #plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:,0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=4)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        print('Score :', res.score(x_poly, y_train))
        pred_an = poly_reg.predict(poly.transform([[11]]))

        pred_an = list(pred_an.flatten())
        Ph = pred_an[-1]

        print('predicted angle : ', Ph)

        pred_ph_deg = math.degrees(Ph)
        v2  = np.rad2deg(Ph)
        print('predicted angle in degrees : {}, {}'.format(pred_ph_deg, v2))


        ax2.set_title('Regression w.r.t Angle')
        ax2.set_xlabel('Lambda')
        ax2.set_ylabel('Angle')
        ax2.plot(x_train, poly_reg.predict(x_poly), c='blue', label='Predicted Line')
        ax2.legend(loc="upper right")

        f_name = '{0}/results/' + f'Phase[Model].png'
        plt.savefig(os.path.realpath(f_name.format(utils.get_dir_path())))

    def model_xcoord(self) -> None:
        X = np.array(self._df['Lambda']).reshape(-1, 1)
        y = np.array(self._df['x_center'])

        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False

        _, ax3 = plt.subplots(1,1)

        ax3.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax3.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax3.legend(loc="upper left")
        #plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:,0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=9)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        print('Score :', res.score(x_poly, y_train))

        pred_Xc = poly_reg.predict(poly.transform([[11]]))
        pred_Xc = list(pred_Xc.flatten())
        Xc = pred_Xc[-1]

        print('predicted X : ', Xc)

        ax3.set_title('Regression w.r.t X_Center')
        ax3.set_xlabel('Lambda')
        ax3.set_ylabel('X_Center')
        ax3.plot(x_train, poly_reg.predict(x_poly), c='blue', label='Predicted Line')
        ax3.legend(loc="upper right")

        f_name = '{0}/results/' + f'X-Coordinate[Model].png'
        plt.savefig(os.path.realpath(f_name.format(utils.get_dir_path())))

    def model_ycoord(self) -> None:
        X = np.array(self._df['Lambda']).reshape(-1, 1)
        y = np.array(self._df['y_center'])

        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)

        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False

        _, ax4 = plt.subplots(1,1)

        ax4.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
        ax4.scatter(x_train, y_train, c='#8acfd4', label='Training data')
        ax4.legend(loc="upper left")
        #plt.show()

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:,0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        poly = PolynomialFeatures(degree=9)
        x_poly = poly.fit_transform(x_train)

        poly_reg = LinearRegression()
        res = poly_reg.fit(x_poly, y_train)

        print('Score :', res.score(x_poly, y_train))
        pred_Yc = poly_reg.predict(poly.transform([[11]]))

        pred_Yc = list(pred_Yc.flatten())
        Yc = pred_Yc[-1]

        print('predicted Y : ', Yc)

        ax4.set_title('Regression w.r.t y_center')
        ax4.set_xlabel('Lambda')
        ax4.set_ylabel('Y_Center')
        ax4.plot(x_train, poly_reg.predict(x_poly), c='blue', label='Predicted Line')
        ax4.legend(loc="upper left")

        f_name = '{0}/results/' + f'Y-coordinate[Model].png'
        plt.savefig(os.path.realpath(f_name.format(utils.get_dir_path())))

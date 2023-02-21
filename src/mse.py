import os
import logging
import pandas as pd
from utils import Utils
from sklearn.metrics import mean_squared_error

class MeanSquaredError:
    def __init__(self, _final_df) -> None:
        self._final_df = _final_df
        # calculate the Mean Squared Error between the true and predicted magnitude for the frequency w.r.t 
        # as a function of lambda
        #self.load_dataframe()
        self.calculate_mse()

    # def load_dataframe(self):
    #     _res_dir = Utils().get_results_dir_path()
    #     f_name = os.path.realpath('{0}/final_df.csv'.format(_res_dir))
    #     if os.path.isfile(f_name):
    #         self._final_df = pd.read_csv(f_name)
    #     else:
    #         # doesn't exist
    #         logging.error("MeanSquaredError():File not found.")

    def calculate_mse(self):
        _mse_list = []
        # extract the 'Lambda'values from dataframe
        _res_df = self._final_df.copy(deep=True)
        lmda_lst = _res_df.apply(lambda row: row['Lambda'], axis=1).tolist()
        # remove the duplicate values from the list
        lmda_lst = list(dict.fromkeys(lmda_lst))
        for p in lmda_lst:
            rslt_frf_df = _res_df[_res_df['Lambda'].isin([p])]
            true_mag = rslt_frf_df['Org_Mag'].to_list()
            pred_mag = rslt_frf_df['Pred_Mag'].to_list()
            _mse = mean_squared_error(true_mag, pred_mag)
            logging.info(f'Lambda : {p} , MSE = {_mse}')
            _mse_list.append(_mse)

        max_value = max(_mse_list)
        max_index = _mse_list.index(max_value)
        logging.info(f'Max Value of mse:{max_value}, Lambda : {lmda_lst[max_index]}')
        _mse_df = pd.DataFrame(list(zip(lmda_lst, _mse_list)), columns=['Lambda', 'MSE'])
        _name = '{0}/' + f'mse_df.csv'
        _mse_df.to_csv(os.path.realpath(_name.format(Utils().get_results_dir_path())))
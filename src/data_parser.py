import os
import natsort
import pandas as pd
import numpy as np
import utils

# fetch the file directory and filename
dirpath = utils.get_dir_path()
fname = os.path.realpath('{0}/data/200_ds.txt'.format(dirpath))
fsplit_path = os.path.realpath('{0}/data/file_split'.format(dirpath))
fout_path = os.path.realpath('{0}/data/file_output'.format(dirpath))
ext = ('.txt')
lambda_val = []


class DataParser():
    def parse_freq_data(self):
        # Split the data into small chunks based on a literal '#Parameters' as split criteria
        #print("File Path : ", fsplit_path)
        with open(fname) as f:
            wsplit = ''
            f_out = None
            i = 0
            for line in f:
                # when a match is found , create a new output file
                if line.startswith('#Parameters'):
                    wsplit = line.split(';')[2]
                    wsplit = wsplit.split('=')[1]
                    lambda_val.append(float(wsplit))
                    i = i + 1
                    title = 'file-' + str(i)
                    # print(title)
                    if f_out:
                        f_out.close()
                    f_out = open(f'{fsplit_path}\\{title}.txt', 'w')
                if f_out:
                    f_out.write(line)
            if f_out:
                f_out.close()
        # end of file split, convert file output
        self.create_file_output()

    def create_file_output(self):
        # drop the lines with text (this done for creating the dataframe just with the values)
        id = 1
        flist = os.listdir(fsplit_path)
        flist = natsort.natsorted(flist)

        for file in flist:
            if file.endswith(ext):
                # print(id, " : ", file)
                with open(f'{fsplit_path}\\{file}', 'r') as fin:
                    data = fin.read().splitlines(True)
                with open(f'{fout_path}\\file{id}_out.txt', 'w') as fout:
                    fout.writelines(data[3:])  # drop the text inside the file
                    id = id + 1
                if fout:
                    fout.close()
            else:
                continue

    def process_data(self):
        columns_header = ['Frequency', 'S1_Real[RE]',
                          'S1_Imaginary[Im]', 'Ref.Imp. [Re]', 'Ref.Imp. [Im]']
        self.df_dict = {}
        i = 0

        files = os.listdir(fout_path)
        files = natsort.natsorted(files)

        for f in files:
            if f.endswith(ext) and i < len(lambda_val):
                val = pd.read_csv(f'{fout_path}\\{f}',
                                  delim_whitespace=True, names=columns_header)
                self.df_dict[lambda_val[i]] = val
                i += 1
            else:
                continue

    def create_data_frame(self):
        df_list = []
        for k, val in self.df_dict.items():
            val.drop(val.iloc[:, 3:], inplace=True, axis='columns')
            val['Lambda'] = k
            # update the data frame dictionary with lambda param(key)
            # and corresponding data as value
            self.df_dict.update({k: val})
            # list of dataframes containing data corresponding to each lambda param
            df_list.append(val)

        #print(len(df_list))

        # calculate the magnitude using real and imaginary values from data
        for dframe in df_list:
            for idx in dframe.index:
                # dframe.at[idx, 'Magnitude'] = sqrt(pow(dframe['S1_Real[RE]'][idx], 2) + pow(dframe['S1_Imaginary[Im]'][idx], 2))
                cn = complex(dframe['S1_Real[RE]'][idx],
                             dframe['S1_Imaginary[Im]'][idx])
                dframe.at[idx, 'Magnitude'] = abs(cn)
                dframe.at[idx, 'Phase(Deg)'] = np.angle(cn, deg=True)
                dframe.at[idx, 'Phase(Rad)'] = np.angle(cn)

        self.frf_df = pd.concat(df_list, ignore_index=True)
        self.frf_df.to_csv(os.path.realpath(
            '{0}/data/final_frf_data.csv'.format(dirpath)))
        self.frf_df.to_csv(os.path.realpath(
            '{0}/data/final_frf_data.txt'.format(dirpath)), sep='\t', index=False)
        return self.frf_df

    def get_freq_data(self):
        self.frf_df = pd.read_csv(os.path.realpath(
            '{0}/data/final_frf_data.csv'.format(dirpath)))
        return self.frf_df
import os
import utils
import matplotlib
import random
from scipy import optimize
from numpy import *
from data_visualizer import DataVisualizer
import plotly.graph_objects as go
import numpy as np


class LeastSquaresCircle():
    # reference : https://stackoverflow.com/a/55828367
    def generate_random_color():
        hex_colors_dic = {}
        rgb_colors_dic = {}
        hex_colors_only = []
        for name, hex in matplotlib.colors.cnames.items():
            hex_colors_only.append(hex)
            hex_colors_dic[name] = hex
            rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

        # getting random color from list of hex colors
        return random.choice(hex_colors_only)

    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x-xc)**2 + (self.y-yc)**2)

    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()

    # Return circle center and radius
    def least_square_circle_extraction(self, x, y):
        # coordinates of the barycenter
        x_m = mean(x)
        y_m = mean(y)

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(self.f_2, center_estimate)

        center_2 = tuple(center_2)
        xc_2, yc_2 = center_2
        Ri_2 = self.calc_R(*center_2)
        R_2 = Ri_2.mean()
        residu_2 = sum((Ri_2 - R_2)**2)
        # print("center : {0}, radius : {1}".format(center_2, R_2))
        return xc_2, yc_2, R_2

    # extract the 20 adjacent points of the picked frequency
    def extract_ten_neighbours(self, idx):
        # copy list into new variable so we don't change it
        orgDFs = self.orgDF_list[:]

        left_slice = idx - 40  # start of range
        left_slice = min(max(0, left_slice), len(
            orgDFs) - 80)  # account for edges
        right_slice = left_slice + 80  # end of range is straightforward now

        return orgDFs[left_slice:right_slice]

    def process_circle_extraction(self):
        # initialize the plot figure
        fig = go.Figure()

        # Set plot axes properties
        fig.update_xaxes(zeroline=False)
        fig.update_yaxes(zeroline=False)

        # set the graph template in plotly
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
        # ----------------------------------------------------------------------------------------------------------------

        # create a copy of the dataset
        frf_df_cp = DataVisualizer().get_freq_dataframe()
        frf_df6 = frf_df_cp.copy(deep=True)
        # frf_df6 = frf_df6.sort_values('Lambda')

        self.orgDF_list = frf_df6.apply(lambda row:
                                        [row['Frequency']]
                                        + [row['Lambda']]
                                        + [row['S1_Real[RE]']]
                                        + [row['S1_Imaginary[Im]']], axis=1).to_list()

        print('Length Original dataframe : ', len(self.orgDF_list))

        # ----------------------------------------------------------------------------------------------------------------

        # pick a random frequency from the data , 'n' denotes : number of samples to be picked
        # In this case we just pick one frequency and create a model for that.
        df_elements = frf_df6.sample(n=1)
        # contains the row of the picked frequency
        print('\n<--------- Randomly Picked Row --------->\n', df_elements)
        print('-------------------------------------------')

        # ----------------------------------------------------------------------------------------------------------------

        # frq = df_elements['Frequency'].to_list() # extract the freq. to a list
        # frq = frq[0]
        # frq = 28.75
        frq = 34.270000457764
        print('\nPicked Frequency :', frq)  # extracted freq.

        # For the picked frequency extract all the matching rows
        if not frf_df6.empty:
            result_df = frf_df6[frf_df6['Frequency'].isin([frq])]
            result_df.to_csv('Extracted_Frequency_df.csv')
        else:
            print('DF is empty!!, cannot continue...')

        # (Convert to dictionary) working dict after the data extraction based on the picked frequency
        wrk_dict = result_df.apply(lambda row:
                                   [row['Frequency']]
                                   + [row['Lambda']]
                                   + [row['S1_Real[RE]']]
                                   + [row['S1_Imaginary[Im]']], axis=1).to_dict()

        # contains the indexes of the picked frequency along with the other information
        print('Length of working dataframe : ', len(wrk_dict))

        # ----------------------------------------------------------------------------------------------------------------
        """
        - For each 'lambda' value in the working dataframe, we pick the index and extract the adjacent data points
        for the picked index, which inturn contains the adjacent frequencies data.
        - Pass the extracted real and imaginarly values as tuple to plot the circle which passes through these 3 points.
        - With circle plot we also have circle 'Radius' and 'Center' information.
        """
        # ----------------------------------------------------------------------------------------------------------------
        lambda_list = []
        radii_list = []

        for k, item in wrk_dict.items():
            info = self.extract_ten_neighbours(k)
            frequency_value = item[0]
            lambda_value = item[1]
            frq_name = f'Frequency = {frequency_value}'
            lambda_name = f'Lambda = {lambda_value}'
            print(
                '<=========================================================================================>')
            print('\nExtracted Info : {0}'.format(info))
            print(f'{frq_name} , {lambda_name}')

            # extract the coordinate tuples from the info
            # coord = []
            self.x = []
            self.y = []
            freq_pts = []
            for count, val in enumerate(info):
                # 0 : Freq and 1 : Lambda from 2: Coordinates
                coord = tuple(val[2:])
                # extract the first element of the tuple into list
                self.x.append(coord[0])
                # extract the second element of the tuple into a list
                self.y.append(coord[1])
                freq_pts.append(val[0])
            self.x = r_[self.x]  # convert to numpy array
            self.y = r_[self.y]
            print('-------------------------------------------------------------')
            # print("\nExtracted Coordinates {0}: \n|-> Adj. Frequencies : {1}".format(coord, freq_pts))
            print(
                "\nExtracted X-Points {0}: \nExtracted Y-Points {1}: \nAdj. Frequencies : {2}".format(self.x, self.y, freq_pts))

            # Now pass this extracted 3 x-y points and calculate the center and radius
            # h,k,radius = np.float64(three_point_circle(*coord))
            h, k, radius = np.float64(
                self.least_square_circle_extraction(self.x, self.y))
            print("center(h : {0}, k : {1}), radius : {2}".format(
                h, k, radius))

            lambda_list.append(lambda_value)
            radii_list.append(radius)

            # Scatter plot
            fig.add_trace(go.Scatter(x=[lambda_value], y=[radius],
                                     mode='markers',
                                     name=f'{lambda_name} , radius = {radius}'))
        # Line plot
        fig.add_trace(go.Scatter(x=lambda_list, y=radii_list, mode='lines'))

        # Set figure size
        fig.update_layout(title=f'Lambda Vs Radius (Frequency = {frq})',
                          xaxis=dict(
                              range=[7, 11],
                              tickmode='linear',
                              dtick=0.5),
                          yaxis=dict(
                              range=[0.3, 0.8],
                              tickmode='linear',
                              dtick=0.1),
                          template=large_rockwell_template,
                          width=1000, height=600,
                          showlegend=True)

        # Change grid color and axis colors
        fig.update_xaxes(gridcolor='black', griddash='dot')
        fig.update_yaxes(gridcolor='black', griddash='dot')

        path = '{0}/results/' + f'LambdasVsRadius{frq}.html'
        fig.write_html(os.path.realpath(path.format(utils.get_dir_path())))

import os
import utils
import matplotlib
import random
from scipy import optimize
from numpy import *
from data_visualizer import DataVisualizer
import plotly.graph_objects as go
import numpy as np
import math
import pandas as pd

"""
Circle approximation using Least Squares Method.
Ref: https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
"""


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
        """ 
        calculate the algebraic distance between the data points 
        and the mean circle centered at c=(xc, yc) 
        """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()

    # Return circle center and radius
    """
    Computes given x and y points and approximates the best possible
    circle with the given x and y points
    Parameters:
    ------------
    x - List
    list of X-coordinates in the which the circle should pass
    y - List
    list of Y-coordinates in the which the circle should pass
    Returns:
    ------------
    Circle center(Xc, Yc) and Radius (R)
    """

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
        # residu_2 = sum((Ri_2 - R_2)**2)
        # print("center : {0}, radius : {1}".format(center_2, R_2))
        return xc_2, yc_2, R_2

    # extract the adjacent points of the picked frequency
    def extract_neighbours(self, idx):
        orgDFs = self.orgDF_list[:]  # copy list into new variable

        val = orgDFs[idx]
        extract_xy = val[-2:]  # type list
        xy_tuple = tuple(extract_xy)

        left_slice = idx - 5  # start of range
        left_slice = min(max(0, left_slice), len(
            orgDFs) - 10)  # account for edges
        right_slice = left_slice + 10  # end of range

        return orgDFs[left_slice:right_slice], xy_tuple

    def process_circle_extraction(self):
        # initialize the plot figure
        fig = go.Figure()
        fig1 = go.Figure()
        fig2 = go.Figure()
        fig3 = go.Figure()

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

        print('Length original dataframe : ', len(self.orgDF_list))

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
        frq = 30.0
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
        - For each 'lambda' value in the working dataframe, we pick the index and 
          extract the adjacent data points for the picked index, which inturn contains 
          the adjacent frequencies data.
        - Pass the extracted real and imaginarly values as tuple to plot the circle 
          which passes through these 3 points.
        - With circle plot we also have circle 'Radius' and 'Center' information.
        """
        # ----------------------------------------------------------------------------------------------------------------
        lambda_list = []
        radii_list = []
        phase_list = []
        h_list = []
        k_list = []
        xy_list = []

        for k, item in wrk_dict.items():
            # extract the neighbours
            info, xy_tuple = self.extract_neighbours(k)
            frequency_value = item[0]
            lambda_value = item[1]
            frq_name = f'Frequency = {frequency_value}'
            lambda_name = f'Lambda = {lambda_value}'
            print(
                '\n-----------------------------------------------------------------------------------------')
            print('\n=> Extracted Info : {0} \n=> (x,y) : {1}'.format(
                info, xy_tuple))
            print(f'\n=> {frq_name} , {lambda_name}')

            # extract the coordinate tuples from the info
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
            # print('--------------------------------------------------------------------------------------')
            # print("\nExtracted Coordinates {0}: \n|-> Adj. Frequencies : {1}".format(coord, freq_pts))
            print(
                "\n=> Extracted X-Points : {0} \n=> Extracted Y-Points : {1}" +
                "\n=> Adj. Frequencies : {2}".format(self.x, self.y, freq_pts))

            # Now pass this extracted 3 x-y points and calculate the center and radius
            # h,k,radius = np.float64(three_point_circle(*coord))
            h, k, radius = np.float64(
                self.least_square_circle_extraction(self.x, self.y))
            print("\n=> center(h : {0}, k : {1}), radius : {2}".format(
                h, k, radius))

            # Calculate the angle
            # angle = atan2(y2-y1, x2-x1) where (x1,y1) is center and (x2, y2) is point on circle
            # ref : https://math.stackexchange.com/questions/94379/calculating-angle-in-circle
            angle_rad = math.atan2(xy_tuple[1] - k, xy_tuple[0] - h)
            if angle_rad < 0:
                angle_rad = (angle_rad + (2*math.pi))
            else:
                pass
            print("\n=> Angle in radians : {}".format(angle_rad))

            lambda_list.append(lambda_value)
            radii_list.append(radius)
            phase_list.append(angle_rad)
            h_list.append(h)
            k_list.append(k)
            xy_list.append(xy_tuple)  # contains list of xy tuples

            # Scatter plot
            fig.add_trace(go.Scatter(x=[lambda_value], y=[radius],
                                     mode='markers',
                                     name=f'{lambda_name} , radius = {radius}'))

            fig1.add_trace(go.Scatter(
                x=[lambda_value],
                y=[angle_rad],
                mode='markers',
                name=f'{lambda_name} , angle = {angle_rad}'))

            fig2.add_trace(go.Scatter(
                x=[lambda_value],
                y=[h],
                mode='markers',
                name=f'{lambda_name} , h = {h}'))

            fig3.add_trace(go.Scatter(
                x=[lambda_value],
                y=[k],
                mode='markers',
                name=f'{lambda_name} , k = {k}'))

        # Line plot
        fig.add_trace(go.Scatter(x=lambda_list, y=radii_list, mode='lines'))
        fig1.add_trace(go.Scatter(x=lambda_list, y=phase_list, mode='lines'))
        fig2.add_trace(go.Scatter(x=lambda_list, y=h_list, mode='lines'))
        fig3.add_trace(go.Scatter(x=lambda_list, y=k_list, mode='lines'))

        # Set figure size
        fig.update_layout(title=f'Radius vs Lambda (Frequency = {frq})',
                          xaxis=dict(
                              range=[7, 11],
                              tickmode='linear',
                              dtick=0.5),
                          yaxis=dict(
                              range=[0.3, 1.0],
                              tickmode='linear',
                              dtick=0.1),
                          template=large_rockwell_template,
                          width=1000, height=600,
                          showlegend=True)

        # Change grid color and x&y axis colors
        fig.update_xaxes(gridcolor='black', griddash='dot')
        fig.update_yaxes(gridcolor='black', griddash='dot')

        plt = '{0}/results/' + f'Radius_Vs_Lambda{frq}.html'
        fig.write_html(os.path.realpath(plt.format(utils.get_dir_path())))

        # -----------------------------------------------------------------------------
        # Set figure1 size
        fig1.update_layout(title=f'Phase vs Lambda (Frequency = {frq})',
                           template=large_rockwell_template,
                           showlegend=True)
        fig1.update_xaxes(gridcolor='black', griddash='dot')
        fig1.update_yaxes(gridcolor='black', griddash='dot')
        plt1 = '{0}/results/' + f'Phase_Vs_Lambda{frq}.html'
        fig1.write_html(os.path.realpath(plt1.format(utils.get_dir_path())))

        # -----------------------------------------------------------------------------
        # Set figure2 size
        fig2.update_layout(title=f'X-coord vs Lambda (Frequency = {frq})',
                           xaxis=dict(
                               range=[7, 11],
                               tickmode='linear',
                               dtick=1.0),
                           yaxis=dict(
                               range=[0.3, 0.6],
                               tickmode='linear',
                               dtick=0.1),
                           template=large_rockwell_template,
                           showlegend=True)
        fig2.update_xaxes(gridcolor='black', griddash='dot')
        fig2.update_yaxes(gridcolor='black', griddash='dot')
        plt2 = '{0}/results/' + f'X-Coord_Vs_Lambda{frq}.html'
        fig2.write_html(os.path.realpath(plt2.format(utils.get_dir_path())))

        # -----------------------------------------------------------------------------
        # Set figure size
        fig3.update_layout(title=f'y-coord vs Lambda (Frequency = {frq})',
                           xaxis=dict(
                               range=[7, 11],
                               tickmode='linear',
                               dtick=1.0),
                           yaxis=dict(
                               range=[-0.4, -0.1],
                               tickmode='linear',
                               dtick=0.1),
                           template=large_rockwell_template,
                           showlegend=True)
        fig3.update_xaxes(gridcolor='black', griddash='dot')
        fig3.update_yaxes(gridcolor='black', griddash='dot')
        plt3 = '{0}/results/' + f'Y-Coord_Vs_Lambda{frq}.html'
        fig3.write_html(os.path.realpath(plt3.format(utils.get_dir_path())))

        # create dataframe with the desired for regression modeling
        self._df = pd.DataFrame(list(zip(lambda_list, radii_list, phase_list, h_list, k_list, xy_list)),
                                columns=['Lambda', 'Radius', 'Angle', 'x_center', 'y_center', 'xy_coord'])

    def _get_dataframe(self):
        return self._df

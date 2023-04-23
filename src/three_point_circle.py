'''
Three Points Circle : Initialize version

Implemented by Monica Dasi
Developed for Master's Thesis
Topic : Surrogate Modeling For Freequency Domain Simulation Data
Semester : SoSe 2023
Department of Computer Science (Fb2), Frankfurt University of Applied Sciences
'''

import logging
import math
import os
import time 
from humanfriendly import format_timespan
import numpy as np
import pandas as pd

from sympy import Point, Circle
import matplotlib, random
from data_parser import DataParser
import plotly.graph_objects as go
from utils import Utils

log = logging.getLogger(__name__)

class ThreePointCircle:

    def __init__(self) -> None:
        self._plt_dir = Utils().get_plots_dir_path()
        self._df_list = []
        self._plot = Utils()._draw_plots()
    #-------------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------------

    # Return circle center and radius
    def three_point_circle(self, t1,t2,t3):
        log.debug("\n three_point_circle() : \n tuple1 : {0},\n tuple2 : {1},\n tuple3 : {2}\n".format(t1,t2,t3))
        c2 = Circle(Point(t1), Point(t2), Point(t3))
        h = c2.center[0]
        k = c2.center[1]
        r = c2.radius.evalf()
        return h,k,r
    #-------------------------------------------------------------------------------

    # extract the adjacent points of the picked frequency
    def extract_neighbours(self, idx):
        res = []
        if idx < 15: # skip the 1st index : 0 as it does't have 2 adjacent neighbours
            pass
        elif idx > (len(self.orgDF_list) - 15): # skip the last indx too.:
            pass
        else:
            res = [self.orgDF_list[idx - 15], self.orgDF_list[idx], self.orgDF_list[idx + 15]]
        return res, self.orgDF_list[idx]
    
    def process_circle_extraction(self):
        # create a copy of the dataset
        frf_df_cp = DataParser().get_freq_data()
        frf_df6 = frf_df_cp.copy(deep=True)

        # Pick the each row of these columns containing frequency, lambda, S1 real, S1 imaginary
        # and form a list of each row containing these column values.
        self.orgDF_list = frf_df6.apply(lambda row:
                                        [row['Frequency']]
                                        + [row['Lambda']]
                                        + [row['S1_Real[RE]']]
                                        + [row['S1_Imaginary[Im]']], axis=1).to_list()

        log.info(f'Length original dataframe : {len(self.orgDF_list)}')

        # ----------------------------------------------------------------------------------------------------------------

        # pick a random frequency from the data , 'n' denotes : number of samples to be picked
        # In this case we just pick one frequency and create a model for that.
        df_elements = frf_df6.sample(n=3)
        # # contains the row of the picked frequency
        # log.info(f'<--------- Randomly Picked Row --------->\n{df_elements}')

        start_time = time.monotonic()
        log.info('------------------- START --------------------------')

        # ----------------------------------------------------------------------------------------------------------------

        # extract the freq. to a list
        frq_list = df_elements['Frequency'].to_list()

        #frq_list = frf_df6['Frequency'].to_list()
        _frqs = list(dict.fromkeys(frq_list))  # remove the duplicate freq.
        # log.info(f'Frequency List Size : {len(_frqs)}')

        log.info(f'Frequency List Size : {len(_frqs)}')
        for i, frq in enumerate(_frqs, start=1):
            # extracted frequency.
            log.info(
                'Picked Frequency : {0} , Processing freq. {1}'.format(frq, i))
            # For the picked frequency extract all the matching rows
            if not frf_df6.empty:
                result_df = frf_df6[frf_df6['Frequency'].isin([frq])]
                result_df.to_csv(os.path.realpath(
                    '{0}/extracted_frequency_df.csv'.format(Utils().get_data_dir_path())))
                # result_df.to_csv('extracted_frequency_df.csv')
            else:
                log.info('Dataframe is empty,cannot continue...!!')

            # (Convert to dictionary) working dict after the data extraction based on the picked frequency
            wrk_dict = result_df.apply(lambda row:
                                       [row['Frequency']]
                                       + [row['Lambda']]
                                       + [row['S1_Real[RE]']]
                                       + [row['S1_Imaginary[Im]']], axis=1).to_dict()

            # contains the indexes of the picked frequency along with the other information
            # log.info(f'Length of working dataframe : {len(wrk_dict)}')

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
            _freqs = []
            lambda_list = []
            radii_list = []
            phase_list = []
            h_list = []
            k_list = []
            xy_list = []
            # -----------------------------------------------------------------------------------------------------------------
            if self._plot:
                # Initialize the plot figures
                fig = go.Figure()
                fig1 = go.Figure()
                fig2 = go.Figure()
                fig3 = go.Figure()
                # set the graph template in plotly
                large_rockwell_template = dict(
                    layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
            # -----------------------------------------------------------------------------------------------------------------
            for k, item in wrk_dict.items():
                # extract the neighbours
                info, xy_tuple = self.extract_neighbours(k)
                frequency_value = item[0]
                lambda_value = item[1]
                frq_name = f'Frequency = {frequency_value}'
                lambda_name = f'Lambda = {lambda_value}'
                log.debug(
                    '\n-----------------------------------------------------------------------------------------')
                log.debug('\n=> Extracted Info : {0}'.format(info))
                log.debug(f'\n=> {frq_name} , {lambda_name}')

                # extract the coordinate tuples from the info
                coord = []
                freq_pts = []
                for _, val in enumerate(info):
                    coord.append(tuple(val[2:])) # 0 : Freq and 1 : Lambda from 2: Coordinates
                    freq_pts.append(item[0])

                # print('--------------------------------------------------------------------------------------')
                log.debug("\nExtracted Coordinates {0}: \n|-> Adj. Frequencies : {1}".format(coord, freq_pts))

                # Now pass this extracted 3 x-y points and calculate the center and radius
                h,k,radius = self.three_point_circle(*coord)
                # log.debug("\n=> center(h : {0}, k : {1}), radius : {2}".format(
                #     h, k, radius))

                # Calculate the angle
                # angle = atan2(y2-y1, x2-x1) where (x1,y1) is center and (x2, y2) is point on circle
                # ref : https://math.stackexchange.com/questions/94379/calculating-angle-in-circle
                angle_rad = math.atan2(xy_tuple[1] - k, xy_tuple[0] - h)
                if angle_rad < 0:
                    angle_rad = (angle_rad + (2*math.pi))
                else:
                    pass
                # log.debug("\n=> Angle in radians : {0}".format(angle_rad))
                log.debug("(lambda:{0}) => radius : {1}, angle : {2}, center(h : {3}, k : {4})".format(
                    lambda_value, radius, h, k, angle_rad))

                lambda_list.append(lambda_value)
                radii_list.append(radius)
                phase_list.append(angle_rad)
                h_list.append(h)
                k_list.append(k)
                xy_list.append(xy_tuple)  # contains list of xy tuples
                _freqs.append(frq)

                if self._plot:
                    # Set plot axes properties
                    fig.update_xaxes(zeroline=False)
                    fig.update_yaxes(zeroline=False)

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
            # endfor
            if self._plot:
                # Line plots
                fig.add_trace(go.Scatter(
                    x=lambda_list, y=radii_list, mode='lines'))
                fig1.add_trace(go.Scatter(
                    x=lambda_list, y=phase_list, mode='lines'))
                fig2.add_trace(go.Scatter(
                    x=lambda_list, y=h_list, mode='lines'))
                fig3.add_trace(go.Scatter(
                    x=lambda_list, y=k_list, mode='lines'))

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

                # Change grid color and x and y axis colors
                fig.update_xaxes(gridcolor='black', griddash='dot')
                fig.update_yaxes(gridcolor='black', griddash='dot')

                # plt = '{0}/'+ f'Radius_Vs_Lambda_{frq}.html'
                plt = '{0}/' + f'Radius_Vs_Lambda_{frq}.png'
                f_name = os.path.realpath(plt.format(self._plt_dir))
                # fig.write_html(f_name)
                fig.write_image(f_name, engine="orca",
                                format="png", width=800, height=400)

                # -----------------------------------------------------------------------------
                # Set figure1 size
                fig1.update_layout(title=f'Phase vs Lambda (Frequency = {frq})',
                                   template=large_rockwell_template,
                                   showlegend=True)
                fig1.update_xaxes(gridcolor='black', griddash='dot')
                fig1.update_yaxes(gridcolor='black', griddash='dot')
                # plt1 = '{0}/'+ f'Phase_Vs_Lambda_{frq}.html'
                plt1 = '{0}/' + f'Phase_Vs_Lambda_{frq}.png'
                # fig1.write_html(os.path.realpath(plt1.format(self._plt_dir)))
                f1_name = os.path.realpath(plt1.format(self._plt_dir))
                fig.write_image(f1_name, engine="orca",
                                format="png", width=800, height=400)

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
                plt2 = '{0}/' + f'X-Coord_Vs_Lambda_{frq}.png'
                f2_name = os.path.realpath(plt2.format(self._plt_dir))
                fig2.write_image(f2_name, engine="orca",
                                 format="png", width=800, height=400)

                # -----------------------------------------------------------------------------
                # Set figure3 size
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
                plt3 = '{0}/' + f'Y-Coord_Vs_Lambda_{frq}.png'
                fig3.write_image(os.path.realpath(plt3.format(
                    self._plt_dir)), engine="orca", format="png", width=800, height=400)
            # create dataframe with the desired attributes for regression modeling
            _df = pd.DataFrame(list(zip(_freqs, lambda_list, radii_list, phase_list, h_list, k_list, xy_list)),
                               columns=['Frequency', 'Lambda', 'Radius', 'Angle', 'x_center', 'y_center', 'coordinates'])
            self._df_list.append(_df)
        end_time = time.monotonic()
        log.info(f'Finished in {format_timespan(end_time - start_time)}')
        log.info('-------------------- END ---------------------------')

    def _get_df_list(self):
        log.debug(f'LeastSquareCircle(): _get_df_list = {len(self._df_list)}')
        return self._df_list
import os
import random
import utils
import json
import plotly.graph_objects as go

from pathlib import Path
from data_parser import DataParser
from singleton import Singleton


class DataVisualizer(metaclass=Singleton):

    def __init__(self):
        dataparser = DataParser()
        with open('config.json', 'r') as f:
            config = json.load(f)
        if config['parse_data'] == "True":
            print("Processing the frequency data ...")
            dataparser.parse_freq_data()
            dataparser.process_data()
            self.frf_df = dataparser.create_data_frame()
        else:
            print("data is available...fast forwarding...")
            self.frf_df = dataparser.get_freq_data()

    def get_freq_dataframe(self):
        return self.frf_df

    def plot_frf_data(self):
        plt_name = os.path.realpath(
            '{0}/results/frf_plot.html'.format(utils.get_dir_path()))
        if Path(plt_name).is_file():
            print("Frequency Plot exists, skipping...")
        else:
            print("plotting frequency curves...")
            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))

            frf_df2 = self.frf_df.copy(deep=True)

            # Create traces
            fig = go.Figure()

            # extract the 'Lambda'values from dataframe
            lm_lst = frf_df2.apply(lambda row: row['Lambda'], axis=1).tolist()
            # print(type(lm_lst))

            # remove the duplicate values from the list
            lm_lst = list(dict.fromkeys(lm_lst))
            # print(len(lm_lst))

            for b in lm_lst:
                if not frf_df2.empty:
                    rslt_frf_df = frf_df2[frf_df2['Lambda'].isin([b])]
                    x_val = rslt_frf_df['Frequency'].to_list()
                    y_val = rslt_frf_df['Magnitude'].to_list()
                    lm_name = f'Lambda = {b}'
                    fig.add_trace(go.Scatter(x=x_val, y=y_val,
                                             mode='lines',
                                             name=lm_name))
                    rslt_lst = rslt_frf_df.index
                    frf_df2 = frf_df2.drop(rslt_lst, axis=0)
                else:
                    continue

            fig.update_layout(title='S-Parameters [Magnitude]',
                              xaxis=dict(
                                  tickmode='linear',
                                  tick0=25,
                                  dtick=1),
                              yaxis=dict(
                                  tickmode='linear',
                                  dtick=0.1),
                              template=large_rockwell_template)

            # Change grid color and axis colors
            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            fig.update_traces(line=dict(width=1))

            fig.write_html(plt_name)

    def draw_mag_parameter_plot(self):
        plt_name = os.path.realpath(
            '{0}/results/lambda_mag_plot.html'.format(utils.get_dir_path()))
        if Path(plt_name).is_file():
            print("Magnitude Vs S-Paramater Plot exists, skipping...")
        else:
            print("Plotting Magnitude Vs S-Parameter...")
            # set the graph template in plotly
            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))

            frf_df1 = self.frf_df.copy(deep=True)
            fq_lst = frf_df1.apply(
                lambda row: row['Frequency'], axis=1).tolist()

            # print(type(fq_lst))
            # print(len(fq_lst))

            # remove the duplicate values from the list
            # this is done to extract the values of lambda and magnitude at a given frequency
            fq_lst = list(dict.fromkeys(fq_lst))
            # print(len(fq_lst))

            # Create traces
            fig = go.Figure()

            for a in range(25):
                b = random.choice(fq_lst)
                if not frf_df1.empty:
                    rslt_frf_df = frf_df1[frf_df1['Frequency'].isin([b])]

                    x_val = rslt_frf_df['Lambda'].to_list()
                    y_val = rslt_frf_df['Magnitude'].to_list()
                    fq_name = f'Frequency = {b}'

                    # Add the x and y values to trace
                    fig.add_trace(go.Scatter(x=x_val, y=y_val,
                                             mode='lines',
                                             name=fq_name))
                else:
                    continue

            fig.update_layout(
                xaxis=dict(
                    tickmode='linear',
                    tick0=25,
                    dtick=0.5
                ),
                yaxis=dict(
                    tickmode='linear',
                    dtick=0.1
                ),
                template=large_rockwell_template
            )
            # Change grid color and axis colors
            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            fig.update_traces(line=dict(width=1))
            # fig.show()
            fig.write_html(plt_name)

    def draw_polar_plot(self):
        plt_name = os.path.realpath(
            '{0}/results/polar_plot.html'.format(utils.get_dir_path()))
        if Path(plt_name).is_file():
            print("Polar Plot exists, skipping...")
        else:
            print("Plotting polar plot")
            frf_df3 = self.frf_df.copy(deep=True)
            frf_df3 = frf_df3.sort_values('Lambda')

            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))

            # Create traces
            fig = go.Figure()

            # extract the 'Lambda' values from dataframe
            lm_lst = frf_df3.apply(lambda row: row['Lambda'], axis=1).tolist()
            # remove the duplicate lambda values
            lm_lst = list(dict.fromkeys(lm_lst))
            # print(len(lm_lst))

            for b in lm_lst:
                if not frf_df3.empty:
                    rslt_frf_df = frf_df3[frf_df3['Lambda'].isin([b])]
                    r_val = rslt_frf_df['Magnitude'].to_list()
                    theta_list = rslt_frf_df['Phase(Deg)'].to_list()

                    lab = f'Lambda = {b}'
                    fig.add_trace(go.Scatterpolar(
                        r=r_val,
                        theta=theta_list,
                        mode='markers',
                        name=lab))

                    rslt_lst = rslt_frf_df.index
                    frf_df3 = frf_df3.drop(rslt_lst, axis=0)
                else:
                    continue

            fig.update_layout(title=' S-Parameters [Polar Plot]',
                              polar=dict(
                                  radialaxis=dict(range=[0, 1], dtick=0.2),
                                  angularaxis=dict(
                                      thetaunit="degrees", dtick=30)
                              ), template=large_rockwell_template
                              )

            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            fig.update_traces(marker=dict(size=5))
            # save the plot to the local disk and open image from the disk
            fig.write_html(plt_name)

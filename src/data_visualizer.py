import matplotlib
import file_parser

class DataVisualizer:

    def __init__(self):
        file_parser.parseDataFile()
        #self.parse_data()


    def parse_data(self):
        print(self.df_csv)

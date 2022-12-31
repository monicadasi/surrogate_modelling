import os

def get_dir_path():
    return os.path.normpath(os.getcwd() + os.sep + os.pardir)

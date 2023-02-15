import logging

FORMAT_STR = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FMT = '%Y-%m-%d %H:%M:%S'

class BaseLogger:
    def __init__(self, log_level) -> None:
        root = logging.getLogger()
        root.setLevel(logging.NOTSET)

        # set up logging to console
        console = logging.StreamHandler()
        console.setLevel(log_level)

        # set a format which is simpler for console use
        formatter = logging.Formatter(FORMAT_STR, datefmt=DATE_FMT)
        console.setFormatter(formatter)

        # add the handler to the root logger
        root.addHandler(console)
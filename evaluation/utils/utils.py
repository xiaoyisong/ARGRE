import time
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(logdir, logname = None, mode='a+'):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(message)s') 
    logfile = f'{logdir}/run_{time.strftime("%Y%m%d_%H%M%S")}.log' if logname is None else f"{logdir}/{logname}.log"
    print(logfile)
    file_handler = RotatingFileHandler(
        logfile,  
        mode=mode,  
        backupCount=5  
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
import logging

def getmylogger(name):
    file_formatter = logging.Formatter('%(processName)s:%(asctime)s.%(msecs)03d~%(levelname)s~%(message)s~module:%(module)s~function:%(module)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(processName)s:%(asctime)s.%(msecs)03d %(levelname)s -- %(message)s', datefmt='%H:%M:%S')
    
    file_handler = logging.FileHandler("./log/logfile.log")
    file_handler.setLevel(logging.WARN)
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    
    return logger
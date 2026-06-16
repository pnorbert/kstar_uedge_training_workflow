import logging


# ------------------------------------------------------------------------------
def init_logger(level):

    LOG_FMT = '%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s - %(message)s'

    logger = logging.getLogger()
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(LOG_FMT))
    logger.addHandler(sh)


# ------------------------------------------------------------------------------
def print_dict(_, indent=0):
    assert isinstance(_, dict)
    assert isinstance(indent, int)
    istr = '   '.join(['' for _ in range(indent)])
    for k,v in _.items():
        if isinstance(v, dict):
            print(f'{istr}[{k}] ==>')
            print_dict(v, indent+2)
        else:
            print(f'{istr}[{k}] ==> [{v}]')

# ------------------------------------------------------------------------------

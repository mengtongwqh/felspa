import logging
import time
import numpy as np

# setup the logger to directly output to stdout
logger = logging.getLogger("felspanalytic")
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


def int_to_string(number, n_digits=0):
    '''
    Convert integer to string with n_digits of padded zeros.
    '''
    if n_digits:
        numstr = str(number)
        n_existing_digits = len(numstr)
        if (n_existing_digits > n_digits):
            raise RuntimeError(
                "Digit length is too short for the provided number.")
        return (n_digits - n_existing_digits) * '0' + numstr
    else:
        return str(number)


def current_function_name(level=1):
    import sys
    return sys._getframe(level).f_code.co_name


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer:
    '''
    Code execution timer
    '''

    def __init__(self, explanatory_string=None):
        self._start_time = None
        self._explanatory_string = explanatory_string

    def __enter__(self):
        self._label_text = current_function_name(2)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        '''
        Start new timer
        The function name will also be recorded for reporting
        '''
        if self._start_time is not None:
            raise RuntimeError(
                "Timer is running. Use stop() to stop the previous run.")
        self._label_text = current_function_name(2)
        self._start_time = time.perf_counter()

    def stop(self):
        '''
        Stop the timer and return the elapsed time.
        Also log the time to the global logger at loglevel "TIMER"
        '''
        if self._start_time is None:
            raise RuntimeError("Timer is not yet started")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        logger.info(f"{BColors.OKBLUE}{self._explanatory_string } in function "
                    f"[{self._label_text}] elpased {elapsed_time} "
                    f"seconds{BColors.ENDC}")
        return elapsed_time
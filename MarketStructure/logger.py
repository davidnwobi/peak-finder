import functools
import logging


def setup_logger(logger_name, log_file):
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


flow_logger = setup_logger('flow_logger', 'flow.log')


def log_function_call(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Log the function name and its arguments
            # flow_logger.info(f"Calling function '{func.__name__}' with arguments: {args}, {kwargs}")
            # Call the wrapped function
            result = func(self, *args, **kwargs)
            # Log the result (you can customize this part based on your needs)
            # flow_logger.info(f"Function '{func.__name__}' returned: {result}")
            return result

        except Exception as e:
            # Log the exception
            # logging.error(f"Function '{func.__name__}' raised an exception: {e}")
            raise  #

    return wrapper

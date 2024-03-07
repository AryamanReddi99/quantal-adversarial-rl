import timeit

class Timer:
    """
    Timing class which can be used to log the time taken for a function to run.
    """
    def __init__(self, logger=None):
        self.logger = logger

    def time_function(self, function_to_time, *args, **kwargs):
        start = timeit.default_timer()
        res = function_to_time(*args, **kwargs)
        end = timeit.default_timer()
        runtime = end - start

        time_str = f"{int(runtime//3600)} hours {int((runtime%3600)//60)} minutes {int((runtime%3600)%60)} seconds"
        msg = f"{function_to_time.__name__}() took {time_str}"
        print(msg)
        if self.logger:
            self.logger.info(msg)
        return res

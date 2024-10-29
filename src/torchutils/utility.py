import time

def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """
    

    def dump_decorator(func):
        def dump_inner(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            print(
                "%s elapsed time: %s ms"
                % (customized_msg if customized_msg else func.__qualname__, round((end - start) * 1000, 2))
            )
            return res

        return dump_inner

    return dump_decorator
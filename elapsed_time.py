import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', kw['func_id'])
            kw['log_time'][name] = int((te - ts) * 1000)
        return result
    return timed

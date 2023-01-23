import inspect
import time
import weakref
import signal 
import os

from settings import settings


# Taken from: https://stackoverflow.com/a/31270973/8132000
class Singleton(type):
    def __init__(self, *args):
        super(Singleton, self).__init__(*args)
        self._instance = super(Singleton, self).__call__()

    def __call__(self, *args, **kwargs):
        return self._instance


class Logger(metaclass=Singleton):
    silenced = settings.logger.quiet
    n_logs_to_keep = 5

    def __init__(self) -> None:

        # Create lig file, failing if the file already exists
        log_file_path = f"{settings.logger.out_dir}/{time.time_ns()}.log"
        open(log_file_path, mode="x") 
        self.file = open(log_file_path, mode="w")

        # Register cleanup function to be called when this is destroyed
        weakref.finalize(self, self.cleanup)

        # Remove old log files
        files = list(map(lambda f: os.path.join(settings.logger.out_dir, f), os.listdir(settings.logger.out_dir)))
        if len(files) > self.n_logs_to_keep:
            files.sort(key=lambda p: os.path.getmtime(p))
            for f in files[:-self.n_logs_to_keep]:
                os.remove(f)

    def log(self, *args, quiet=False, **kwargs):
        if not self.silenced and not quiet:
            print(*args, **kwargs)

        # caller_frame = inspect.currentframe().f_back
        # file_path, file_line, *_ = inspect.getframeinfo(caller_frame)
        # file_name = file_path.split("\\")[-1]
        # print(f"{file_name}:{file_line}", end="\t", file=self.file)

        print(*args, **kwargs, file=self.file)

    def cleanup(self):
        try:
            self.file.flush()
        except:
            pass

        self.file.close()


log = Logger().log
assert Logger() is Logger()


# Stop logger printing stuff after keyboard interrupt
_original_sigint_handler = signal.getsignal(signal.SIGINT)
def sigint_handler(signal, frame):
    Logger().silenced = True
    _original_sigint_handler(signal, frame)
signal.signal(signal.SIGINT, sigint_handler)

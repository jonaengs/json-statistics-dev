import inspect
import time
import types
import weakref
import signal 
import os

from munch import Munch

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
    n_logs_to_keep = 10

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

        self.log_settings()

    def log(self, *args, quiet=False, **kwargs):
        if not self.silenced and not quiet:
            print(*args, **kwargs)

        # caller_frame = inspect.currentframe().f_back
        # file_path, file_line, *_ = inspect.getframeinfo(caller_frame)
        # file_name = file_path.split("\\")[-1]
        # print(f"{file_name}:{file_line}", end="\t", file=self.file)

        print(*args, **kwargs, file=self.file)

    def log_settings(self, quiet=True):
        def rec_log(root, indent):
            for key, child in root.items():
                if type(child) == Munch:
                    self.log('\t'*indent + str(key) + ":", quiet=quiet)
                    rec_log(child, indent+1)
                else:
                    if type(child) == types.MethodType:
                        self.log('\t'*indent + f"{key}:\t<bound method {child.__name__}>", quiet=quiet)
                    elif type(child) in (list, tuple):
                        self.log('\t'*indent + f"{key}:\t[{', '.join(f'{e}' for e in child)}]", quiet=quiet)
                        # self.log('\t'*indent + f"{key}:\t{child.__class__(map(str, child))}", quiet=quiet)
                    else:
                        self.log('\t'*indent + f"{key}:\t{child}", quiet=quiet)

        self.log("\nPROGRAM SETTINGS:", quiet=quiet)
        rec_log(settings, 0)
        self.log(quiet=quiet)

    def cleanup(self):
        try:
            self.file.flush()
        except:
            pass

        self.file.close()


logger = Logger()
log = logger.log

# Stop logger printing stuff after keyboard interrupt
_original_sigint_handler = signal.getsignal(signal.SIGINT)
def sigint_handler(signal, frame):
    Logger().silenced = True
    _original_sigint_handler(signal, frame)
signal.signal(signal.SIGINT, sigint_handler)

if __name__ == '__main__':
    assert Logger() is Logger()  # Assert that the singleton metaclass works as expected

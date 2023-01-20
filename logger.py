import time
import weakref
import signal 

from settings import settings


# Taken from: https://stackoverflow.com/a/31270973/8132000
class Singleton(type):
    def __init__(self, *args):
        super(Singleton, self).__init__(*args)
        self._instance = super(Singleton, self).__call__()

    def __call__(self, *args, **kwargs):
        return self._instance


class Logger(metaclass=Singleton):
    quiet = settings.logger.quiet
    

    def __init__(self) -> None:
        log_file_path = f"logs/{time.time_ns()}.log"

        open(log_file_path, mode="x") # Fail if file already exists
        self.file = open(log_file_path, mode="w")

        # Register cleanup function to be called when object is destroyed
        weakref.finalize(self, self.cleanup)

    def log(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

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
    Logger().quiet = True
    _original_sigint_handler(signal, frame)
signal.signal(signal.SIGINT, sigint_handler)
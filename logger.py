import inspect
import shutil
import sys
import time
import traceback
import types
import weakref
import signal 
import os
from datetime import datetime as dt

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
    silenced = settings.logger.silenced
    n_logs_to_keep = 5

    def __init__(self) -> None:
        # Create lig file, failing if the file already exists
        if settings.logger.store_output:
            # file_name = f"{sys.argv[0][:-3]}_{dt.now().strftime('%H.%M.%S')}.log"
            # file_name = f"{dt.now().strftime('%H.%M.%S')}_{sys.argv[0][:-3]}.log"
            # file_name = f"{sys.argv[0][:-3]}_{time.time_ns()}.log"
            file_name = f"{str(time.time_ns())[:-6]}_{sys.argv[0][:-3]}.log"
            log_file_path = os.path.join(settings.logger.out_dir, file_name)
            open(log_file_path, mode="x")
            self.file = open(log_file_path, mode="w")
        else:
            self.file = open(os.devnull, "w")

        # Register cleanup function to be called when this is destroyed
        weakref.finalize(self, self.cleanup)

        # Move old log files to a subdirectory
        old_logs_path = os.path.join(settings.logger.out_dir, "old/")
        if not os.path.exists(old_logs_path):
            os.makedirs(old_logs_path)

        # files = [
        #     p
        #     for f in os.listdir(settings.logger.out_dir)
        #     if os.path.isfile(p := os.path.join(settings.logger.out_dir, f))
        # ]
        files = list(
           filter(os.path.isfile,
              map(lambda f: os.path.join(settings.logger.out_dir, f), os.listdir(settings.logger.out_dir))
        ))
        if len(files) > self.n_logs_to_keep:
            # Sort files in ascending order, meaning from oldest to most recent
            files.sort(key=lambda p: os.path.getmtime(p))
            for f in files[:-(self.n_logs_to_keep+1)]:
                shutil.move(f, os.path.join(old_logs_path, os.path.basename(f)))

        self.log(dt.now(), quiet=True)
        # self.log_settings()


        # Set custom except hook, so exceptions are logged as well

        def custom_excepthook(ex_type, ex_val, ex_traceback):
            with self.block_log(quiet=True) as block:
                block.big_header(f"Exception: {ex_type}")
                ex_str = "".join(traceback.format_exception(ex_type, ex_val, ex_traceback))
                block.log(ex_str)

            return sys.__excepthook__(ex_type, ex_val, ex_traceback)

        sys.excepthook = custom_excepthook

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
                if isinstance(child, Munch):
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

        with self.block_log(quiet=quiet) as block:
            block.small_header("PROGRAM SETTINGS:")
            rec_log(settings, 0)

        # self.log("\nPROGRAM SETTINGS:", quiet=quiet)
        # self.log(quiet=quiet)

    def cleanup(self):
        self._flush()
        self.file.close()

    def _flush(self):
        try:
            self.file.flush()
        except:
            pass

    def block_log(_self, quiet=False, **kwargs):
        _log = lambda *a, **k: _self.log(*a, **k, quiet=quiet)
        class LogBlock:
            def __init__(self, **kwargs):
                self.print_footer = False

            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                _self._flush()
                if self.print_footer:
                    _log("="*20 + "\n")
                else:
                    _log()

            def big_header(self, header):
                self.print_footer = True
                _log("\n" + "="*20 + f"\n{header}\n" + "="*20)
                # _log(f"\n{'='*20}\n{header}\n{'='*20}")

            def small_header(self, header):
                _log(f"\n{header}")

            def log(*args, **kwargs):
                _log(*args, **kwargs)

        return LogBlock(**kwargs)


logger = Logger()
log = logger.log

# Stop logger printing stuff after keyboard interrupt
_original_sigint_handler = signal.getsignal(signal.SIGINT)
def sigint_handler(signal, frame):
    logger.silenced = True
    logger._flush()
    _original_sigint_handler(signal, frame)
signal.signal(signal.SIGINT, sigint_handler)

if __name__ == '__main__':
    assert Logger() is Logger()  # Assert that the singleton metaclass works as expected

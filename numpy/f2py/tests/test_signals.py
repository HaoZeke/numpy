import contextlib
import signal

import pytest

import numpy as np

from . import util


@contextlib.contextmanager
def sigint_handler(handler):
    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.slow
class TestCheckSignals(util.F2PyTest):
    # gh-20148
    sources = [util.getpath("tests", "src", "signals", "check_signals.f90")]

    @pytest.mark.parametrize("name", ["spin", "spin_threadsafe"])
    def test_interrupt_stops_loop(self, name):
        iters = np.zeros(1, dtype=np.intc)
        with sigint_handler(signal.default_int_handler):
            with pytest.raises(KeyboardInterrupt):
                getattr(self.module, name)(signal.SIGINT, 5, iters)
        assert iters[0] == 1

    def test_handler_without_exception(self):
        received = []
        iters = np.zeros(1, dtype=np.intc)
        with sigint_handler(lambda signum, frame: received.append(signum)):
            self.module.spin(signal.SIGINT, 5, iters)
        assert received == [signal.SIGINT]
        assert iters[0] == 5

    def test_pending_exception_reported_until_return(self):
        npending = np.zeros(1, dtype=np.intc)
        with sigint_handler(signal.default_int_handler):
            with pytest.raises(KeyboardInterrupt):
                self.module.count_pending(signal.SIGINT, 3, npending)
        assert npending[0] == 3

import subprocess
import sys
import textwrap

import numpy as np


def test_installed_f2py_stays_in_the_star_import():
    assert "f2py" in np.__all__
    namespace = {}
    exec("from numpy import *", namespace)
    assert namespace["f2py"].__name__ == "numpy.f2py"


def test_star_import_when_f2py_is_absent():
    # gh-28016: `from numpy import *` must not import a missing f2py.
    script = textwrap.dedent(
        """
        import importlib.util
        import sys
        real_find_spec = importlib.util.find_spec

        def find_spec(name, package=None):
            if name == "numpy.f2py" or name.startswith("numpy.f2py."):
                return None
            return real_find_spec(name, package)

        importlib.util.find_spec = find_spec

        class Block:
            def find_spec(self, fullname, path, target=None):
                if fullname == "numpy.f2py" or fullname.startswith("numpy.f2py."):
                    raise ModuleNotFoundError(fullname)
                return None

        import numpy as np
        namespace = {}
        exec("from numpy import *", namespace)
        assert "f2py" not in np.__all__
        assert "f2py" not in namespace
        sys.meta_path.insert(0, Block())
        missing = False
        try:
            np.f2py
        except AttributeError:
            missing = True
        assert missing
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

#!/bin/bash
# s390x probe for gh-11831: live t1 behaviour + generated return path
set -ex
python -m pytest --pyargs numpy.f2py.tests.test_return_character \
  -v --no-header -rX --runxfail || true
mkdir -p /tmp/probe && cd /tmp/probe
printf '       function t1(value)\n         character*1 value\n         character*1 t1\n         t1 = value\n       end\n' > tc.f
python -m numpy.f2py -c tc.f -m tcprobe --build-dir bld > f2py.log 2>&1 || tail -20 f2py.log
python -c 'import tcprobe; print("PROBE t1:", repr(tcprobe.t1(b"B")))'
echo '---- generated t1 return path ----'
grep -n -B2 -A6 'Py_BuildValue\|return_value' bld/tcprobemodule.c | head -70
echo '---- fortran wrapper ----'
cat bld/tcprobe-f2pywrappers.f
cd /numpy
python -m pytest --pyargs numpy.f2py.tests.test_kind -v --no-header

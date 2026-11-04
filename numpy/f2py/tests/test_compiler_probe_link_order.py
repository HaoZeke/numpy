from types import SimpleNamespace

import pytest

from numpy.f2py import _compiler_context, crackfortran


@pytest.mark.parametrize('link_args', [
    ('/opt/target/lib/libFortranRuntime.a',),
    ('-Wl,--as-needed', '-L/opt/target/lib', '-lFortranRuntime'),
    ('-Wl,--start-group', '-lfirst', '-lsecond', '-Wl,--end-group'),
])
def test_probe_object_precedes_libraries_without_changing_flag_order(
    monkeypatch, link_args,
):
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

    context = _compiler_context.CompilerContext(
        command=('compiler-cache', 'target-fc', '--target=aarch64-linux-gnu'),
        compile_args=('-O2',), link_args=link_args,
        runner=('qemu-aarch64', '-L', '/target'), selected=True,
    )
    monkeypatch.setattr(crackfortran.subprocess, 'run', run)
    crackfortran._selected_real_kind_program.cache_clear()
    try:
        owner, executable = crackfortran._selected_real_kind_program(
            context, (8, 8, 8))
        object_file = str(executable.with_suffix('.o'))
        assert commands[0] == [
            *context.command, '-O2', '-c',
            str(executable.with_suffix('.f90')), '-o', object_file,
        ]
        assert commands[1] == [
            *context.command, object_file, *link_args, '-o', str(executable),
        ]
        assert len(commands) == 2
        assert owner.name == str(executable.parent)
    finally:
        crackfortran._selected_real_kind_program.cache_clear()

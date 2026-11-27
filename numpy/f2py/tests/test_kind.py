import platform
import sys

import pytest

from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
    _selected_real_kind_func as selected_real_kind,
)

from . import util

IS_PPC_OR_AIX = platform.machine().lower().startswith("ppc") or platform.system() == 'AIX'

@pytest.mark.slow
class TestKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "foo.f90")]

    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1,
                        reason="Fails for 32 bit machines")
    def test_int(self):
        """Test `int` kind_func for integers up to 10**40."""
        selectedintkind = self.module.selectedintkind

        for i in range(40):
            assert selectedintkind(i) == selected_int_kind(
                i
            ), f"selectedintkind({i}): expected {selected_int_kind(i)!r} but got {selectedintkind(i)!r}"

    def test_real(self):
        """
        Test (processor-dependent) `real` kind_func for real numbers
        of up to 31 digits precision (extended/quadruple).
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"

    @pytest.mark.xfail(IS_PPC_OR_AIX,
                       reason="Some PowerPC may not support full IEEE 754 precision")
    def test_quad_precision(self):
        """
        Test kind_func for quadruple precision [`real(16)`] of 32+ digits .
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32, 40):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"


class TestSelectedRealKindBelow4:
    @pytest.mark.parametrize('fc', [
        'flang',
        'flang-new',
        'flang-new -O2',
        'ccache flang',
        'x86_64-linux-gnu-flang',
        r'C:\Program Files\LLVM\bin\flang-new.exe',
        r'"C:\Program Files\LLVM\bin\flang-new.exe"',
        r'C:\Program Files\LLVM\bin\flang-new.exe -O2',
    ])
    def test_flang_keeps_kind_4_and_warns(self, monkeypatch, capsys, fc):
        monkeypatch.setenv('FC', fc)
        for p in range(4):
            assert selected_real_kind(p) == 4
        err = capsys.readouterr().err
        assert err.count('kind 2') == 4
        assert 'stays 4' in err
        assert selected_real_kind(4) == 4
        assert capsys.readouterr().err == ''

    def test_range_and_radix_keep_the_table(self, monkeypatch, capsys):
        monkeypatch.setenv('FC', 'flang-new')
        assert selected_real_kind(3, 37) == 4
        assert selected_real_kind(3, 200) == 4
        assert selected_real_kind(3, 0, 2) == 4
        assert capsys.readouterr().err == ''

    @pytest.mark.parametrize('fc', [
        'gfortran',
        'aarch64-linux-gnu-gfortran',
        r'C:\flang\bin\gfortran.exe',
        r'C:\Program Files\gfortran\bin\gfortran.exe',
        'false',
    ])
    def test_other_compilers_stay_4(self, monkeypatch, capsys, fc):
        monkeypatch.setenv('FC', fc)
        assert selected_real_kind(0) == 4
        assert selected_real_kind(3) == 4
        assert capsys.readouterr().err == ''

    def test_missing_compiler_keeps_kind_4(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('FC', str(tmp_path / 'missing-fc'))
        assert selected_real_kind(0) == 4
        assert capsys.readouterr().err == ''

    def test_no_compiler_keeps_kind_4(self, monkeypatch, tmp_path, capsys):
        monkeypatch.delenv('FC', raising=False)
        monkeypatch.setenv('PATH', str(tmp_path))
        assert selected_real_kind(0) == 4
        assert selected_real_kind(3) == 4
        assert capsys.readouterr().err == ''

    def test_fc_is_not_executed(self, monkeypatch, tmp_path, capsys):
        marker = tmp_path / 'ran'
        fc = tmp_path / 'flang-new'
        fc.write_text(f'#!/bin/sh\ntouch {marker}\nexit 1\n')
        fc.chmod(0o755)
        monkeypatch.setenv('FC', str(fc))
        assert selected_real_kind(0) == 4
        assert not marker.exists()
        assert 'kind 2' in capsys.readouterr().err

import os
import platform
import re
import shlex
import subprocess
import sys
import textwrap
from collections import namedtuple
from pathlib import Path

import pytest

from numpy.f2py.f2py2e import get_newer_options, main as f2pycli
from numpy.testing._private.utils import NOGIL_BUILD

from . import util

#######################
# F2PY Test utilities #
######################

# Tests for CLI commands which call meson will fail if no compilers are present, these are to be skipped

def compiler_check_f2pycli():
    if not util.has_fortran_compiler():
        pytest.skip("CLI command needs a Fortran compiler")
    else:
        f2pycli()

#########################
# CLI utils and classes #
#########################


PPaths = namedtuple("PPaths", "finp, f90inp, pyf, wrap77, wrap90, cmodf")


def get_io_paths(fname_inp, mname="untitled"):
    """Takes in a temporary file for testing and returns the expected output and input paths

    Here expected output is essentially one of any of the possible generated
    files.

    ..note::

         Since this does not actually run f2py, none of these are guaranteed to
         exist, and module names are typically incorrect

    Parameters
    ----------
    fname_inp : str
                The input filename
    mname : str, optional
                The name of the module, untitled by default

    Returns
    -------
    genp : NamedTuple PPaths
            The possible paths which are generated, not all of which exist
    """
    bpath = Path(fname_inp)
    return PPaths(
        finp=bpath.with_suffix(".f"),
        f90inp=bpath.with_suffix(".f90"),
        pyf=bpath.with_suffix(".pyf"),
        wrap77=bpath.with_name(f"{mname}-f2pywrappers.f"),
        wrap90=bpath.with_name(f"{mname}-f2pywrappers2.f90"),
        cmodf=bpath.with_name(f"{mname}module.c"),
    )


################
# CLI Fixtures #
################


@pytest.fixture(scope="session")
def hello_world_f90(tmpdir_factory):
    """Generates a single f90 file for testing"""
    fdat = util.getpath("tests", "src", "cli", "hiworld.f90").read_text()
    fn = tmpdir_factory.getbasetemp() / "hello.f90"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def gh23598_warn(tmpdir_factory):
    """F90 file for testing warnings in gh23598"""
    fdat = util.getpath("tests", "src", "crackfortran", "gh23598Warn.f90").read_text()
    fn = tmpdir_factory.getbasetemp() / "gh23598Warn.f90"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def gh22819_cli(tmpdir_factory):
    """F90 file for testing disallowed CLI arguments in ghff819"""
    fdat = util.getpath("tests", "src", "cli", "gh_22819.pyf").read_text()
    fn = tmpdir_factory.getbasetemp() / "gh_22819.pyf"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def gh25654_mix(tmp_path_factory):
    """Signature + Fortran sources for gh-25654 mixed-input warning."""
    fdat = util.getpath("tests", "src", "cli", "gh25654.f").read_text()
    pdat = util.getpath("tests", "src", "cli", "gh25654.pyf").read_text()
    base = tmp_path_factory.mktemp("gh25654")
    fpath = base / "gh25654.f"
    ppath = base / "gh25654.pyf"
    fpath.write_text(fdat, encoding="ascii")
    ppath.write_text(pdat, encoding="ascii")
    return ppath, fpath


@pytest.fixture(scope="session")
def hello_world_f77(tmpdir_factory):
    """Generates a single f77 file for testing"""
    fdat = util.getpath("tests", "src", "cli", "hi77.f").read_text()
    fn = tmpdir_factory.getbasetemp() / "hello.f"
    fn.write_text(fdat, encoding="ascii")
    return fn


@pytest.fixture(scope="session")
def retreal_f77(tmpdir_factory):
    """Generates a single f77 file for testing"""
    fdat = util.getpath("tests", "src", "return_real", "foo77.f").read_text()
    fn = tmpdir_factory.getbasetemp() / "foo.f"
    fn.write_text(fdat, encoding="ascii")
    return fn

@pytest.fixture(scope="session")
def f2cmap_f90(tmpdir_factory):
    """Generates a single f90 file for testing"""
    fdat = util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90").read_text()
    f2cmap = util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap").read_text()
    fn = tmpdir_factory.getbasetemp() / "f2cmap.f90"
    fmap = tmpdir_factory.getbasetemp() / "mapfile"
    fn.write_text(fdat, encoding="ascii")
    fmap.write_text(f2cmap, encoding="ascii")
    return fn

#########
# Tests #
#########

def test_gh22819_cli(capfd, gh22819_cli, monkeypatch):
    """Check that module names are handled correctly
    gh-22819
    Essentially, the -m name cannot be used to import the module, so the module
    named in the .pyf needs to be used instead

    CLI :: -m and a .pyf file
    """
    ipath = Path(gh22819_cli)
    monkeypatch.setattr(sys, "argv", f"f2py -m blah {ipath}".split())
    with util.switchdir(ipath.parent):
        f2pycli()
        gen_paths = [item.name for item in ipath.parent.rglob("*") if item.is_file()]
        assert "blahmodule.c" not in gen_paths  # shouldn't be generated
        assert "blah-f2pywrappers.f" not in gen_paths
        assert "test_22819-f2pywrappers.f" in gen_paths
        assert "test_22819module.c" in gen_paths


def test_gh22819_many_pyf(capfd, gh22819_cli, monkeypatch):
    """Only one .pyf file allowed
    gh-22819
    CLI :: .pyf files
    """
    ipath = Path(gh22819_cli)
    monkeypatch.setattr(sys, "argv", f"f2py -m blah {ipath} hello.pyf".split())
    with util.switchdir(ipath.parent):
        with pytest.raises(ValueError, match="Only one .pyf file per call"):
            f2pycli()


<<<<<<< HEAD
=======
def test_gh25654_pyf_fortran_mix_warn(gh25654_mix):
    """Warn when .pyf and Fortran sources are passed together.

    gh-25654
    CLI :: -m with mixed signature and Fortran inputs
    Runs in a subprocess: in-process f2py on a .pyf leaks do-lower=0
    into crackfortran.options and poisons later parses on this worker.
    """
    ppath, fpath = gh25654_mix
    res = subprocess.run(
        [sys.executable, "-m", "numpy.f2py", "-m", "fibx", "--lower",
         str(ppath), str(fpath)],
        cwd=ppath.parent, check=True, capture_output=True, text=True,
    )
    out = res.stdout + res.stderr
    assert "Warning:" in out
    assert ".pyf directives are ignored" in out
    assert str(ppath.name) in out
    assert str(fpath.name) in out

>>>>>>> origin/fix/f2py-pyf-mix-warning
def test_gh23598_warn(capfd, gh23598_warn, monkeypatch):
    foutl = get_io_paths(gh23598_warn, mname="test")
    ipath = foutl.f90inp
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test'.split())

    with util.switchdir(ipath.parent):
        f2pycli()  # Generate files
        wrapper = foutl.wrap90.read_text()
        assert "intproductf2pywrap, intpr" not in wrapper


def test_gh25777_allocatable_accessor_intp(tmp_path_factory, monkeypatch):
    """Module allocatable accessors must match f2py_init_func (npy_intp dims).

    gh-25777: generated accessors must declare dimensions as ``npy_intp*``,
    matching ``f2py_init_func`` / ``f2py_set_data_func`` (an ``int*`` there
    is -Wincompatible-pointer-types and truncates on LP64). The init
    wrapper must also carry no dead ``use mod, only : <allocatable>``; the
    getdims helper has its own use-association.
    """
    fdat = util.getpath(
        "tests", "src", "modules", "module_data_docstring.f90"
    ).read_text()
    fn = tmp_path_factory.mktemp("gh25777") / "moddata.f90"
    fn.write_text(fdat, encoding="ascii")
    mname = "moddata"
    foutl = get_io_paths(fn, mname=mname)
    monkeypatch.setattr(sys, "argv", f"f2py -m {mname} {fn}".split())

    with util.switchdir(fn.parent):
        f2pycli()
        cmod = foutl.cmodf.read_text()
        wrapper = foutl.wrap90.read_text()

    # Accessor / setup signatures use npy_intp, not plain int (LP64 width).
    assert (
        "void (*b)(int*,npy_intp*,void(*)(char*,npy_intp*),int*)" in cmod
    )
    assert "void (*b)(int*,int*,void(*)(char*,int*),int*)" not in cmod
    assert "void (*)(int*,npy_intp*,void(*)(char*,npy_intp*),int*)" in cmod

    # getdims binds the allocatable; f2pyinit must not also `use` it unused.
    assert "use mod, only: d => b" in wrapper
    assert "use mod, only : b" not in wrapper
    # Non-allocatable module vars still need an explicit use.
    assert "use mod, only : i" in wrapper


def test_gen_pyf(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file is generated via the CLI
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    opath = Path(hello_world_f90).stem + ".pyf"
    monkeypatch.setattr(sys, "argv", f'f2py -h {opath} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()  # Generate wrappers
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out
        assert Path(f'{opath}').exists()


def test_gen_pyf_stdout(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file can be dumped to stdout
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -h stdout {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out
        assert "function hi() ! in " in out


def test_gen_pyf_no_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the CLI refuses to overwrite signature files
    CLI :: -h without --overwrite-signature
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -h faker.pyf {ipath}'.split())

    with util.switchdir(ipath.parent):
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        with pytest.raises(SystemExit):
            f2pycli()  # Refuse to overwrite
            _, err = capfd.readouterr()
            assert "Use --overwrite-signature to overwrite" in err


@pytest.mark.skipif(sys.version_info <= (3, 12), reason="Python 3.12 required")
@pytest.mark.slow
def test_untitled_cli(capfd, hello_world_f90, monkeypatch):
    """Check that modules are named correctly

    CLI :: defaults
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f"f2py --backend meson -c {ipath}".split())
    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        out, _ = capfd.readouterr()
        assert "untitledmodule.c" in out

@pytest.mark.slow
def test_no_distutils_backend(capfd, hello_world_f90, monkeypatch):
    """Check that distutils backend and related options fail
    CLI :: --fcompiler --help-link --backend distutils
    """
    MNAME = "hi"
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    monkeypatch.setattr(
        sys, "argv", f"f2py {ipath} -c --fcompiler=gfortran -m {MNAME}".split()
    )
    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        out, _ = capfd.readouterr()
        assert "--fcompiler cannot be used with meson" in out

    monkeypatch.setattr(
        sys, "argv", ["f2py", "--help-link"]
    )
    with pytest.raises(SystemExit):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Unknown option --help-link" in out

    monkeypatch.setattr(
        sys, "argv", ["f2py", "--backend", "distutils"]
    )
    with pytest.raises(SystemExit):
        compiler_check_f2pycli()
        f2pycli()
        out, _ = capfd.readouterr()
        assert "'distutils' backend was removed" in out

@pytest.mark.xfail
def test_f2py_skip(capfd, retreal_f77, monkeypatch):
    """Tests that functions can be skipped
    CLI :: skip:
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    remaining = "td s0"
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test skip: {toskip}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not found the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in remaining.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_f2py_only(capfd, retreal_f77, monkeypatch):
    """Test that functions can be kept by only:
    CLI :: only:
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    tokeep = "td s0"
    monkeypatch.setattr(
        sys, "argv",
        f'f2py {ipath} -m test only: {tokeep}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_file_processing_switch(capfd, hello_world_f90, retreal_f77,
                                monkeypatch):
    """Tests that it is possible to return to file processing mode
    CLI :: :
    BUG: numpy-gh #20520
    """
    foutl = get_io_paths(retreal_f77, mname="test")
    ipath = foutl.finp
    toskip = "t0 t4 t8 sd s8 s4"
    ipath2 = Path(hello_world_f90)
    tokeep = "td s0 hi"  # hi is in ipath2
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -m {mname} only: {tokeep} : {ipath2}'.split(
        ),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, err = capfd.readouterr()
        for skey in toskip.split():
            assert (
                f'buildmodule: Could not find the body of interfaced routine "{skey}". Skipping.'
                in err)
        for rkey in tokeep.split():
            assert f'Constructing wrapper function "{rkey}"' in out


def test_mod_gen_f77(capfd, hello_world_f90, monkeypatch):
    """Checks the generation of files based on a module name
    CLI :: -m
    """
    MNAME = "hi"
    foutl = get_io_paths(hello_world_f90, mname=MNAME)
    ipath = foutl.f90inp
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m {MNAME}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()

    # Always generate C module
    assert Path.exists(foutl.cmodf)
    # File contains a function, check for F77 wrappers
    assert Path.exists(foutl.wrap77)


def test_mod_gen_gh25263(capfd, hello_world_f77, monkeypatch):
    """Check that pyf files are correctly generated with module structure
    CLI :: -m <name> -h pyf_file
    BUG: numpy-gh #20520
    """
    MNAME = "hi"
    foutl = get_io_paths(hello_world_f77, mname=MNAME)
    ipath = foutl.finp
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m {MNAME} -h hi.pyf'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        with Path('hi.pyf').open() as hipyf:
            pyfdat = hipyf.read()
            assert "python module hi" in pyfdat


def test_lower_cmod(capfd, hello_world_f77, monkeypatch):
    """Lowers cases by flag or when -h is present

    CLI :: --[no-]lower
    """
    foutl = get_io_paths(hello_world_f77, mname="test")
    ipath = foutl.finp
    capshi = re.compile(r"HI\(\)")
    capslo = re.compile(r"hi\(\)")
    # Case I: --lower is passed
    monkeypatch.setattr(sys, "argv", f'f2py {ipath} -m test --lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None
    # Case II: --no-lower is passed
    monkeypatch.setattr(sys, "argv",
                        f'f2py {ipath} -m test --no-lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None


def test_lower_sig(capfd, hello_world_f77, monkeypatch):
    """Lowers cases in signature files by flag or when -h is present

    CLI :: --[no-]lower -h
    """
    foutl = get_io_paths(hello_world_f77, mname="test")
    ipath = foutl.finp
    # Signature files
    capshi = re.compile(r"Block: HI")
    capslo = re.compile(r"Block: hi")
    # Case I: --lower is implied by -h
    # TODO: Clean up to prevent passing --overwrite-signature
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature'.split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None

    # Case II: --no-lower overrides -h
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature --no-lower'
        .split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None


def test_build_dir(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --build-dir
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    odir = "tttmp"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --build-dir {odir}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert f"Wrote C/API module \"{mname}\"" in out


def test_overwrite(capfd, hello_world_f90, monkeypatch):
    """Ensures that the build directory can be specified

    CLI :: --overwrite-signature
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(
        sys, "argv",
        f'f2py -h faker.pyf {ipath} --overwrite-signature'.split())

    with util.switchdir(ipath.parent):
        Path("faker.pyf").write_text("Fake news", encoding="ascii")
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Saving signatures to file" in out


def test_latexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --latex-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --latex-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" in out
        with Path(f"{mname}module.tex").open() as otex:
            assert "\\documentclass" in otex.read()


def test_nolatexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-latex-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-latex-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" not in out

@pytest.mark.slow
def test_latex_doc_gh30268(tmp_path):

    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler found")

    fsource = textwrap.dedent("""
        subroutine foo
        end
    """)

    fpath = tmp_path / "test_latex.f90"
    with open(fpath, "w") as f:
        f.write(fsource)

    cmd = [sys.executable, "-m", "numpy.f2py", "-c", str(fpath), "-m", "test_latex", "--latex-doc"]
    subprocess.check_call(cmd, cwd=tmp_path)


def test_shortlatex(capfd, hello_world_f90, monkeypatch):
    """Ensures that truncated documentation is written out

    TODO: Test to ensure this has no effect without --latex-doc
    CLI :: --latex-doc --short-latex
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} --latex-doc --short-latex'.split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Documentation is saved to file" in out
        with Path(f"./{mname}module.tex").open() as otex:
            assert "\\documentclass" not in otex.read()


def test_restdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that RsT documentation is written out

    CLI :: --rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --rest-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "ReST Documentation is saved to file" in out
        with Path(f"./{mname}module.rest").open() as orst:
            assert r".. -*- rest -*-" in orst.read()


def test_norestexdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that TeX documentation is written out

    CLI :: --no-rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-rest-doc'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "ReST Documentation is saved to file" not in out


def test_debugcapi(capfd, hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers are written

    CLI :: --debug-capi
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --debug-capi'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f"./{mname}module.c").open() as ocmod:
            assert r"#define DEBUGCFUNCS" in ocmod.read()


@pytest.mark.skip(reason="Consistently fails on CI; noisy so skip not xfail.")
def test_debugcapi_bld(hello_world_f90, monkeypatch):
    """Ensures that debugging wrappers work

    CLI :: --debug-capi -c
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} -c --debug-capi'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split(f"{sys.executable} -c \"import blah; blah.hi()\"")
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        eerr = textwrap.dedent("""\
debug-capi:Python C/API function blah.hi()
debug-capi:float hi=:output,hidden,scalar
debug-capi:hi=0
debug-capi:Fortran subroutine `f2pywraphi(&hi)'
debug-capi:hi=0
debug-capi:Building return value.
debug-capi:Python C/API function blah.hi: successful.
debug-capi:Freeing memory.
        """)
        assert rout.stdout == eout
        assert rout.stderr == eerr


def test_wrapfunc_def(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 are included by default

    CLI :: --[no]-wrap-functions
    """
    # Implied
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv", f'f2py -m {mname} {ipath}'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
    out, _ = capfd.readouterr()
    assert r"Fortran 77 wrappers are saved to" in out

    # Explicit
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" in out


def test_nowrapfunc(capfd, hello_world_f90, monkeypatch):
    """Ensures that fortran subroutine wrappers for F77 can be disabled

    CLI :: --no-wrap-functions
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(sys, "argv",
                        f'f2py -m {mname} {ipath} --no-wrap-functions'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" not in out


def test_inclheader(capfd, hello_world_f90, monkeypatch):
    """Add to the include directories

    CLI :: -include
    TODO: Document this in the help string
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    monkeypatch.setattr(
        sys,
        "argv",
        f'f2py -m {mname} {ipath} -include<stdbool.h> -include<stdio.h> '.
        split(),
    )

    with util.switchdir(ipath.parent):
        f2pycli()
        with Path(f"./{mname}module.c").open() as ocmod:
            ocmr = ocmod.read()
            assert "#include <stdbool.h>" in ocmr
            assert "#include <stdio.h>" in ocmr

@pytest.mark.skipif((platform.system() != 'Linux'), reason='Compiler required')
@pytest.mark.slow
def test_cli_obj(capfd, hello_world_f90, monkeypatch):
    """Ensures that the extra object can be specified when using meson backend
    """
    ipath = Path(hello_world_f90)
    mname = "blah"
    odir = "tttmp"
    obj = "extra.o"
    monkeypatch.setattr(sys, "argv",
                        f'f2py --backend meson --build-dir {odir} -m {mname} -c {obj} {ipath}'.split())

    with util.switchdir(ipath.parent):
        Path(obj).touch()
        compiler_check_f2pycli()
        with Path(f"{odir}/meson.build").open() as mesonbuild:
            mbld = mesonbuild.read()
            assert "objects:" in mbld
            assert f"'''{obj}'''" in mbld


def _meson_template(deps, fortran_args=None):
    """Build a MesonTemplate for pure-Python meson.build inspection."""
    from numpy.f2py._backends._meson import MesonTemplate
    return MesonTemplate(
        "testmod",
        [Path("src.f90")],
        deps,
        [],
        [],
        [],
        [],
        [],
        fortran_args or [],
        "release",
        sys.executable,
    )


def test_meson_openmp_dep_gh27163():
    """--dep openmp must apply to C and Fortran and set the Fortran linker.

    CLI :: --dep openmp
    Regression for gh-27163 / gh-30804: bare dependency('openmp') only
    supplies C OpenMP flags, so Fortran sources miss -fopenmp/-qopenmp and
    the C linker leaves omp_* undefined (Intel ifx).
    """
    src = _meson_template(["openmp"]).generate_meson_build()
    assert "dependency('openmp', language: 'c')" in src
    assert "dependency('openmp', language: 'fortran')" in src
    # bare form defaults to C-only and must not appear
    assert re.search(r"dependency\('openmp'\)", src) is None
    assert "link_language: 'fortran'" in src


def test_meson_openmp_dep_case_and_mixed():
    """OpenMP special-case is case-insensitive and coexists with other deps."""
    src = _meson_template(["lapack", "OpenMP", "blas"]).generate_meson_build()
    assert "dependency('lapack')" in src
    assert "dependency('blas')" in src
    assert "dependency('openmp', language: 'c')" in src
    assert "dependency('openmp', language: 'fortran')" in src
    assert "link_language: 'fortran'" in src


def test_meson_non_openmp_dep_no_link_language():
    """Non-OpenMP --dep values keep the previous single-language form."""
    src = _meson_template(["lapack"]).generate_meson_build()
    assert "dependency('lapack')" in src
    assert "link_language" not in src
    # the interpreter path may contain 'openmp' (e.g. a venv name), so
    # only assert on the dependency call
    assert "dependency('openmp'" not in src


def test_inclpath(monkeypatch):
    """Add to the include directories

    CLI :: --include-paths
    """
    monkeypatch.setattr(os, "pathsep", ";")

    include_paths, ftcompat, remain = get_newer_options(
        ["--include-paths", r"C:\inc1"]
    )

    assert include_paths == [r"C:\inc1"]
    assert ftcompat is None
    assert remain == []

    include_paths, ftcompat, remain = get_newer_options(
        ["--include-paths", r"C:\inc1;D:\inc2"]
    )

    assert set(include_paths) == {r"C:\inc1", r"D:\inc2"}
    assert ftcompat is None
    assert remain == []

    include_paths, ftcompat, remain = get_newer_options(
        ["--include_paths", r"C:\inc1;D:\inc2"]
    )

    assert set(include_paths) == {r"C:\inc1", r"D:\inc2"}
    assert ftcompat is None
    assert remain == []

    include_paths, ftcompat, remain = get_newer_options(
        ["-I", r"C:\inc1;D:\inc2"]
    )

    assert include_paths == [r"C:\inc1;D:\inc2"]
    assert ftcompat is None
    assert remain == []


def test_hlink():
    """Add to the include directories

    CLI :: --help-link
    """
    # TODO: populate
    pass


def test_f2cmap(capfd, f2cmap_f90, monkeypatch):
    """Check that Fortran-to-Python KIND specs can be passed

    CLI :: --f2cmap
    """
    ipath = Path(f2cmap_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --f2cmap mapfile'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "Reading f2cmap from 'mapfile' ..." in out
        assert "Mapping \"real(kind=real32)\" to \"float\"" in out
        assert "Mapping \"real(kind=real64)\" to \"double\"" in out
        assert "Mapping \"integer(kind=int64)\" to \"long_long\"" in out
        assert "Successfully applied user defined f2cmap changes" in out


def test_quiet(capfd, hello_world_f90, monkeypatch):
    """Reduce verbosity

    CLI :: --quiet
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --quiet'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert len(out) == 0


def test_verbose(capfd, hello_world_f90, monkeypatch):
    """Increase verbosity

    CLI :: --verbose
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} --verbose'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert "analyzeline" in out


def test_version(capfd, monkeypatch):
    """Ensure version

    CLI :: -v
    """
    monkeypatch.setattr(sys, "argv", ["f2py", "-v"])
    # TODO: f2py2e should not call sys.exit() after printing the version
    with pytest.raises(SystemExit):
        f2pycli()
        out, _ = capfd.readouterr()
        import numpy as np
        assert np.__version__ == out.strip()


@pytest.mark.skip(reason="Consistently fails on CI; noisy so skip not xfail.")
def test_npdistop(hello_world_f90, monkeypatch):
    """
    CLI :: -c
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c'.split())

    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split(f"{sys.executable} -c \"import blah; blah.hi()\"")
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout


@pytest.mark.skipif((platform.system() != 'Linux') or sys.version_info <= (3, 12),
                    reason='Compiler and Python 3.12 or newer required')
@pytest.mark.slow
def test_no_freethreading_compatible(hello_world_f90, monkeypatch):
    """
    CLI :: --no-freethreading-compatible
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c --no-freethreading-compatible'.split())

    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        cmd = f"{sys.executable} -c \"import blah; blah.hi();"
        if NOGIL_BUILD:
            cmd += "import sys; assert sys._is_gil_enabled() is True\""
        else:
            cmd += "\""
        cmd_run = shlex.split(cmd)
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout
        if NOGIL_BUILD:
            assert "The global interpreter lock (GIL) has been enabled to load module 'blah'" in rout.stderr
        assert rout.returncode == 0


@pytest.mark.skipif((platform.system() != 'Linux') or sys.version_info <= (3, 12),
                    reason='Compiler and Python 3.12 or newer required')
def test_freethreading_compatible(hello_world_f90, monkeypatch):
    """
    CLI :: --freethreading_compatible
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, "argv", f'f2py -m blah {ipath} -c --freethreading-compatible'.split())

    with util.switchdir(ipath.parent):
        compiler_check_f2pycli()
        cmd = f"{sys.executable} -c \"import blah; blah.hi();"
        if NOGIL_BUILD:
            cmd += "import sys; assert sys._is_gil_enabled() is False\""
        else:
            cmd += "\""
        cmd_run = shlex.split(cmd)
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout
        if "LSAN_OPTIONS" not in os.environ:
            assert rout.stderr == ""
        assert rout.returncode == 0


def test_gh30167_c23_no_empty_prototypes(tmp_path):
    """callstatement without callprotoargument must not emit K&R ().

    Under C23, ``()`` means ``(void)``, so calls that pass arguments are
    constraint violations (gcc 15+).  Codegen-only: run f2py, inspect module.c.
    """
    pyf = tmp_path / "gh30167.pyf"
    pyf.write_text(textwrap.dedent("""\
        python module gh30167
            interface
                subroutine foo(a, b)
                    !f2py callstatement (*f2py_func)(&a, &b)
                    integer intent(in) :: a
                    integer intent(out) :: b
                end subroutine foo
            end interface
        end python module gh30167
        """), encoding="ascii")
    # subprocess: in-process f2py on a .pyf leaks do-lower=0 into
    # crackfortran.options and poisons later parses on this worker
    subprocess.run(
        [sys.executable, "-m", "numpy.f2py", str(pyf)],
        cwd=tmp_path, check=True, capture_output=True,
    )
    text = (tmp_path / "gh30167module.c").read_text(encoding="utf-8")

    # K&R empty parameter list must never appear on f2py_func or the extern.
    assert "void (*f2py_func)()" not in text
    assert re.search(r"F_FUNC\s*\(\s*foo\s*,\s*FOO\s*\)\s*\(\s*\)", text) is None
    # Derived prototype from the signature (int in/out → int*).
    assert re.search(r"void\s*\(\*f2py_func\)\s*\(\s*int\s*\*\s*,\s*int\s*\*\s*\)", text)
    assert "(*f2py_func)(&a, &b)" in text


def test_gh30167_explicit_callprotoargument(tmp_path):
    """Explicit callprotoargument is still honoured over derived types."""
    pyf = tmp_path / "gh30167_explicit.pyf"
    pyf.write_text(textwrap.dedent("""\
        python module gh30167_explicit
            interface
                subroutine foo(a, b)
                    !f2py callstatement (*f2py_func)(&a, &b)
                    !f2py callprotoargument char*, size_t
                    integer intent(in) :: a
                    integer intent(out) :: b
                end subroutine foo
            end interface
        end python module gh30167_explicit
        """), encoding="ascii")
    # subprocess: in-process f2py on a .pyf leaks do-lower=0 into
    # crackfortran.options and poisons later parses on this worker
    subprocess.run(
        [sys.executable, "-m", "numpy.f2py", str(pyf)],
        cwd=tmp_path, check=True, capture_output=True,
    )
    text = (tmp_path / "gh30167_explicitmodule.c").read_text(encoding="utf-8")

    assert "void (*f2py_func)()" not in text
    assert re.search(r"void\s*\(\*f2py_func\)\s*\(\s*char\s*\*\s*,\s*size_t\s*\)", text)


# Numpy distutils flags
# TODO: These should be tested separately

def test_npd_fcompiler():
    """
    CLI :: -c --fcompiler
    """
    # TODO: populate
    pass


def test_npd_compiler():
    """
    CLI :: -c --compiler
    """
    # TODO: populate
    pass


def test_npd_help_fcompiler():
    """
    CLI :: -c --help-fcompiler
    """
    # TODO: populate
    pass


def test_npd_f77exec():
    """
    CLI :: -c --f77exec
    """
    # TODO: populate
    pass


def test_npd_f90exec():
    """
    CLI :: -c --f90exec
    """
    # TODO: populate
    pass


def test_npd_f77flags():
    """
    CLI :: -c --f77flags
    """
    # TODO: populate
    pass


def test_npd_f90flags():
    """
    CLI :: -c --f90flags
    """
    # TODO: populate
    pass


def test_npd_opt():
    """
    CLI :: -c --opt
    """
    # TODO: populate
    pass


def test_npd_arch():
    """
    CLI :: -c --arch
    """
    # TODO: populate
    pass


def test_npd_noopt():
    """
    CLI :: -c --noopt
    """
    # TODO: populate
    pass


def test_npd_noarch():
    """
    CLI :: -c --noarch
    """
    # TODO: populate
    pass


def test_npd_debug():
    """
    CLI :: -c --debug
    """
    # TODO: populate
    pass


def test_npd_link_auto():
    """
    CLI :: -c --link-<resource>
    """
    # TODO: populate
    pass


def test_npd_lib():
    """
    CLI :: -c -L/path/to/lib/ -l<libname>
    """
    # TODO: populate
    pass


def test_npd_define():
    """
    CLI :: -D<define>
    """
    # TODO: populate
    pass


def test_npd_undefine():
    """
    CLI :: -U<name>
    """
    # TODO: populate
    pass


def test_npd_incl():
    """
    CLI :: -I/path/to/include/
    """
    # TODO: populate
    pass


def test_npd_linker():
    """
    CLI :: <filename>.o <filename>.so <filename>.a
    """
    # TODO: populate
    pass

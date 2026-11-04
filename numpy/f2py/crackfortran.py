"""
crackfortran --- read fortran (77,90) code and extract declaration information.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.


Usage of crackfortran:
======================
Command line keys: -quiet,-verbose,-fix,-f77,-f90,-show,-h <pyffilename>
                   -m <module name for f77 routines>,--ignore-contains
Functions: crackfortran, crack2fortran
The following Fortran statements/constructions are supported
(or will be if needed):
   block data,byte,call,character,common,complex,contains,data,
   dimension,double complex,double precision,end,external,function,
   implicit,integer,intent,interface,intrinsic,
   logical,module,optional,parameter,private,public,
   program,real,(sequence?),subroutine,type,use,virtual,
   include,pythonmodule
Note: 'virtual' is mapped to 'dimension'.
Note: 'implicit integer (z) static (z)' is 'implicit static (z)' (this is minor bug).
Note: code after 'contains' will be ignored until its scope ends.
Note: 'common' statement is extended: dimensions are moved to variable definitions
Note: f2py directive: <commentchar>f2py<line> is read as <line>
Note: pythonmodule is introduced to represent Python module

Usage:
  `postlist=crackfortran(files)`
  `postlist` contains declaration information read from the list of files `files`.
  `crack2fortran(postlist)` returns a fortran code to be saved to pyf-file

  `postlist` has the following structure:
 *** it is a list of dictionaries containing `blocks':
     B = {'block','body','vars','parent_block'[,'name','prefix','args','result',
          'implicit','externals','interfaced','common','sortvars',
          'commonvars','note']}
     B['block'] = 'interface' | 'function' | 'subroutine' | 'module' |
                  'program' | 'block data' | 'type' | 'pythonmodule' |
                  'abstract interface'
     B['body'] --- list containing `subblocks' with the same structure as `blocks'
     B['parent_block'] --- dictionary of a parent block:
                             C['body'][<index>]['parent_block'] is C
     B['vars'] --- dictionary of variable definitions
     B['sortvars'] --- dictionary of variable definitions sorted by dependence (independent first)
     B['name'] --- name of the block (not if B['block']=='interface')
     B['prefix'] --- prefix string (only if B['block']=='function')
     B['args'] --- list of argument names if B['block']== 'function' | 'subroutine'
     B['result'] --- name of the return value (only if B['block']=='function')
     B['implicit'] --- dictionary {'a':<variable definition>,'b':...} | None
     B['externals'] --- list of variables being external
     B['interfaced'] --- list of variables being external and defined
     B['common'] --- dictionary of common blocks (list of objects)
     B['commonvars'] --- list of variables used in common blocks (dimensions are moved to variable definitions)
     B['from'] --- string showing the 'parents' of the current block
     B['use'] --- dictionary of modules used in current block:
         {<modulename>:{['only':<0|1>],['map':{<local_name1>:<use_name1>,...}]}}
     B['note'] --- list of LaTeX comments on the block
     B['f2pyenhancements'] --- optional dictionary
          {'threadsafe':'','fortranname':<name>,
           'callstatement':<C-expr>|<multi-line block>,
           'callprotoargument':<C-expr-list>,
           'usercode':<multi-line block>|<list of multi-line blocks>,
           'pymethoddef:<multi-line block>'
           }
     B['entry'] --- dictionary {entryname:argslist,..}
     B['varnames'] --- list of variable names given in the order of reading the
                       Fortran code, useful for derived types.
     B['saved_interface'] --- a string of scanned routine signature, defines explicit interface
 *** Variable definition is a dictionary
     D = B['vars'][<variable name>] =
     {'typespec'[,'attrspec','kindselector','charselector','=','typename']}
     D['typespec'] = 'byte' | 'character' | 'complex' | 'double complex' |
                     'double precision' | 'integer' | 'logical' | 'real' | 'type'
     D['attrspec'] --- list of attributes (e.g. 'dimension(<arrayspec>)',
                       'external','intent(in|out|inout|hide|c|callback|cache|aligned4|aligned8|aligned16)',
                       'optional','required', etc)
     K = D['kindselector'] = {['*','kind']} (only if D['typespec'] =
                         'complex' | 'integer' | 'logical' | 'real' )
     C = D['charselector'] = {['*','len','kind','f2py_len']}
                             (only if D['typespec']=='character')
     D['='] --- initialization expression string
     D['typename'] --- name of the type if D['typespec']=='type'
     D['dimension'] --- list of dimension bounds
     D['intent'] --- list of intent specifications
     D['depend'] --- list of variable names on which current variable depends on
     D['check'] --- list of C-expressions; if C-expr returns zero, exception is raised
     D['note'] --- list of LaTeX comments on the variable
 *** Meaning of kind/char selectors (few examples):
     D['typespec>']*K['*']
     D['typespec'](kind=K['kind'])
     character*C['*']
     character(len=C['len'],kind=C['kind'], f2py_len=C['f2py_len'])
     (see also fortran type declaration statement formats below)

Fortran 90 type declaration statement format
============================================

F77 is subset of F90

(Main source: IBM XL Fortran 5.1 Language Reference Manual)
type declaration = <typespec> [[<attrspec>]::] <entitydecl>
<typespec> = byte                          |
             character[<charselector>]     |
             complex[<kindselector>]       |
             double complex                |
             double precision              |
             integer[<kindselector>]       |
             logical[<kindselector>]       |
             real[<kindselector>]          |
             type(<typename>)
<charselector> = * <charlen>               |
             ([len=]<len>[,[kind=]<kind>]) |
             (kind=<kind>[,len=<len>])
<kindselector> = * <intlen>                |
             ([kind=]<kind>)
<attrspec> = comma separated list of attributes.
             Only the following attributes are used in
             building up the interface:
                external
                (parameter --- affects '=' key)
                optional
                intent
             Other attributes are ignored.
<intentspec> = in | out | inout
<arrayspec> = comma separated list of dimension bounds.
<entitydecl> = <name> [[*<charlen>][(<arrayspec>)] | [(<arrayspec>)]*<charlen>]
                      [/<init_expr>/ | =<init_expr>] [,<entitydecl>]

In addition, the following attributes are used: check,depend,note

TODO:
    * Apply 'parameter' attribute (e.g. 'integer parameter :: i=2' 'real x(i)'
                                   -> 'real x(2)')
    The above may be solved by creating appropriate preprocessor program, for example.

"""
import ast
import codecs
import collections.abc
import copy
import fileinput
import functools
import itertools
import math
import os
import re
import string
import sys
from fractions import Fraction
from pathlib import Path

try:
    import charset_normalizer
except ImportError:
    charset_normalizer = None

import numpy as np

from . import __version__, symbolic

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *

f2py_version = __version__.version

# Global flags:
strictf77 = 1          # Ignore `!' comments unless line[0]=='!'
sourcecodeform = 'fix'  # 'fix','free'
quiet = 0              # Be verbose if 0 (Obsolete: not used any more)
verbose = 1            # Be quiet if 0, extra verbose if > 1.
tabchar = 4 * ' '
pyffilename = ''
f77modulename = ''
skipemptyends = 0      # for old F77 programs without 'program' statement
ignorecontains = 1
dolowercase = 1
debug = []

# Global variables
beginpattern = ''
currentfilename = ''
expectbegin = 1
f90moduleparams = {}
f90modulevars = {}
filepositiontext = ''

# Kind names from iso_fortran_env.
# f2py's kind maps already use the byte size as the kind number:
# integer 1, 2, 4 and 8, and real 4 and 8.
# The standard names the bit width and leaves the kind number to the compiler.
# real128 is left unresolved, as c_long_double is.
# Unit numbers and IOSTAT values are left to the compiler.
# The generated Fortran uses the compiler's symbols for those.


def _make_param(typespec, value):
    return {'typespec': typespec, 'attrspec': ['parameter'], '=': str(value)}


_ISO_FORTRAN_ENV_VARS = {
    'int8': _make_param('integer', 1),
    'int16': _make_param('integer', 2),
    'int32': _make_param('integer', 4),
    'int64': _make_param('integer', 8),
    'real32': _make_param('integer', 4),
    'real64': _make_param('integer', 8),
}

# The kinds of iso_c_binding are those of the companion C processor
# (F2018 18.2.2), such as c_long_double, which is 10 for gfortran on
# x86-64, so they stay names and capi_maps maps them to C types.
_INTRINSIC_MODULE_VARS = {
    'iso_fortran_env': _ISO_FORTRAN_ENV_VARS,
}
gotnextfile = 1
groupcache = None
groupcounter = 0
grouplist = {groupcounter: []}
groupname = ''
include_paths = []
neededmodule = -1
onlyfuncs = []
previous_context = None
skipblocksuntil = -1
skipfuncs = []
skipfunctions = []
usermodules = []


def reset_global_f2py_vars():
    global groupcounter, grouplist, neededmodule, expectbegin
    global skipblocksuntil, usermodules, f90modulevars, gotnextfile
    global filepositiontext, currentfilename, skipfunctions, skipfuncs
    global onlyfuncs, include_paths, previous_context, f90moduleparams
    global strictf77, sourcecodeform, quiet, verbose, tabchar, pyffilename
    global f77modulename, skipemptyends, ignorecontains, dolowercase, debug

    # flags
    strictf77 = 1
    sourcecodeform = 'fix'
    quiet = 0
    verbose = 1
    tabchar = 4 * ' '
    pyffilename = ''
    f77modulename = ''
    skipemptyends = 0
    ignorecontains = 1
    dolowercase = 1
    debug = []
    # variables
    groupcounter = 0
    grouplist = {groupcounter: []}
    neededmodule = -1
    expectbegin = 1
    skipblocksuntil = -1
    usermodules = []
    f90moduleparams = {}
    f90modulevars = {}
    gotnextfile = 1
    filepositiontext = ''
    currentfilename = ''
    skipfunctions = []
    skipfuncs = []
    onlyfuncs = []
    include_paths = []
    previous_context = None


def outmess(line, flag=1):
    global filepositiontext

    if not verbose:
        return
    if not quiet:
        if flag:
            sys.stdout.write(filepositiontext)
        sys.stdout.write(line)


defaultimplicitrules = {}
for c in "abcdefghopqrstuvwxyz$_":
    defaultimplicitrules[c] = {'typespec': 'real'}
for c in "ijklmn":
    defaultimplicitrules[c] = {'typespec': 'integer'}
badnames = {}
invbadnames = {}
for n in ['int', 'double', 'float', 'char', 'short', 'long', 'void', 'case', 'while',
          'return', 'signed', 'unsigned', 'if', 'for', 'typedef', 'sizeof', 'union',
          'struct', 'static', 'register', 'new', 'break', 'do', 'goto', 'switch',
          'continue', 'else', 'inline', 'extern', 'delete', 'const', 'auto',
          'len', 'rank', 'shape', 'index', 'slen', 'size', '_i',
          'max', 'min',
          'flen', 'fshape',
          'string', 'complex_double', 'float_double', 'stdin', 'stderr', 'stdout',
          'type', 'default']:
    badnames[n] = n + '_bn'
    invbadnames[n + '_bn'] = n


def rmbadname1(name):
    if name in badnames:
        errmess(f'rmbadname1: Replacing "{name}" with "{badnames[name]}".\n')
        return badnames[name]
    return name


def rmbadname(names):
    return [rmbadname1(_m) for _m in names]


def undo_rmbadname1(name):
    if name in invbadnames:
        errmess(f'undo_rmbadname1: Replacing "{name}" with "{invbadnames[name]}".\n')
        return invbadnames[name]
    return name


def undo_rmbadname(names):
    return [undo_rmbadname1(_m) for _m in names]


_has_f_header = re.compile(r'-\*-\s*fortran\s*-\*-', re.I).search
_has_f90_header = re.compile(r'-\*-\s*f90\s*-\*-', re.I).search
_has_fix_header = re.compile(r'-\*-\s*fix\s*-\*-', re.I).search
_free_f90_start = re.compile(r'[^c*]\s*[^\s\d\t]', re.I).match

# Extensions
COMMON_FREE_EXTENSIONS = ['.f90', '.f95', '.f03', '.f08']
COMMON_FIXED_EXTENSIONS = ['.for', '.ftn', '.f77', '.f']


def openhook(filename, mode):
    """Ensures that filename is opened with correct encoding parameter.

    This function uses charset_normalizer package, when available, for
    determining the encoding of the file to be opened. When charset_normalizer
    is not available, the function detects only UTF encodings, otherwise, ASCII
    encoding is used as fallback.
    """
    # Reads in the entire file. Robust detection of encoding.
    # Correctly handles comments or late stage unicode characters
    # gh-22871
    if charset_normalizer is not None:
        encoding = charset_normalizer.from_path(filename).best().encoding
    else:
        # hint: install charset_normalizer for correct encoding handling
        # No need to read the whole file for trying with startswith
        nbytes = min(32, os.path.getsize(filename))
        with open(filename, 'rb') as fhandle:
            raw = fhandle.read(nbytes)
            if raw.startswith(codecs.BOM_UTF8):
                encoding = 'UTF-8-SIG'
            elif raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
                encoding = 'UTF-32'
            elif raw.startswith((codecs.BOM_LE, codecs.BOM_BE)):
                encoding = 'UTF-16'
            else:
                # Fallback, without charset_normalizer
                encoding = 'ascii'
    return open(filename, mode, encoding=encoding)


def is_free_format(fname):
    """Check if file is in free format Fortran."""
    # f90 allows both fixed and free format, assuming fixed unless
    # signs of free format are detected.
    result = False
    if Path(fname).suffix.lower() in COMMON_FREE_EXTENSIONS:
        result = True
    with openhook(fname, 'r') as fhandle:
        line = fhandle.readline()
        n = 15  # the number of non-comment lines to scan for hints
        if _has_f_header(line):
            n = 0
        elif _has_f90_header(line):
            n = 0
            result = True
        while n > 0 and line:
            if line[0] != '!' and line.strip():
                n -= 1
                if (line[0] != '\t' and _free_f90_start(line[:5])) or line[-2:-1] == '&':
                    result = True
                    break
            line = fhandle.readline()
    return result


# Read fortran (77,90) code
def readfortrancode(ffile, dowithline=show, istop=1):
    """
    Read fortran codes from files and
     1) Get rid of comments, line continuations, and empty lines; lower cases.
     2) Call dowithline(line) on every line.
     3) Recursively call itself when statement \"include '<filename>'\" is met.
    """
    global gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77
    global beginpattern, quiet, verbose, dolowercase, include_paths

    if not istop:
        saveglobals = gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77,\
            beginpattern, quiet, verbose, dolowercase
    if ffile == []:
        return
    localdolowercase = dolowercase
    # cont: set to True when the content of the last line read
    # indicates statement continuation
    cont = False
    finalline = ''
    ll = ''
    includeline = re.compile(
        r'\s*include\s*(\'|")(?P<name>[^\'"]*)(\'|")', re.I)
    cont1 = re.compile(r'(?P<line>.*)&\s*\Z')
    cont2 = re.compile(r'(\s*&|)(?P<line>.*)')
    mline_mark = re.compile(r".*?'''")
    if istop:
        dowithline('', -1)
    ll, l1 = '', ''
    spacedigits = [' '] + [str(_m) for _m in range(10)]
    filepositiontext = ''
    fin = fileinput.FileInput(ffile, openhook=openhook)
    while True:
        try:
            l = fin.readline()
        except UnicodeDecodeError as msg:
            raise Exception(
                f'readfortrancode: reading {fin.filename()}#{fin.lineno()}'
                f' failed with\n{msg}.\nIt is likely that installing charset_normalizer'
                ' package will help f2py determine the input file encoding'
                ' correctly.')
        if not l:
            break
        if fin.isfirstline():
            filepositiontext = ''
            currentfilename = fin.filename()
            gotnextfile = 1
            l1 = l
            strictf77 = 0
            sourcecodeform = 'fix'
            ext = os.path.splitext(currentfilename)[1]
            if Path(currentfilename).suffix.lower() in COMMON_FIXED_EXTENSIONS and \
                    not (_has_f90_header(l) or _has_fix_header(l)):
                strictf77 = 1
            elif is_free_format(currentfilename) and not _has_fix_header(l):
                sourcecodeform = 'free'
            if strictf77:
                beginpattern = beginpattern77
            else:
                beginpattern = beginpattern90
            outmess(f"\tReading file {currentfilename!r} "
                    f"(format:{sourcecodeform}{',strict' if strictf77 else ''})\n")

        l = l.expandtabs().replace('\xa0', ' ')
        # Get rid of newline characters
        while not l == '':
            if l[-1] not in "\n\r\f":
                break
            l = l[:-1]
        # Do not lower for directives, gh-2547, gh-27697, gh-26681
        is_f2py_directive = False
        # Unconditionally remove comments
        (l, rl) = split_by_unquoted(l, '!')
        l += ' '
        if rl[:5].lower() == '!f2py':  # f2py directive
            l, _ = split_by_unquoted(l + 4 * ' ' + rl[5:], '!')
            is_f2py_directive = True
        if l.strip() == '':  # Skip empty line
            if sourcecodeform == 'free':
                # In free form, a statement continues in the next line
                # that is not a comment line [3.3.2.4^1], lines with
                # blanks are comment lines [3.3.2.3^1]. Hence, the
                # line continuation flag must retain its state.
                pass
            else:
                # In fixed form, statement continuation is determined
                # by a non-blank character at the 6-th position. Empty
                # line indicates a start of a new statement
                # [3.3.3.3^1]. Hence, the line continuation flag must
                # be reset.
                cont = False
            continue
        if sourcecodeform == 'fix':
            if l[0] in ['*', 'c', '!', 'C', '#']:
                if l[1:5].lower() == 'f2py':  # f2py directive
                    l = '     ' + l[5:]
                    is_f2py_directive = True
                else:  # Skip comment line
                    cont = False
                    is_f2py_directive = False
                    continue
            elif strictf77:
                if len(l) > 72:
                    l = l[:72]
            if l[0] not in spacedigits:
                raise Exception('readfortrancode: Found non-(space,digit) char '
                                'in the first column.\n\tAre you sure that '
                                f'this code is in fix form?\n\tline={l!r}')

            if (not cont or strictf77) and (len(l) > 5 and not l[5] == ' '):
                # Continuation of a previous line
                ll = ll + l[6:]
                finalline = ''
                origfinalline = ''
            else:
                r = cont1.match(l)
                if r:
                    l = r.group('line')  # Continuation follows ..
                if cont:
                    ll = ll + cont2.match(l).group('line')
                    finalline = ''
                    origfinalline = ''
                else:
                    # clean up line beginning from possible digits.
                    l = '     ' + l[5:]
                    # f2py directives are already stripped by this point
                    if localdolowercase:
                        finalline = ll.lower()
                    else:
                        finalline = ll
                    origfinalline = ll
                    ll = l

        elif sourcecodeform == 'free':
            if not cont and ext == '.pyf' and mline_mark.match(l):
                l = l + '\n'
                while True:
                    lc = fin.readline()
                    if not lc:
                        errmess(
                            'Unexpected end of file when reading multiline\n')
                        break
                    l = l + lc
                    if mline_mark.match(lc):
                        break
                l = l.rstrip()
            r = cont1.match(l)
            if r:
                l = r.group('line')  # Continuation follows ..
            if cont:
                ll = ll + cont2.match(l).group('line')
                finalline = ''
                origfinalline = ''
            else:
                if localdolowercase:
                    # only skip lowering for C style constructs
                    # gh-2547, gh-27697, gh-26681, gh-28014
                    finalline = ll.lower() if not (is_f2py_directive and iscstyledirective(ll)) else ll
                else:
                    finalline = ll
                origfinalline = ll
                ll = l
            cont = (r is not None)
        else:
            raise ValueError(
                f"Flag sourcecodeform must be either 'fix' or 'free': {repr(sourcecodeform)}")
        filepositiontext = (f'Line #{fin.filelineno() - 1} '
                            f'in {currentfilename}:"{l1}"\n\t')
        m = includeline.match(origfinalline)
        if m:
            fn = m.group('name')
            if os.path.isfile(fn):
                readfortrancode(fn, dowithline=dowithline, istop=0)
            else:
                include_dirs = [
                    os.path.dirname(currentfilename)] + include_paths
                foundfile = 0
                for inc_dir in include_dirs:
                    fn1 = os.path.join(inc_dir, fn)
                    if os.path.isfile(fn1):
                        foundfile = 1
                        readfortrancode(fn1, dowithline=dowithline, istop=0)
                        break
                if not foundfile:
                    outmess(f'readfortrancode: could not find include file {fn!r} '
                            f'in {os.pathsep.join(include_dirs)}. Ignoring.\n')
        else:
            dowithline(finalline)
        l1 = ll
    # Last line should never have an f2py directive anyway
    if localdolowercase:
        finalline = ll.lower()
    else:
        finalline = ll
    origfinalline = ll
    filepositiontext = f'Line #{fin.filelineno() - 1} in {currentfilename}:"{l1}"\n\t'
    m = includeline.match(origfinalline)
    if m:
        fn = m.group('name')
        if os.path.isfile(fn):
            readfortrancode(fn, dowithline=dowithline, istop=0)
        else:
            include_dirs = [os.path.dirname(currentfilename)] + include_paths
            foundfile = 0
            for inc_dir in include_dirs:
                fn1 = os.path.join(inc_dir, fn)
                if os.path.isfile(fn1):
                    foundfile = 1
                    readfortrancode(fn1, dowithline=dowithline, istop=0)
                    break
            if not foundfile:
                outmess(f'readfortrancode: could not find include file {fn!r} '
                        f'in {os.pathsep.join(include_dirs)}. Ignoring.\n')
    else:
        dowithline(finalline)
    filepositiontext = ''
    fin.close()
    if istop:
        dowithline('', 1)
    else:
        gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77,\
            beginpattern, quiet, verbose, dolowercase = saveglobals


# Crack line
beforethisafter = r'\s*(?P<before>%s(?=\s*(\b(%s)\b)))'\
    r'\s*(?P<this>(\b(%s)\b))'\
    r'\s*(?P<after>%s)\s*\Z'
##
fortrantypes = r'character|logical|integer|real|complex|double\s*(precision\s*(complex|)|complex)|type(?=\s*\([\w\s,=(*).+-]*\))|class(?=\s*\([\w\s,=(*).+-]*\))|byte'
typespattern = re.compile(
    beforethisafter % ('', fortrantypes, fortrantypes, '.*'), re.I), 'type'
typespattern4implicit = re.compile(beforethisafter % (
    '', fortrantypes + '|static|automatic|undefined', fortrantypes + '|static|automatic|undefined', '.*'), re.I)
#
functionpattern = re.compile(beforethisafter % (
    r'([a-z]+[\w\s(=*+-/)]*?|)', 'function', 'function', '.*'), re.I), 'begin'
subroutinepattern = re.compile(beforethisafter % (
    r'[a-z\s]*?', 'subroutine', 'subroutine', '.*'), re.I), 'begin'
# modulepattern=re.compile(beforethisafter%('[a-z\s]*?','module','module','.*'),re.I),'begin'
#
groupbegins77 = r'program|block\s*data'
beginpattern77 = re.compile(
    beforethisafter % ('', groupbegins77, groupbegins77, '.*'), re.I), 'begin'
# "type is (...)" is a type guard of SELECT TYPE (F2018 11.1.11), not the
# definition of a type named is
groupbegins90 = groupbegins77 + \
    r'|module(?!\s*procedure)|python\s*module|(abstract|)\s*interface|'\
    r'type(?!\s*\()(?!\s*is\s*\()'
beginpattern90 = re.compile(
    beforethisafter % ('', groupbegins90, groupbegins90, '.*'), re.I), 'begin'
groupends = (r'end|endprogram|endblockdata|endmodule|endpythonmodule|'
             r'endinterface|endsubroutine|endfunction')
endpattern = re.compile(
    beforethisafter % ('', groupends, groupends, '.*'), re.I), 'end'
# block, the Fortran 2008 construct needs special handling in the rest of the file
endifs = r'end\s*(if|do|where|select|while|forall|associate|'\
         r'critical|enum|team)'
endifpattern = re.compile(
    beforethisafter % (r'[\w]*?', endifs, endifs, '.*'), re.I), 'endif'
#
moduleprocedures = r'module\s*procedure'
moduleprocedurepattern = re.compile(
    beforethisafter % ('', moduleprocedures, moduleprocedures, '.*'), re.I), \
    'moduleprocedure'
implicitpattern = re.compile(
    beforethisafter % ('', 'implicit', 'implicit', '.*'), re.I), 'implicit'
dimensionpattern = re.compile(beforethisafter % (
    '', 'dimension|virtual', 'dimension|virtual', '.*'), re.I), 'dimension'
externalpattern = re.compile(
    beforethisafter % ('', 'external', 'external', '.*'), re.I), 'external'
optionalpattern = re.compile(
    beforethisafter % ('', 'optional', 'optional', '.*'), re.I), 'optional'
requiredpattern = re.compile(
    beforethisafter % ('', 'required', 'required', '.*'), re.I), 'required'
publicpattern = re.compile(
    beforethisafter % ('', 'public', 'public', '.*'), re.I), 'public'
privatepattern = re.compile(
    beforethisafter % ('', 'private', 'private', '.*'), re.I), 'private'
intrinsicpattern = re.compile(
    beforethisafter % ('', 'intrinsic', 'intrinsic', '.*'), re.I), 'intrinsic'
intentpattern = re.compile(beforethisafter % (
    '', 'intent|depend|note|check', 'intent|depend|note|check', r'\s*\(.*?\).*'), re.I), 'intent'
parameterpattern = re.compile(
    beforethisafter % ('', 'parameter', 'parameter', r'\s*\(.*'), re.I), 'parameter'
datapattern = re.compile(
    beforethisafter % ('', 'data', 'data', '.*'), re.I), 'data'
callpattern = re.compile(
    beforethisafter % ('', 'call', 'call', '.*'), re.I), 'call'
entrypattern = re.compile(
    beforethisafter % ('', 'entry', 'entry', '.*'), re.I), 'entry'
callfunpattern = re.compile(
    beforethisafter % ('', 'callfun', 'callfun', '.*'), re.I), 'callfun'
commonpattern = re.compile(
    beforethisafter % ('', 'common', 'common', '.*'), re.I), 'common'
usepattern = re.compile(
    beforethisafter % ('', 'use', 'use', '.*'), re.I), 'use'
containspattern = re.compile(
    beforethisafter % ('', 'contains', 'contains', ''), re.I), 'contains'
sequencepattern = re.compile(
    beforethisafter % ('', 'sequence', 'sequence', ''), re.I), 'sequence'
formatpattern = re.compile(
    beforethisafter % ('', 'format', 'format', '.*'), re.I), 'format'
# Non-fortran and f2py-specific statements
f2pyenhancementspattern = re.compile(beforethisafter % ('', 'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef',
                                                        'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef', '.*'), re.I | re.S), 'f2pyenhancements'
multilinepattern = re.compile(
    r"\s*(?P<before>''')(?P<this>.*?)(?P<after>''')\s*\Z", re.S), 'multiline'
##

def split_by_unquoted(line, characters):
    """
    Splits the line into (line[:i], line[i:]),
    where i is the index of first occurrence of one of the characters
    not within quotes, or len(line) if no such index exists
    """
    assert not (set('"\'') & set(characters)), "cannot split by unquoted quotes"
    r = re.compile(
        r"\A(?P<before>({single_quoted}|{double_quoted}|{not_quoted})*)"
        r"(?P<after>{char}.*)\Z".format(
            not_quoted=f"[^\"'{re.escape(characters)}]",
            char=f"[{re.escape(characters)}]",
            single_quoted=r"('([^'\\]|(\\.))*')",
            double_quoted=r'("([^"\\]|(\\.))*")'))
    m = r.match(line)
    if m:
        d = m.groupdict()
        return (d["before"], d["after"])
    return (line, "")


_quoted_string_re = re.compile(
    r"('([^'\\]|(\\.))*')|(\"([^\"\\]|(\\.))*\")")


def mask_string_literals(expr):
    """Blank the contents of Fortran string literals in *expr*.

    Dependency scanning matches variable names against the right-hand
    side of a declaration. A character parameter such as
    ``mkdir = 'mkdir '`` must not be treated as depending on the
    identifiers that happen to appear inside its own literal value, so
    the quoted text is collapsed to empty quotes before the scan while
    identifiers outside the quotes (e.g. in ``'x '//name``) are kept.
    """
    return _quoted_string_re.sub("''", expr)


def _simplifyargs(argsline):
    a = []
    for n in markoutercomma(argsline).split('@,@'):
        for r in '(),':
            n = n.replace(r, '_')
        a.append(n)
    return ','.join(a)


crackline_re_1 = re.compile(r'\s*(?P<result>\b[a-z]+\w*\b)\s*=.*', re.I)
crackline_bind_1 = re.compile(r'\s*(?P<bind>\b[a-z]+\w*\b)\s*=.*', re.I)
crackline_bindlang = re.compile(r'\s*bind\(\s*(?P<lang>[^,]+)\s*,\s*name\s*=\s*"(?P<lang_name>[^"]+)"\s*\)', re.I)

def crackline(line, reset=0):
    """
    reset=-1  --- initialize
    reset=0   --- crack the line
    reset=1   --- final check if mismatch of blocks occurred

    Cracked data is saved in grouplist[0].
    """
    global beginpattern, groupcounter, groupname, groupcache, grouplist
    global filepositiontext, currentfilename, neededmodule, expectbegin
    global skipblocksuntil, skipemptyends, previous_context, gotnextfile

    _, has_semicolon = split_by_unquoted(line, ";")
    if has_semicolon and not (f2pyenhancementspattern[0].match(line) or
                               multilinepattern[0].match(line)):
        # XXX: non-zero reset values need testing
        assert reset == 0, repr(reset)
        # split line on unquoted semicolons
        line, semicolon_line = split_by_unquoted(line, ";")
        while semicolon_line:
            crackline(line, reset)
            line, semicolon_line = split_by_unquoted(semicolon_line[1:], ";")
        crackline(line, reset)
        return
    if reset < 0:
        groupcounter = 0
        groupname = {groupcounter: ''}
        groupcache = {groupcounter: {}}
        grouplist = {groupcounter: []}
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = ''
        groupcache[groupcounter]['name'] = ''
        neededmodule = -1
        skipblocksuntil = -1
        return
    if reset > 0:
        fl = 0
        if f77modulename and neededmodule == groupcounter:
            fl = 2
        while groupcounter > fl:
            outmess(f'crackline: groupcounter={groupcounter!r} groupname={groupname!r}\n')
            outmess(
                'crackline: Mismatch of blocks encountered. Trying to fix it by assuming "end" statement.\n')
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1
        if f77modulename and neededmodule == groupcounter:
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end interface
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end module
            neededmodule = -1
        return
    if line == '':
        return
    flag = 0
    for pat in [dimensionpattern, externalpattern, intentpattern, optionalpattern,
                requiredpattern,
                parameterpattern, datapattern, publicpattern, privatepattern,
                intrinsicpattern,
                endifpattern, endpattern,
                formatpattern,
                beginpattern, functionpattern, subroutinepattern,
                implicitpattern, typespattern, commonpattern,
                callpattern, usepattern, containspattern, sequencepattern,
                entrypattern,
                f2pyenhancementspattern,
                multilinepattern,
                moduleprocedurepattern
                ]:
        m = pat[0].match(line)
        if m:
            break
        flag = flag + 1
    if not m:
        re_1 = crackline_re_1
        if 0 <= skipblocksuntil <= groupcounter:
            return
        if 'externals' in groupcache[groupcounter]:
            for name in groupcache[groupcounter]['externals']:
                if name in invbadnames:
                    name = invbadnames[name]
                if 'interfaced' in groupcache[groupcounter] and name in groupcache[groupcounter]['interfaced']:
                    continue
                m1 = re.match(
                    rf'(?P<before>[^"]*)\b{name}\b\s*@\(@(?P<args>[^@]*)@\)@.*\Z',
                    markouterparen(line), re.I)
                if m1:
                    m2 = re_1.match(m1.group('before'))
                    a = _simplifyargs(m1.group('args'))
                    if m2:
                        line = f"callfun {name}({a}) result ({m2.group('result')})"
                    else:
                        line = f'callfun {name}({a})'
                    m = callfunpattern[0].match(line)
                    if not m:
                        outmess(
                            f'crackline: could not resolve function call for line={repr(line)}.\n')
                        return
                    analyzeline(m, 'callfun', line)
                    return
        if verbose > 1 or (verbose == 1 and currentfilename.lower().endswith('.pyf')):
            previous_context = None
            outmess(f'crackline:{groupcounter}: No pattern for line\n')
        return
    elif pat[1] == 'end':
        if 0 <= skipblocksuntil < groupcounter:
            groupcounter = groupcounter - 1
            if skipblocksuntil <= groupcounter:
                return
        if groupcounter <= 0:
            raise Exception(f'crackline: groupcounter(={groupcounter}) is nonpositive. '
                            'Check the blocks.')
        m1 = beginpattern[0].match(line)
        if (m1) and (not m1.group('this') == groupname[groupcounter]):
            raise Exception(f'crackline: End group {m1.group("this")!r} '
                            'does not match with previous Begin group '
                            f'{groupname[groupcounter]!r}\n\t{filepositiontext}')
        if skipblocksuntil == groupcounter:
            skipblocksuntil = -1
        grouplist[groupcounter - 1].append(groupcache[groupcounter])
        grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
        del grouplist[groupcounter]
        groupcounter = groupcounter - 1
        if not skipemptyends:
            expectbegin = 1
    elif pat[1] == 'begin':
        if 0 <= skipblocksuntil <= groupcounter:
            groupcounter = groupcounter + 1
            return
        gotnextfile = 0
        analyzeline(m, pat[1], line)
        expectbegin = 0
    elif pat[1] == 'endif':
        pass
    elif pat[1] == 'moduleprocedure':
        analyzeline(m, pat[1], line)
    elif pat[1] == 'contains':
        if ignorecontains:
            return
        if 0 <= skipblocksuntil <= groupcounter:
            return
        skipblocksuntil = groupcounter
    elif pat[1] == 'sequence':
        # SEQUENCE statement inside a type block (F2018 7.5.2.3).
        # Add 'sequence' to the type's attrspec so wrapping code can
        # detect it via issequencetype().
        if groupcache[groupcounter].get('block') == 'type':
            tname = groupcache[groupcounter].get('name', '')
            # The parent block stores type attributes in vars[typename]
            if groupcounter > 0:
                parent = groupcache[groupcounter - 1]
                if tname in parent.get('vars', {}):
                    attrs = parent['vars'][tname].get('attrspec', [])
                    if 'sequence' not in attrs:
                        attrs.append('sequence')
                    parent['vars'][tname]['attrspec'] = attrs
        return
    else:
        if 0 <= skipblocksuntil <= groupcounter:
            return
        analyzeline(m, pat[1], line)


def markouterparen(line):
    l = ''
    f = 0
    for c in line:
        if c == '(':
            f = f + 1
            if f == 1:
                l = l + '@(@'
                continue
        elif c == ')':
            f = f - 1
            if f == 0:
                l = l + '@)@'
                continue
        l = l + c
    return l


def markoutercomma(line, comma=','):
    l = ''
    f = 0
    before, after = split_by_unquoted(line, comma + '()[]')
    l += before
    while after:
        if (after[0] == comma) and (f == 0):
            l += '@' + comma + '@'
        else:
            l += after[0]
            if after[0] in '([':
                f += 1
            elif after[0] in ')]':
                f -= 1
        before, after = split_by_unquoted(after[1:], comma + '()[]')
        l += before
    assert not f, repr((f, line, l))
    return l

def unmarkouterparen(line):
    r = line.replace('@(@', '(').replace('@)@', ')')
    return r


def appenddecl(decl, decl2, force=1):
    if not decl:
        decl = {}
    if not decl2:
        return decl
    if decl is decl2:
        return decl
    for k in list(decl2.keys()):
        if k == 'typespec':
            if force or k not in decl:
                decl[k] = decl2[k]
        elif k == 'attrspec':
            for l in decl2[k]:
                decl = setattrspec(decl, l, force)
        elif k == 'kindselector':
            decl = setkindselector(decl, decl2[k], force)
        elif k == 'charselector':
            decl = setcharselector(decl, decl2[k], force)
        elif k in ['=', 'typename']:
            if force or k not in decl:
                decl[k] = decl2[k]
        elif k == 'note':
            pass
        elif k in ['intent', 'check', 'dimension', 'optional',
                   'required', 'depend']:
            errmess(f'appenddecl: "{k}" not implemented.\n')
        else:
            raise Exception('appenddecl: Unknown variable definition key: ' +
                            str(k))
    return decl


selectpattern = re.compile(
    r'\s*(?P<this>(@\(@.*?@\)@|\*[\d*]+|\*\s*@\(@.*?@\)@|))(?P<after>.*)\Z', re.I)
typedefpattern = re.compile(
    r'(?:,(?P<attributes>[\w(),]+))?(::)?(?P<name>\b[a-z$_][\w$]*\b)'
    r'(?:\((?P<params>[\w,]*)\))?\Z', re.I)
nameargspattern = re.compile(
    r'\s*(?P<name>\b[\w$]+\b)\s*(@\(@\s*(?P<args>[\w\s,]*)\s*@\)@|)\s*((result(\s*@\(@\s*(?P<result>\b[\w$]+\b)\s*@\)@|))|(bind\s*@\(@\s*(?P<bind>(?:(?!@\)@).)*)\s*@\)@))*\s*\Z', re.I)
operatorpattern = re.compile(
    r'\s*(?P<scheme>(operator|assignment))'
    r'@\(@\s*(?P<name>[^)]+)\s*@\)@\s*\Z', re.I)
callnameargspattern = re.compile(
    r'\s*(?P<name>\b[\w$]+\b)\s*@\(@\s*(?P<args>.*)\s*@\)@\s*\Z', re.I)
real16pattern = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\d*\.\d+))[dD]((?:[-+]?\d+)?)')
real8pattern = re.compile(
    r'([-+]?((?:\d+(?:\.\d*)?|\d*\.\d+))[eE]((?:[-+]?\d+)?)|(\d+\.\d*))')

_intentcallbackpattern = re.compile(r'intent\s*\(.*?\bcallback\b', re.I)


def _is_intent_callback(vdecl):
    for a in vdecl.get('attrspec', []):
        if _intentcallbackpattern.match(a):
            return 1
    return 0


def _resolvetypedefpattern(line):
    line = ''.join(line.split())  # removes whitespace
    m1 = typedefpattern.match(line)
    if m1:
        attrs = m1.group('attributes')
        attrs = [a.lower() for a in attrs.split(',')] if attrs else []
        return m1.group('name'), attrs, m1.group('params')
    return None, [], None

def parse_name_for_bind(line):
    pattern = re.compile(r'bind\(\s*(?P<lang>[^,]+)(?:\s*,\s*name\s*=\s*["\'](?P<name>[^"\']+)["\']\s*)?\)', re.I)
    match = pattern.search(line)
    bind_statement = None
    if match:
        bind_statement = match.group(0)
        # Remove the 'bind' construct from the line.
        line = line[:match.start()] + line[match.end():]
    return line, bind_statement

def _resolvenameargspattern(line):
    line, bind_cname = parse_name_for_bind(line)
    line = markouterparen(line)
    m1 = nameargspattern.match(line)
    if m1:
        return m1.group('name'), m1.group('args'), m1.group('result'), bind_cname
    m1 = operatorpattern.match(line)
    if m1:
        name = m1.group('scheme') + '(' + m1.group('name') + ')'
        return name, [], None, None
    m1 = callnameargspattern.match(line)
    if m1:
        return m1.group('name'), m1.group('args'), None, None
    return None, [], None, None


def analyzeline(m, case, line):
    """
    Reads each line in the input file in sequence and updates global vars.

    Effectively reads and collects information from the input file to the
    global variable groupcache, a dictionary containing info about each part
    of the fortran module.

    At the end of analyzeline, information is filtered into the correct dict
    keys, but parameter values and dimensions are not yet interpreted.
    """
    global groupcounter, groupname, groupcache, grouplist, filepositiontext
    global currentfilename, f77modulename, neededinterface, neededmodule
    global expectbegin, gotnextfile, previous_context

    block = m.group('this')
    if case != 'multiline':
        previous_context = None
    if expectbegin and case not in ['begin', 'call', 'callfun', 'type'] \
       and not skipemptyends and groupcounter < 1:
        newname = os.path.basename(currentfilename).split('.')[0]
        outmess(
            f'analyzeline: no group yet. Creating program group with name "{newname}".\n')
        gotnextfile = 0
        groupcounter = groupcounter + 1
        groupname[groupcounter] = 'program'
        groupcache[groupcounter] = {}
        grouplist[groupcounter] = []
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = 'program'
        groupcache[groupcounter]['name'] = newname
        groupcache[groupcounter]['from'] = 'fromsky'
        expectbegin = 0
    if case in ['begin', 'call', 'callfun']:
        # Crack line => block,name,args,result
        block = block.lower()
        if re.match(r'block\s*data', block, re.I):
            block = 'block data'
        elif re.match(r'python\s*module', block, re.I):
            block = 'python module'
        elif re.match(r'abstract\s*interface', block, re.I):
            block = 'abstract interface'
        if block == 'type':
            name, attrs, _ = _resolvetypedefpattern(m.group('after'))
            groupcache[groupcounter]['vars'][name] = {'attrspec': attrs}
            args = []
            result = None
        else:
            name, args, result, bindcline = _resolvenameargspattern(m.group('after'))
        if name is None:
            if block == 'block data':
                name = '_BLOCK_DATA_'
            else:
                name = ''
            if block not in ['interface', 'block data', 'abstract interface']:
                outmess('analyzeline: No name/args pattern found for line.\n')

        previous_context = (block, name, groupcounter)
        if args:
            args = rmbadname([x.strip()
                              for x in markoutercomma(args).split('@,@')])
        else:
            args = []
        if '' in args:
            while '' in args:
                args.remove('')
            outmess(
                'analyzeline: argument list is malformed (missing argument).\n')

        # end of crack line => block,name,args,result
        needmodule = 0
        needinterface = 0

        if case in ['call', 'callfun']:
            needinterface = 1
            if 'args' not in groupcache[groupcounter]:
                return
            if name not in groupcache[groupcounter]['args']:
                return
            for it in grouplist[groupcounter]:
                if it['name'] == name:
                    return
            if name in groupcache[groupcounter]['interfaced']:
                return
            block = {'call': 'subroutine', 'callfun': 'function'}[case]
        if f77modulename and neededmodule == -1 and groupcounter <= 1:
            neededmodule = groupcounter + 2
            needmodule = 1
            if block not in ['interface', 'abstract interface']:
                needinterface = 1
        # Create new block(s)
        groupcounter = groupcounter + 1
        groupcache[groupcounter] = {}
        grouplist[groupcounter] = []
        if needmodule:
            if verbose > 1:
                outmess(f'analyzeline: Creating module block {f77modulename!r}\n', 0)
            groupname[groupcounter] = 'module'
            groupcache[groupcounter]['block'] = 'python module'
            groupcache[groupcounter]['name'] = f77modulename
            groupcache[groupcounter]['from'] = ''
            groupcache[groupcounter]['body'] = []
            groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['interfaced'] = []
            groupcache[groupcounter]['vars'] = {}
            groupcounter = groupcounter + 1
            groupcache[groupcounter] = {}
            grouplist[groupcounter] = []
        if needinterface:
            if verbose > 1:
                outmess('analyzeline: Creating additional interface block '
                        f'({groupcounter=}).\n', 0)
            groupname[groupcounter] = 'interface'
            groupcache[groupcounter]['block'] = 'interface'
            groupcache[groupcounter]['name'] = 'unknown_interface'
            prev_group = groupcache[groupcounter - 1]
            groupcache[groupcounter]['from'] = f"{prev_group['from']}:{prev_group['name']}"
            groupcache[groupcounter]['body'] = []
            groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['interfaced'] = []
            groupcache[groupcounter]['vars'] = {}
            groupcounter = groupcounter + 1
            groupcache[groupcounter] = {}
            grouplist[groupcounter] = []
        groupname[groupcounter] = block
        groupcache[groupcounter]['block'] = block
        if not name:
            name = 'unknown_' + block.replace(' ', '_')
        groupcache[groupcounter]['prefix'] = m.group('before')
        # a derived type keeps its name, which declarations, extends()
        # and the attributes stored above refer to, and which only ever
        # appears in C inside longer identifiers
        groupcache[groupcounter]['name'] = (
            name if block == 'type' else rmbadname1(name))
        groupcache[groupcounter]['result'] = result
        if groupcounter == 1:
            groupcache[groupcounter]['from'] = currentfilename
        elif f77modulename and groupcounter == 3:
            groupcache[groupcounter]['from'] = f"{groupcache[groupcounter - 1]['from']}:{currentfilename}"
        else:
            prev_group = groupcache[groupcounter - 1]
            groupcache[groupcounter]['from'] = f"{prev_group['from']}:{prev_group['name']}"
        for k in list(groupcache[groupcounter].keys()):
            if not groupcache[groupcounter][k]:
                del groupcache[groupcounter][k]

        groupcache[groupcounter]['args'] = args
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['externals'] = []
        groupcache[groupcounter]['interfaced'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['entry'] = {}
        # end of creation
        if block == 'type':
            groupcache[groupcounter]['varnames'] = []

        if case in ['call', 'callfun']:  # set parents variables
            if name not in groupcache[groupcounter - 2]['externals']:
                groupcache[groupcounter - 2]['externals'].append(name)
            groupcache[groupcounter]['vars'] = copy.deepcopy(
                groupcache[groupcounter - 2]['vars'])
            try:
                del groupcache[groupcounter]['vars'][name][
                    groupcache[groupcounter]['vars'][name]['attrspec'].index('external')]
            except Exception:
                pass
        if block in ['function', 'subroutine']:  # set global attributes
            # name is fortran name
            if bindcline:
                bindcdat = re.search(crackline_bindlang, bindcline)
                if bindcdat:
                    groupcache[groupcounter]['bindlang'] = {name: {}}
                    groupcache[groupcounter]['bindlang'][name]["lang"] = bindcdat.group('lang')
                    if bindcdat.group('lang_name'):
                        groupcache[groupcounter]['bindlang'][name]["name"] = bindcdat.group('lang_name')
            try:
                groupcache[groupcounter]['vars'][name] = appenddecl(
                    groupcache[groupcounter]['vars'][name], groupcache[groupcounter - 2]['vars'][''])
            except Exception:
                pass
            if case == 'callfun':  # return type
                if result and result in groupcache[groupcounter]['vars']:
                    if not name == result:
                        groupcache[groupcounter]['vars'][name] = appenddecl(
                            groupcache[groupcounter]['vars'][name], groupcache[groupcounter]['vars'][result])
            # if groupcounter>1: # name is interfaced
            try:
                groupcache[groupcounter - 2]['interfaced'].append(name)
            except Exception:
                pass
        if block == 'function':
            t = typespattern[0].match(m.group('before') + ' ' + name)
            if t:
                typespec, selector, attr, edecl = cracktypespec0(
                    t.group('this'), t.group('after'))
                updatevars(typespec, selector, attr, edecl)

        if case in ['call', 'callfun']:
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end routine
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end interface

    elif case == 'entry':
        name, args, result, _ = _resolvenameargspattern(m.group('after'))
        if name is not None:
            if args:
                args = rmbadname([x.strip()
                                  for x in markoutercomma(args).split('@,@')])
            else:
                args = []
            assert result is None, repr(result)
            groupcache[groupcounter]['entry'][name] = args
            previous_context = ('entry', name, groupcounter)
    elif case == 'type':
        typespec, selector, attr, edecl = cracktypespec0(
            block, m.group('after'))
        last_name = updatevars(typespec, selector, attr, edecl)
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case in ['dimension', 'intent', 'optional', 'required', 'external', 'public', 'private', 'intrinsic']:
        edecl = groupcache[groupcounter]['vars']
        ll = m.group('after').strip()
        i = ll.find('::')
        if i < 0 and case == 'intent':
            i = markouterparen(ll).find('@)@') - 2
            ll = ll[:i + 1] + '::' + ll[i + 1:]
            i = ll.find('::')
            if ll[i:] == '::' and 'args' in groupcache[groupcounter]:
                outmess('All arguments will have attribute '
                        f'{m.group("this")}{ll[:i]}\n')
                ll = ll + ','.join(groupcache[groupcounter]['args'])
        if i < 0:
            i = 0
            pl = ''
        else:
            pl = ll[:i].strip()
            ll = ll[i + 2:]
        ch = markoutercomma(pl).split('@,@')
        if len(ch) > 1:
            pl = ch[0]
            outmess("analyzeline: cannot handle multiple attributes without "
                    f"type specification. Ignoring {','.join(ch[1:])!r}.\n")
        last_name = None

        for e in [x.strip() for x in markoutercomma(ll).split('@,@')]:
            m1 = namepattern.match(e)
            if not m1:
                if case in ['public', 'private']:
                    k = ''
                else:
                    print(m.groupdict())
                    outmess(f'analyzeline: no name pattern found in {case} statement '
                            f'for {e!r}. Skipping.\n')
                    continue
            else:
                k = rmbadname1(m1.group('name'))
            if case in ['public', 'private'] and k in {'operator', 'assignment'}:
                k += m1.group('after')
            if k not in edecl:
                edecl[k] = {}
            if case == 'dimension':
                ap = case + m1.group('after')
            if case == 'intent':
                ap = m.group('this') + pl
                if _intentcallbackpattern.match(ap):
                    if k not in groupcache[groupcounter]['args']:
                        if groupcounter > 1:
                            if '__user__' not in groupcache[groupcounter - 2]['name']:
                                outmess(
                                    'analyzeline: missing __user__ module (could be nothing)\n')
                            # fixes ticket 1693
                            if k != groupcache[groupcounter]['name']:
                                outmess(f"analyzeline: appending intent(callback) {k}"
                                        f" to {groupcache[groupcounter]['name']} "
                                        "arguments\n")
                                groupcache[groupcounter]['args'].append(k)
                        else:
                            errmess(
                                f'analyzeline: intent(callback) {k} is ignored\n')
                    else:
                        errmess(f'analyzeline: intent(callback) {k} is already'
                                ' in argument list\n')
            if case in ['optional', 'required', 'public', 'external', 'private', 'intrinsic']:
                ap = case
            if 'attrspec' in edecl[k]:
                edecl[k]['attrspec'].append(ap)
            else:
                edecl[k]['attrspec'] = [ap]
            if case == 'external':
                if groupcache[groupcounter]['block'] == 'program':
                    outmess('analyzeline: ignoring program arguments\n')
                    continue
                if k not in groupcache[groupcounter]['args']:
                    continue
                if 'externals' not in groupcache[groupcounter]:
                    groupcache[groupcounter]['externals'] = []
                groupcache[groupcounter]['externals'].append(k)
            last_name = k
        groupcache[groupcounter]['vars'] = edecl
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'moduleprocedure':
        groupcache[groupcounter]['implementedby'] = \
            [x.strip() for x in m.group('after').split(',')]
    elif case == 'parameter':
        edecl = groupcache[groupcounter]['vars']
        ll = m.group('after').strip()[1:-1]
        last_name = None
        for e in markoutercomma(ll).split('@,@'):
            try:
                k, initexpr = [x.strip() for x in e.split('=', 1)]
            except Exception:
                outmess(
                    f'analyzeline: could not extract name,expr in parameter statement "{e}" of "{ll}\"\n')
                continue
            k = rmbadname1(k)
            if k not in edecl:
                edecl[k] = {}
            if '=' in edecl[k] and (not edecl[k]['='] == initexpr):
                outmess(f'analyzeline: Overwriting the value of parameter "{k}" '
                        f'("{edecl[k]["="]}") with "{initexpr}".\n')
            # get_parameters evaluates it with the other parameters, by the
            # rules of Fortran, as for a parameter declared with its type
            edecl[k]['='] = initexpr
            if 'attrspec' in edecl[k]:
                edecl[k]['attrspec'].append('parameter')
            else:
                edecl[k]['attrspec'] = ['parameter']
            last_name = k
        groupcache[groupcounter]['vars'] = edecl
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'implicit':
        if m.group('after').strip().lower() == 'none':
            groupcache[groupcounter]['implicit'] = None
        elif m.group('after'):
            impl = groupcache[groupcounter].get('implicit', {})
            if impl is None:
                outmess(
                    'analyzeline: Overwriting earlier "implicit none" statement.\n')
                impl = {}
            for e in markoutercomma(m.group('after')).split('@,@'):
                decl = {}
                m1 = re.match(
                    r'\s*(?P<this>.*?)\s*(\(\s*(?P<after>[a-z-, ]+)\s*\)\s*|)\Z', e, re.I)
                if not m1:
                    outmess(
                        f'analyzeline: could not extract info of implicit statement part "{e}\"\n')
                    continue
                m2 = typespattern4implicit.match(m1.group('this'))
                if not m2:
                    outmess(
                        f'analyzeline: could not extract types pattern of implicit statement part "{e}\"\n')
                    continue
                typespec, selector, attr, edecl = cracktypespec0(
                    m2.group('this'), m2.group('after'))
                kindselect, charselect, typename = cracktypespec(
                    typespec, selector)
                decl['typespec'] = typespec
                decl['kindselector'] = kindselect
                decl['charselector'] = charselect
                decl['typename'] = typename
                for k in list(decl.keys()):
                    if not decl[k]:
                        del decl[k]
                for r in markoutercomma(m1.group('after')).split('@,@'):
                    if '-' in r:
                        try:
                            begc, endc = [x.strip() for x in r.split('-')]
                        except Exception:
                            outmess(
                                f'analyzeline: expected "<char>-<char>" instead of "{r}" in range list of implicit statement\n')
                            continue
                    else:
                        begc = endc = r.strip()
                    if not len(begc) == len(endc) == 1:
                        outmess(
                            f'analyzeline: expected "<char>-<char>" instead of "{r}" in range list of implicit statement (2)\n')
                        continue
                    for o in range(ord(begc), ord(endc) + 1):
                        impl[chr(o)] = decl
            groupcache[groupcounter]['implicit'] = impl
    elif case == 'data':
        ll = []
        dl = ''
        il = ''
        f = 0
        fc = 1
        inp = 0
        for c in m.group('after'):
            if not inp:
                if c == "'":
                    fc = not fc
                if c == '/' and fc:
                    f = f + 1
                    continue
            if c == '(':
                inp = inp + 1
            elif c == ')':
                inp = inp - 1
            if f == 0:
                dl = dl + c
            elif f == 1:
                il = il + c
            elif f == 2:
                dl = dl.strip()
                if dl.startswith(','):
                    dl = dl[1:].strip()
                ll.append([dl, il])
                dl = c
                il = ''
                f = 0
        if f == 2:
            dl = dl.strip()
            if dl.startswith(','):
                dl = dl[1:].strip()
            ll.append([dl, il])
        vars = groupcache[groupcounter].get('vars', {})
        last_name = None
        for l in ll:
            l[0], l[1] = l[0].strip().removeprefix(','), l[1].strip()
            if l[0].startswith('('):
                outmess(f'analyzeline: implied-DO list "{l[0]}" is not supported. Skipping.\n')
                continue
            for idx, v in enumerate(rmbadname([x.strip() for x in markoutercomma(l[0]).split('@,@')])):
                if v.startswith('('):
                    outmess(f'analyzeline: implied-DO list "{v}" is not supported. Skipping.\n')
                    # XXX: subsequent init expressions may get wrong values.
                    # Ignoring since data statements are irrelevant for
                    # wrapping.
                    continue
                if '!' in l[1]:
                    # Fixes gh-24746 pyf generation
                    # XXX: This essentially ignores the value for generating the pyf which is fine:
                    # integer dimension(3) :: mytab
                    # common /mycom/ mytab
                    # Since in any case it is initialized in the Fortran code
                    outmess(f'Comment line in declaration "{l[1]}" is not supported. Skipping.\n')
                    continue
                vars.setdefault(v, {})
                vtype = vars[v].get('typespec')
                vdim = getdimension(vars[v])
                matches = re.findall(r"\(.*?\)", l[1]) if vtype == 'complex' else l[1].split(',')
                try:
                    new_val = f"(/{', '.join(matches)}/)" if vdim else matches[idx]
                except IndexError:
                    # gh-24746
                    # Runs only if above code fails. Fixes the line
                    # DATA IVAR1, IVAR2, IVAR3, IVAR4, EVAR5 /4*0,0.0D0/
                    # by expanding to ['0', '0', '0', '0', '0.0d0']
                    if any("*" in m for m in matches):
                        expanded_list = []
                        for match in matches:
                            if "*" in match:
                                try:
                                    multiplier, value = match.split("*")
                                    expanded_list.extend([value.strip()] * int(multiplier))
                                except ValueError:  # if int(multiplier) fails
                                    expanded_list.append(match.strip())
                            else:
                                expanded_list.append(match.strip())
                        matches = expanded_list
                    new_val = f"(/{', '.join(matches)}/)" if vdim else matches[idx]
                current_val = vars[v].get('=')
                if current_val and (current_val != new_val):
                    outmess(f'analyzeline: changing init expression of "{v}" ("{current_val}") to "{new_val}\"\n')
                vars[v]['='] = new_val
                last_name = v
        groupcache[groupcounter]['vars'] = vars
        if last_name:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'common':
        line = m.group('after').strip()
        if not line[0] == '/':
            line = '//' + line

        cl = []
        [_, bn, ol] = re.split('/', line, maxsplit=2)
        bn = bn.strip()
        if not bn:
            bn = '_BLNK_'
        cl.append([bn, ol])
        commonkey = {}
        if 'common' in groupcache[groupcounter]:
            commonkey = groupcache[groupcounter]['common']
        for c in cl:
            if c[0] not in commonkey:
                commonkey[c[0]] = []
            for i in [x.strip() for x in markoutercomma(c[1]).split('@,@')]:
                if i:
                    commonkey[c[0]].append(i)
        groupcache[groupcounter]['common'] = commonkey
        previous_context = ('common', bn, groupcounter)
    elif case == 'use':
        m1 = re.match(
            r'\A\s*(?P<name>\b\w+\b)\s*((,(\s*\bonly\b\s*:|(?P<notonly>))\s*(?P<list>.*))|)\s*\Z', m.group('after'), re.I)
        if m1:
            mm = m1.groupdict()
            if 'use' not in groupcache[groupcounter]:
                groupcache[groupcounter]['use'] = {}
            name = m1.group('name')
            groupcache[groupcounter]['use'][name] = {}
            isonly = 0
            if 'list' in mm and mm['list'] is not None:
                if 'notonly' in mm and mm['notonly'] is None:
                    isonly = 1
                groupcache[groupcounter]['use'][name]['only'] = isonly
                ll = [x.strip() for x in mm['list'].split(',')]
                rl = {}
                for l in ll:
                    if '=' in l:
                        m2 = re.match(
                            r'\A\s*(?P<local>\b\w+\b)\s*=\s*>\s*(?P<use>\b\w+\b)\s*\Z', l, re.I)
                        if m2:
                            rl[m2.group('local').strip()] = m2.group(
                                'use').strip()
                        else:
                            outmess(
                                f'analyzeline: Not local=>use pattern found in {repr(l)}\n')
                    else:
                        rl[l] = l
                    groupcache[groupcounter]['use'][name]['map'] = rl
        else:
            print(m.groupdict())
            outmess('analyzeline: Could not crack the use statement.\n')
    elif case in ['f2pyenhancements']:
        if 'f2pyenhancements' not in groupcache[groupcounter]:
            groupcache[groupcounter]['f2pyenhancements'] = {}
        d = groupcache[groupcounter]['f2pyenhancements']
        if m.group('this') == 'usercode' and 'usercode' in d:
            if isinstance(d['usercode'], str):
                d['usercode'] = [d['usercode']]
            d['usercode'].append(m.group('after'))
        else:
            d[m.group('this')] = m.group('after')
    elif case == 'multiline':
        if previous_context is None:
            if verbose:
                outmess('analyzeline: No context for multiline block.\n')
            return
        gc = groupcounter
        appendmultiline(groupcache[gc],
                        previous_context[:2],
                        m.group('this'))
    elif verbose > 1:
        print(m.groupdict())
        outmess('analyzeline: No code implemented for line.\n')


def appendmultiline(group, context_name, ml):
    if 'f2pymultilines' not in group:
        group['f2pymultilines'] = {}
    d = group['f2pymultilines']
    if context_name not in d:
        d[context_name] = []
    d[context_name].append(ml)


def cracktypespec0(typespec, ll):
    selector = None
    attr = None
    if re.match(r'double\s*complex', typespec, re.I):
        typespec = 'double complex'
    elif re.match(r'double\s*precision', typespec, re.I):
        typespec = 'double precision'
    else:
        typespec = typespec.strip().lower()
    m1 = selectpattern.match(markouterparen(ll))
    if not m1:
        outmess(
            'cracktypespec0: no kind/char_selector pattern found for line.\n')
        return
    d = m1.groupdict()
    for k in list(d.keys()):
        d[k] = unmarkouterparen(d[k])
    if typespec in ['complex', 'integer', 'logical', 'real', 'character', 'type', 'class']:
        selector = d['this']
        ll = d['after']
    # Normalize class(T) to type(T) but preserve polymorphic marker
    # (F2018 7.3.2.3: CLASS type specifier for polymorphic entities)
    is_polymorphic = False
    is_unlimited_poly = False
    if typespec == 'class':
        typespec = 'type'
        # Check for CLASS(*) -- unlimited polymorphism
        if selector and selector.strip() == '(*)':
            is_unlimited_poly = True
        else:
            is_polymorphic = True
    i = ll.find('::')
    if i >= 0:
        attr = ll[:i].strip()
        ll = ll[i + 2:]
    # Inject polymorphic attribute so downstream code can detect it
    if is_polymorphic:
        if attr:
            attr = attr + ', polymorphic'
        else:
            attr = 'polymorphic'
    elif is_unlimited_poly:
        if attr:
            attr = attr + ', unlimited_polymorphic'
        else:
            attr = 'unlimited_polymorphic'
    return typespec, selector, attr, ll


#####
namepattern = re.compile(r'\s*(?P<name>\b\w+\b)\s*(?P<after>.*)\s*\Z', re.I)
kindselector = re.compile(
    r'\s*(\(\s*(kind\s*=)?\s*(?P<kind>.*)\s*\)|\*\s*(?P<kind2>.*?))\s*\Z', re.I)
charselector = re.compile(
    r'\s*(\((?P<lenkind>.*)\)|\*\s*(?P<charlen>.*))\s*\Z', re.I)
typenamepattern = re.compile(
    r'\s*\(\s*(?P<name>\w+)\s*(?P<params>\(.*\))?\s*\)', re.I)
lenkindpattern = re.compile(
    r'\s*(kind\s*=\s*(?P<kind>.*?)\s*(@,@\s*len\s*=\s*(?P<len>.*)|)'
    r'|(len\s*=\s*|)(?P<len2>.*?)\s*(@,@\s*(kind\s*=\s*|)(?P<kind2>.*)'
    r'|(f2py_len\s*=\s*(?P<f2py_len>.*))|))\s*\Z', re.I)
lenarraypattern = re.compile(
    r'\s*(@\(@\s*(?!/)\s*(?P<array>.*?)\s*@\)@\s*\*\s*(?P<len>.*?)|(\*\s*(?P<len2>.*?)|)\s*(@\(@\s*(?!/)\s*(?P<array2>.*?)\s*@\)@|))\s*(=\s*(?P<init>.*?)|(@\(@|)/\s*(?P<init2>.*?)\s*/(@\)@|)|)\s*\Z', re.I)


def removespaces(expr):
    expr = expr.strip()
    if len(expr) <= 1:
        return expr
    expr2 = expr[0]
    for i in range(1, len(expr) - 1):
        # relations such as 3 > 2 and .not. x are operators too
        if (expr[i] == ' ' and
            ((expr[i + 1] in "()[]{}=+-/*<>. ") or
                (expr[i - 1] in "()[]{}=+-/*<>. "))):
            continue
        expr2 = expr2 + expr[i]
    expr2 = expr2 + expr[-1]
    return expr2


def markinnerspaces(line):
    """
    The function replace all spaces in the input variable line which are
    surrounded with quotation marks, with the triplet "@_@".

    For instance, for the input "a 'b c'" the function returns "a 'b@_@c'"

    Parameters
    ----------
    line : str

    Returns
    -------
    str

    """
    fragment = ''
    inside = False
    current_quote = None
    escaped = ''
    for c in line:
        if escaped == '\\' and c in ['\\', '\'', '"']:
            fragment += c
            escaped = c
            continue
        if not inside and c in ['\'', '"']:
            current_quote = c
        if c == current_quote:
            inside = not inside
        elif c == ' ' and inside:
            fragment += '@_@'
            continue
        fragment += c
        escaped = c  # reset to non-backslash
    return fragment


def updatevars(typespec, selector, attrspec, entitydecl):
    """
    Returns last_name, the variable name without special chars, parenthesis
        or dimension specifiers.

    Alters groupcache to add the name, typespec, attrspec (and possibly value)
    of current variable.
    """
    global groupcache, groupcounter

    last_name = None
    kindselect, charselect, typename = cracktypespec(typespec, selector)
    typeparams = cracktypeparams(typespec, selector)
    # Clean up outer commas, whitespace and undesired chars from attrspec
    if attrspec:
        attrspec = [x.strip() for x in markoutercomma(attrspec).split('@,@')]
        l = []
        c = re.compile(r'(?P<start>[a-zA-Z]+)')
        for a in attrspec:
            if not a:
                continue
            m = c.match(a)
            if m:
                s = m.group('start').lower()
                a = s + a[len(s):]
            l.append(a)
        attrspec = l
    # Strip pointer default initialization before entity splitting.
    # F2018 R741: component-initialization is "=> null-init" or
    # "=> initial-data-target" for pointer components (7.5.4.6).
    # This must happen before space-based splitting, otherwise
    # "data(:) => null()" becomes three tokens: "data(:)", "=>", "null()".
    entitydecl_clean = re.sub(
        r'\s*=>\s*null\s*\(\s*\)', '', entitydecl, flags=re.I)
    el = [x.strip() for x in markoutercomma(entitydecl_clean).split('@,@')]
    el1 = []
    for e in el:
        for e1 in [x.strip() for x in markoutercomma(removespaces(markinnerspaces(e)), comma=' ').split('@ @')]:
            if e1:
                el1.append(e1.replace('@_@', ' '))
    for e in el1:
        m = namepattern.match(e)
        if not m:
            outmess(
                f'updatevars: no name pattern found for entity={repr(e)}. Skipping.\n')
            continue
        ename = rmbadname1(m.group('name'))
        edecl = {}
        if ename in groupcache[groupcounter]['vars']:
            edecl = groupcache[groupcounter]['vars'][ename].copy()
            not_has_typespec = 'typespec' not in edecl
            if not_has_typespec:
                edecl['typespec'] = typespec
            elif typespec and (not typespec == edecl['typespec']):
                current_typespec = edecl['typespec']
                outmess(f'updatevars: attempt to change the type of "{ename}" '
                        f'("{current_typespec}") to "{typespec}". Ignoring.\n')
            if 'kindselector' not in edecl:
                edecl['kindselector'] = copy.copy(kindselect)
            elif kindselect:
                for k in list(kindselect.keys()):
                    if k in edecl['kindselector'] and (not kindselect[k] == edecl['kindselector'][k]):
                        current_kind = edecl['kindselector'][k]
                        outmess('updatevars: attempt to change the kindselector '
                                f'"{k}" of "{ename}" ("{current_kind}") to '
                                f'"{kindselect[k]}". Ignoring.\n')
                    else:
                        edecl['kindselector'][k] = copy.copy(kindselect[k])
            if 'charselector' not in edecl and charselect:
                if not_has_typespec:
                    edecl['charselector'] = charselect
                else:
                    errmess(f'updatevars:{ename}: attempt to change empty charselector '
                            f'to {charselect!r}. Ignoring.\n')
            elif charselect:
                for k in list(charselect.keys()):
                    if k in edecl['charselector'] and (not charselect[k] == edecl['charselector'][k]):
                        outmess(f'updatevars: attempt to change the charselector "{k}" '
                                f'of "{ename}" ("{edecl["charselector"][k]}") to '
                                f'"{charselect[k]}". Ignoring.\n')
                    else:
                        edecl['charselector'][k] = copy.copy(charselect[k])
            if 'typename' not in edecl:
                edecl['typename'] = typename
            elif typename and (not edecl['typename'] == typename):
                outmess(f'updatevars: attempt to change the typename of "{ename}" '
                        f'("{edecl["typename"]}") to "{typename}". Ignoring.\n')
            if typeparams and 'typeparams' not in edecl:
                edecl['typeparams'] = typeparams
            if 'attrspec' not in edecl:
                edecl['attrspec'] = copy.copy(attrspec)
            elif attrspec:
                for a in attrspec:
                    if a not in edecl['attrspec']:
                        edecl['attrspec'].append(a)
        else:
            edecl['typespec'] = copy.copy(typespec)
            edecl['kindselector'] = copy.copy(kindselect)
            edecl['charselector'] = copy.copy(charselect)
            edecl['typename'] = typename
            if typeparams:
                edecl['typeparams'] = typeparams
            edecl['attrspec'] = copy.copy(attrspec)
        if 'external' in (edecl.get('attrspec') or []) and e in groupcache[groupcounter]['args']:
            if 'externals' not in groupcache[groupcounter]:
                groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['externals'].append(e)
        after = m.group('after')
        # a coarray-spec follows the array-spec and precedes the length and
        # the initialization (F2018 R803); brackets inside parentheses or
        # after = construct arrays
        depth, bracket = 0, -1
        for i, c in enumerate(after):
            depth += {'(': 1, ')': -1}.get(c, 0)
            if depth == 0 and c in '=[':
                bracket = i if c == '[' else -1
                break
        if bracket != -1:
            closing = after.find(']', bracket)
            if closing != -1:
                edecl['attrspec'] = [*(edecl.get('attrspec') or []),
                                     f'codimension{after[bracket:closing + 1]}']
                after = after[:bracket] + after[closing + 1:]
        if after:
            m1 = lenarraypattern.match(markouterparen(after))
            if m1:
                d1 = m1.groupdict()
                for lk in ['len', 'array', 'init']:
                    if d1[lk + '2'] is not None:
                        d1[lk] = d1[lk + '2']
                        del d1[lk + '2']
                for k in list(d1.keys()):
                    if d1[k] is not None:
                        d1[k] = unmarkouterparen(d1[k])
                    else:
                        del d1[k]

                if 'len' in d1 and 'array' in d1:
                    if d1['len'] == '':
                        d1['len'] = d1['array']
                        del d1['array']
                    elif typespec == 'character':
                        if ('charselector' not in edecl) or (not edecl['charselector']):
                            edecl['charselector'] = {}
                        if 'len' in edecl['charselector']:
                            del edecl['charselector']['len']
                        edecl['charselector']['*'] = d1['len']
                        del d1['len']
                    else:
                        d1['array'] = d1['array'] + ',' + d1['len']
                        del d1['len']
                        array_spec = d1['array']
                        errmess(f'updatevars: "{typespec} {e}" is mapped to '
                                f'"{typespec} {ename}({array_spec})"\n')

                if 'len' in d1:
                    if typespec in ['complex', 'integer', 'logical', 'real']:
                        if ('kindselector' not in edecl) or (not edecl['kindselector']):
                            edecl['kindselector'] = {}
                        edecl['kindselector']['*'] = d1['len']
                        del d1['len']
                    elif typespec == 'character':
                        if ('charselector' not in edecl) or (not edecl['charselector']):
                            edecl['charselector'] = {}
                        if 'len' in edecl['charselector']:
                            del edecl['charselector']['len']
                        edecl['charselector']['*'] = d1['len']
                        del d1['len']

                if 'init' in d1:
                    if '=' in edecl and (not edecl['='] == d1['init']):
                        current_init = edecl['=']
                        new_init = d1['init']
                        outmess('updatevars: attempt to change the init expression of '
                                f'"{ename}" ("{current_init}") to "{new_init}". '
                                f'Ignoring.\n')
                    else:
                        edecl['='] = d1['init']

                if 'array' in d1:
                    dm = f"dimension({d1['array']})"
                    if 'attrspec' not in edecl or (not edecl['attrspec']):
                        edecl['attrspec'] = [dm]
                    else:
                        edecl['attrspec'].append(dm)
                        for dm1 in edecl['attrspec']:
                            if dm1[:9] == 'dimension' and dm1 != dm:
                                del edecl['attrspec'][-1]
                                errmess(f'updatevars:{ename}: attempt to change '
                                        f'{dm1!r} to {dm!r}. Ignoring.\n')
                                break

            else:
                outmess('updatevars: could not crack entity declaration "%s". Ignoring.\n' % (
                    ename + after))
        for k in list(edecl.keys()):
            if not edecl[k]:
                del edecl[k]
        groupcache[groupcounter]['vars'][ename] = edecl
        if 'varnames' in groupcache[groupcounter]:
            groupcache[groupcounter]['varnames'].append(ename)
        last_name = ename
    return last_name


def cracktypespec(typespec, selector):
    kindselect = None
    charselect = None
    typename = None
    if selector:
        if typespec in ['complex', 'integer', 'logical', 'real']:
            kindselect = kindselector.match(selector)
            if not kindselect:
                outmess(
                    f'cracktypespec: no kindselector pattern found for {repr(selector)}\n')
                return
            kindselect = kindselect.groupdict()
            kindselect['*'] = kindselect['kind2']
            del kindselect['kind2']
            for k in list(kindselect.keys()):
                if not kindselect[k]:
                    del kindselect[k]
            for k, i in list(kindselect.items()):
                kindselect[k] = rmbadname1(i)
        elif typespec == 'character':
            charselect = charselector.match(selector)
            if not charselect:
                outmess(
                    f'cracktypespec: no charselector pattern found for {repr(selector)}\n')
                return
            charselect = charselect.groupdict()
            charselect['*'] = charselect['charlen']
            del charselect['charlen']
            if charselect['lenkind']:
                lenkind = lenkindpattern.match(
                    markoutercomma(charselect['lenkind']))
                lenkind = lenkind.groupdict()
                for lk in ['len', 'kind']:
                    if lenkind[lk + '2']:
                        lenkind[lk] = lenkind[lk + '2']
                    charselect[lk] = lenkind[lk]
                    del lenkind[lk + '2']
                if lenkind['f2py_len'] is not None:
                    # used to specify the length of assumed length strings
                    charselect['f2py_len'] = lenkind['f2py_len']
            del charselect['lenkind']
            for k in list(charselect.keys()):
                if not charselect[k]:
                    del charselect[k]
            for k, i in list(charselect.items()):
                charselect[k] = rmbadname1(i)
        elif typespec == 'type':
            # Match type(Name) or type(Name(params)) for parameterized types
            typename = typenamepattern.match(selector)
            if typename:
                typename = typename.group('name')
            else:
                outmess(f'cracktypespec: no typename found in {typespec + selector}\n')
        else:
            outmess(f'cracktypespec: no selector used for {repr(selector)}\n')
    return kindselect, charselect, typename


def cracktypeparams(typespec, selector):
    """Return the type parameter list of a ``type(name(...))`` selector."""
    if typespec != 'type' or not selector:
        return None
    m = typenamepattern.match(selector)
    if m and m.group('params'):
        return m.group('params')[1:-1].strip()
    return None
######


def setattrspec(decl, attr, force=0):
    if not decl:
        decl = {}
    if not attr:
        return decl
    if 'attrspec' not in decl:
        decl['attrspec'] = [attr]
        return decl
    if force:
        decl['attrspec'].append(attr)
    if attr in decl['attrspec']:
        return decl
    if attr == 'static' and 'automatic' not in decl['attrspec']:
        decl['attrspec'].append(attr)
    elif attr == 'automatic' and 'static' not in decl['attrspec']:
        decl['attrspec'].append(attr)
    elif attr == 'public':
        if 'private' not in decl['attrspec']:
            decl['attrspec'].append(attr)
    elif attr == 'private':
        if 'public' not in decl['attrspec']:
            decl['attrspec'].append(attr)
    else:
        decl['attrspec'].append(attr)
    return decl


def setkindselector(decl, sel, force=0):
    if not decl:
        decl = {}
    if not sel:
        return decl
    if 'kindselector' not in decl:
        decl['kindselector'] = sel
        return decl
    for k in list(sel.keys()):
        if force or k not in decl['kindselector']:
            decl['kindselector'][k] = sel[k]
    return decl


def setcharselector(decl, sel, force=0):
    if not decl:
        decl = {}
    if not sel:
        return decl
    if 'charselector' not in decl:
        decl['charselector'] = sel
        return decl

    for k in list(sel.keys()):
        if force or k not in decl['charselector']:
            decl['charselector'][k] = sel[k]
    return decl


def getblockname(block, unknown='unknown'):
    if 'name' in block:
        return block['name']
    return unknown

# post processing


def setmesstext(block):
    global filepositiontext

    try:
        filepositiontext = f"In: {block['from']}:{block['name']}\n"
    except Exception:
        pass


def get_usedict(block):
    usedict = {}
    if 'parent_block' in block:
        usedict = get_usedict(block['parent_block'])
    if 'use' in block:
        usedict.update(block['use'])
    return usedict


def get_useparameters(block, param_map=None):
    global f90modulevars

    if param_map is None:
        param_map = {}
    usedict = get_usedict(block)
    if not usedict:
        return param_map
    for usename, mapping in list(usedict.items()):
        usename = usename.lower()
        if usename not in f90modulevars:
            # Inject kind names from an intrinsic module when that module is used.
            if usename in _INTRINSIC_MODULE_VARS:
                f90modulevars[usename] = _INTRINSIC_MODULE_VARS[usename]
                f90moduleparams[usename] = get_parameters(
                    f90modulevars[usename])
            else:
                outmess(f'get_useparameters: no module {usename} info used by '
                        f'{block.get("name")}\n')
                continue
        params = f90moduleparams[usename]
        if not params:
            continue
        # Apply use...only / renaming mapping (F2018 11.2.2)
        if mapping:
            only = mapping.get('only', 0)
            name_map = mapping.get('map', {})
            if only and name_map:
                # use module, only: a, b => c
                # Only import listed names, applying renames
                filtered = {}
                for local_name, mod_name in name_map.items():
                    mod_name_l = mod_name.lower()
                    if mod_name_l in params:
                        filtered[local_name.lower()] = params[mod_name_l]
                params = filtered
            elif name_map:
                # use module, a => b (rename without only)
                renamed = dict(params)
                for local_name, mod_name in name_map.items():
                    mod_name_l = mod_name.lower()
                    if mod_name_l in renamed:
                        renamed[local_name.lower()] = renamed.pop(mod_name_l)
                params = renamed
        for k, v in list(params.items()):
            if k in param_map:
                outmess(f'get_useparameters: overriding parameter {k!r} with'
                        f' value from module {usename!r}\n')
            param_map[k] = v

    return param_map


def _block_uses_intrinsic_module(block):
    """Check if block or children use an intrinsic module."""
    if isinstance(block, list):
        return any(_block_uses_intrinsic_module(b) for b in block)
    usedict = block.get('use', {})
    for modname in usedict:
        if modname.lower() in _INTRINSIC_MODULE_VARS:
            return True
    for child in block.get('body', []):
        if _block_uses_intrinsic_module(child):
            return True
    return False


def postcrack2(block, tab='', param_map=None):
    global f90modulevars

    if not f90modulevars and not _block_uses_intrinsic_module(block):
        return block
    if isinstance(block, list):
        ret = [postcrack2(g, tab=tab + '\t', param_map=param_map)
               for g in block]
        return ret
    setmesstext(block)
    outmess(f"{tab}Block: {block['name']}\n", 0)

    if param_map is None:
        param_map = get_useparameters(block)

    if param_map is not None and 'vars' in block:
        vars = block['vars']
        for n in list(vars.keys()):
            var = vars[n]
            if 'kindselector' in var:
                kind = var['kindselector']
                if 'kind' in kind:
                    val = kind['kind']
                    if val in param_map:
                        kind['kind'] = param_map[val]
    new_body = [postcrack2(b, tab=tab + '\t', param_map=param_map)
                for b in block['body']]
    block['body'] = new_body

    return block


def postcrack(block, args=None, tab=''):
    """
    TODO:
          function return values
          determine expression types if in argument list
    """
    global usermodules, onlyfunctions

    if isinstance(block, list):
        gret = []
        uret = []
        for g in block:
            setmesstext(g)
            g = postcrack(g, tab=tab + '\t')
            # sort user routines to appear first
            if 'name' in g and '__user__' in g['name']:
                uret.append(g)
            else:
                gret.append(g)
        return uret + gret
    setmesstext(block)
    if not isinstance(block, dict) and 'block' not in block:
        raise Exception('postcrack: Expected block dictionary instead of ' +
                        str(block))
    if 'name' in block and not block['name'] == 'unknown_interface':
        outmess(f"{tab}Block: {block['name']}\n", 0)
    if (block.get('block') in ('function', 'subroutine')
            and 'saved_interface' not in block):
        block['saved_interface'] = crack2fortrangen(
            block, '\n' + ' ' * 6, as_interface=True)
    block = analyzeargs(block)
    block = analyzecommon(block)
    block['vars'] = analyzevars(block)
    block['sortvars'] = sortvarnames(block['vars'])
    if block.get('args'):
        args = block['args']
    block['body'] = analyzebody(block, args, tab=tab)

    userisdefined = []
    if 'use' in block:
        useblock = block['use']
        for k in list(useblock.keys()):
            if '__user__' in k:
                userisdefined.append(k)
    else:
        useblock = {}
    name = ''
    if 'name' in block:
        name = block['name']
    # and not userisdefined: # Build a __user__ module
    if block.get('externals'):
        interfaced = []
        if 'interfaced' in block:
            interfaced = block['interfaced']
        mvars = copy.copy(block['vars'])
        if name:
            mname = name + '__user__routines'
        else:
            mname = 'unknown__user__routines'
        if mname in userisdefined:
            i = 1
            while f"{mname}_{i}" in userisdefined:
                i = i + 1
            mname = f"{mname}_{i}"
        interface = {'block': 'interface', 'body': [],
                     'vars': {}, 'name': name + '_user_interface'}
        for e in block['externals']:
            if e in interfaced:
                edef = []
                j = -1
                for b in block['body']:
                    j = j + 1
                    if b['block'] == 'interface':
                        i = -1
                        for bb in b['body']:
                            i = i + 1
                            if 'name' in bb and bb['name'] == e:
                                edef = copy.copy(bb)
                                del b['body'][i]
                                break
                        if edef:
                            if not b['body']:
                                del block['body'][j]
                            del interfaced[interfaced.index(e)]
                            break
                interface['body'].append(edef)
            elif e in mvars and not isexternal(mvars[e]):
                interface['vars'][e] = mvars[e]
        if interface['vars'] or interface['body']:
            block['interfaced'] = interfaced
            mblock = {'block': 'python module', 'body': [
                interface], 'vars': {}, 'name': mname, 'interfaced': block['externals']}
            useblock[mname] = {}
            usermodules.append(mblock)
    if useblock:
        block['use'] = useblock
    return block


def _sort_cyclic_dependents(dep, vars):
    """Order one cyclic depend group by real dependency edges.

    ``sortvarnames`` calls this only after a cycle has been reported.
    The usual f2py cycle is an input array dimensioned by hide scalars
    that are computed from that array (``n = shape(a, ...)``,
    ``lda = MAX(1, n)``). Those array-to-hide edges are dropped so
    ``getarrdims`` can blank them, and every remaining edge is honored
    by a topological sort. An initializer therefore runs after the
    names it uses, independent of declaration order.

    A cycle that is not one of those shape back-edges is left in
    ``dep`` order. The caller has already reported it. This function
    does not recurse.
    """
    remaining = set(dep)

    def blocking(name):
        var = vars[name]
        blocked_by = []
        for dep_name in var.get('depend') or []:
            if dep_name not in remaining or dep_name == name:
                continue
            target = vars.get(dep_name, {})
            # Dimension back-edge onto a hide value or an initializer.
            if (isarray(var) and isintent_nothide(var)
                    and (isintent_hide(target) or '=' in target)):
                continue
            blocked_by.append(dep_name)
        return blocked_by

    ordered = []
    while remaining:
        ready = [name for name in dep
                 if name in remaining and not blocking(name)]
        if not ready:
            ordered.extend(name for name in dep if name in remaining)
            break
        ordered.append(ready[0])
        remaining.remove(ready[0])
    return ordered


def sortvarnames(vars):
    indep = []
    dep = []
    for v in list(vars.keys()):
        if 'depend' in vars[v] and vars[v]['depend']:
            dep.append(v)
        else:
            indep.append(v)
    n = len(dep)
    i = 0
    while dep:  # XXX: How to catch dependence cycles correctly?
        v = dep[0]
        fl = 0
        for w in dep[1:]:
            if w in vars[v]['depend']:
                fl = 1
                break
        if fl:
            dep = dep[1:] + [v]
            i = i + 1
            if i > n:
                errmess('sortvarnames: failed to compute dependencies because'
                        ' of cyclic dependencies between '
                        + ', '.join(dep) + '\n')
                indep = indep + _sort_cyclic_dependents(dep, vars)
                break
        else:
            indep.append(v)
            dep = dep[1:]
            n = len(dep)
            i = 0
    return indep


def analyzecommon(block):
    if not hascommon(block):
        return block
    commonvars = []
    for k in list(block['common'].keys()):
        comvars = []
        for e in block['common'][k]:
            m = re.match(
                r'\A\s*\b(?P<name>.*?)\b\s*(\((?P<dims>.*?)\)|)\s*\Z', e, re.I)
            if m:
                dims = []
                if m.group('dims'):
                    dims = [x.strip()
                            for x in markoutercomma(m.group('dims')).split('@,@')]
                n = rmbadname1(m.group('name').strip())
                if n in block['vars']:
                    if 'attrspec' in block['vars'][n]:
                        block['vars'][n]['attrspec'].append(
                            f"dimension({','.join(dims)})")
                    else:
                        block['vars'][n]['attrspec'] = [
                            f"dimension({','.join(dims)})"]
                elif dims:
                    block['vars'][n] = {
                        'attrspec': [f"dimension({','.join(dims)})"]}
                else:
                    block['vars'][n] = {}
                if n not in commonvars:
                    commonvars.append(n)
            else:
                n = e
                errmess(
                    f'analyzecommon: failed to extract "<name>[(<dims>)]" from "{e}" in common /{k}/.\n')
            comvars.append(n)
        block['common'][k] = comvars
    if 'commonvars' not in block:
        block['commonvars'] = commonvars
    else:
        block['commonvars'] = block['commonvars'] + commonvars
    return block


def analyzebody(block, args, tab=''):
    global usermodules, skipfuncs, onlyfuncs, f90modulevars

    setmesstext(block)

    maybe_private = {
        key: value
        for key, value in block['vars'].items()
        if 'attrspec' not in value or 'public' not in value['attrspec']
    }

    body = []
    for b in block['body']:
        b['parent_block'] = block
        if b['block'] in ['function', 'subroutine']:
            if args is not None and b['name'] not in args:
                continue
            else:
                as_ = b['args']
            # Add private members to skipfuncs for gh-23879
            if b['name'] in maybe_private.keys():
                skipfuncs.append(b['name'])
            if b['name'] in skipfuncs:
                continue
            if onlyfuncs and b['name'] not in onlyfuncs:
                continue
            b['saved_interface'] = crack2fortrangen(
                b, '\n' + ' ' * 6, as_interface=True)

        else:
            as_ = args
        b = postcrack(b, as_, tab=tab + '\t')
        if b['block'] in ['interface', 'abstract interface'] and \
           not b['body'] and not b.get('implementedby'):
            if 'f2pyenhancements' not in b:
                continue
        if b['block'].replace(' ', '') == 'pythonmodule':
            usermodules.append(b)
        else:
            if b['block'] == 'module':
                f90modulevars[b['name']] = b['vars']
            body.append(b)
    return body


def buildimplicitrules(block):
    setmesstext(block)
    implicitrules = defaultimplicitrules
    attrrules = {}
    if 'implicit' in block:
        if block['implicit'] is None:
            implicitrules = None
            if verbose > 1:
                outmess(
                    f"buildimplicitrules: no implicit rules for routine {repr(block['name'])}.\n")
        else:
            for k in list(block['implicit'].keys()):
                if block['implicit'][k].get('typespec') not in ['static', 'automatic']:
                    implicitrules[k] = block['implicit'][k]
                else:
                    attrrules[k] = block['implicit'][k]['typespec']
    return implicitrules, attrrules


def myeval(e, g=None, l=None):
    """ Like `eval` but returns only integers and floats """
    r = eval(e, g, l)
    if type(r) in [int, float]:
        return r
    raise ValueError(f'r={r!r}')


getlincoef_re_1 = re.compile(r'\A\b\w+\b\Z', re.I)


def getlincoef(e, xset):  # e = a*x+b ; x in xset
    """
    Obtain ``a`` and ``b`` when ``e == "a*x+b"``, where ``x`` is a symbol in
    xset.

    >>> getlincoef('2*x + 1', {'x'})
    (2, 1, 'x')
    >>> getlincoef('3*x + x*2 + 2 + 1', {'x'})
    (5, 3, 'x')
    >>> getlincoef('0', {'x'})
    (0, 0, None)
    >>> getlincoef('0*x', {'x'})
    (0, 0, 'x')
    >>> getlincoef('x*x', {'x'})
    (None, None, None)

    This can be tricked by sufficiently complex expressions

    >>> getlincoef('(x - 0.5)*(x - 1.5)*(x - 1)*x + 2*x + 3', {'x'})
    (2.0, 3.0, 'x')
    """
    try:
        c = int(myeval(e, {}, {}))
        return 0, c, None
    except Exception:
        pass
    if getlincoef_re_1.match(e):
        return 1, 0, e
    len_e = len(e)
    for x in xset:
        if len(x) > len_e:
            continue
        if re.search(r'\w\s*\([^)]*\b' + x + r'\b', e):
            # skip function calls having x as an argument, e.g max(1, x)
            continue
        re_1 = re.compile(r'(?P<before>.*?)\b' + x + r'\b(?P<after>.*)', re.I)
        m = re_1.match(e)
        if m:
            try:
                m1 = re_1.match(e)
                while m1:
                    ee = f"{m1.group('before')}({0}){m1.group('after')}"
                    m1 = re_1.match(ee)
                b = myeval(ee, {}, {})
                m1 = re_1.match(e)
                while m1:
                    ee = f"{m1.group('before')}({1}){m1.group('after')}"
                    m1 = re_1.match(ee)
                a = myeval(ee, {}, {}) - b
                m1 = re_1.match(e)
                while m1:
                    ee = f"{m1.group('before')}({0.5}){m1.group('after')}"
                    m1 = re_1.match(ee)
                c = myeval(ee, {}, {})
                # computing another point to be sure that expression is linear
                m1 = re_1.match(e)
                while m1:
                    ee = f"{m1.group('before')}({1.5}){m1.group('after')}"
                    m1 = re_1.match(ee)
                c2 = myeval(ee, {}, {})
                if (a * 0.5 + b == c and a * 1.5 + b == c2):
                    return a, b, x
            except Exception:
                pass
            break
    return None, None, None


word_pattern = re.compile(r'\b[a-z][\w$]*\b', re.I)


def _get_depend_dict(name, vars, deps):
    # ``deps[name]`` is stored before walking dependencies so a cycle
    # returns the partial list instead of recursing without bound.
    if name in deps:
        return deps[name]
    if name not in vars:
        outmess(f'_get_depend_dict: no dependence info for {repr(name)}\n')
        deps[name] = []
        return []
    deps[name] = []
    # Extend an existing depend list in place, matching the previous
    # behavior of walking that same list object.
    words = vars[name].get('depend')
    if words is None:
        words = []
    if '=' in vars[name] and not isstring(vars[name]):
        # Same literal mask as analyzevars (gh-28700). Names that occur
        # only inside quotes are not dependencies.
        rhs = mask_string_literals(vars[name]['='])
        for word in word_pattern.findall(rhs):
            if word not in words and word in vars and word != name:
                words.append(word)
    if isparameter(vars[name]):
        # the value of a parameter takes the kind and length it declares
        for selector in ('kindselector', 'charselector'):
            for text in (vars[name].get(selector) or {}).values():
                for word in word_pattern.findall(str(text)):
                    if word not in words and word in vars and word != name:
                        words.append(word)
    for word in list(words):
        for w in _get_depend_dict(word, vars, deps):
            if w not in words and w != name:
                words.append(w)
    deps[name] = words
    return words


def _calc_depend_dict(vars):
    names = list(vars.keys())
    depend_dict = {}
    for n in names:
        _get_depend_dict(n, vars, depend_dict)
    return depend_dict


def get_sorted_names(vars):
    depend_dict = _calc_depend_dict(vars)
    names = []
    for name in list(depend_dict.keys()):
        if not depend_dict[name]:
            names.append(name)
            del depend_dict[name]
    while depend_dict:
        progressed = False
        for name, lst in list(depend_dict.items()):
            if name not in depend_dict:
                continue
            new_lst = [n for n in lst if n in depend_dict]
            if not new_lst:
                names.append(name)
                del depend_dict[name]
                progressed = True
            else:
                depend_dict[name] = new_lst
        if not progressed:
            errmess('get_sorted_names: failed to compute dependencies because'
                    ' of cyclic dependencies between '
                    + ', '.join(depend_dict) + '\n')
            names.extend(depend_dict.keys())
            break
    return [name for name in names if name in vars]


# Fortran 2018 16.4 describes the numbers of each kind by a model: integers
# by their radix and digits, reals by their radix, digits and exponent
# range.  gfortran, whose kinds f2py maps to C types, stores integer(k) as a
# two's complement integer of k bytes and real(k) in an IEEE 754 binary
# format, so the model of each kind follows from its storage.
_integer_kinds = (1, 2, 4, 8, 16)


def _real_formats_for_long_double(nmant, nexp, maxexp):
    # Kinds 4 and 8 are binary32 and binary64. gfortran has both on every
    # target (KIND Type Parameters, The GNU Fortran Compiler).
    formats = {4: (8, 24), 8: (11, 53)}
    # Kind 10 is 80-bit extended precision. gfortran's kind is the mode
    # precision rounded up to bytes, and XFmode is kind 10
    # (gcc/fortran/trans-types.cc, GCC 13.3). C long double has that
    # shape when NumPy reports 63 fraction bits, 15 exponent bits and
    # max exponent 16384. capi_maps.py maps real(10) from the 63 bits.
    # Any other long double has no real(10).
    if (nmant, nexp, maxexp) == (63, 15, 16384):
        formats[10] = (15, 64)
    # Kind 16 is IEEE binary128, (15, 113). gfortran's real(16) is the
    # IBM extended double, two IEEE doubles, when long double has
    # binary64's max exponent and a significand wider than 53 bits
    # (trans-types.cc, the pnan < p branch; numpy/_core/getlimits.py).
    # That kind is left out. It is also left out when long double is
    # itself binary128: powerpc64le -mabi=ieeelongdouble keeps kind 16
    # for the IBM format and uses kind 17 for IEEE binary128
    # (trans-types.cc). A binary128 long double does not identify
    # kind 16, so those targets have no kind 16 here either.
    if (maxexp == 1024 and nmant > 52) or (nmant, maxexp) == (112, 16384):
        return formats
    formats[16] = (15, 113)
    return formats


_ldinfo = np.finfo(np.longdouble)
_real_formats = _real_formats_for_long_double(
    _ldinfo.nmant, _ldinfo.nexp, _ldinfo.maxexp)
# character kinds by the names selected_char_kind takes
_character_kinds = {'ascii': 1, 'iso_10646': 4}
# Logical kinds f2py holds, the same byte widths as the integer kinds.
# Any other kind is rejected, because f2py has no representation for it.
_logical_kinds = (1, 2, 4, 8, 16)
_default_kinds = {'integer': 4, 'real': 4, 'complex': 4, 'logical': 4,
                  'character': 1, 'double precision': 8}


@functools.cache
def _model(typespec, kind):
    """Return the radix, digits and least and greatest exponent of the
    model numbers of a kind, with no exponents for an integer."""
    if typespec == 'integer' and kind in _integer_kinds:
        # every bit but the sign is a digit
        return 2, 8 * kind - 1, None, None
    if typespec in ('real', 'complex') and kind in _real_formats:
        bits, digits = _real_formats[kind]
        # IEEE 754 writes significands from 1 to 2 and the model from 1/2
        # to 1, so the model exponents are one more than IEEE's
        emax = 2 ** (bits - 1)
        return 2, digits, 3 - emax, emax
    raise TypeError(f'f2py knows no {typespec} of kind {kind!r}')


def _huge_number(typespec, kind):
    radix, digits, _, emax = _model(typespec, kind)
    if emax is None:
        return radix ** digits - 1
    return (1 - Fraction(radix) ** -digits) * radix ** emax


def _decimal_exponent(x):
    # the d with 10**d <= x < 10**(d + 1) of a number x > 0, exactly, so
    # INT(LOG10(x)) where x is at least 1
    x = Fraction(x)
    d = math.floor(math.log10(x.numerator) - math.log10(x.denominator))
    while Fraction(10) ** d > x:
        d -= 1
    while Fraction(10) ** (d + 1) <= x:
        d += 1
    return d


@functools.cache
def _precision(kind):
    radix, digits, _, _ = _model('real', kind)
    # INT((p - 1) * LOG10(b)), as the radix is no power of 10
    return _decimal_exponent(Fraction(radix) ** (digits - 1))


@functools.cache
def _range(typespec, kind):
    radix, _, emin, _ = _model(typespec, kind)
    huge = _decimal_exponent(_huge_number(typespec, kind))
    if emin is None:
        return huge
    # INT(MIN(LOG10(huge), -LOG10(tiny))), where tiny is b**(emin - 1)
    return min(huge, _decimal_exponent(Fraction(radix) ** (1 - emin)))


def _default_integer(value):
    return _Integer(value, _default_kinds['integer'])


def _selected_int_kind_func(r):
    # 2**q - 1 holds all of -10**r < n < 10**r exactly when its decimal
    # range is at least r
    kinds = [k for k in _integer_kinds if _range('integer', k) >= r]
    if not kinds:
        return _default_integer(-1)
    # of those the one of least range, and then the least kind
    return _default_integer(min(kinds,
                                key=lambda k: (_range('integer', k), k)))


def _selected_real_kind_func(p=None, r=None, radix=None):
    if p is None and r is None and radix is None:
        raise TypeError('selected_real_kind takes at least one argument')
    p = 0 if p is None else p
    r = 0 if r is None else r
    kinds = [k for k in _real_formats
             if radix is None or _model('real', k)[0] == radix]
    with_p = {k for k in kinds if _precision(k) >= p}
    with_r = {k for k in kinds if _range('real', k) >= r}
    if with_p & with_r:
        # the one of least precision, and then the least kind
        return _default_integer(min(with_p & with_r,
                                    key=lambda k: (_precision(k), k)))
    if not kinds:
        return _default_integer(-5)
    if with_p and with_r:
        return _default_integer(-4)
    return _default_integer(-1 if with_r else -2 if with_p else -3)


def _selected_char_kind_func(name):
    # the name is read without regard to case or trailing blanks
    name = name.rstrip(' ').lower()
    if name == 'default':
        return _default_integer(_default_kinds['character'])
    return _default_integer(_character_kinds.get(name, -1))


class _Integer(int):
    """An integer of a kind, which is None for a kind f2py does not know."""

    def __new__(cls, value, kind=None):
        self = super().__new__(cls, value)
        self.kind = kind
        return self


class _Real:
    """A real of a kind, held exactly as that kind holds it."""

    __slots__ = ('value', 'kind')

    def __init__(self, value, kind=None):
        self.value = _round(Fraction(value), kind)
        self.kind = kind

    def __eq__(self, other):
        if isinstance(other, _Real):
            return self.value == other.value
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __float__(self):
        try:
            return float(self.value)
        except OverflowError:
            return math.copysign(math.inf, self.value)

    def __str__(self):
        return _real_text(self.value, self.kind)

    __repr__ = __str__


class _Logical:
    """A logical of a kind."""

    __slots__ = ('value', 'kind')

    def __init__(self, value, kind=None):
        if kind is not None and kind not in _logical_kinds:
            raise TypeError(f'f2py knows no logical of kind {kind!r}')
        self.value = bool(value)
        self.kind = kind

    def __bool__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, (_Logical, bool)):
            return self.value == bool(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return '.true.' if self.value else '.false.'

    __repr__ = __str__


class _Character(str):
    """A character value of a kind."""

    def __new__(cls, value, kind=None):
        if kind is not None and kind not in _character_kinds.values():
            raise TypeError(f'f2py knows no character of kind {kind!r}')
        self = super().__new__(cls, value)
        self.kind = kind
        return self


class _Complex:
    """A complex of a kind, its parts held exactly as that kind holds them."""

    __slots__ = ('real', 'imag', 'kind')

    def __init__(self, real, imag, kind=None):
        self.real = _round(Fraction(real), kind)
        self.imag = _round(Fraction(imag), kind)
        self.kind = kind

    def __eq__(self, other):
        if isinstance(other, _Complex):
            return (self.real, self.imag) == (other.real, other.imag)
        if (isinstance(other, (int, float, complex))
                and not isinstance(other, bool)):
            other = complex(other)
            return self.real == other.real and self.imag == other.imag
        return NotImplemented

    def __hash__(self):
        if not self.imag:
            return hash(self.real)
        return hash(complex(self))

    def __complex__(self):
        return complex(float(_Real(self.real)), float(_Real(self.imag)))

    def __str__(self):
        return (f'({_real_text(self.real, self.kind)}, '
                f'{_real_text(self.imag, self.kind)})')

    __repr__ = __str__


class _Unknown:
    """A name of known type and kind but unknown value, such as a variable,
    which only inquiries take."""

    __slots__ = ('typespec', 'kind')

    def __init__(self, typespec, kind):
        self.typespec = typespec
        self.kind = kind

    def __bool__(self):
        raise TypeError('the value is not known')


_value_types = {_Integer: 'integer', _Real: 'real', _Complex: 'complex',
                _Logical: 'logical', _Character: 'character'}
_numeric_types = ('integer', 'real', 'complex')


def _type_kind(x):
    """Return the type and kind of a value, or of the elements of an
    array."""
    if isinstance(x, (dict, list)):
        elements = list(x.values()) if isinstance(x, dict) else x
        if not elements:
            raise TypeError('an empty array has no type f2py knows')
        x = elements[0]
    for cls, typespec in _value_types.items():
        if isinstance(x, cls):
            return typespec, x.kind
    raise TypeError(f'{x!r} is not a Fortran value')


def _typed(value):
    """Give a value from Python, such as a parameter that a caller passes,
    its Fortran type, of a kind f2py does not know."""
    if isinstance(value, bool):
        return _Logical(value)
    if isinstance(value, int) and not isinstance(value, _Integer):
        return _Integer(value)
    if isinstance(value, float):
        return _Real(value)
    if isinstance(value, dict) and not isinstance(value, _ParameterArray):
        return _ParameterArray({k: _typed(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_typed(v) for v in value]
    return value


def _integer(value, kind):
    """Return an integer of a kind, which must hold it."""
    if kind is not None:
        digits = _model('integer', kind)[1]
        # two's complement holds -huge - 1 as well
        if not -2 ** digits <= value < 2 ** digits:
            raise OverflowError(f'{value} overflows integer({kind})')
    return _Integer(value, kind)


def _exact(x):
    # the exact value of an integer or real
    if isinstance(x, _Real):
        return x.value
    if isinstance(x, _Integer):
        return Fraction(int(x))
    raise TypeError(f'{x!r} is not an integer or real')


def _parts(x):
    # the exact real and imaginary parts of a number
    if isinstance(x, _Complex):
        return x.real, x.imag
    return _exact(x), Fraction(0)


def _exponent(a):
    # the exponent e of a > 0 in the model, 2**(e - 1) <= a < 2**e
    e = a.numerator.bit_length() - a.denominator.bit_length()
    return e + 1 if a >= Fraction(2) ** e else e


def _round(x, kind):
    """Round x to the nearest real of a kind, from halfway to even, as
    IEEE 754 rounds."""
    if kind is None or not x:
        return x
    radix, digits, emin, _ = _model('real', kind)
    a = abs(x)
    # below the least exponent the numbers are spaced as at it
    spacing = Fraction(radix) ** (max(_exponent(a), emin) - digits)
    n, rest = divmod(a, spacing)
    if 2 * rest > spacing or 2 * rest == spacing and n % 2:
        n += 1
    if n * spacing > _huge_number('real', kind):
        raise OverflowError(f'{x} overflows real({kind})')
    return n * spacing if x > 0 else -n * spacing


def _real_text(x, kind):
    """Write a real as Python writes a float, with the fewest digits that
    read back as the same real of its kind."""
    if kind is None:
        try:
            return repr(float(x))
        except OverflowError:
            return repr(math.copysign(math.inf, x))
    if not x:
        return '0.0'
    radix, digits, emin, _ = _model('real', kind)
    a = abs(x)
    e = _exponent(a)
    spacing = Fraction(radix) ** (max(e, emin) - digits)
    # the reals that read as a lie up to halfway to its neighbours, the one
    # below being nearer at a power of the radix, and halfway reads as
    # whichever is even
    below = spacing
    if a == Fraction(radix) ** (e - 1) and e > emin:
        below = spacing / radix
    low, high = a - below / 2, a + spacing / 2
    closed = (a / spacing) % 2 == 0
    point = _decimal_exponent(a) + 1
    for n in itertools.count(1):
        scale = Fraction(10) ** (point - n)
        least = (math.ceil(low / scale) if closed
                 else math.floor(low / scale) + 1)
        most = (math.floor(high / scale) if closed
                else math.ceil(high / scale) - 1)
        if least <= most:
            d = str(min(max(round(a / scale), least), most))
            break
    # a is 0.<significand> * 10**decimal_point
    significand = d.rstrip('0')
    decimal_point = point - n + len(d)
    text = _float_text(significand, decimal_point)
    return '-' + text if x < 0 else text


def _float_text(digits, point):
    # as repr writes a float, positionally from 1e-4 to below 1e16
    if -4 < point <= 16:
        if point <= 0:
            return '0.' + '0' * -point + digits
        if point >= len(digits):
            return digits + '0' * (point - len(digits)) + '.0'
        return digits[:point] + '.' + digits[point:]
    mantissa = digits[0] + ('.' + digits[1:] if len(digits) > 1 else '')
    return f'{mantissa}e{point - 1:+03d}'


def _exponent_bounds(kind):
    # the exponents of the powers of 2 outside of which a real of the kind,
    # or of any kind if kind is None, overflows or rounds to zero
    kinds = _real_formats if kind is None else [kind]
    models = [_model('real', k) for k in kinds]
    return (min(emin - digits - 1 for _, digits, emin, _ in models),
            max(emax + 1 for _, _, _, emax in models))


def _real_power(x, n, kind):
    """Raise an exact real x to an integer power n, for a real of a
    kind."""
    if not x:
        if n < 0:
            raise ZeroDivisionError('0 to a negative power')
        return _Real(Fraction(not n), kind)
    if abs(x) != 1:
        # a power far outside the kind's exponents overflows or rounds to
        # zero, and need not be computed
        low, high = _exponent_bounds(kind)
        log2 = n * (math.log2(abs(x.numerator)) - math.log2(x.denominator))
        if log2 > high:
            raise OverflowError(f'{x}**{n} overflows real({kind})')
        if log2 < low:
            return _Real(0, kind)
    return _Real(x ** n, kind)


def _promote(ta, ka, tb, kb):
    """Return the type and kind of the result of a numeric operation: the
    kind of the operand of greater range of integers and of greater
    precision of reals (Fortran 2018 10.1.5.2.1)."""
    typespec = max(ta, tb, key=_numeric_types.index)
    kinds = [k for t, k in ((ta, ka), (tb, kb))
             if t != 'integer' or typespec == 'integer']
    if None in kinds:
        return typespec, None
    if typespec == 'integer':
        return typespec, max(kinds, key=lambda k: (_range('integer', k), k))
    return typespec, max(kinds, key=lambda k: (_precision(k), k))


def _convert(x, typespec, kind):
    """Convert a value to a type and kind, as assignment does, an integer
    or real taking the real part of a complex."""
    t = _type_kind(x)[0]
    if typespec in _numeric_types and t in _numeric_types:
        real, imag = _parts(x)
        if typespec == 'integer':
            return _integer(math.trunc(real), kind)
        if typespec == 'real':
            return _Real(real, kind)
        return _Complex(real, imag, kind)
    if typespec == t == 'logical':
        return _Logical(x.value, kind)
    if typespec == t == 'character':
        return _Character(x, kind)
    raise TypeError(f'{t} does not convert to {typespec}')


_arithmetic = {'Add': lambda x, y: x + y, 'Sub': lambda x, y: x - y,
               'Mult': lambda x, y: x * y}


def _binary(op, a, b):
    """Apply a binary operator to two values, as Fortran does."""
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        # Python would repeat or join lists, where Fortran works elementwise
        raise TypeError('array arithmetic is not supported')
    (ta, ka), (tb, kb) = _type_kind(a), _type_kind(b)
    if op == 'MatMult':
        # the // of Fortran joins characters of one kind
        if not ta == tb == 'character' or ka != kb:
            raise TypeError('// joins characters of one kind')
        return _Character(str(a) + str(b), ka)
    if ta not in _numeric_types or tb not in _numeric_types:
        raise TypeError(f'{op} is not defined for {ta} and {tb}')
    typespec, kind = _promote(ta, ka, tb, kb)
    if op == 'Pow' and tb == 'integer' and ta != 'integer':
        # an integer power is of the type and kind of its base
        if ta == 'complex':
            return _complex_power(*_parts(a), int(b), ka)
        return _real_power(_exact(a), int(b), ka)
    if typespec == 'complex':
        return _complex_binary(op, *_parts(_convert(a, typespec, kind)),
                               *_parts(_convert(b, typespec, kind)), kind)
    x = _exact(_convert(a, typespec, kind))
    y = _exact(_convert(b, typespec, kind))
    if op == 'Pow':
        if typespec == 'integer':
            return _integer_power(int(x), int(y), kind)
        # a real power is evaluated only when it is an integer, and Fortran
        # forbids a negative base
        if y.denominator != 1 or x < 0:
            raise TypeError(f'{x}**{y} is not supported')
        return _real_power(x, int(y), kind)
    if op == 'Div':
        if not y:
            raise ZeroDivisionError('division by zero')
        if typespec == 'integer':
            # the quotient of integers is truncated
            return _integer(math.trunc(x / y), kind)
        return _Real(x / y, kind)
    if op not in _arithmetic:
        raise TypeError(f'{op} is not Fortran')
    if typespec == 'integer':
        return _integer(int(_arithmetic[op](x, y)), kind)
    return _Real(_arithmetic[op](x, y), kind)


def _complex_binary(op, p, q, r, s, kind):
    # the operation on p + q i and r + s i, each part rounded once
    if op == 'Add':
        return _Complex(p + r, q + s, kind)
    if op == 'Sub':
        return _Complex(p - r, q - s, kind)
    if op == 'Mult':
        return _Complex(p * r - q * s, p * s + q * r, kind)
    if op == 'Div':
        d = r * r + s * s
        if not d:
            raise ZeroDivisionError('division by zero')
        return _Complex((p * r + q * s) / d, (q * r - p * s) / d, kind)
    raise TypeError(f'{op} of a complex is not supported')


def _complex_power(p, q, n, kind):
    """Raise p + q i to an integer power n, for a complex of a kind."""
    if not (p or q):
        if n < 0:
            raise ZeroDivisionError('0 to a negative power')
        return _Complex(Fraction(not n), 0, kind)
    m = p * p + q * q
    if m != 1:
        # a power far outside the kind's exponents overflows or rounds to
        # zero, and need not be computed
        low, high = _exponent_bounds(kind)
        log2 = n * (math.log2(m.numerator) - math.log2(m.denominator)) / 2
        if log2 > high:
            raise OverflowError(f'({p}, {q})**{n} overflows complex({kind})')
        if log2 < low:
            return _Complex(0, 0, kind)
    if n < 0:
        p, q, n = p / m, -q / m, -n
    real, imag = Fraction(1), Fraction(0)
    while n:
        if n % 2:
            real, imag = real * p - imag * q, real * q + imag * p
        p, q, n = p * p - q * q, 2 * p * q, n // 2
    return _Complex(real, imag, kind)


def _integer_power(x, n, kind):
    if n < 0:
        # 1 / x**-n truncated is 0 unless x is 1 or -1
        if not x:
            raise ZeroDivisionError('0 to a negative power')
        return _integer(x ** (n % 2) if abs(x) == 1 else 0, kind)
    # a power of at least 2 beyond the bits of the widest integer overflows
    # it, and need not be computed
    bits = _model('integer', max(_integer_kinds) if kind is None else kind)[1]
    if abs(x) > 1 and n > bits:
        raise OverflowError(f'{x}**{n} overflows integer({kind})')
    return _integer(x ** n, kind)


def _unary(op, a):
    """Apply a unary operator to a value, as Fortran does."""
    if isinstance(a, (list, dict)):
        raise TypeError('array arithmetic is not supported')
    typespec, kind = _type_kind(a)
    if op == 'Not':
        if typespec != 'logical':
            raise TypeError('.not. takes a logical')
        return _Logical(not a, kind)
    if typespec not in _numeric_types:
        raise TypeError(f'{op} is not defined for {typespec}')
    if op == 'UAdd':
        return a
    if typespec == 'integer':
        return _integer(-int(a), kind)
    if typespec == 'complex':
        return _Complex(-a.real, -a.imag, kind)
    return _Real(-a.value, kind)


_relations = {'Eq': lambda x, y: x == y, 'NotEq': lambda x, y: x != y,
              'Lt': lambda x, y: x < y, 'LtE': lambda x, y: x <= y,
              'Gt': lambda x, y: x > y, 'GtE': lambda x, y: x >= y}


def _relational(op, a, b):
    """Compare two values, as Fortran does."""
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        raise TypeError('array comparison is not supported')
    (ta, ka), (tb, kb) = _type_kind(a), _type_kind(b)
    if ta == tb == 'character' and ka == kb:
        # the shorter is compared as if it were padded with blanks
        n = max(len(a), len(b))
        x, y = a.ljust(n), b.ljust(n)
    elif ta in _numeric_types and tb in _numeric_types:
        typespec, kind = _promote(ta, ka, tb, kb)
        x = _parts(_convert(a, typespec, kind))
        y = _parts(_convert(b, typespec, kind))
        if typespec != 'complex':
            x, y = x[0], y[0]
        elif op not in ('Eq', 'NotEq'):
            raise TypeError('complex values are not ordered')
    else:
        raise TypeError(f'{ta} and {tb} do not compare')
    return _Logical(_relations[op](x, y), _default_kinds['logical'])


def _logical(op, *values):
    """Apply .and. or .or. to logical values."""
    if any(isinstance(v, (list, dict)) or _type_kind(v)[0] != 'logical'
           for v in values):
        raise TypeError(f'{op} takes logicals')
    kinds = {v.kind for v in values}
    value = all(values) if op == 'And' else any(values)
    kind = kinds.pop() if len(kinds) == 1 else _default_kinds['logical']
    return _Logical(value, kind)


def _complex_literal(real, imag):
    """Return the complex literal (real, imag), of the kind of its real part
    of greater precision, or of the default real if both are integers."""
    (tr, kr), (ti, ki) = _type_kind(real), _type_kind(imag)
    if not {tr, ti} <= {'integer', 'real'}:
        raise TypeError('the parts of a complex literal are numbers')
    kinds = [k for t, k in ((tr, kr), (ti, ki)) if t == 'real']
    if not kinds:
        kind = _default_kinds['real']
    elif None in kinds:
        kind = None
    else:
        kind = max(kinds, key=lambda k: (_precision(k), k))
    return _Complex(_exact(real), _exact(imag), kind)


_operations = {'_binary': _binary, '_unary': _unary,
               '_relational': _relational, '_logical': _logical,
               '_complex_literal': _complex_literal}


def _operation(name, op, *args):
    return ast.Call(ast.Name(name, ast.Load()), [ast.Constant(op), *args], [])


class _FortranOperations(ast.NodeTransformer):
    """Make each operation of an expression a call of one of _operations,
    which follow Fortran's rules for the types and kinds of operands.
    Python that is not also Fortran is refused."""

    _nodes = (ast.Expression, ast.Name, ast.Load, ast.Call, ast.keyword,
              ast.List, ast.Tuple, ast.BinOp, ast.UnaryOp, ast.BoolOp,
              ast.Compare, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)

    def generic_visit(self, node):
        if not isinstance(node, self._nodes):
            raise TypeError(f'{type(node).__name__} is not Fortran')
        return super().generic_visit(node)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise TypeError('only named functions are called')
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        # (re, im) is a complex literal
        self.generic_visit(node)
        if len(node.elts) != 2:
            raise TypeError('a complex literal has two parts')
        return ast.Call(ast.Name('_complex_literal', ast.Load()), node.elts,
                        [])

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return _operation('_binary', type(node.op).__name__, node.left,
                          node.right)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        return _operation('_unary', type(node.op).__name__, node.operand)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        return _operation('_logical', type(node.op).__name__, *node.values)

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) > 1:
            raise TypeError('Fortran relations do not chain')
        return _operation('_relational', type(node.ops[0]).__name__,
                          node.left, node.comparators[0])


# The parts of Fortran that Python reads differently or not at all:
# character literals, which double their delimiter, with an optional kind
# before them; //; logical literals; the dotted and /= operators; and
# numbers, with an optional exponent and kind, a d exponent making them
# double precision and the q exponent of gfortran real(16).
_dotted_names = 'eq|ne|lt|le|gt|ge|not|and|or|eqv|neqv|true|false'
_fortran_token_re = re.compile(
    r"(?:\b(?P<char_kind>\w+)_)?(?P<char>'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")"
    r'|(?P<concat>//)'
    r'|\.(?P<logical>true|false)\.(?:_(?P<logical_kind>\w+))?'
    r'|(?P<operator>\.(?:eq|ne|lt|le|gt|ge|not|and|or)\.|/=)'
    rf'|(?<!\w)(?P<number>\d+(?:\.(?!(?:{_dotted_names})\.)\d*)?|\.\d+)'
    r'(?:(?P<letter>[deq])(?P<exponent>[-+]?\d+))?(?:_(?P<kind>\w+))?',
    re.I)
_python_operators = {'.eq.': '==', '.ne.': '!=', '.lt.': '<', '.le.': '<=',
                     '.gt.': '>', '.ge.': '>=', '.not.': 'not',
                     '.and.': 'and', '.or.': 'or', '/=': '!='}


def _literal_kind(kind, typespec, names):
    """Return the kind a literal names, or the default kind."""
    if kind is None:
        return _default_kinds[typespec]
    if kind.isdigit():
        return int(kind)
    try:
        value = names[kind]
    except KeyError:
        raise NameError(f'name {kind!r} is not defined') from None
    if _type_kind(value)[0] != 'integer':
        raise TypeError(f'kind {kind} is not an integer')
    return int(value)


def _number(m, names, c_literals):
    """Return the value of a number literal."""
    digits, letter, kind = m['number'], m['letter'], m['kind']
    if letter is None and '.' not in digits:
        value = int(digits)
        if kind is not None:
            return _integer(value, _literal_kind(kind, 'integer', names))
        try:
            return _integer(value, _default_kinds['integer'])
        except OverflowError:
            # gfortran refuses such a literal, and flang reads it as
            # integer(8), so its kind is not known
            return _Integer(value)
    letter = (letter or 'e').lower()
    if letter == 'e':
        # C reads a floating constant without a suffix as a double
        real_kind = (_literal_kind(kind, 'real', names) if kind is not None
                     else 8 if c_literals else _default_kinds['real'])
    elif kind is None:
        # q is gfortran's real(16); d is real(8)
        real_kind = 8 if letter == 'd' else 16
    else:
        raise TypeError(f'a {letter} exponent takes no kind')
    mantissa, scale = Fraction(digits), int(m['exponent'] or 0)
    if mantissa:
        # a power of ten far outside the kind's exponents overflows or rounds
        # to zero, and need not be computed
        low, high = _exponent_bounds(real_kind)
        log2 = (scale + math.log10(mantissa)) * math.log2(10)
        if log2 > high:
            raise OverflowError(f'{m[0]} overflows real({real_kind})')
        if log2 < low:
            mantissa = 0
    return _Real(mantissa * Fraction(10) ** scale, real_kind)


def _translate(expr, names, c_literals=False):
    """Translate a Fortran expression to Python, each literal in it becoming
    a name for its value, which the returned dict gives."""
    literals = {}

    def literal(value):
        name = f'_literal{len(literals)}'
        literals[name] = value
        return name

    def token(m):
        if m['char'] is not None:
            s = m['char']
            return literal(_Character(
                s[1:-1].replace(2 * s[0], s[0]),
                _literal_kind(m['char_kind'], 'character', names)))
        if m['concat']:
            return ' @ '
        if m['logical']:
            return literal(_Logical(
                m['logical'].lower() == 'true',
                _literal_kind(m['logical_kind'], 'logical', names)))
        if m['operator']:
            return f" {_python_operators[m['operator'].lower()]} "
        return literal(_number(m, names, c_literals))

    source = _fortran_token_re.sub(token, expr)
    # (/ and /) delimit array constructors as [ and ] do
    return source.replace('(/', '[').replace('/)', ']').strip(), literals


class _ParameterArray(dict):
    # pa(2) in an expression is an element of the parameter array pa
    def __call__(self, *index):
        return self[index if len(index) > 1 else index[0]]


class _Names(collections.abc.Mapping):
    """The values of the names in an expression, in their Fortran types."""

    def __init__(self, *maps):
        self.maps = maps

    def __getitem__(self, name):
        for m in self.maps:
            # text is a value that f2py could not evaluate, and a later map
            # may know at least its type
            if name in m and (not isinstance(m[name], str)
                              or isinstance(m[name], _Character)):
                return _typed(m[name])
        raise KeyError(name)

    def __iter__(self):
        return iter({name for m in self.maps for name in m})

    def __len__(self):
        return len({name for m in self.maps for name in m})


def _fortran_eval(expr, g_params, params, c_literals=False):
    """Evaluate a Fortran expression, each value of it of a type and kind.

    With c_literals a real literal without a kind is a double, as in the C
    that f2py writes the initial values of arguments in.
    """
    names = _Names(params, g_params)
    source, literals = _translate(expr, names, c_literals)
    tree = _FortranOperations().visit(ast.parse(source, mode='eval'))
    code = compile(ast.fix_missing_locations(tree), '<fortran>', 'eval')
    # no builtins, as those of Python are not Fortran's
    return eval(code, {'__builtins__': {}, **_operations, **_intrinsics,
                       **literals}, names)


def _elements(array):
    # a parameter array is a dict of its elements, keyed by their indices
    return list(array.values()) if isinstance(array, dict) else list(array)


def _kind_arg(kind, typespec):
    # the kind a kind= argument gives, or the default kind
    if kind is None:
        return _default_kinds[typespec]
    if _type_kind(kind)[0] != 'integer':
        raise TypeError('a kind is an integer')
    return int(kind)


def _real_arg(a):
    if isinstance(a, (list, dict)) or _type_kind(a)[0] != 'real':
        raise TypeError(f'{a!r} is not a real')
    return a.value


def _character_arg(c):
    if isinstance(c, (list, dict)) or _type_kind(c)[0] != 'character':
        raise TypeError(f'{c!r} is not a character')
    return c


def _common_type(args):
    """Return the type and kind that integer or real arguments promote
    to, as gfortran and flang promote the arguments of max and mod."""
    if any(isinstance(a, (list, dict)) for a in args):
        raise TypeError('array arguments are not supported')
    types = [_type_kind(a) for a in args]
    if {t for t, _ in types} not in ({'integer'}, {'real'}):
        raise TypeError('the arguments are integers or reals')
    return functools.reduce(lambda x, y: _promote(*x, *y), types)


def _extremum(pick, args):
    typespec, kind = _common_type(args)
    values = [_exact(_convert(a, typespec, kind)) for a in args]
    return _convert(_Real(pick(values)), typespec, kind)


def _nint(a, kind=None):
    v = _real_arg(a)
    # halves round away from zero
    n = math.floor(abs(v) + Fraction(1, 2))
    return _integer(n if v >= 0 else -n, _kind_arg(kind, 'integer'))


def _remainder(a, p, quotient):
    typespec, kind = _common_type([a, p])
    x = _exact(_convert(a, typespec, kind))
    y = _exact(_convert(p, typespec, kind))
    if not y:
        raise ZeroDivisionError('division by zero')
    return _convert(_Real(x - quotient(x / y) * y), typespec, kind)


def _sign(a, b):
    _common_type([a, b])
    x = abs(_exact(a))
    # the result is of the type and kind of a
    return _convert(_Real(x if _exact(b) >= 0 else -x), *_type_kind(a))


def _reduce(op, array):
    elements = _elements(array)
    if not elements:
        raise TypeError('an empty array is not supported')
    return functools.reduce(functools.partial(_binary, op), elements)


def _integer_arg(i):
    if isinstance(i, (list, dict)) or _type_kind(i)[0] != 'integer':
        raise TypeError(f'{i!r} is not an integer')
    return int(i)


def _char(i, kind=None):
    kind = _kind_arg(kind, 'character')
    # a character of kind k holds a code of 8 * k bits
    if not 0 <= _integer_arg(i) < 2 ** (8 * kind):
        raise TypeError(f'{i} is not a code of a character of kind {kind}')
    return _Character(chr(i), kind)


def _ichar(c, kind=None):
    if len(_character_arg(c)) != 1:
        raise TypeError(f'{c!r} is not of length 1')
    return _integer(ord(c), _kind_arg(kind, 'integer'))


def _repeat(string, ncopies):
    if _integer_arg(ncopies) < 0:
        raise TypeError('repeat takes no negative ncopies')
    return _Character(_character_arg(string) * int(ncopies), string.kind)


def _len(string, kind=None):
    # of a character array, the length of each element
    if isinstance(string, (list, dict)):
        string = _elements(string)[0]
    return _integer(len(_character_arg(string)), _kind_arg(kind, 'integer'))


def _size(array, dim=None, kind=None):
    if not isinstance(array, (list, dict)):
        raise TypeError(f'{array!r} is not an array')
    n = len(array) if dim is None else _extent(array, _integer_arg(dim))
    return _integer(n, _kind_arg(kind, 'integer'))


def _inquired(x):
    # the type and kind of a value, an array or a name of unknown value
    if isinstance(x, _Unknown):
        return x.typespec, x.kind
    return _type_kind(x)


def _complex_arg(z):
    if isinstance(z, (list, dict)) or _type_kind(z)[0] != 'complex':
        raise TypeError(f'{z!r} is not a complex')
    return z


def _cmplx(x, y=None, kind=None):
    # without kind=, of the default real kind, whatever the kinds of x and y
    kind = _kind_arg(kind, 'real')
    if _type_kind(x)[0] == 'complex':
        if y is not None:
            raise TypeError('cmplx of a complex takes no y')
        return _convert(x, 'complex', kind)
    return _Complex(_exact(x), 0 if y is None else _exact(y), kind)


def _real(a, kind=None):
    # of the kind of a complex a without kind=
    if kind is None and _type_kind(a)[0] == 'complex':
        return _convert(a, 'real', a.kind)
    return _convert(a, 'real', _kind_arg(kind, 'real'))


def _kind_func(x):
    kind = _inquired(x)[1]
    if kind is None:
        raise TypeError(f'f2py does not know the kind of {x!r}')
    return _default_integer(kind)


def _inquiry(typespecs, result):
    """Return an inquiry function of the model numbers of a kind, which
    takes a value, an array or a name of one of typespecs."""
    def inquiry(x):
        typespec, kind = _inquired(x)
        if typespec not in typespecs or kind is None:
            raise TypeError(f'the {typespec} of kind {kind} has no such model')
        return result(typespec, kind, *_model(typespec, kind))
    return inquiry


# Fortran intrinsics by the Fortran names of their arguments, so that keyword
# calls work too, each with the type and kind of result Fortran gives
_intrinsics = {
    'int': lambda a, kind=None: _convert(a, 'integer',
                                         _kind_arg(kind, 'integer')),
    'nint': _nint,
    'ceiling': lambda a, kind=None: _integer(math.ceil(_real_arg(a)),
                                             _kind_arg(kind, 'integer')),
    'floor': lambda a, kind=None: _integer(math.floor(_real_arg(a)),
                                           _kind_arg(kind, 'integer')),
    'real': _real,
    'float': lambda a: _convert(a, 'real', _default_kinds['real']),
    'dble': lambda a: _convert(a, 'real', _default_kinds['double precision']),
    'cmplx': _cmplx,
    # an extension of gfortran, of double precision
    'dcmplx': lambda x, y=None: _cmplx(
        x, y, _default_integer(_default_kinds['double precision'])),
    'aimag': lambda z: _Real(_complex_arg(z).imag, z.kind),
    'conjg': lambda z: _Complex(_complex_arg(z).real, -z.imag, z.kind),
    'abs': lambda a: _convert(_Real(abs(_exact(a))), *_type_kind(a)),
    'max': lambda *a: _extremum(max, a),
    'min': lambda *a: _extremum(min, a),
    'mod': lambda a, p: _remainder(a, p, math.trunc),
    'modulo': lambda a, p: _remainder(a, p, math.floor),
    'sign': _sign,
    'merge': lambda tsource, fsource, mask:
        tsource if _logical('And', mask) else fsource,
    'achar': _char,
    'char': _char,
    'iachar': _ichar,
    'ichar': _ichar,
    'repeat': _repeat,
    'trim': lambda string: _Character(_character_arg(string).rstrip(' '),
                                      string.kind),
    'len_trim': lambda string, kind=None: _integer(
        len(_character_arg(string).rstrip(' ')), _kind_arg(kind, 'integer')),
    'len': _len,
    'size': _size,
    'sum': lambda array: _reduce('Add', array),
    'product': lambda array: _reduce('Mult', array),
    'maxval': lambda array: _extremum(max, _elements(array)),
    'minval': lambda array: _extremum(min, _elements(array)),
    'any': lambda mask: _logical('Or', *_elements(mask)),
    'all': lambda mask: _logical('And', *_elements(mask)),
    'kind': _kind_func,
    'selected_int_kind': _selected_int_kind_func,
    'selected_real_kind': _selected_real_kind_func,
    'selected_char_kind': _selected_char_kind_func,
    # the model numbers of 16.4 with radix b, digits q or p and exponents
    # emin to emax, as 16.9 defines these inquiries of them
    'bit_size': _inquiry(('integer',), lambda t, k, b, q, emin, emax:
                         _integer(q + 1, k)),
    'digits': _inquiry(('integer', 'real'), lambda t, k, b, p, emin, emax:
                       _default_integer(p)),
    'epsilon': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                        _Real(Fraction(b) ** (1 - p), k)),
    'huge': _inquiry(('integer', 'real'), lambda t, k, b, p, emin, emax:
                     (_integer if t == 'integer' else _Real)(
                         _huge_number(t, k), k)),
    'maxexponent': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                            _default_integer(emax)),
    'minexponent': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                            _default_integer(emin)),
    'precision': _inquiry(('real', 'complex'), lambda t, k, b, p, emin, emax:
                          _default_integer(_precision(k))),
    'radix': _inquiry(('integer', 'real'), lambda t, k, b, p, emin, emax:
                      _default_integer(b)),
    'range': _inquiry(('integer', 'real', 'complex'),
                      lambda t, k, b, p, emin, emax:
                      _default_integer(_range(t, k))),
    'tiny': _inquiry(('real',), lambda t, k, b, p, emin, emax:
                     _Real(Fraction(b) ** (emin - 1), k)),
}


def _fit_length(value, var, g_params, params):
    """Blank pad or cut a character value to the length of var."""
    sel = var.get('charselector') or {}
    try:
        length = _fortran_eval(sel.get('len', sel.get('*', '1')), g_params,
                               params)
    except Exception:
        # an assumed length, e.g. len=*, is the length of the value
        return value
    if not isinstance(length, int):
        return value

    def fit(s):
        if isinstance(s, _Character):
            return _Character(s.ljust(length)[:length], s.kind)
        return s
    if isinstance(value, dict):
        return {k: fit(s) for k, s in value.items()}
    return fit(value)


def _declared_type(name, var, params, implicitrules=None):
    """Return the type and kind that the declaration of a name gives it, or
    its implicit type, with a kind of None where f2py cannot tell it."""
    typespec = var.get('typespec')
    kindselector = var.get('kindselector') or {}
    charselector = var.get('charselector') or {}
    if not typespec and implicitrules and name[:1].lower() in implicitrules:
        rule = implicitrules[name[0].lower()]
        typespec = rule.get('typespec')
        kindselector = rule.get('kindselector') or {}
        charselector = rule.get('charselector') or {}
    if typespec in ('double precision', 'double complex'):
        return ('real' if typespec == 'double precision' else 'complex',
                _default_kinds['double precision'])
    if typespec == 'byte':
        return 'integer', 1
    if typespec not in _default_kinds:
        return None, None
    selector = charselector if typespec == 'character' else kindselector
    if 'kind' in selector:
        try:
            kind = _fortran_eval(selector['kind'], {}, params)
            return typespec, _kind_arg(kind, typespec)
        except Exception:
            return typespec, None
    if '*' in selector and typespec != 'character':
        # real*8 counts bytes, of which a complex takes two per kind
        try:
            return typespec, int(selector['*']) // (
                2 if typespec == 'complex' else 1)
        except ValueError:
            return typespec, None
    return typespec, _default_kinds[typespec]


def _assign(value, typespec, kind):
    """Convert a value to a type and kind, element by element for an array,
    as assigning it does, or keep it if it does not convert."""
    try:
        if isinstance(value, dict):
            return {k: _convert(_typed(v), typespec, kind)
                    for k, v in value.items()}
        return _convert(_typed(value), typespec, kind)
    except (TypeError, OverflowError):
        return value


class _Declarations(collections.abc.Mapping):
    """The names declared in a block as names of unknown value, whose type
    and kind are known for inquiries such as huge(i).  A name of a type or
    procedure has no type declared."""

    def __init__(self, vars, params):
        self.vars, self.params = vars, params

    def __getitem__(self, name):
        if name not in self.vars:
            raise KeyError(name)
        typespec, kind = _declared_type(name, self.vars[name], self.params)
        if typespec is None:
            raise KeyError(name)
        return _Unknown(typespec, kind)

    def __iter__(self):
        return iter(self.vars)

    def __len__(self):
        return len(self.vars)


def get_parameters(vars, global_params={}, implicitrules=None):
    params = copy.copy(global_params)
    g_params = collections.ChainMap(global_params,
                                    _Declarations(vars, params))
    param_names = []
    for n in get_sorted_names(vars):
        if 'attrspec' in vars[n] and 'parameter' in vars[n]['attrspec']:
            param_names.append(n)
    for n in param_names:
        if '=' in vars[n]:
            v = vars[n]['=']
            dimspec = ([s.removeprefix('dimension').strip()
                        for s in vars[n]['attrspec']
                       if s.startswith('dimension')] or [None])[0]
            try:
                params[n] = param_eval(v, g_params, params, dimspec=dimspec)
            except Exception as msg:
                params[n] = v
                outmess(f'get_parameters: got "{msg}" on {n!r}\n')
            if (dimspec or isstring_or_stringarray(vars[n])) and params[n] == v:
                # unlike a number, an array or a string cannot stand for
                # its expression
                del params[n]
                continue
            typespec, kind = _declared_type(n, vars[n], params, implicitrules)
            if typespec:
                params[n] = _assign(params[n], typespec, kind)
            if isstring(vars[n]) and isinstance(params[n], int):
                params[n] = _Character(chr(params[n]),
                                       _default_kinds['character'])
            if isstring_or_stringarray(vars[n]):
                params[n] = _fit_length(params[n], vars[n], g_params, params)
            nl = n.lower()
            if nl != n:
                params[nl] = params[n]
        else:
            print(vars[n])
            outmess(f'get_parameters:parameter {n!r} does not have value?!\n')
    return params


def _eval_dimension(d, params, declared={}):
    """Evaluate a dimension, length or kind that is a constant integer
    expression, in which an inquiry may ask about a declared variable.

    param_parse substitutes parameters and leaves the arithmetic to
    symbolic, which knows no intrinsics and folds 2 * (n / 2) to n.
    """
    try:
        value = _fortran_eval(d, declared, params)
    except Exception:
        return d
    return str(value) if isinstance(value, int) else d


def _eval_length(length, params):
    if length in ['(:)', '(*)', '*']:
        return '(*)'
    return _eval_scalar(length, params)


_real_literal = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[dDeEqQ][-+]?\d+)?(?:_\w+)?'
_real_literal_re = re.compile(
    rf'\s*(?:{_real_literal}|\(\s*{_real_literal}\s*,\s*{_real_literal}\s*\))'
    r'\s*')


def _eval_scalar(value, params, c_literals=False):
    try:
        # TODO: use symbolic from PR #19805
        result = _fortran_eval(value, {}, params, c_literals)
    except (NameError, SyntaxError, TypeError):
        return value
    except Exception as msg:
        errmess(f'"{msg}" in evaluating {value!r} '
                f'(available names: {list(params)})\n')
        return value
    if isinstance(result, (list, dict)):
        return value
    if isinstance(result, (_Real, _Complex)) and not c_literals:
        if _real_literal_re.fullmatch(value):
            # a literal already is its value, and keeps its kind as written
            return value.strip()
        if result.kind not in (None, _default_kinds['real']):
            # without its kind, Fortran would read the value as a default real
            if isinstance(result, _Real):
                return f'{result}_{result.kind}'
            parts = (_real_text(x, result.kind) for x in _parts(result))
            return '({}_{k}, {}_{k})'.format(*parts, k=result.kind)
    return (repr if isinstance(result, str) else str)(result)


def analyzevars(block):
    """
    Sets correct dimension information for each variable/parameter
    """

    global f90modulevars

    setmesstext(block)
    implicitrules, attrrules = buildimplicitrules(block)
    vars = copy.copy(block['vars'])
    if block['block'] == 'function' and block['name'] not in vars:
        vars[block['name']] = {}
    if '' in block['vars']:
        del vars['']
        if 'attrspec' in block['vars']['']:
            gen = block['vars']['']['attrspec']
            for n in set(vars) | {b['name'] for b in block['body']}:
                for k in ['public', 'private']:
                    if k in gen:
                        vars[n] = setattrspec(vars.get(n, {}), k)
    svars = []
    args = block['args']
    for a in args:
        try:
            vars[a]
            svars.append(a)
        except KeyError:
            pass
    for n in list(vars.keys()):
        if n not in args:
            svars.append(n)

    use_params = get_useparameters(block)
    host = block.get('parent_block', {})
    if block['block'] == 'type' and host.get('block') == 'module':
        # components see the parameters of the module by host association,
        # unless a name of the type itself hides them (F2018 19.5.1.4)
        use_params = {**{k: v for k, v in f90moduleparams.get(
            host['name'], {}).items() if k not in vars}, **use_params}
    params = get_parameters(vars, use_params, implicitrules)
    declared = _Declarations(vars, params)
    if block['block'] == 'module':
        # kept for get_useparameters, as the analysis below reduces the
        # dimensions of arrays to their sizes
        f90moduleparams[block['name']] = {
            k: v for k, v in params.items() if k not in use_params}
    # At this point, params are read and interpreted, but
    # the params used to define vars are not yet parsed
    dep_matches = {}
    name_match = re.compile(r'[A-Za-z][\w$]*').match
    for v in list(vars.keys()):
        m = name_match(v)
        if m:
            n = v[m.start():m.end()]
            try:
                dep_matches[n]
            except KeyError:
                dep_matches[n] = re.compile(rf'.*\b{v}\b', re.I).match
    for n in svars:
        if n[0] in list(attrrules.keys()):
            vars[n] = setattrspec(vars[n], attrrules[n[0]])
        if 'typespec' not in vars[n]:
            if not ('attrspec' in vars[n] and 'external' in vars[n]['attrspec']):
                if implicitrules:
                    ln0 = n[0].lower()
                    for k in list(implicitrules[ln0].keys()):
                        if k == 'typespec' and implicitrules[ln0][k] == 'undefined':
                            continue
                        if k not in vars[n]:
                            vars[n][k] = implicitrules[ln0][k]
                        elif k == 'attrspec':
                            for l in implicitrules[ln0][k]:
                                vars[n] = setattrspec(vars[n], l)
                elif n in block['args']:
                    outmess(f"analyzevars: typespec of variable {n!r} is not defined "
                            f"in routine {block['name']}.\n")
        if 'charselector' in vars[n]:
            if 'len' in vars[n]['charselector']:
                vars[n]['charselector']['len'] = _eval_dimension(
                    vars[n]['charselector']['len'], params, declared)
            # an assumed length parameter has the length of its value
            selector = vars[n]['charselector']
            key = 'len' if 'len' in selector else '*'
            value = params.get(n)
            values = value.values() if isinstance(value, dict) else [value]
            lengths = {len(s) for s in values if isinstance(s, str)}
            if (selector.get(key) in ('*', '(*)') and len(lengths) == 1
                    and isparameter(vars[n])):
                selector[key] = str(lengths.pop())

        if 'kindselector' in vars[n]:
            if 'kind' in vars[n]['kindselector']:
                vars[n]['kindselector']['kind'] = _eval_dimension(
                    vars[n]['kindselector']['kind'], params, declared)

        dimension_exprs = {}
        if 'attrspec' in vars[n]:
            attr = vars[n]['attrspec']
            attr.reverse()
            vars[n]['attrspec'] = []
            dim, intent, depend, check, note = None, None, None, None, None
            for a in attr:
                if a[:9] == 'dimension':
                    dim = (a[9:].strip())[1:-1]
                elif a[:6] == 'intent':
                    intent = (a[6:].strip())[1:-1]
                elif a[:6] == 'depend':
                    depend = (a[6:].strip())[1:-1]
                elif a[:5] == 'check':
                    check = (a[5:].strip())[1:-1]
                elif a[:4] == 'note':
                    note = (a[4:].strip())[1:-1]
                else:
                    vars[n] = setattrspec(vars[n], a)
                if intent:
                    if 'intent' not in vars[n]:
                        vars[n]['intent'] = []
                    for c in [x.strip() for x in markoutercomma(intent).split('@,@')]:
                        # Remove spaces so that 'in out' becomes 'inout'
                        tmp = c.replace(' ', '')
                        if tmp not in vars[n]['intent']:
                            vars[n]['intent'].append(tmp)
                    intent = None
                if note:
                    note = note.replace('\\n\\n', '\n\n')
                    note = note.replace('\\n ', '\n')
                    if 'note' not in vars[n]:
                        vars[n]['note'] = [note]
                    else:
                        vars[n]['note'].append(note)
                    note = None
                if depend is not None:
                    if 'depend' not in vars[n]:
                        vars[n]['depend'] = []
                    for c in rmbadname([x.strip() for x in markoutercomma(depend).split('@,@')]):
                        if c not in vars[n]['depend']:
                            vars[n]['depend'].append(c)
                    depend = None
                if check is not None:
                    if 'check' not in vars[n]:
                        vars[n]['check'] = []
                    for c in [x.strip() for x in markoutercomma(check).split('@,@')]:
                        if c not in vars[n]['check']:
                            vars[n]['check'].append(c)
                    check = None
            if dim and 'dimension' not in vars[n]:
                vars[n]['dimension'] = []
                for axis, d in enumerate(rmbadname(
                        [x.strip() for x in markoutercomma(dim).split('@,@')]
                ), start=1):
                    # d is the expression inside the dimension declaration
                    # Evaluate `d` with respect to params
                    try:
                        # the dimension for this variable depends on a
                        # previously defined parameter
                        d = param_parse(_eval_dimension(d, params, declared),
                                        params)
                    except (ValueError, IndexError, KeyError):
                        outmess(
                            'analyzevars: could not parse dimension for '
                            f'variable {d!r}\n'
                        )

                    dim_char = ':' if d == ':' else '*'
                    if d == dim_char:
                        dl = [dim_char]
                    else:
                        dl = markoutercomma(d, ':').split('@:@')
                    if len(dl) == 2 and '*' in dl:  # e.g. dimension(5:*)
                        dl = ['*']
                        d = '*'
                    if (d == '*' and isparameter(vars[n])
                            and isinstance(params.get(n), dict)):
                        # an implied shape parameter has the extent of its value
                        dl = [str(_extent(params[n], axis))]
                    if len(dl) == 1 and dl[0] != dim_char:
                        dl = ['1', dl[0]]
                    if len(dl) == 2:
                        d1, d2 = map(symbolic.Expr.parse, dl)
                        dsize = d2 - d1 + 1
                        d = dsize.tostring(language=symbolic.Language.C)
                        # find variables v that define d as a linear
                        # function, `d == a * v + b`, and store
                        # coefficients a and b for further analysis.
                        solver_and_deps = {}
                        for v in block['vars']:
                            s = symbolic.as_symbol(v)
                            if dsize.contains(s):
                                try:
                                    a, b = dsize.linear_solve(s)

                                    def solve_v(s, a=a, b=b):
                                        return (s - b) / a

                                    all_symbols = set(a.symbols())
                                    all_symbols.update(b.symbols())
                                except RuntimeError as msg:
                                    # d is not a linear function of v,
                                    # however, if v can be determined
                                    # from d using other means,
                                    # implement the corresponding
                                    # solve_v function here.
                                    solve_v = None
                                    all_symbols = set(dsize.symbols())
                                v_deps = {
                                    s.data for s in all_symbols
                                    if s.data in vars}
                                solver_and_deps[v] = solve_v, list(v_deps)
                        # Note that dsize may contain symbols that are
                        # not defined in block['vars']. Here we assume
                        # these correspond to Fortran/C intrinsic
                        # functions or that are defined by other
                        # means. We'll let the compiler validate the
                        # definiteness of such symbols.
                        dimension_exprs[d] = solver_and_deps
                    vars[n]['dimension'].append(d)

        if 'check' not in vars[n] and 'args' in block and n in block['args']:
            # n is an argument that has no checks defined. Here we
            # generate some consistency checks for n, and when n is an
            # array, generate checks for its dimensions and construct
            # initialization expressions.
            n_deps = vars[n].get('depend', [])
            n_checks = []
            n_is_input = l_or(isintent_in, isintent_inout,
                              isintent_inplace)(vars[n])
            if isarray(vars[n]):  # n is array
                for i, d in enumerate(vars[n]['dimension']):
                    coeffs_and_deps = dimension_exprs.get(d)
                    if coeffs_and_deps is None:
                        # d is `:` or `*` or a constant expression
                        pass
                    elif n_is_input:
                        # n is an input array argument and its shape
                        # may define variables used in dimension
                        # specifications.
                        for v, (solver, deps) in coeffs_and_deps.items():
                            def compute_deps(v, deps):
                                for v1 in coeffs_and_deps.get(v, [None, []])[1]:
                                    if v1 not in deps:
                                        deps.add(v1)
                                        compute_deps(v1, deps)
                            all_deps = set()
                            compute_deps(v, all_deps)
                            if (v in n_deps
                                 or '=' in vars[v]
                                 or 'depend' in vars[v]):
                                # Skip a variable that
                                # - n depends on
                                # - has user-defined initialization expression
                                # - has user-defined dependencies
                                continue
                            if solver is not None and v not in all_deps:
                                # v can be solved from d, hence, we
                                # make it an optional argument with
                                # initialization expression:
                                is_required = False
                                init = solver(symbolic.as_symbol(
                                    f'shape({n}, {i})'))
                                init = init.tostring(
                                    language=symbolic.Language.C)
                                vars[v]['='] = init
                                # n needs to be initialized before v. So,
                                # making v dependent on n and on any
                                # variables in solver or d.
                                vars[v]['depend'] = [n] + deps
                                if 'check' not in vars[v]:
                                    # add check only when no
                                    # user-specified checks exist
                                    vars[v]['check'] = [
                                        f'shape({n}, {i}) == {d}']
                            else:
                                # d is a non-linear function on v,
                                # hence, v must be a required input
                                # argument that n will depend on
                                is_required = True
                                if 'intent' not in vars[v]:
                                    vars[v]['intent'] = []
                                if 'in' not in vars[v]['intent']:
                                    vars[v]['intent'].append('in')
                                # v needs to be initialized before n
                                n_deps.append(v)
                                n_checks.append(
                                    f'shape({n}, {i}) == {d}')
                            v_attr = vars[v].get('attrspec', [])
                            if not ('optional' in v_attr
                                    or 'required' in v_attr):
                                v_attr.append(
                                    'required' if is_required else 'optional')
                            if v_attr:
                                vars[v]['attrspec'] = v_attr
                    if coeffs_and_deps is not None:
                        # extend v dependencies with ones specified in attrspec
                        for v, (solver, deps) in coeffs_and_deps.items():
                            v_deps = vars[v].get('depend', [])
                            for aa in vars[v].get('attrspec', []):
                                if aa.startswith('depend'):
                                    aa = ''.join(aa.split())
                                    v_deps.extend(aa[7:-1].split(','))
                            if v_deps:
                                vars[v]['depend'] = list(set(v_deps))
                            if n not in v_deps:
                                n_deps.append(v)
            elif isstring(vars[n]):
                if 'charselector' in vars[n]:
                    if '*' in vars[n]['charselector']:
                        length = _eval_length(vars[n]['charselector']['*'],
                                              params)
                        vars[n]['charselector']['*'] = length
                    elif 'len' in vars[n]['charselector']:
                        length = _eval_length(vars[n]['charselector']['len'],
                                              params)
                        del vars[n]['charselector']['len']
                        vars[n]['charselector']['*'] = length
            if n_checks:
                vars[n]['check'] = n_checks
            if n_deps:
                vars[n]['depend'] = list(set(n_deps))

        if '=' in vars[n]:
            if 'attrspec' not in vars[n]:
                vars[n]['attrspec'] = []
            if ('optional' not in vars[n]['attrspec']) and \
               ('required' not in vars[n]['attrspec']):
                vars[n]['attrspec'].append('optional')
            if 'depend' not in vars[n]:
                vars[n]['depend'] = []
                # Mask string-literal contents so an identifier appearing
                # only inside a character value is not harvested as a
                # dependency (gh-28700).
                rhs = mask_string_literals(vars[n]['='])
                for v, m in list(dep_matches.items()):
                    if m(rhs):
                        vars[n]['depend'].append(v)
                if not vars[n]['depend']:
                    del vars[n]['depend']
            if isscalar(vars[n]):
                # the initial value of an argument is C
                vars[n]['='] = _eval_scalar(vars[n]['='], params, n in args)

    for n in list(vars.keys()):
        if n == block['name']:  # n is block name
            if 'note' in vars[n]:
                block['note'] = vars[n]['note']
            if block['block'] == 'function':
                if 'result' in block and block['result'] in vars:
                    vars[n] = appenddecl(vars[n], vars[block['result']])
                if 'prefix' in block:
                    pr = block['prefix']
                    pr1 = pr.replace('pure', '')
                    ispure = (not pr == pr1)
                    pr = pr1.replace('recursive', '')
                    isrec = (not pr == pr1)
                    m = typespattern[0].match(pr)
                    if m:
                        typespec, selector, attr, edecl = cracktypespec0(
                            m.group('this'), m.group('after'))
                        kindselect, charselect, typename = cracktypespec(
                            typespec, selector)
                        vars[n]['typespec'] = typespec
                        try:
                            if block['result']:
                                vars[block['result']]['typespec'] = typespec
                        except Exception:
                            pass
                        if kindselect:
                            if 'kind' in kindselect:
                                try:
                                    kindselect['kind'] = eval(
                                        kindselect['kind'], {}, params)
                                except Exception:
                                    pass
                            vars[n]['kindselector'] = kindselect
                        if charselect:
                            vars[n]['charselector'] = charselect
                        if typename:
                            vars[n]['typename'] = typename
                            typeparams = cracktypeparams(typespec, selector)
                            if typeparams:
                                vars[n]['typeparams'] = typeparams
                        if ispure:
                            vars[n] = setattrspec(vars[n], 'pure')
                        if isrec:
                            vars[n] = setattrspec(vars[n], 'recursive')
                    else:
                        outmess(
                            f"analyzevars: prefix ({repr(block['prefix'])}) were not used\n")
    if block['block'] not in ['module', 'pythonmodule', 'python module', 'block data']:
        if 'commonvars' in block:
            neededvars = copy.copy(block['args'] + block['commonvars'])
        else:
            neededvars = copy.copy(block['args'])
        for n in list(vars.keys()):
            if l_or(isintent_callback, isintent_aux)(vars[n]):
                neededvars.append(n)
        if 'entry' in block:
            neededvars.extend(list(block['entry'].keys()))
            for k in list(block['entry'].keys()):
                for n in block['entry'][k]:
                    if n not in neededvars:
                        neededvars.append(n)
        if block['block'] == 'function':
            if 'result' in block:
                neededvars.append(block['result'])
            else:
                neededvars.append(block['name'])
        if block['block'] in ['subroutine', 'function']:
            name = block['name']
            if name in vars and 'intent' in vars[name]:
                block['intent'] = vars[name]['intent']
        if block['block'] == 'type':
            neededvars.extend(list(vars.keys()))
        for n in list(vars.keys()):
            if n not in neededvars:
                del vars[n]
    return vars


analyzeargs_re_1 = re.compile(r'\A[a-z]+[\w$]*\Z', re.I)

# the control of an implied-do loop, e.g. the i = 1 in (i, i = 1, n)
_implied_do_re = re.compile(r'\s*([a-z]\w*)\s*=(?!=)', re.I)


def _eval_items(item, g_params, params):
    """Evaluate an item of an array constructor to a list of elements.

    Arrays are flattened and implied-do loops such as (i, 2 * i, i = 1, n)
    are expanded, as in Fortran.
    """
    item = item.strip()
    if item[:2] == '(/' and item[-2:] == '/)':
        item = f'[{item[2:-2]}]'
    if item[:1] + item[-1:] in ('[]', '()'):
        parts = markoutercomma(item[1:-1]).split('@,@')
        if item[0] == '[':
            return [x for part in parts
                    for x in _eval_items(part, g_params, params)]
        for j in (-2, -3):
            m = len(parts) > -j and _implied_do_re.match(parts[j])
            if m:
                start, end, step = (
                    _fortran_eval(e, g_params, params)
                    for e in [parts[j][m.end():], *parts[j + 1:], '1'][:3])
                # the loop includes its end, where a Python range does not
                end += 1 if step > 0 else -1
                return [x for i in range(start, end, step)
                        for part in parts[:j]
                        for x in _eval_items(part, g_params, {
                            **params, m[1]: _default_integer(i)})]
    value = _fortran_eval(item, g_params, params)
    if isinstance(value, dict):
        return list(value.values())
    return value if isinstance(value, list) else [value]


def _element_order(shape):
    """Yield the subscripts of an array from 0 in array element order.

    The first subscript varies fastest, as Fortran stores arrays.
    """
    for s in itertools.product(*map(range, reversed(shape))):
        yield s[::-1]


def _extent(array, dim):
    # a parameter array is keyed by its subscripts, a tuple for rank > 1
    return len({k[dim - 1] if isinstance(k, tuple) else k for k in array})


_reshape_re = re.compile(r'\s*reshape\s*\((?P<args>.*)\)\s*\Z', re.I)


def _eval_reshape(args, g_params, params):
    """Evaluate reshape(source, shape, pad, order) given its arguments.

    Returns the elements of the result in array element order and its
    shape.
    """
    names = ['source', 'shape', 'pad', 'order']
    texts = {}
    for name, arg in zip(names, markoutercomma(args).split('@,@')):
        m = re.match(r'\s*(source|shape|pad|order)\s*=(?!=)', arg, re.I)
        texts[m[1].lower() if m else name] = arg[m.end():] if m else arg
    source, shape, pad, order = (
        _eval_items(texts[k], g_params, params) if k in texts else []
        for k in names)
    size = math.prod(shape)
    values = (source + pad * size)[:size]
    if order:
        # the result is filled with subscript order(1) varying fastest
        filled = {}
        for s, x in zip(_element_order([shape[k - 1] for k in order]), values):
            filled[tuple(s[order.index(k)] for k in range(1, len(shape) + 1))] = x
        values = [filled[s] for s in _element_order(shape)]
    return values, shape


def param_eval(v, g_params, params, dimspec=None):
    """
    Creates a dictionary of indices and values for each parameter in a
    parameter array to be evaluated later.

    A multidimensional array parameter, e.g. dimension(-3:1, 4, 3:5), is
    keyed by tuples of its indices. Its value is a RESHAPE, as Fortran
    has no other constructor for it, or a scalar.
    One issue remains: if the user wants to access the array parameter from
    python, we should either
    1) allow them to access the parameter array using python standard indexing
       (which is often incompatible with the original fortran indexing)
    2) allow the parameter array to be accessed in python as a dictionary with
       fortran indices as keys
    We are choosing 2 for now.
    """
    if dimspec is None:
        try:
            p = _fortran_eval(v, g_params, params)
        except Exception as msg:
            p = v
            outmess(f'param_eval: got "{msg}" on {v!r}\n')
        return p

    # This is an array parameter.
    # First, we parse the dimension information
    if len(dimspec) < 2 or dimspec[::len(dimspec) - 1] != "()":
        raise ValueError(f'param_eval: dimension {dimspec} can\'t be parsed')
    bounds = []
    for dimrange in markoutercomma(dimspec[1:-1]).split('@,@'):
        # e.g. dimension(2), dimension(-1:1) or dimension(0:*)
        dimrange = dimrange.split(':')
        # now, dimrange is a list of 1 or 2 elements
        if len(dimrange) == 1:
            dimrange = ['1'] + dimrange
        lbound = int(param_parse(
            _eval_dimension(dimrange[0], params, g_params), params))
        ubound = param_parse(_eval_dimension(dimrange[1], params, g_params),
                             params)
        bounds.append((lbound, ubound.strip()))
    # an implied shape array has the shape of its value
    implied = any(ubound == '*' for _, ubound in bounds)

    # Parse parameter value
    m = _reshape_re.match(v)
    if m:
        v_eval, shape = _eval_reshape(m['args'], g_params, params)
    else:
        if v.startswith('(/'):
            v = v[2:-2]
        elif v.startswith('['):
            v = v[1:-1]
        v_eval = []
        for item in markoutercomma(v).split('@,@'):
            try:
                v_eval.extend(_eval_items(item, g_params, params))
            except Exception as msg:
                if implied:
                    raise
                outmess(f'param_eval: got "{msg}" on {item!r}\n')
                v_eval.append(None)
        shape = [len(v_eval)]

    extents = [shape[i] if ubound == '*' else int(ubound) - lbound + 1
               for i, (lbound, ubound) in enumerate(bounds)]
    if len(v_eval) == 1:
        # a scalar sets every element
        v_eval *= math.prod(extents)
    p = {}
    for s, x in zip(_element_order(extents), v_eval):
        index = tuple(lbound + i for (lbound, _), i in zip(bounds, s))
        if x is not None:
            p[index if len(index) > 1 else index[0]] = x

    return p


def param_parse(d, params):
    """Recursively parse array dimensions.

    Parses the declaration of an array variable or parameter
    `dimension` keyword, and is called recursively if the
    dimension for this array is a previously defined parameter
    (found in `params`).

    Parameters
    ----------
    d : str
        Fortran expression describing the dimension of an array.
    params : dict
        Previously parsed parameters declared in the Fortran source file.

    Returns
    -------
    out : str
        Parsed dimension expression.

    Examples
    --------

    * If the line being analyzed is

      `integer, parameter, dimension(2) :: pa = (/ 3, 5 /)`

      then `d = 2` and we return immediately, with

    >>> d = '2'
    >>> param_parse(d, params)
    2

    * If the line being analyzed is

      `integer, parameter, dimension(pa) :: pb = (/1, 2, 3/)`

      then `d = 'pa'`; since `pa` is a previously parsed parameter,
      and `pa = 3`, we call `param_parse` recursively, to obtain

    >>> d = 'pa'
    >>> params = {'pa': 3}
    >>> param_parse(d, params)
    3

    * If the line being analyzed is

      `integer, parameter, dimension(pa(1)) :: pb = (/1, 2, 3/)`

      then `d = 'pa(1)'`; since `pa` is a previously parsed parameter,
      and `pa(1) = 3`, we call `param_parse` recursively, to obtain

    >>> d = 'pa(1)'
    >>> params = dict(pa={1: 3, 2: 5})
    >>> param_parse(d, params)
    3
    """
    dname = d[:d.find("(")] if "(" in d else ""
    if dname:
        # Text precedes the first "(", so this is array-parameter
        # indexing such as pa(1) or myparamarray(nested(dim)).
        ddims = d[d.find("(") + 1:d.rfind(")")]
        # the index is itself an expression over params; parse it recursively
        index = int(param_parse(ddims, params))
        return str(params[dname][index])
    elif d in params:
        return str(params[d])
    else:
        for p in params:
            re_1 = re.compile(
                r'(?P<before>.*?)\b' + p + r'\b(?P<after>.*)', re.I
            )
            m = re_1.match(d)
            while m:
                d = m.group('before') + \
                    str(params[p]) + m.group('after')
                m = re_1.match(d)
        return d


def expr2name(a, block, args=[]):
    orig_a = a
    a_is_expr = not analyzeargs_re_1.match(a)
    if a_is_expr:  # `a` is an expression
        implicitrules, attrrules = buildimplicitrules(block)
        at = determineexprtype(a, block['vars'], implicitrules)
        na = 'e_'
        for c in a:
            c = c.lower()
            if c not in string.ascii_lowercase + string.digits:
                c = '_'
            na = na + c
        if na[-1] == '_':
            na = na + 'e'
        else:
            na = na + '_e'
        a = na
        while a in block['vars'] or a in block['args']:
            a = a + 'r'
    if a in args:
        k = 1
        while a + str(k) in args:
            k = k + 1
        a = a + str(k)
    if a_is_expr:
        block['vars'][a] = at
    else:
        if a not in block['vars']:
            block['vars'][a] = block['vars'].get(orig_a, {})
        if 'externals' in block and orig_a in block['externals'] + block['interfaced']:
            block['vars'][a] = setattrspec(block['vars'][a], 'external')
    return a


def analyzeargs(block):
    setmesstext(block)
    implicitrules, _ = buildimplicitrules(block)
    if 'args' not in block:
        block['args'] = []
    args = []
    for a in block['args']:
        a = expr2name(a, block, args)
        args.append(a)
    block['args'] = args
    if 'entry' in block:
        for k, args1 in list(block['entry'].items()):
            for a in args1:
                if a not in block['vars']:
                    block['vars'][a] = {}

    for b in block['body']:
        if b['name'] in args:
            if 'externals' not in block:
                block['externals'] = []
            if b['name'] not in block['externals']:
                block['externals'].append(b['name'])
    if 'result' in block and block['result'] not in block['vars']:
        block['vars'][block['result']] = {}
    return block


determineexprtype_re_1 = re.compile(r'\A\(.+?,.+?\)\Z', re.I)
determineexprtype_re_2 = re.compile(r'\A[+-]?\d+(_(?P<name>\w+)|)\Z', re.I)
determineexprtype_re_3 = re.compile(
    r'\A[+-]?[\d.]+[-\d+de.]*(_(?P<name>\w+)|)\Z', re.I)
determineexprtype_re_4 = re.compile(r'\A\(.*\)\Z', re.I)
determineexprtype_re_5 = re.compile(r'\A(?P<name>\w+)\s*\(.*?\)\s*\Z', re.I)


def _ensure_exprdict(r):
    if isinstance(r, int):
        return {'typespec': 'integer'}
    if isinstance(r, float):
        return {'typespec': 'real'}
    if isinstance(r, complex):
        return {'typespec': 'complex'}
    if isinstance(r, dict):
        return r
    raise AssertionError(repr(r))


def determineexprtype(expr, vars, rules={}):
    if expr in vars:
        return _ensure_exprdict(vars[expr])
    expr = expr.strip()
    if determineexprtype_re_1.match(expr):
        return {'typespec': 'complex'}
    m = determineexprtype_re_2.match(expr)
    if m:
        if 'name' in m.groupdict() and m.group('name'):
            outmess(
                f'determineexprtype: selected kind types not supported ({repr(expr)})\n')
        return {'typespec': 'integer'}
    m = determineexprtype_re_3.match(expr)
    if m:
        if 'name' in m.groupdict() and m.group('name'):
            outmess(
                f'determineexprtype: selected kind types not supported ({repr(expr)})\n')
        return {'typespec': 'real'}
    for op in ['+', '-', '*', '/']:
        for e in [x.strip() for x in markoutercomma(expr, comma=op).split('@' + op + '@')]:
            if e in vars:
                return _ensure_exprdict(vars[e])
    t = {}
    if determineexprtype_re_4.match(expr):  # in parenthesis
        t = determineexprtype(expr[1:-1], vars, rules)
    else:
        m = determineexprtype_re_5.match(expr)
        if m:
            rn = m.group('name')
            t = determineexprtype(m.group('name'), vars, rules)
            if t and 'attrspec' in t:
                del t['attrspec']
            if not t:
                if rn[0] in rules:
                    return _ensure_exprdict(rules[rn[0]])
    if expr[0] in '\'"':
        return {'typespec': 'character', 'charselector': {'*': '*'}}
    if not t:
        outmess(
            f'determineexprtype: could not determine expressions ({repr(expr)}) type.\n')
    return t

######


def crack2fortrangen(block, tab='\n', as_interface=False):
    global skipfuncs, onlyfuncs

    setmesstext(block)
    ret = ''
    if isinstance(block, list):
        for g in block:
            if g and g['block'] in ['function', 'subroutine']:
                if g['name'] in skipfuncs:
                    continue
                if onlyfuncs and g['name'] not in onlyfuncs:
                    continue
            ret = ret + crack2fortrangen(g, tab, as_interface=as_interface)
        return ret
    prefix = ''
    name = ''
    args = ''
    blocktype = block['block']
    if blocktype == 'program':
        return ''
    argsl = []
    if 'name' in block:
        name = block['name']
    if 'args' in block:
        vars = block['vars']
        for a in block['args']:
            a = expr2name(a, block, argsl)
            if not isintent_callback(vars[a]):
                argsl.append(a)
        if block['block'] == 'function' or argsl:
            args = f"({','.join(argsl)})"
    f2pyenhancements = ''
    if 'f2pyenhancements' in block:
        for k in list(block['f2pyenhancements'].keys()):
            f2pyenhancements = (f"{f2pyenhancements}{tab + tabchar}{k} "
                                f"{block['f2pyenhancements'][k]}")
    intent_lst = block.get('intent', [])[:]
    if blocktype == 'function' and 'callback' in intent_lst:
        intent_lst.remove('callback')
    if intent_lst:
        f2pyenhancements = (f"{f2pyenhancements}{tab + tabchar}"
                            f"intent({','.join(intent_lst)}) {name}")
    use = ''
    if 'use' in block:
        use = use2fortran(block['use'], tab + tabchar)
    common = ''
    if 'common' in block:
        common = common2fortran(block['common'], tab + tabchar)
    if name == 'unknown_interface':
        name = ''
    result = ''
    if 'result' in block:
        result = f" result ({block['result']})"
        if block['result'] not in argsl:
            argsl.append(block['result'])
    body = crack2fortrangen(block['body'], tab + tabchar, as_interface=as_interface)
    vars = vars2fortran(
        block, block['vars'], argsl, tab + tabchar, as_interface=as_interface)
    mess = ''
    if 'from' in block and not as_interface:
        mess = f"! in {block['from']}"
    if 'entry' in block:
        entry_stmts = ''
        for k, i in list(block['entry'].items()):
            entry_stmts = f"{entry_stmts}{tab + tabchar}entry {k}({','.join(i)})"
        body = body + entry_stmts
    if blocktype == 'block data' and name == '_BLOCK_DATA_':
        name = ''
    ret = (f'{tab}{prefix}{blocktype} {name}{args}{result} '
           f'{mess}{f2pyenhancements}{use}{vars}{common}{body}{tab}end '
           f'{blocktype} {name}')
    return ret


def common2fortran(common, tab=''):
    ret = ''
    for k in list(common.keys()):
        if k == '_BLNK_':
            ret = f"{ret}{tab}common {','.join(common[k])}"
        else:
            ret = f"{ret}{tab}common /{k}/ {','.join(common[k])}"
    return ret


def use2fortran(use, tab=''):
    ret = ''
    for m in list(use.keys()):
        ret = f'{ret}{tab}use {m},'
        if use[m] == {}:
            if ret and ret[-1] == ',':
                ret = ret[:-1]
            continue
        if 'only' in use[m] and use[m]['only']:
            ret = f'{ret} only:'
        if 'map' in use[m] and use[m]['map']:
            c = ' '
            for k in list(use[m]['map'].keys()):
                if k == use[m]['map'][k]:
                    ret = f'{ret}{c}{k}'
                    c = ','
                else:
                    ret = f"{ret}{c}{k}=>{use[m]['map'][k]}"
                    c = ','
        if ret and ret[-1] == ',':
            ret = ret[:-1]
    return ret


def true_intent_list(var):
    lst = var['intent']
    ret = []
    for intent in lst:
        try:
            f = globals()[f'isintent_{intent}']
        except KeyError:
            pass
        else:
            if f(var):
                ret.append(intent)
    return ret


def vars2fortran(block, vars, args, tab='', as_interface=False):
    setmesstext(block)
    ret = ''
    nout = []
    for a in args:
        if a in block['vars']:
            nout.append(a)
    if 'commonvars' in block:
        for a in block['commonvars']:
            if a in vars:
                if a not in nout:
                    nout.append(a)
            else:
                errmess(
                    f'vars2fortran: Confused?!: "{a}" is not defined in vars.\n')
    if 'varnames' in block:
        nout.extend(block['varnames'])
    if not as_interface:
        for a in list(vars.keys()):
            if a not in nout:
                nout.append(a)
    for a in nout:
        if 'depend' in vars[a]:
            for d in vars[a]['depend']:
                if d in vars and 'depend' in vars[d] and a in vars[d]['depend']:
                    errmess(
                        f'vars2fortran: Warning: cross-dependence between variables "{a}" and "{d}\"\n')
        if 'externals' in block and a in block['externals']:
            if isintent_callback(vars[a]):
                ret = f'{ret}{tab}intent(callback) {a}'
            # An explicit interface gives the dummy its EXTERNAL attribute.
            if not as_interface or a not in block.get('interfaced', []):
                ret = f'{ret}{tab}external {a}'
            if isoptional(vars[a]):
                ret = f'{ret}{tab}optional {a}'
            if a in vars and 'typespec' not in vars[a]:
                continue
            cont = 1
            for b in block['body']:
                if a == b['name'] and b['block'] == 'function':
                    cont = 0
                    break
            if cont:
                continue
        if a not in vars:
            show(vars)
            outmess(f'vars2fortran: No definition for argument "{a}".\n')
            continue
        if a == block['name']:
            if block['block'] != 'function' or block.get('result'):
                # 1) skip declaring a variable that name matches with
                #    subroutine name
                # 2) skip declaring function when its type is
                #    declared via `result` construction
                continue
        if 'typespec' not in vars[a]:
            if 'attrspec' in vars[a] and 'external' in vars[a]['attrspec']:
                if a in args:
                    ret = f'{ret}{tab}external {a}'
                continue
            show(vars[a])
            outmess(f'vars2fortran: No typespec for argument "{a}".\n')
            continue
        vardef = vars[a]['typespec']
        if vardef == 'type' and 'typename' in vars[a]:
            typename = vars[a]['typename']
            if 'typeparams' in vars[a]:
                typename = f"{typename}({vars[a]['typeparams']})"
            vardef = f'{vardef}({typename})'
        selector = {}
        if 'kindselector' in vars[a]:
            selector = vars[a]['kindselector']
        elif 'charselector' in vars[a]:
            selector = vars[a]['charselector']
        if '*' in selector:
            if selector['*'] in ['*', ':']:
                vardef = f"{vardef}*({selector['*']})"
            else:
                vardef = f"{vardef}*{selector['*']}"
        elif 'len' in selector:
            vardef = f"{vardef}(len={selector['len']}"
            if 'kind' in selector:
                vardef = f"{vardef},kind={selector['kind']})"
            else:
                vardef = f'{vardef})'
        elif 'kind' in selector:
            vardef = f"{vardef}(kind={selector['kind']})"
        c = ' '
        if 'attrspec' in vars[a]:
            attr = [l for l in vars[a]['attrspec']
                    if l not in ['external']]
            if as_interface and 'intent(in)' in attr and 'intent(out)' in attr:
                # In Fortran, intent(in, out) are conflicting while
                # intent(in, out) can be specified only via
                # `!f2py intent(out) ..`.
                # So, for the Fortran interface, we'll drop
                # intent(out) to resolve the conflict.
                attr.remove('intent(out)')
            if attr:
                vardef = f"{vardef}, {','.join(attr)}"
                c = ','
        if 'dimension' in vars[a]:
            vardef = f"{vardef}{c}dimension({','.join(vars[a]['dimension'])})"
            c = ','
        if 'intent' in vars[a]:
            lst = true_intent_list(vars[a])
            if lst:
                vardef = f"{vardef}{c}intent({','.join(lst)})"
            c = ','
        if 'check' in vars[a]:
            vardef = f"{vardef}{c}check({','.join(vars[a]['check'])})"
            c = ','
        if 'depend' in vars[a]:
            vardef = f"{vardef}{c}depend({','.join(vars[a]['depend'])})"
            c = ','
        if '=' in vars[a]:
            vardef = f"{vardef} :: {a}={vars[a]['=']}"
        else:
            vardef = f'{vardef} :: {a}'
        ret = f'{ret}{tab}{vardef}'
    return ret
######


# We expose post_processing_hooks as global variable so that
# user-libraries could register their own hooks to f2py.
post_processing_hooks = []


def crackfortran(files):
    global usermodules, post_processing_hooks

    outmess('Reading fortran codes...\n', 0)
    readfortrancode(files, crackline)
    outmess('Post-processing...\n', 0)
    usermodules = []
    postlist = postcrack(grouplist[0])
    outmess('Applying post-processing hooks...\n', 0)
    for hook in post_processing_hooks:
        outmess(f'  {hook.__name__}\n', 0)
        postlist = traverse(postlist, hook)
    outmess('Post-processing (stage 2)...\n', 0)
    postlist = postcrack2(postlist)
    return usermodules + postlist


def crack2fortran(block):
    global f2py_version

    pyf = crack2fortrangen(block) + '\n'
    header = """!    -*- f90 -*-
! Note: the context of this file is case sensitive.
"""
    footer = f"""
! This file was auto-generated with f2py (version:{f2py_version}).
! See:
! https://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e
"""
    return header + pyf + footer


def _is_visit_pair(obj):
    return (isinstance(obj, tuple)
            and len(obj) == 2
            and isinstance(obj[0], (int, str)))


def traverse(obj, visit, parents=[], result=None, *args, **kwargs):
    '''Traverse f2py data structure with the following visit function:

    def visit(item, parents, result, *args, **kwargs):
        """

        parents is a list of key-"f2py data structure" pairs from which
        items are taken from.

        result is a f2py data structure that is filled with the
        return value of the visit function.

        item is 2-tuple (index, value) if parents[-1][1] is a list
        item is 2-tuple (key, value) if parents[-1][1] is a dict

        The return value of visit must be None, or of the same kind as
        item, that is, if parents[-1] is a list, the return value must
        be 2-tuple (new_index, new_value), or if parents[-1] is a
        dict, the return value must be 2-tuple (new_key, new_value).

        If new_index or new_value is None, the return value of visit
        is ignored, that is, it will not be added to the result.

        If the return value is None, the content of obj will be
        traversed, otherwise not.
        """
    '''

    if _is_visit_pair(obj):
        if obj[0] == 'parent_block':
            # avoid infinite recursion
            return obj
        new_result = visit(obj, parents, result, *args, **kwargs)
        if new_result is not None:
            assert _is_visit_pair(new_result)
            return new_result
        parent = obj
        result_key, obj = obj
    else:
        parent = (None, obj)
        result_key = None

    if isinstance(obj, list):
        new_result = []
        for index, value in enumerate(obj):
            new_index, new_item = traverse((index, value), visit,
                                           parents + [parent], result,
                                           *args, **kwargs)
            if new_index is not None:
                new_result.append(new_item)
    elif isinstance(obj, dict):
        new_result = {}
        for key, value in obj.items():
            new_key, new_value = traverse((key, value), visit,
                                          parents + [parent], result,
                                          *args, **kwargs)
            if new_key is not None:
                new_result[new_key] = new_value
    else:
        new_result = obj

    if result_key is None:
        return new_result
    return result_key, new_result


def character_backward_compatibility_hook(item, parents, result,
                                          *args, **kwargs):
    """Previously, Fortran character was incorrectly treated as
    character*1. This hook fixes the usage of the corresponding
    variables in `check`, `dimension`, `=`, and `callstatement`
    expressions.

    The usage of `char*` in `callprotoargument` expression can be left
    unchanged because C `character` is C typedef of `char`, although,
    new implementations should use `character*` in the corresponding
    expressions.

    See https://github.com/numpy/numpy/pull/19388 for more information.

    """
    parent_key, parent_value = parents[-1]
    key, value = item

    def fix_usage(varname, value):
        value = re.sub(r'[*]\s*\b' + varname + r'\b', varname, value)
        value = re.sub(r'\b' + varname + r'\b\s*[\[]\s*0\s*[\]]',
                       varname, value)
        return value

    if parent_key in ['dimension', 'check']:
        assert parents[-3][0] == 'vars'
        vars_dict = parents[-3][1]
    elif key == '=':
        assert parents[-2][0] == 'vars'
        vars_dict = parents[-2][1]
    else:
        vars_dict = None

    new_value = None
    if vars_dict is not None:
        new_value = value
        for varname, vd in vars_dict.items():
            if ischaracter(vd):
                new_value = fix_usage(varname, new_value)
    elif key == 'callstatement':
        vars_dict = parents[-2][1]['vars']
        new_value = value
        for varname, vd in vars_dict.items():
            if ischaracter(vd):
                # replace all occurrences of `<varname>` with
                # `&<varname>` in argument passing
                new_value = re.sub(
                    r'(?<![&])\b' + varname + r'\b', '&' + varname, new_value)

    if new_value is not None:
        if new_value != value:
            # We report the replacements here so that downstream
            # software could update their source codes
            # accordingly. However, such updates are recommended only
            # when BC with numpy 1.21 or older is not required.
            outmess(f'character_bc_hook[{parent_key}.{key}]:'
                    f' replaced `{value}` -> `{new_value}`\n', 1)
        return (key, new_value)


post_processing_hooks.append(character_backward_compatibility_hook)


if __name__ == "__main__":
    files = []
    funcs = []
    f = 1
    f2 = 0
    f3 = 0
    showblocklist = 0
    for l in sys.argv[1:]:
        if l == '':
            pass
        elif l[0] == ':':
            f = 0
        elif l == '-quiet':
            quiet = 1
            verbose = 0
        elif l == '-verbose':
            verbose = 2
            quiet = 0
        elif l == '-fix':
            if strictf77:
                outmess(
                    'Use option -f90 before -fix if Fortran 90 code is in fix form.\n', 0)
            skipemptyends = 1
            sourcecodeform = 'fix'
        elif l == '-skipemptyends':
            skipemptyends = 1
        elif l == '--ignore-contains':
            ignorecontains = 1
        elif l == '-f77':
            strictf77 = 1
            sourcecodeform = 'fix'
        elif l == '-f90':
            strictf77 = 0
            sourcecodeform = 'free'
            skipemptyends = 1
        elif l == '-h':
            f2 = 1
        elif l == '-show':
            showblocklist = 1
        elif l == '-m':
            f3 = 1
        elif l[0] == '-':
            errmess(f'Unknown option {repr(l)}\n')
        elif f2:
            f2 = 0
            pyffilename = l
        elif f3:
            f3 = 0
            f77modulename = l
        elif f:
            try:
                open(l).close()
                files.append(l)
            except OSError as detail:
                errmess(f'OSError: {detail!s}\n')
        else:
            funcs.append(l)
    if not strictf77 and f77modulename and not skipemptyends:
        outmess("""\
  Warning: You have specified module name for non Fortran 77 code that
  should not need one (expect if you are scanning F90 code for non
  module blocks but then you should use flag -skipemptyends and also
  be sure that the files do not contain programs without program
  statement).
""", 0)

    postlist = crackfortran(files)
    if pyffilename:
        outmess(f'Writing fortran code to file {repr(pyffilename)}\n', 0)
        pyf = crack2fortran(postlist)
        with open(pyffilename, 'w') as f:
            f.write(pyf)
    if showblocklist:
        show(postlist)

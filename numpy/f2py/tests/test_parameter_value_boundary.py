from numpy.f2py import crackfortran, f90mod_rules


def integer_parameter(value, kind="8", dimensions=None):
    attributes = ["parameter"]
    if dimensions is not None:
        attributes.append("dimension" + dimensions)
    return dict(typespec="integer", kindselector=dict(kind=kind),
                attrspec=attributes, **{"=": value})


def test_public_integer_value_preserves_private_kind():
    variables = dict(n=integer_parameter("5_8 / 2_8 * 2_8"))
    internal = crackfortran._get_typed_parameters(variables)
    public = crackfortran.get_parameters(variables)
    assert isinstance(internal["n"], crackfortran._Integer)
    assert internal["n"].kind == 8
    assert crackfortran._fortran_eval("kind(n)", {}, internal) == 8
    assert type(public["n"]) is int
    assert public["n"] == internal["n"] == 4


def test_public_indexed_array_preserves_private_element_kinds():
    variables = dict(n=integer_parameter("5_8 / 2_8 * 2_8"),
                     values=integer_parameter("(/ n, 1_8 /)", dimensions="(-1:0)"))
    internal = crackfortran._get_typed_parameters(variables)
    public = crackfortran.get_parameters(variables)
    assert type(public["n"]) is int
    assert public["values"] == {-1: 4, 0: 1}
    assert type(public["values"]) is dict
    assert all(type(value) is int for value in public["values"].values())
    assert all(value.kind == 8 for value in internal["values"].values())
    assert crackfortran._fortran_eval("kind(values(-1))", {}, internal) == 8


def test_used_parameter_cache_keeps_kind_metadata(tmp_path):
    source = tmp_path / "parameter_kinds.f90"
    source.write_text(
        "module boundary_origin\n"
        "integer(8), parameter :: n = 5_8 / 2_8 * 2_8\n"
        "end module boundary_origin\n"
        "module boundary_consumer\n"
        "use boundary_origin\n"
        "integer, parameter :: width = kind(n)\n"
        "real :: values(width)\n"
        "end module boundary_consumer\n",
        encoding="ascii",
    )
    blocks = {block["name"]: block for block in crackfortran.crackfortran([str(source)])}
    assert blocks["boundary_consumer"]["vars"]["values"]["dimension"] == ["8"]
    internal = crackfortran.f90moduleparams["boundary_origin"]["n"]
    assert isinstance(internal, crackfortran._Integer)
    assert internal.kind == 8
    public = crackfortran.get_parameters(blocks["boundary_origin"]["vars"])
    assert type(public["n"]) is int
    assert public["n"] == 4
    assert crackfortran.f90moduleparams["boundary_origin"]["n"] is internal


def test_module_storage_resolves_kind_inquiry_in_typed_namespace():
    word = dict(typespec="character", attrspec=["parameter"],
                charselector=dict(len="kind(n)"), **{"=": "'abcdefgh'"})
    module = dict(block="module", name="boundary_storage", body=[],
                  vars=dict(n=integer_parameter("4_8"), word=word))
    resolved = f90mod_rules._parameter_variable(word, module)
    assert resolved["charselector"]["f2py_len"] == "8"
    assert word["charselector"]["len"] == "kind(n)"

``f2py`` folds character literals only when kinds are equal
-----------------------------------------------------------
``normalize()`` merges adjacent character literals only when their kinds are equal, and the result keeps that kind.
``"AB" // "CD"`` still becomes ``"ABCD"``, and ``4_"AB" // 4_"CD"`` becomes ``4_"ABCD"``.
``"AB" // ck_"CD"`` stays a concatenation, so it no longer raises ``TypeError``.
``4_"AB" // 1_"CD"`` is no longer folded into ``4_"ABCD"``.

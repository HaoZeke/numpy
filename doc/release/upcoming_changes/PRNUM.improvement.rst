``f2py -h`` keeps call-back intents
-----------------------------------

Signature files written by ``f2py -h`` now keep the ``callback`` and
``hide`` intents of such call-backs, so modules built from them no
longer fail to import or expose a hidden call-back as a required
argument.

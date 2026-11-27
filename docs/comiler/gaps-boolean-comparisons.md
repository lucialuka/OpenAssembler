# Logical Boolean comparison semantics

Status: original boundaries recorded; isolated candidate has initial offline evidence; production unchanged.

After Boolean scalar and [storage admission](gaps-boolean-storage.md), direct
Boolean comparisons still reject in the Python frontend. The [original source
probe](../../compiler_exp/gap_closure/python_boolean_abi/comparisons/baseline.json)
records four scalar/storage/helper/chained cases with exact source locations.
The compiler discovers the source without executing its top-level exception.

The [experiment contract](../../compiler_exp/gap_closure/python_boolean_abi/comparisons/README.md)
requires complete false/true comparison tables, original once-only operand order,
short-circuit chain effects, explicit type/literal rules, helper/caller isolation
and source/native ownership. Full independent offline/A100/public scalar/buffer
and integration gates are required before production admission. Existing explicit
integer conversions are a workaround, not proof that the original source works.
Boolean arithmetic/atomics, identity/membership and general mixed unions remain
separate contracts. The full 51-gap program remains incomplete.


The [initial offline candidate gate](../../compiler_exp/gap_closure/python_boolean_abi/comparisons/candidate-offline.json)
passes **2,784 complete executions / 116 rows** for six comparison relations,
literal/branch/helper/chain/predicate-result forms and independent once-only atomic
short-circuit effects. Predicates become canonical zero/one u32 operands for the
existing integer relation. This establishes an initial CPU gate; it does not admit
production comparisons or supply original-source/A100/public-run/full integration
evidence. The [production storage integration](gaps-boolean-storage.md) is frozen
separately before this comparison candidate can be promoted.

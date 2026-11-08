# Scalar latency, scheduling and registers

## Corrected A100 latency sweep

The corrected 42-case sweep separates fixed-pipe RAW dependencies from
predicate-to-guard dependencies. The measured minima are:

* approximately 4 cycles for tested same-pipe ALU/FMA pairs;
* approximately 5 cycles for tested cross-class pairs;
* approximately 4 cycles from a predicate definition to tested `SEL` use;
* approximately 13 cycles from a predicate definition to tested guarded
  instructions and branches.

These are minimum safe distances for the tested pair and operand forms. They
are not a complete A100 latency table, and they do not describe HMMA issue rate.

## Scoreboards and WAR

Variable-latency loads, operand-collection instructions and asynchronous memory
operations use dependency scoreboards. A later writer of an address register
must wait for the prior reader to finish; dropping that read-WAR edge can
produce an apparently shorter but incorrect kernel. Scoreboard generations must
retire before a slot is reused.

## Register allocation

The compiler lowers PTX through virtual values, removes dead/copy-only values,
allocates aligned groups for wide and tensor operands, and then emits physical
register text. Current scheduling still has important ordering constraints:
physical choices can affect later movement, and the tensor path intentionally
keeps conservative source order. The experiment scheduler under
`exp/tensor_scheduler` explores scheduling over lowered blocks but is not a
production replacement.

## Production decision

The measured scalar classes are used in the scalar backend with conservative
fallbacks. A mined-latency scheduler promotion was rejected after full
`nvdisasm` validation produced invalid SASS on ordinary kernels. This is a
key reverse-engineering lesson: matching control fields on a probe does not
prove that an arbitrary reordered block is legal.


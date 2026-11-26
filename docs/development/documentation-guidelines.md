# Documentation maintenance guidelines

Documentation is part of the compiler contract. Keep user instructions,
implementation limits, validation and research status in phase with source.

## Where information belongs

- Keep the root README focused on project purpose, target, a short entry path
  and links.
- Put task navigation in the [documentation home](../README.md).
- Put exact CLI syntax in COMMANDS.md, user workflow in REPOSITORY_WORKFLOW.md,
  and current architecture/status in STATUS.md.
- Put current implementation contract, known gaps and validation in compiler/.
  Keep hypotheses, measurements, failed attempts and integration decisions in
  research/.
- Keep compact reproducible records in evidence/ and hardware/encoding
  observations in reverse_eng_results/.
- Do not create placeholder pages or duplicate long tables just to mirror an
  ideal tree. Add pages for a real task, maintained contract or reproducible
  result; link to canonical existing details.

## Feature and support claims

Name the admitted source form and relevant types/dtypes, shapes, layouts,
attributes, architecture, execution mode and verification scope. Link to owner
implementation or tests. Discovery and parsing alone do not establish support.
For model operators, state whether the evidence covers forward, backward,
inference, training or replay. For kernels, record launch ABI and memory
requirements. For PTX, qualify operands and modifiers.

## Performance claims

Identify what was timed, workload and shape/dtype, GPU and source revision,
comparator, warmups/repetitions and statistic, correctness gate, and whether
the scope is a kernel or complete application. Link method and raw results.
State limitations and failed comparisons with positive results. Do not turn a
kernel timing into a whole-model claim.

## Research and evidence

Research pages distinguish hypothesis, method, result, limitations,
reproduction and integration status. Update the dossier and research
catalogue together. Retain failed evidence when it explains a decision. Do
not describe unavailable hardware, skipped tests or unmeasured candidates as
negative results.

## Page templates

Use this outline for a new feature or operator contract:

    # Feature name

    ## Summary
    What the feature does and who should use it.

    ## Status
    Implemented, experimental or unsupported. Record the revision last checked.

    ## Supported contract
    Inputs, types, shapes, attributes, target and execution mode.

    ## Example
    Minimal runnable input and expected output.

    ## Limitations
    Unsupported cases and failure behavior.

    ## Implementation and validation
    Source files, tests, hardware, correctness method and evidence.

Use this outline for a research or benchmark record:

    # Experiment name

    ## Question and conclusion
    What was tested and what the results establish.

    ## Setup and baselines
    Hardware, software, source revision, workloads and exact comparators.

    ## Method
    Correctness gate, warmup, repetitions, timing method and statistic.

    ## Results
    Measurements, variability and failed cases.

    ## Interpretation and limitations
    Scope, uncertainty and applicability.

    ## Reproduction and evidence
    Commands, scripts, result files and artifact locations.

    ## Integration status
    Whether the result is used by the current compiler or runtime.

## Validate documentation changes

From the repository root, run:

    python3 tools/check_docs.py

Use the supported project Python if the system Python is too old to import the
compiler. The checker covers local links/anchors, research catalogue
consistency, status structure and CLI reference coverage. Run any
feature-specific tests or commands required by the claim as well: a link check
cannot establish technical accuracy.

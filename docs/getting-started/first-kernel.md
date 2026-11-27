# Compile and run your first kernel

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

This walkthrough uses the checked-in typed Python vector-add kernel. It compiles
to an OpenAssembler-owned cubin, then describes a launch for two 128-element
buffers.

## Prerequisites

- Install OpenAssembler as described in [installation](installation.md).
- Run from this source checkout so the example source and launch file exist.
- For execution, authenticate Modal or configure an SSH A100. Compilation and
  emulation verification do not need a GPU.
- Local Linux compilation needs CUDA toolkit tools; local execution also needs
  an NVIDIA driver and A100.

The command syntax below was checked against the CLI parser at source revision
0ca89bd, then run with Python 3.14.5 and CUDA toolkit 12.8.61. Compilation and
emulator verification passed six random trials bit-exact against the PTX
interpreter. This did not execute the cubin on an A100.

## Compile

    compile examples/vector_add.py \
      --backend local --out-dir output/first-kernel --verify

Local compilation uses the current Linux host. On a workstation without local
CUDA tools, choose Modal or an SSH target for the build. The verify option
checks the standalone kernel with OpenAssembler's emulator; it does not launch
the kernel on a GPU.

The output root contains a manifest, cubin/vector_add.cubin and
sass/vector_add.sass, with intermediate files and compiler reports under
.build/. The manifest records the entry name, arguments and ownership. A cubin
is a driver-loaded GPU module, not an executable.

## Run

The checked-in launch spec describes the buffers, arguments, grid and block:

    run output/first-kernel \
      --spec examples/run_spec.json --backend modal

This launches the cubin on the selected Modal A100. Replace modal with an SSH
alias or user@host to use your own machine. The report records launch status,
requested output buffers and timing. This standalone run does not compare its
output against a vendor reference.

For another kernel, set buffer lengths and launch dimensions to match its
parameters. See [launch specification syntax](../COMMANDS.md#describe-an-actual-launch-with---spec)
and the [typed Python frontend](../FRONTENDS.md).

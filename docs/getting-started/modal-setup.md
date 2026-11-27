# Run on Modal or an A100 over SSH

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

OpenAssembler separates a workload from its transport. Choose Modal for a
managed remote A100, SSH for an A100 you administer, or local on a compatible
Linux GPU host.

## Modal

The Homebrew installation includes the Modal CLI. Authenticate once:

    modal setup

Then compile and run a repository:

    compile /path/to/repo --backend modal --verify
    run /path/to/repo --backend modal

Or combine the stages:

    cnr /path/to/repo --backend modal --verify

The default requested GPU is A100-40GB. Reports record returned hardware; the
request string does not prove which device the provider returned. See
[Modal transport details](../REPOSITORY_WORKFLOW.md#modal-from-a-workstation).

## SSH to an A100

Configure a host in ~/.ssh/config:

    Host training-a100
        HostName gpu.example.com
        User your-user
        IdentityFile ~/.ssh/id_ed25519

Check noninteractive access and the remote environment, then build and run:

    ssh -o BatchMode=yes training-a100 'python3 --version; nvidia-smi'
    ssh training-a100 'ptxas --version; nvdisasm --version; nvcc --version'
    cnr /path/to/repo --backend training-a100 --verify

CUDA toolkit tools are needed on the SSH host to compile native inputs; the
driver and A100 are needed to execute there. Model capture also requires the
documented PyTorch/CUDA runtime. See the [SSH guide](../REPOSITORY_WORKFLOW.md#ssh-from-your-workstation)
and [commands inside the GPU host](../REPOSITORY_WORKFLOW.md#inside-an-ssh-terminal-on-the-gpu-host).

## Inspect without a job

inspect PATH --json reads repository files and source syntax
without importing user code. A supported compile/run dry-run describes planned
work without submitting or launching it. Neither checks remote credentials,
compiles a cubin, nor proves that a GPU job will pass.

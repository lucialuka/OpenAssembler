class Openassembler < Formula
  desc "Python/CUDA C/C++/PTX to NVIDIA SASS and CUDA binary compiler"
  homepage "https://github.com/lucialuka/OpenAssembler"
  url "https://github.com/lucialuka/OpenAssembler/archive/refs/heads/CLI.tar.gz"
  version "0.2.0"
  license "Apache-2.0"

  # OpenAssembler's compiler and emulator use only the Python standard library.
  # Modal is required so the documented --backend modal workflow works directly
  # after `brew install`; its own formula manages its Python runtime.
  depends_on "modal"
  depends_on "python@3.12"

  def install
    # Install the source tree rather than rebuilding a Python wheel. This keeps
    # the formula relocatable and preserves the repository-relative data roots
    # used by the encoder, oracle, research tools and remote-build transport.
    libexec.install "openassembler", "tools", "specs", "corpus", "examples", "docs", "env",
                    "README.md", "NOTICE", "LICENSE", "pyproject.toml"

    (bin / "openassembler").write <<~EOS
      #!/bin/bash
      set -euo pipefail
      export PYTHONPATH="#{libexec}${PYTHONPATH:+:$PYTHONPATH}"
      exec "#{formula_opt_bin("python@3.12")}/python3.12" -m openassembler "$@"
    EOS

    (bin / "openassembler-cuda").write <<~EOS
      #!/bin/bash
      set -euo pipefail
      exec "#{libexec}/env/cuda.sh" "$@"
    EOS

    (bin / "openassembler-fetch-cuda").write <<~EOS
      #!/bin/bash
      set -euo pipefail
      exec "#{libexec}/env/fetch_cuda.sh" "$@"
    EOS
  end

  def caveats
    <<~EOS
      OpenAssembler is installed with its complete CLI, compiler data, examples,
      documentation and research tools.

      Remote Modal workflows are ready after:
        modal setup

      Native compilation/execution still needs an NVIDIA CUDA toolkit and an
      SM80/A100 GPU or compatible SSH host. On macOS, local toolkit commands
      use Colima:
        brew install colima
        colima start
        openassembler-fetch-cuda

      The usual repository workflow is:
        openassembler compile /path/to/repo --backend modal --verify
        openassembler run /path/to/repo --backend modal
        openassembler cnr /path/to/repo --backend modal --verify
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/openassembler --version")
    help = shell_output("#{bin}/openassembler --help")
    assert_match "compile", help
    assert_match "cnr", help
    assert_match "modal", help.downcase
    assert_path_exists libexec / "env/cuda.sh"
    assert_path_exists bin / "openassembler-cuda"
    assert_path_exists bin / "openassembler-fetch-cuda"
  end
end

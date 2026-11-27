class Openassembler < Formula
  desc "Python/CUDA C/C++/PTX to NVIDIA SASS and CUDA binary compiler"
  homepage "https://github.com/lucialuka/OpenAssembler"
  url "https://github.com/lucialuka/OpenAssembler/archive/cb1fd97fe2559e11ab553a884980db247a3e4178.tar.gz"
  sha256 "b1a1e695a0d15c78c5e986d33ee543965a88c35601f4c3eb30fe90792edfdf15"
  version "0.3.1"
  revision 1
  license "Apache-2.0"

  # OpenAssembler's compiler and emulator use only the Python standard library.
  # Modal is required so the documented --backend modal workflow works directly
  # after `brew install`; its own formula manages its Python runtime.
  depends_on "modal"
  depends_on "python@3.12"

  # This compiler input was removed from later source archives as generated
  # data. Fetch its identical, validated bytes from the retained source commit.
  resource "sm80-encoder-data" do
    url "https://raw.githubusercontent.com/lucialuka/OpenAssembler/731ec36620f5451797ddbcc6d48fa9934c87f1bf/corpus/sm_80/pairs.jsonl"
    sha256 "c619303020b016b27eb8b93eb68652563f11c7a7a78bf7eaba4eca5d172bb491"
  end

  # Modal detects macOS system proxies, but its Homebrew package omits the
  # optional CONNECT/SOCKS transport. This pure-Python helper is imported by
  # the Modal subprocess through our bundled PYTHONPATH (asyncio is stdlib).
  resource "python-socks" do
    url "https://files.pythonhosted.org/packages/95/c6/4036f0b1bbcbd2bb863dbc82bc1b48c304553f070cc067cdeb797268af6c/python_socks-3.2.2.tar.gz"
    sha256 "307cca18f0b464cdc48341578e521832c4878690a6f87b82dfad7bad7474b78f"
  end

  resource "aiohttp-socks" do
    url "https://files.pythonhosted.org/packages/18/1d/a306e0111222180e60f17131a3f5d9bc694dd999a8115959a7dd76c2238e/aiohttp_socks-0.12.0.tar.gz"
    sha256 "3caf9f5a4164611122d412bc11b2f9114fd29c85e1ba27bb38060d3c236bdc8d"
  end

  def install
    # Install the source tree rather than rebuilding a Python wheel. This keeps
    # the formula relocatable and preserves the repository-relative data roots
    # used by the encoder, oracle, research tools and remote-build transport.
    libexec.install "openassembler", "tools", "specs", "corpus", "examples", "docs", "env", "third_party",
                    "README.md", "NOTICE", "LICENSE", "pyproject.toml"

    resource("sm80-encoder-data").stage do
      (libexec / "corpus/sm_80").install "pairs.jsonl"
    end

    resource("python-socks").stage do
      libexec.install "python_socks"
      (libexec / "third_party/python-socks").install "LICENSE.txt"
    end
    resource("aiohttp-socks").stage do
      libexec.install "aiohttp_socks"
      (libexec / "third_party/aiohttp-socks").install "LICENSE.txt"
    end
    # Keep existing documentation links working with the source archive's
    # historical directory spelling.
    if (libexec / "docs/comiler").directory? && !(libexec / "docs/compiler").exist?
      (libexec / "docs/compiler").make_relative_symlink(libexec / "docs/comiler")
    end

    (bin / "openassembler").write <<~EOS
      #!/bin/bash
      set -euo pipefail
      unset PYTHONHOME PYTHONUSERBASE
      export PYTHONPATH="#{libexec}"
      export PATH="#{formula_opt_bin("modal")}:$PATH"
      exec "#{formula_opt_bin("python@3.12")}/python3.12" -s -P -m openassembler "$@"
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

      Start the interactive shell from your terminal:
        openassembler

      Then enter commands at its prompt:
        compile /path/to/repo --backend modal --verify
        run /path/to/repo --backend modal
        cnr /path/to/repo --backend modal --verify
        exit

      If an active Python environment shadows this command, leave that
      environment with `deactivate`, or launch this installation directly:
        #{opt_bin}/openassembler
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/openassembler --version")
    help = shell_output("#{bin}/openassembler --help")
    assert_match "compile", help
    assert_match "cnr", help
    assert_match "modal", help.downcase
    shell = pipe_output("#{bin}/openassembler", "help compile\nexit\n", 0)
    assert_match "--backend", shell
    # Homebrew runs tests outside the compiler source tree, just as users do
    # from an ML repository. The shell must retain that working directory.
    (testpath / "kernels").mkpath
    (testpath / "kernels/probe.ptx").write <<~PTX
      .version 8.0
      .target sm_80
      .address_size 64
      .visible .entry probe() { ret; }
    PTX
    report = pipe_output("#{bin}/openassembler", "inspect . --json\nexit\n", 0)
    assert_match '"path": "kernels/probe.ptx"', report
    plan = shell_output("#{bin}/openassembler compile . --backend modal --dry-run")
    assert_match "nothing was compiled or uploaded", plan
    ENV["PYTHONPATH"] = libexec.to_s
    system Formula["python@3.12"].opt_bin / "python3.12", "-s", "-P", "-c",
           "from openassembler.isa.encode import Encoder; assert Encoder('sm_80').encode('NOP')"
    system Formula["modal"].opt_libexec / "bin/python", "-c",
           "from python_socks.async_.asyncio import Proxy; from aiohttp_socks import ProxyConnector"
    assert_path_exists libexec / "env/cuda.sh"
    assert_path_exists bin / "openassembler-cuda"
    assert_path_exists bin / "openassembler-fetch-cuda"
  end
end

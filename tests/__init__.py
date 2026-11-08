"""Repository-local regression test package.

Keeping an explicit package marker prevents an unrelated site-package named
``tests`` from shadowing these helpers when pytest imports ``tests.backend_harness``.
"""


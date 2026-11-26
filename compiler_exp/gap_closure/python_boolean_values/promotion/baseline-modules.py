"""Static local Python device-helper source graphs, with no user-code execution."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PREFIX = '// openassembler-pyfront-source-graph-v1 '


def source_root(path, explicit=None):
    path = Path(path).resolve()
    if explicit is not None:
        root = Path(explicit).resolve()
        if not root.is_dir() or not path.is_relative_to(root):
            raise ValueError('Python entry must be inside --python-source-root')
        return root
    root = path.parent
    while (root/'__init__.py').is_file():
        root = root.parent
    return root


class SourceGraph:
    """One compilation snapshot; helper bodies keep their defining names and lines."""

    def __init__(self, path, tree, names, root=None):
        self.path = Path(path).resolve()
        self.root = source_root(self.path, root)
        self.modules = {}
        self.files = {}
        self.active = []
        self.resolving = []
        self.calls = {}
        self.comparisons = {}
        from .static_values import StaticValues
        self.static_values = StaticValues(self)
        self.cross_module = False
        self.kernel = None
        self.total_bytes = 0
        relative = self.path.relative_to(self.root)
        parts = list(relative.with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        self.entry = '.'.join(parts)
        if not self.entry or any(not part.isidentifier() for part in parts):
            # Entry filenames need not be valid imported module names.
            self.entry = '<entry>'
        self._register(self.entry, self.path, tree, names)
        if self.entry != '<entry>':
            for count in range(1, len(parts)):
                initializer = self.root.joinpath(*parts[:count])/'__init__.py'
                if initializer.is_file():
                    self._load('.'.join(parts[:count]), None)

    def fail(self, message, node=None, path=None):
        from .compile import NotSupported
        raise NotSupported(message, node, path=str(path) if path is not None else None)

    def _read(self, path, node=None):
        path = Path(path).absolute()
        if not path.is_relative_to(self.root) or path.is_symlink() or path.resolve() != path:
            self.fail('Python helper dependency escapes source root or uses a symlink', node)
        if path in self.files:
            return self.files[path]
        if len(self.files) >= 64:
            self.fail('Python helper source graph exceeds 64 files', node)
        try:
            data = path.read_bytes()
        except OSError as error:
            self.fail('cannot read Python helper source: '+str(error), node)
        self.total_bytes += len(data)
        if self.total_bytes > 8*1024*1024:
            self.fail('Python helper source graph exceeds 8 MiB', node)
        self.files[path] = data
        return data

    def _register(self, module, path, tree=None, names=None, node=None):
        from . import compile as c
        from .helpers import _ModuleBindings
        data = self._read(path, node)
        if tree is None:
            try:
                tree = ast.parse(data, filename=str(path))
            except SyntaxError as error:
                raise c.NotSupported('cannot parse Python helper source: '+error.msg,
                                     path=str(path), lineno=error.lineno) from error
            except (ValueError, UnicodeDecodeError) as error:
                self.fail('cannot parse Python helper source: '+str(error), node, path)
        if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(data), include_attributes=False):
            self.fail('Python source changed before graph snapshot', node, path)
        for item in ast.walk(tree):
            if path != self.path:
                item._oa_source_path = str(path)
        if names is None:
            try:
                names = c._Names(tree)
            except c.NotSupported as error:
                raise error.with_path(path) from None
        binding = _ModuleBindings()
        binding.visit(tree)
        info = dict(name=module, path=path, tree=tree, names=names,
                    bindings=binding.counts, imports={}, mutations=set(), stars=False)
        package = module if path.name == '__init__.py' else module.rpartition('.')[0]
        for item in tree.body:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    info['imports'][alias.asname or alias.name.split('.')[0]] = (
                        'module', alias.name if alias.asname else alias.name.split('.')[0])
            elif isinstance(item, ast.ImportFrom):
                base = item.module or ''
                if item.level:
                    parts = package.split('.') if package else []
                    if item.level > len(parts):
                        self.fail('relative Python helper import escapes source root', item, path)
                    base = '.'.join(parts[:len(parts)-item.level+1]+([base] if base else []))
                for alias in item.names:
                    if alias.name == '*':
                        if not (not item.level and item.module in c._Names._PF_MODULES):
                            info['stars'] = True
                    else:
                        info['imports'][alias.asname or alias.name] = ('member', base, alias.name)
            # Track module attribute mutation, including inside module control flow.
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for value in ast.walk(item):
                    if isinstance(value, (ast.Attribute, ast.Subscript)) and isinstance(value.ctx, (ast.Store, ast.Del)):
                        head = value
                        while isinstance(head, (ast.Attribute, ast.Subscript)):
                            head = head.value
                        if isinstance(head, ast.Name):
                            info['mutations'].add(head.id)
        names.graph = self
        names.module_name = module
        names.module_path = path
        self.modules[module] = info
        return info

    def _load(self, name, node):
        if name in self.modules:
            return self.modules[name]
        if not name or any(not part.isidentifier() for part in name.split('.')):
            self.fail('invalid local Python helper module: '+name, node)
        if len(self.modules) >= 64:
            self.fail('Python helper source graph exceeds 64 modules', node)
        parts = name.split('.')
        for count in range(1, len(parts)):
            directory = self.root.joinpath(*parts[:count])
            if not directory.is_dir() or directory.resolve() != directory:
                self.fail('missing or symlinked Python helper package: '+'.'.join(parts[:count]), node)
            initializer = directory/'__init__.py'
            if initializer.is_file():
                parent = self._load('.'.join(parts[:count]), node)
                if parts[count] in parent['mutations']:
                    self.fail('Python helper package attribute is mutated: '+'.'.join(parts[:count+1]), node)
        base = self.root.joinpath(*parts)
        choices = [base.with_suffix('.py'), base/'__init__.py']
        matches = [path for path in choices if path.is_file()]
        if len(matches) > 1 or matches and base.is_dir() and matches[0] == base.with_suffix('.py'):
            self.fail('ambiguous local Python helper module: '+name, node)
        if not matches:
            self.fail('local Python helper module not found: '+name, node)
        return self._register(name, matches[0], node=node)

    def resolve(self, names, expression):
        parts = []
        node = expression
        while isinstance(node, ast.Attribute):
            parts.insert(0, node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        info = self.modules[names.module_name]
        name = node.id
        binding = info['imports'].get(name)
        if binding is None:
            return None
        if info['stars']:
            self.fail('wildcard Python helper imports require explicit bindings', expression)
        if info['bindings'][name] != 1 or name in info['mutations']:
            self.fail('Python helper import target is rebound or mutated: '+name, expression)
        if binding[0] == 'module':
            if not parts:
                self.fail('a module is not a device helper', expression)
            module, function = '.'.join([binding[1], *parts[:-1]]), parts[-1]
        else:
            base, member = binding[1:]
            if parts:
                module, function = '.'.join([base, member, *parts[:-1]]), parts[-1]
            else:
                module, function = base, member
        result = self._target(module, function, expression)
        self.cross_module |= result[0].module_path != self.path
        return result

    def _target(self, module, function, node):
        key = (module, function)
        if key in self.resolving or len(self.resolving) >= 32:
            self.fail('cyclic or excessive Python helper re-export chain', node)
        self.resolving.append(key)
        try:
            info = self._load(module, node)
            if info['bindings'][function] != 1 or function in info['mutations']:
                self.fail('Python helper export is missing, rebound or mutated: '+module+'.'+function, node)
            if function in info['names'].devices.definitions:
                return info['names'], function
            if function in info['imports']:
                value = ast.copy_location(ast.Name(id=function, ctx=ast.Load()), node)
                value._oa_source_path = str(info['path'])
                return self.resolve(info['names'], value)
            self.fail('import target is not a typed @device helper: '+module+'.'+function, node)
        finally:
            self.resolving.pop()

    def note_call(self, names, name, fn, kinds, result):
        relative = names.module_path.relative_to(self.root).as_posix()
        key = relative+':'+name
        record = self.calls.setdefault(key, dict(path=relative, function=name, line=fn.lineno,
            inputs=list(kinds), output=result,
            body_sha256=hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest(),
            kernels={}))
        record['kernels'][self.kernel] = record['kernels'].get(self.kernel, 0)+1

    def note_comparison(self, names, function, expression):
        relative = names.module_path.relative_to(self.root).as_posix()
        key = f'{relative}:{function.name}:{expression.lineno}:{expression.col_offset}'
        digest = hashlib.sha256(ast.dump(expression, include_attributes=False).encode()).hexdigest()
        row = self.comparisons.setdefault(key, dict(path=relative, function=function.name,
            line=expression.lineno, column=expression.col_offset,
            operators=[type(operator).__name__ for operator in expression.ops],
            specialized_expression_sha256=digest, kernels={}))
        if row['specialized_expression_sha256'] != digest:
            self.fail('comparison chain specialization differs at original location', expression)
        row['kernels'][self.kernel] = row['kernels'].get(self.kernel, 0) + 1

    def contract(self):
        if not self.cross_module and not self.static_values.records and not self.static_values.iterations and not getattr(self, 'early_returns', False) and not self.comparisons:
            return None
        result = dict(schema=1, strategy='static-local-python-helper-source-graph',
                    entry=self.path.relative_to(self.root).as_posix(),
                    files=[dict(path=path.relative_to(self.root).as_posix(),
                                sha256=hashlib.sha256(data).hexdigest())
                           for path, data in sorted(self.files.items())],
                    helpers={key: self.calls[key] for key in sorted(self.calls)})
        if self.comparisons:
            result['comparison_chains'] = {key:self.comparisons[key] for key in sorted(self.comparisons)}
        if self.static_values.records:
            result['static_values'] = {key:self.static_values.records[key] for key in sorted(self.static_values.records)}
        if self.static_values.iterations:
            result['static_iterations'] = {key:self.static_values.iterations[key] for key in sorted(self.static_values.iterations)}
        return result

    def verify(self):
        for path, data in self.files.items():
            if path.read_bytes() != data:
                self.fail('Python helper source changed during compilation', path=path)

    def prefix(self):
        self.verify()
        contract = self.contract()
        return PREFIX+json.dumps(contract, sort_keys=True, separators=(',', ':'))+'\n' if contract else ''


def contract_from_ptx(text):
    records = [line[len(PREFIX):] for line in text.splitlines() if line.startswith(PREFIX)]
    if not records:
        return None
    if len(records) != 1:
        raise ValueError('duplicate Python helper source graph')
    obj = json.loads(records[0])
    if obj.get('schema') != 1 or obj.get('strategy') != 'static-local-python-helper-source-graph':
        raise ValueError('invalid Python helper source graph schema')
    files = obj.get('files')
    if not isinstance(files, list) or not files or len(files) > 64:
        raise ValueError('invalid Python helper file inventory')
    paths = set()
    for row in files:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or not path.parts or row['path'] in paths:
            raise ValueError('invalid Python helper dependency path')
        paths.add(row['path'])
        digest = row['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid Python helper dependency digest')
    if obj.get('entry') not in paths or not isinstance(obj.get('helpers'), dict) or not (obj['helpers'] or obj.get('static_values') or obj.get('static_iterations') or obj.get('comparison_chains')):
        raise ValueError('invalid Python helper entry/function inventory')
    for key, helper in obj['helpers'].items():
        if helper['path'] not in paths or key != helper['path']+':'+helper['function']:
            raise ValueError('invalid Python helper function identity')
        if not isinstance(helper['kernels'], dict) or not helper['kernels'] or any(type(n) is not int or n <= 0 for n in helper['kernels'].values()):
            raise ValueError('invalid Python helper call inventory')
    if 'static_values' in obj:
        from .static_values import validate_records
        validate_records(obj['static_values'], paths)
    if 'static_iterations' in obj:
        from .static_values import validate_iterations
        validate_iterations(obj['static_iterations'], paths)
    if 'comparison_chains' in obj:
        validate_comparisons(obj['comparison_chains'], paths)
    return obj


def validate_sources(path, contract, root=None):
    if contract is None:
        return
    root = source_root(path, root)
    if Path(path).resolve().relative_to(root).as_posix() != contract['entry']:
        raise ValueError('Python helper graph entry differs from source')
    for row in contract['files']:
        target = root/row['path']
        if target.resolve() != target or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Python helper dependency changed: '+row['path'])


def validate_comparisons(records, paths):
    """Check original chain locations, typed expression identities and use counts."""
    if not isinstance(records, dict) or not records:
        raise ValueError('invalid comparison-chain inventory')
    allowed = {'Lt', 'LtE', 'Gt', 'GtE', 'Eq', 'NotEq'}
    for key, row in records.items():
        if not isinstance(row, dict) or row.get('path') not in paths:
            raise ValueError('invalid comparison-chain source')
        function, line, column = row.get('function'), row.get('line'), row.get('column')
        if (not isinstance(function, str) or not function.isidentifier()
                or type(line) is not int or line <= 0
                or type(column) is not int or column < 0
                or key != f"{row['path']}:{function}:{line}:{column}"):
            raise ValueError('invalid comparison-chain location')
        operators = row.get('operators')
        if (not isinstance(operators, list) or len(operators) < 2
                or any(not isinstance(operator, str) or operator not in allowed for operator in operators)):
            raise ValueError('invalid comparison-chain operators')
        digest = row.get('specialized_expression_sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid comparison-chain expression digest')
        kernels = row.get('kernels')
        if (not isinstance(kernels, dict) or not kernels
                or any(not isinstance(name, str) or not name.isidentifier()
                       or type(count) is not int or count <= 0 for name, count in kernels.items())):
            raise ValueError('invalid comparison-chain use inventory')

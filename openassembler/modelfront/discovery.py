"""Infer model entrypoints and observed shapes without importing user code.

This is deliberately a small static analysis, not execution of training scripts.
Ambiguous models/shapes require a CLI override; missing information is never
replaced with a guessed batch size, optimizer, precision or training objective.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path


_UNKNOWN = object()


@dataclass(frozen=True)
class _Tensor:
    shape: tuple[int, ...]


@dataclass
class _Instance:
    key: tuple[str, str]
    args: list
    kwargs: dict


class _TupleLiteral(list):
    """Shape/example container, excluded from JSON constructor arguments."""


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f'{prefix}.{node.attr}' if prefix else ''
    return ''


def _json_value(value):
    if isinstance(value, _TupleLiteral):
        return False  # A constructor must not silently receive a list for a tuple.
    if value is None or type(value) in (bool, str, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, (tuple, list)):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(type(k) is str and _json_value(v) for k, v in value.items())
    return False


def _shape(value):
    return (isinstance(value, (list, tuple)) and 0 < len(value) <= 8
            and all(type(d) is int and 0 < d <= 2**28 for d in value)
            and math.prod(value) <= 2**28)


def discover_model(root: Path, files=None, *, entrypoint=None, input_shapes=None) -> dict:
    """Return a JSON report with an optional inferred ``model`` configuration."""
    from openassembler import project
    root = Path(root).resolve()
    if files is None:
        files, _ = project.walk(root, root / 'output')
    units, modules = {}, {}
    for path in files:
        if path.suffix != '.py' or path.stat().st_size > 2 * 1024 * 1024:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(), filename=rel)
        except (ValueError, SyntaxError, UnicodeError, OSError):
            continue
        parts = list(Path(rel).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        module = '.'.join(parts)
        modules[module] = rel
        if parts and parts[0] == 'src':
            modules['.'.join(parts[1:])] = rel
        units[rel] = {'tree': tree, 'module': module, 'imports': {}}
    for rel, unit in units.items():
        package = unit['module'].split('.')
        if Path(rel).name != '__init__.py':
            package = package[:-1]
        for node in unit['tree'].body:
            if isinstance(node, ast.Import):
                for item in node.names:
                    unit['imports'][item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                prefix = '.'.join(package[:len(package) - node.level + 1]) if node.level else ''
                origin = '.'.join(p for p in (prefix, node.module) if p)
                for item in node.names:
                    unit['imports'][item.asname or item.name] = f'{origin}.{item.name}'

    def canonical(rel, node):
        name = _dotted(node)
        head, _, tail = name.partition('.')
        return units[rel]['imports'].get(head, head) + ('.' + tail if tail else '')

    def local_key(rel, node, seen=()):
        name = canonical(rel, node)
        if '.' not in name:
            return rel, name
        module, _, symbol = name.rpartition('.')
        key = modules.get(module, ''), symbol
        # Resolve package re-exports, e.g. from models import Net where
        # models/__init__.py imports Net from .network.
        if key[0] and key not in seen and symbol in units[key[0]]['imports']:
            return local_key(key[0], ast.Name(id=symbol), (*seen, key))
        return key

    definitions = {(rel, node.name): node for rel, unit in units.items()
                   for node in unit['tree'].body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    classes = set()
    while True:
        more = {key for key, node in definitions.items() if isinstance(node, ast.ClassDef)
                and any(canonical(key[0], base) == 'torch.nn.Module' or local_key(key[0], base) in classes
                        for base in node.bases)}
        if more <= classes:
            break
        classes |= more
    candidates = set(classes)
    for key, node in definitions.items():
        if not isinstance(node, ast.FunctionDef):
            continue
        returns = [n.value for n in node.body if isinstance(n, ast.Return)]
        if len(returns) == 1 and isinstance(returns[0], ast.Call):
            target = returns[0].func
            builtin = canonical(key[0], target)
            if local_key(key[0], target) in classes or (builtin.startswith('torch.nn.')
                    and builtin.removeprefix('torch.nn.').isidentifier()
                    and builtin.removeprefix('torch.nn.')[:1].isupper()):
                candidates.add(key)

    observations, examples = [], {}

    def value(rel, node, env):
        if isinstance(node, ast.Name):
            return env.get(node.id, _UNKNOWN)
        if isinstance(node, ast.Constant):
            return node.value if _json_value(node.value) else _UNKNOWN
        if isinstance(node, (ast.List, ast.Tuple)):
            items = [value(rel, child, env) for child in node.elts]
            return _TupleLiteral(items) if isinstance(node, ast.Tuple) else items
        if isinstance(node, ast.Dict):
            keys = [value(rel, n, env) if n else _UNKNOWN for n in node.keys]
            return {k: value(rel, v, env) for k, v in zip(keys, node.values)} if all(type(k) is str for k in keys) else _UNKNOWN
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = value(rel, node.operand, env)
            return (-operand if isinstance(node.op, ast.USub) else operand) if type(operand) in (int, float) else _UNKNOWN
        if not isinstance(node, ast.Call):
            return _UNKNOWN
        name = canonical(rel, node.func)
        if name in {'torch.randn', 'torch.rand', 'torch.zeros', 'torch.ones', 'torch.empty'}:
            dtype = next((canonical(rel, kw.value) for kw in node.keywords if kw.arg == 'dtype'), 'torch.float32')
            if dtype not in ('torch.float32', 'torch.float'):
                return _UNKNOWN
            dims = [value(rel, arg, env) for arg in node.args]
            if not dims:
                dims = next((value(rel, kw.value, env) for kw in node.keywords if kw.arg == 'size'), [])
            elif len(dims) == 1 and isinstance(dims[0], list):
                dims = dims[0]
            return _Tensor(tuple(dims)) if _shape(dims) else _UNKNOWN
        key = local_key(rel, node.func)
        if key in candidates:
            args = [value(rel, arg, env) for arg in node.args]
            kwargs = {kw.arg: value(rel, kw.value, env) for kw in node.keywords if kw.arg}
            if (all(kw.arg for kw in node.keywords) and _json_value(args) and _json_value(kwargs)):
                return _Instance(key, args, kwargs)
        target = env.get(_dotted(node.func), _UNKNOWN)
        if isinstance(target, _Instance):
            inputs = [value(rel, arg, env) for arg in node.args]
            if inputs and not node.keywords and all(isinstance(item, _Tensor) for item in inputs):
                observations.append({'entrypoint': target.key[0], 'factory': target.key[1],
                                     'factory_args': target.args, 'factory_kwargs': target.kwargs,
                                     'input_shapes': [list(item.shape) for item in inputs],
                                     'evidence': f'{rel}:{node.lineno}'})
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('eval', 'train', 'cpu', 'contiguous'):
            return value(rel, node.func.value, env)
        return _UNKNOWN

    def scan(rel, body, inherited):
        env = dict(inherited)
        for node in body:
            if isinstance(node, ast.FunctionDef):
                local = dict(env)
                for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                    local.pop(arg.arg, None)
                positional = [*node.args.posonlyargs, *node.args.args]
                for arg, default in zip(positional[-len(node.args.defaults):], node.args.defaults):
                    local[arg.arg] = value(rel, default, env)
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                    if default is not None:
                        local[arg.arg] = value(rel, default, env)
                scan(rel, node.body, local)
            elif isinstance(node, ast.ClassDef):
                continue  # A submodule's forward is not an application example.
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                result = value(rel, node.value, env) if node.value else _UNKNOWN
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = result
                        if target.id in ('example_input', 'example_inputs'):
                            items = result if isinstance(result, list) else [result]
                            if items and all(isinstance(item, _Tensor) for item in items):
                                examples.setdefault(rel, []).append(([list(item.shape) for item in items], f'{rel}:{node.lineno}'))
            elif isinstance(node, (ast.Expr, ast.Return)) and node.value:
                value(rel, node.value, env)
            elif isinstance(node, (ast.If, ast.With, ast.For, ast.While, ast.Try)):
                scan(rel, node.body, env)
                scan(rel, getattr(node, 'orelse', []), env)
                scan(rel, getattr(node, 'finalbody', []), env)
        return env

    for rel, unit in units.items():
        scan(rel, unit['tree'].body, {})

    # Factories are preferable to their implementation classes when neither is
    # observed at a call site. Multiple factories never pick by filename/order.
    defaults = []
    composed = {local_key(rel, call.func) for (rel, name) in classes
                for call in ast.walk(definitions[(rel, name)]) if isinstance(call, ast.Call)
                if local_key(rel, call.func) in classes and local_key(rel, call.func) != (rel, name)}
    for key in sorted(candidates):
        node = definitions[key]
        if isinstance(node, ast.ClassDef):
            node = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
            offset = 1
        else:
            offset = 0
        callable_without_args = node is None or (len(node.args.posonlyargs) + len(node.args.args) - offset <= len(node.args.defaults)
                                                   and all(n is not None for n in node.args.kw_defaults))
        defaults.append({'entrypoint': key[0], 'factory': key[1], 'callable_without_args': callable_without_args})
    if entrypoint:
        entry, sep, factory = entrypoint.rpartition(':')
        if not sep or not factory.isidentifier() or entry not in units:
            raise ValueError('--entrypoint must be a Python path inside SRC followed by :FACTORY, e.g. models/net.py:Net')
        choices = [item for item in observations if (item['entrypoint'], item['factory']) == (entry, factory)]
        selected = {'entrypoint': entry, 'factory': factory, 'factory_args': [], 'factory_kwargs': {}}
    else:
        choices = [item for item in observations if (item['entrypoint'], item['factory']) not in composed]
        eligible = [item for item in defaults if item['callable_without_args'] and not item['factory'].startswith('_')
                    and (item['entrypoint'], item['factory']) not in composed]
        factories = [item for item in eligible if (item['entrypoint'], item['factory']) not in classes]
        implementations = set()
        for item in factories:
            node = definitions[item['entrypoint'], item['factory']]
            returned = next(n.value for n in node.body if isinstance(n, ast.Return))
            implementations.add(local_key(item['entrypoint'], returned.func))
        eligible = [item for item in eligible if (item['entrypoint'], item['factory']) not in implementations]
        selected = {k: v for k, v in eligible[0].items() if k != 'callable_without_args'} if len(eligible) == 1 else None
    # A simple tabular model sometimes has no tensor example: its direct input
    # Linear fixes the feature dimension, and the application declares a literal
    # batch-size default. Require both; never invent a batch or sequence length.
    if not choices and selected and input_shapes is None and not examples.get(selected['entrypoint']):
        key = selected['entrypoint'], selected['factory']
        node = definitions.get(key)
        if isinstance(node, ast.FunctionDef):
            returned = [n.value for n in node.body if isinstance(n, ast.Return)]
            if len(returned) == 1 and isinstance(returned[0], ast.Call) and not returned[0].args and not returned[0].keywords:
                key = local_key(key[0], returned[0].func)
                node = definitions.get(key)
        if isinstance(node, ast.ClassDef):
            methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
            forward, init = methods.get('forward'), methods.get('__init__')
            if (forward and init and len(forward.args.args) >= 2
                    and len(forward.args.args) - len(forward.args.defaults) <= 2):
                tensor_name = forward.args.args[1].arg
                direct = {n.func.attr for n in ast.walk(forward) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                          and n.func.value.id == 'self' and len(n.args) == 1
                          and isinstance(n.args[0], ast.Name) and n.args[0].id == tensor_name}
                env = {arg.arg: value(key[0], default, {}) for arg, default in
                       zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}
                widths = set()
                for assignment in init.body:
                    if not isinstance(assignment, ast.Assign) or not isinstance(assignment.value, ast.Call):
                        continue
                    call = assignment.value
                    if canonical(key[0], call.func) != 'torch.nn.Linear':
                        continue
                    if not any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                               and t.value.id == 'self' and t.attr in direct for t in assignment.targets):
                        continue
                    width = value(key[0], call.args[0], env) if call.args else next(
                        (value(key[0], kw.value, env) for kw in call.keywords if kw.arg == 'in_features'), _UNKNOWN)
                    if type(width) is int and width > 0:
                        widths.add(width)
                batches = {}
                for rel, unit in units.items():
                    for call in ast.walk(unit['tree']):
                        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                                and call.func.attr == 'add_argument' and any(isinstance(a, ast.Constant)
                                and a.value in ('--batch-size', '--batch_size') for a in call.args)):
                            batch = next((value(rel, kw.value, {}) for kw in call.keywords if kw.arg == 'default'), _UNKNOWN)
                            if type(batch) is int and batch > 0:
                                batches[batch] = f'{rel}:{call.lineno}'
                if len(widths) == len(batches) == 1:
                    batch = next(iter(batches))
                    dims = [batch, next(iter(widths))]
                    if _shape(dims):
                        choices = [{**selected, 'input_shapes': [dims],
                                    'evidence': f'{batches[batch]}; {key[0]}: direct input Linear'}]
    if input_shapes is not None:
        if not isinstance(input_shapes, list) or not input_shapes or not all(_shape(s) for s in input_shapes):
            raise ValueError('--input-shapes must be JSON positive integer shapes, e.g. \'[[17, 8]]\' (at most 2**28 elements per input)')
        choices = [{**item, 'input_shapes': input_shapes} for item in choices]
    if not choices and selected:
        shape_options = [(input_shapes, 'CLI --input-shapes')] if input_shapes is not None else examples.get(selected['entrypoint'], [])
        choices = [{**selected, 'input_shapes': shapes, 'evidence': evidence} for shapes, evidence in shape_options]
    unique = {}
    for item in choices:
        key = json.dumps({k: v for k, v in item.items() if k != 'evidence'}, sort_keys=True)
        unique.setdefault(key, item)
    error, model = None, None
    if len(unique) == 1:
        model = dict(next(iter(unique.values())))
        model.pop('evidence', None)
        model.update(mode='inference', math='fp32')
    elif len(unique) > 1:
        error = 'multiple model/constructor/shape contracts were found; select --entrypoint FILE:FACTORY and --input-shapes \'[[BATCH, FEATURES]]\''
    elif candidates or entrypoint:
        error = ('cannot infer one model and its exact FP32 input shapes; use --entrypoint FILE:FACTORY '
                 'and --input-shapes \'[[BATCH, FEATURES]]\', or provide an openassembler.toml [model] contract')
    return {'schema': 1, 'ready': model is not None, 'model': model, 'candidates': defaults,
            'observations': list(unique.values()), 'error': error,
            'scope': 'Forward inference with synthetic FP32 inputs at the discovered shapes; host training loops, losses, optimizers and extension launch logic are not compiled.'}


def config_text(config: dict) -> str:
    """A reproducible model configuration; JSON literals cover these TOML types."""
    fields = ['entrypoint', 'factory', 'input_shapes', 'math', 'mode', 'seed', 'steps',
              'learning_rate', 'rtol', 'atol', 'storage_views', 'reuse_storage', 'tensor_packing',
              'execution', 'sass_listings', 'gpu', 'budget', 'build_backend']
    lines = ['# Inferred by OpenAssembler; source files were not modified.', '[model]']
    lines += [f'{key} = {json.dumps(config[key], ensure_ascii=True)}' for key in fields if key in config]
    # TOML has no null literal; JSON strings preserve optional constructor args.
    for key in ('factory_args', 'factory_kwargs'):
        if config.get(key):
            lines.append(f'{key}_json = {json.dumps(json.dumps(config[key]))}')
    if config.get('build_host'):
        lines.append(f'host = {json.dumps(config["build_host"])}')
    return '\n'.join(lines) + '\n'

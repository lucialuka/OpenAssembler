"""CPU state-comparison utilities and disabled legacy worker compatibility guards.

GPU compilation and execution live in openassembler.modelfront.worker.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
import traceback


TORCH_VERSION = '2.7.1'


def runtime_info():
    import torch
    if torch.__version__.split('+')[0] != TORCH_VERSION:
        raise RuntimeError(f"PyTorch application bundles currently require torch=={TORCH_VERSION}; "
                           f"found {torch.__version__}. Use --build-backend modal for the pinned environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch application capture needs a CUDA GPU. On macOS use "
                           "--build-backend modal (after `pip install modal && modal setup`).")
    capability = list(torch.cuda.get_device_capability())
    return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': sys.version.split()[0],
            'device': torch.cuda.get_device_name(), 'capability': capability}


def _save_state(state):
    import torch
    out = io.BytesIO()
    torch.save(state, out)
    return out.getvalue()


class StateVerificationError(RuntimeError):
    """A failed comparison with JSON-serializable, bounded tensor diagnostics."""

    def __init__(self, message, diagnostics):
        # Error responses also travel through subprocess/Modal JSON and a
        # bounded frontend traceback. Keep group totals even if details must go.
        rendered = json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        while len(rendered) > 12000 and diagnostics.get('failures'):
            diagnostics['failures'].pop()
            diagnostics['failure_details_omitted'] += 1
            rendered = json.dumps(diagnostics, sort_keys=True, allow_nan=False)
        self.diagnostics = diagnostics
        super().__init__(message + '\nstate verification diagnostics: ' +
                         rendered)


def _finite_scalar(value):
    result = float(value)
    return result if math.isfinite(result) else None


def _tensor_diagnostic(reference, actual, rtol, atol):
    """Describe a failed assert_close; never substitute a different pass gate."""
    import torch
    result = {'reference_shape': list(reference.shape), 'reference_dtype': str(reference.dtype),
              'reference_layout': str(reference.layout),
              'relative_tolerance': rtol, 'absolute_tolerance': atol}
    if not isinstance(actual, torch.Tensor):
        return {**result, 'actual_type': type(actual).__name__, 'kind': 'type'}
    result.update(actual_shape=list(actual.shape), actual_dtype=str(actual.dtype),
                  actual_layout=str(actual.layout))
    if reference.shape != actual.shape or reference.dtype != actual.dtype or reference.layout != actual.layout:
        return {**result, 'kind': 'metadata'}
    if reference.layout != torch.strided or reference.is_quantized:
        return {**result, 'kind': 'unsupported-statistics'}
    close = torch.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=False)
    mismatch = ~close
    result.update(kind='values', elements=reference.numel(), mismatched_elements=int(mismatch.sum()))
    samples = []
    flat = mismatch.reshape(-1)
    for start in range(0, flat.numel(), 4096):
        indices = torch.nonzero(flat[start:start + 4096], as_tuple=False).flatten()
        for offset in indices[:8 - len(samples)].tolist():
            linear = start + offset
            index = []
            for size in reversed(reference.shape):
                index.append(linear % size)
                linear //= size
            samples.append(list(reversed(index)))
        if len(samples) == 8:
            break
    result['mismatched_index_sample'] = samples
    result['reference_nonfinite_elements'] = int((~torch.isfinite(reference)).sum())
    result['actual_nonfinite_elements'] = int((~torch.isfinite(actual)).sum())
    # Cast before subtraction: integer subtraction can overflow and BF16 would
    # round the diagnostic itself. Complex magnitudes use the same double width.
    dtype = torch.complex128 if reference.is_complex() else torch.float64
    a, b = reference.to(dtype), actual.to(dtype)
    finite = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[finite], b[finite]
    delta = b - a
    reference_l2 = _finite_scalar(torch.linalg.vector_norm(a))
    difference_l2 = _finite_scalar(torch.linalg.vector_norm(delta))
    result.update(finite_pair_elements=a.numel(),
                  max_abs_difference=_finite_scalar(delta.abs().max()) if delta.numel() else None,
                  reference_l2=reference_l2,
                  actual_l2=_finite_scalar(torch.linalg.vector_norm(b)),
                  difference_l2=difference_l2,
                  relative_l2_difference=(_finite_scalar(difference_l2 / reference_l2)
                                          if reference_l2 and difference_l2 is not None else None),
                  statistics_scope='finite element pairs; null means undefined or overflow')
    return result


def compare_states(reference: bytes, actual: bytes, rtol: float, atol: float,
                   state_rtol: float, state_atol: float):
    """Apply the declared gates to every state; diagnose failures without admitting them."""
    import torch
    expected = torch.load(io.BytesIO(reference), map_location='cpu', weights_only=True)
    got = torch.load(io.BytesIO(actual), map_location='cpu', weights_only=True)
    report = {'schema': 1, 'status': 'failed', 'tensors_checked': 0, 'tensors_failed': 0,
              'non_tensor_values_failed': 0, 'groups': {}, 'failures': [],
              'rtol': rtol, 'atol': atol, 'state_rtol': state_rtol, 'state_atol': state_atol,
              'gradient_scope': 'retained application gradients; raw versus clipped is not inferred'}
    if expected.keys() != got.keys():
        report.update(missing_keys=[key[:256] for key in sorted(expected.keys() - got.keys())[:8]],
                      unexpected_keys=[key[:256] for key in sorted(got.keys() - expected.keys())[:8]],
                      missing_key_count=len(expected.keys() - got.keys()),
                      unexpected_key_count=len(got.keys() - expected.keys()),
                      key_sample_scope='first 8 keys per set, at most 256 characters each')
        raise StateVerificationError('verification failed: eager/compiled execution observed different model or optimizer states', report)
    first_failure = None
    failure_count = 0
    for key in expected:
        e, g = expected[key], got[key]
        prefix = key.split('/', 1)[0]
        group_name = prefix if prefix in ('outputs', 'parameters', 'gradients', 'optimizer') else 'other'
        group = report['groups'].setdefault(group_name, {'tensors_checked': 0, 'tensors_failed': 0,
                                                       'non_tensor_values_failed': 0})
        if isinstance(e, torch.Tensor):
            tensor_rtol, tensor_atol = ((state_rtol, state_atol) if key.startswith(('parameters/', 'optimizer/'))
                                        else (rtol, atol))
            report['tensors_checked'] += 1
            group['tensors_checked'] += 1
            try:
                torch.testing.assert_close(g, e, rtol=tensor_rtol, atol=tensor_atol, equal_nan=False)
                continue
            except (AssertionError, TypeError, ValueError) as exc:
                message = str(exc)
            report['tensors_failed'] += 1
            group['tensors_failed'] += 1
            if len(report['failures']) < 8:
                try:
                    details = _tensor_diagnostic(e, g, tensor_rtol, tensor_atol)
                except Exception as exc:
                    # A reporting limitation must never hide the original failure.
                    details = {'kind': 'statistics-unavailable', 'reason': type(exc).__name__}
                report['failures'].append({'key': key[:256], 'key_truncated': len(key) > 256, **details})
        elif g != e:
            message = f'{g!r} != {e!r}'
            report['non_tensor_values_failed'] += 1
            group['non_tensor_values_failed'] += 1
            if len(report['failures']) < 8:
                report['failures'].append({'key': key[:256], 'key_truncated': len(key) > 256,
                                           'kind': 'non-tensor-value'})
        else:
            continue
        failure_count += 1
        if first_failure is None:
            first_failure = f'verification failed at {key[:256]}: {message[:1200]}'
    if first_failure is not None:
        report['failure_details_omitted'] = failure_count - len(report['failures'])
        raise StateVerificationError(first_failure, report)
    if not report['tensors_checked']:
        raise RuntimeError('verification observed no model tensors; no comparison was performed')
    return {'status': 'ok', 'tensors_checked': report['tensors_checked'], 'rtol': rtol, 'atol': atol,
            'state_rtol': state_rtol, 'state_atol': state_atol,
            'scope': 'observed model outputs, final parameters, gradients and optimizer state',
            'groups': report['groups'], 'gradient_scope': report['gradient_scope']}


def execute_program(root: Path, config: dict, state_path: Path, *, compiled: bool, report_path: Path,
                    timing_path: Path | None = None):
    raise RuntimeError('legacy Inductor/Triton execution is disabled; use the OA native model worker')






def process(request: dict) -> dict:
    """Reject retired vendor capture/replay before initialization or GPU work."""
    raise RuntimeError("Inductor/Triton application worker is disabled; use openassembler.modelfront.worker")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--timing', type=Path)
    parser.add_argument('--compiled', action='store_true')
    a = parser.parse_args()
    if a.execute:
        execute_program(a.execute, json.loads(a.config.read_text()), a.state, compiled=a.compiled,
                        report_path=a.report, timing_path=a.timing)
    else:
        try:
            result = process(json.loads(a.request.read_text()))
        except Exception as exc:
            result = {'error': traceback.format_exc()}
            if isinstance(exc, StateVerificationError):
                result['verification'] = exc.diagnostics
        a.result.write_text(json.dumps(result))
        return 1 if 'error' in result else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

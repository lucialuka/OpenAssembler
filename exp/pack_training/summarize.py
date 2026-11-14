"""Report only zero-exit campaigns; retain failures and their gate reasons."""
import json,statistics,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(name):
    envelope=json.loads((ROOT/'runs'/name/'result.json').read_text())
    if envelope['returncode']!=0 or not envelope['result']['passed']:raise ValueError('failed campaign '+name)
    return envelope['result']

def main():
    first='production-20261114T032106Z';repeat='production-repeat-20261114T033422Z';probe='probe-20261114T031654Z';bridge='bridge-20261114T032828Z'
    heldout='production-heldout-20261114T033558Z'
    a=load(first);b=load(repeat);stress=load(probe);br=load(bridge);hold=load(heldout)
    # Historical envelope's hidden field was overwritten by a loop variable;
    # derive it from the actual compiled plan used by each model.
    with zipfile.ZipFile(ROOT/'runs'/first/'artifacts.zip') as archive:
        for row in a['records']:
            plan=json.loads(archive.read('results/'+row['label']+'/candidate/plan.json'))
            parameter=next(p for p in plan['parameters'] if p['target'].endswith(('up.weight','q.weight')))
            row['hidden']=parameter['shape'][0]
    lines=['# FP16 pack reuse, layout overhead and bounded tails','',
      'The measured implementation is promoted as an explicit `tensor_packing="dual"` policy. `separate` remains the default. All GPU arithmetic, conversion, transpose, crop, loss, gradient and update kernels are OpenAssembler-owned cubins; no vendor computation is an execution fallback.','',
      '## Findings and implementation','',
      'The original recipes launch two operand packs for every forward, dX and dW product. The new schedule converts each logical operand once per forward/backward iteration, produces normal and transposed FP16 layouts together, and shares fixed packed views. A 32×32 shared-memory tile with stride 33 makes both output layouts coalesced. Every forward clears pack validity, so changed inputs and SGD-updated weights are refreshed even in repeated CUDA Graph execution. The schedule never reuses packs across training iterations.','',
      'Odd matrix dimensions use zero-padded operands, the existing OA Tensor Core GEMM, and a guarded FP32 crop with optional bias. The public option is bounded to logical linear dimensions at most 512. Rank-three linear inputs follow existing contiguous flatten/view lowering. FP16 batched attention products and broadcasting remain unsupported; BMM retains the explicit FP32 contract.','',
      '## Complete training measurements','',
      'Both production campaigns use NVIDIA A100-SXM4-40GB, PyTorch 2.7.1+cu128 and CUDA 12.8. Each sample measures four complete MSE/backward/SGD updates in a CUDA Graph with resident inputs, 15 shuffled samples per round and three rounds. GPU event times exclude compilation, CPU capture, allocation and transfers. Reported speedup is baseline time / dual-pack time. Profiler traces are diagnostic; their durations are not added to manufacture an end-to-end total.','',
      '| Workload | Input, hidden | Baseline µs/step (three round medians) | Dual µs/step | Speedup by round | Launches |','|---|---|---|---|---|---|']
    for r in a['records']:
        if not r['existing_contract']:continue
        base=[x['medians_ms']['baseline']*1000 for x in r['timings']];cand=[x['medians_ms']['candidate']*1000 for x in r['timings']]
        lines.append(f"| {r['label']} | {r['shape']}, hidden {r['hidden']} | "+', '.join(f'{x:.2f}' for x in base)+' | '+', '.join(f'{x:.2f}' for x in cand)+' | '+', '.join(f'{x:.4f}×' for x in r['speedups'])+f" | {r['counts']['baseline']}→{r['counts']['candidate']} |")
    lines+=['','Independent production repeat for the three-projection model: '+', '.join(f'{x:.4f}×' for x in b['records'][0]['speedups'])+'. This reproduces the useful complete-training improvement. MLP gains do not reproduce strongly, and the larger-batch MLP is slightly slower, so the policy is not enabled automatically.','',
      'For the three-projection model, pack launches fall from 18 to 5 and complete-step launches from 45 to 32. Private guarded allocations grow from 723,012 to 854,052 bytes (about 18.1%), reflecting the second packed layout. This is a launch/layout optimization, not an improvement to the GEMM instruction schedule.','',
      '## Numerical and ownership gates','',
      'Each case passes four changed-input direct training iterations against CPUReference, including prediction, loss, every dX/dW, updated weights, immutable inputs/targets and allocation canaries. Three additional actual graph replays change inputs/targets and poison workspaces before replay. Profiler kernel names/counts match the runtime ledger exactly. The four cases with an existing separate-pack route also match that schedule bitwise; held-out wider shapes retain the declared tolerance without a bitwise guarantee. The fixed numerical gate is rtol=0.002, atol=0.00002.','',
      f'Exact pack stress passes {len(stress["records"])} shape/pattern combinations, covering random FP32 bit patterns and signed zeros, subnormals, half rounding ties, overflow, infinities and signed canonical NaNs. Both padded layouts, zero fill, input preservation and guards are checked byte-for-byte. Training arithmetic is restricted to finite operands that remain finite after FP16 conversion.','',
      'The configured 17×65 repository passes worker compilation and verification in both direct execution and the public `TrainingReplay` path. Both report zero vendor compute launches and zero GPU code generation during replay. The same plan and runtime code ship through Modal and SSH; an SSH GPU was not separately benchmarked in this campaign.','',
      '## New matrix coverage','',
      '| Complete MLP training input | Hidden | Existing separate FP16 route | Dual route |','|---|---:|---|---|']
    for r in a['records']:
        if not r['existing_contract']:lines.append(f"| {r['shape']} | {r['hidden']} | rejects shape | full direct training and captured replay pass |")
    lines+=['','Held-out input [1,129] with hidden 65 also passes. Wider [128,256] with hidden 512 passes direct and captured training; it is near-neutral in performance. These extend shape coverage rather than demonstrating speedups over an existing unsupported route.','',
      '## Rejected results and limits','',
      'The first implementation scattered transposed writes and failed to improve the MLP despite fewer launches; it remains in `first_candidate.py` and the retained failed/partial runs. A profiler-trace issue required fresh worker subprocesses per case. One early runner envelope incorrectly inherited a child success after a parent failure; that run has returncode=1 and is excluded. The corrected runner requires exit status zero.','',
      'The 1024-wide stress model fails a captured-training gradient gate (888 of 1,048,576 elements in the failing tensor). It is not an admitted contract, its speed is not reported, and the production dual-pack option rejects dimensions above 512. The failure does not establish which compiler/runtime component caused it; further isolation is needed before extending this bound.','',
      'Packing does not eliminate repeated GEMMs in Q/K/V training: they have distinct weights and outputs. General common-subexpression elimination, GEMM fusion, native FP16 BMM/attention, and faster loss reductions for other output sizes are outside this patch.','',
      '## Usage and evidence','',
      '```toml','[model]','entrypoint = "model.py"','input_shapes = [[17,65]]','math = "f16_rne_f32_accum"','tensor_packing = "dual"','mode = "training"','execution = "cuda_graph"','reuse_storage = true','```','',
      'Python: `compile_model(model, examples, out, math_mode="f16_rne_f32_accum", tensor_packing="dual")`.','',
      f'- [Production seven-case training campaign](runs/{first}/result.json)',
      f'- [Independent production three-projection repeat](runs/{repeat}/result.json)',
      f'- [Exact conversion/padding stress](runs/{probe}/result.json)',
      f'- [Direct/public-replay repository bridge](runs/{bridge}/result.json)',
      f'- [Bounded single-row/512-wide held-out repeat](runs/{heldout}/result.json)',
      '- [Excluded 1024-wide stress and retained failure](runs/production-heldout-20261114T033054Z/artifacts.zip)',
      '- [Reproducer and experiment design](README.md)','',
      'Every run directory retains its experiment source, runtime source hashes, full samples and an artifact archive containing OA PTX/cubins and ownership ledgers. Historical sources are frozen; current scripts may include later validation improvements.']
    qkv=next(r for r in a['records'] if r['label']=='qkv128')
    breakdown=['## Packing and matrix trace breakdown','',
        'These are sums of raw kernel activities from one separately profiled complete Q/K/V training invocation (µs). They diagnose where time is spent; they are not paired steady-state distributions and are not the end-to-end latency table above.','',
        '| Region | Separate packs | Dual-layout reuse |','|---|---:|---:|']
    groups={'Packing / transpose':('pack_f16_rne','pack_dual_f16_rne'),
            'Forward GEMMs':('linear',),'dX GEMMs':('linear_dx',),'dW GEMMs':('linear_dw',),
            'Bias / crop':('bias_add','crop_bias'),'Bias gradient reductions':('bias_grad',),'MSE':('mse',),'SGD':('sgd',)}
    measures={}
    with zipfile.ZipFile(ROOT/'runs'/first/'artifacts.zip') as archive:
        for method in ['baseline','candidate']:
            ledger=json.loads(archive.read('results/qkv128/'+method+'/native-ledger.json'))
            roles={n:v['op'] for n,v in ledger['tensor_kernels'].items()}
            roles.update({n:v['signature']['op'] for n,v in ledger['kernels'].items()})
            measures[method]={group:sum(sum(v) for n,v in qkv['audits'][method]['kernel_duration_us'].items() if roles[n] in ops) for group,ops in groups.items()}
    for group in groups:breakdown.append('| '+group+' | '+f"{measures['baseline'][group]:.2f} | {measures['candidate'][group]:.2f} |")
    breakdown+=['','Pack reuse removes conversion/layout work while GEMM kernel times stay effectively unchanged. Serial bias gradients remain expensive. The 256-row MLP and flattened rank-three case additionally expose slow MSE shapes outside the existing 128×128 parallel-loss contract; that is why fewer packing launches barely move their totals.','']
    position=lines.index('## Numerical and ownership gates')
    lines[position:position]=breakdown
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()

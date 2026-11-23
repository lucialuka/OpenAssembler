"""32-bit GPU-relaxed atomics: exact independent and contention linearization gates."""
from collections import Counter
import random
import struct

HEADER = '.version 8.7\n.target sm_80\n.address_size 64\n'
N = 4099
THREADS = 128
GRID = ((N + THREADS - 1) // THREADS, 1, 1)
LANES = GRID[0] * THREADS
BINS = 17
GUARD = b'\xa5' * 256
SENTINEL = 0x13579bdf
CAS_COMPARE = 0x76543210
EDGES = [0, 1, 0xffffffff, 0x7fffffff, 0x80000000, 0xfffffffe, 0x80000001]


def cases():
    for prefix in ('global', 'relaxed.gpu.global'):
        for name, operation, dtype in [('atom', 'add', 'u32'), ('atom', 'add', 's32'),
                                        ('atom', 'exch', 'b32'), ('atom', 'cas', 'b32'),
                                        ('red', 'add', 'u32'), ('red', 'add', 's32')]:
            contract = '.'.join((name, prefix, operation, dtype))
            operands = '%old,[%address+4],%value' if name == 'atom' else '[%address+4],%value'
            if operation == 'cas': operands = '%old,[%address+4],%compare,%value'
            source = HEADER + f'''.visible .entry probe(.param .u64 output,.param .u64 state,.param .u64 input,.param .u32 n,.param .u32 mode) {{
.reg .u64 %output,%state,%input,%offset,%address,%src,%dst;
.reg .b32 %t,%ct,%nt,%i,%n,%mode,%odd,%bin,%value,%compare,%old;
.reg .pred %end,%take,%contended;
ld.param.u64 %output,[output]; ld.param.u64 %state,[state]; ld.param.u64 %input,[input];
ld.param.u32 %n,[n]; ld.param.u32 %mode,[mode]; mov.u32 %t,%tid.x; mov.u32 %ct,%ctaid.x; mov.u32 %nt,%ntid.x;
mad.lo.u32 %i,%ct,%nt,%t; setp.ge.u32 %end,%i,%n; @%end bra DONE;
setp.ne.u32 %contended,%mode,0; mov.u32 %bin,%i; @%contended rem.u32 %bin,%i,{BINS};
mul.wide.u32 %offset,%i,8; add.u64 %src,%input,%offset;
ld.global.b32 %value,[%src]; ld.global.b32 %compare,[%src+4];
mul.wide.u32 %offset,%bin,4; add.u64 %address,%state,%offset;
and.b32 %odd,%i,1; setp.eq.u32 %take,%odd,0; mov.b32 %old,{SENTINEL};
@%take {contract} {operands};
mul.wide.u32 %offset,%i,4; add.u64 %dst,%output,%offset; st.global.u32 [%dst],%old;
DONE: ret; }}
'''
            yield {'name': contract.replace('.', '_'), 'source': source,
                   'contract': {'form': contract, 'family': name, 'operation': operation, 'dtype': dtype,
                                'scope': 'gpu', 'ordering': 'relaxed', 'alignment': 4,
                                'predication': 'even global lanes participate; false lanes keep old result',
                                'address_offset': 4}}


def inputs(case, mode):
    rng = random.Random(14052)
    operation = case['contract']['operation']
    state = [0xa5a5a5a5] + ([CAS_COMPARE] * (BINS if mode else LANES) if operation == 'cas' else
                           [EDGES[i % len(EDGES)] for i in range(BINS if mode else LANES)])
    payload = []
    for i in range(LANES):
        value = 1 if mode and operation == 'add' else i + 1 if mode else rng.getrandbits(32)
        compare = CAS_COMPARE if mode else state[i + 1] if i % 4 == 0 else state[i + 1] ^ 1
        payload += [value, compare]
    return {'state': struct.pack('<' + 'I' * len(state), *state),
            'input': struct.pack('<' + 'I' * len(payload), *payload),
            'output': struct.pack('<' + 'I' * LANES, *([SENTINEL] * LANES))}


def check(case, mode, before, actual):
    state = list(struct.unpack('<' + 'I' * (len(actual['state']) // 4), actual['state']))
    original = list(struct.unpack('<' + 'I' * (len(before['state']) // 4), before['state']))
    output = list(struct.unpack('<' + 'I' * LANES, actual['output']))
    data = list(struct.unpack('<' + 'I' * (len(before['input']) // 4), before['input']))
    operation, returning = case['contract']['operation'], case['contract']['family'] == 'atom'
    failures = []
    if state[0] != original[0]: failures.append('offset predecessor overwritten')
    if actual['input'] != before['input']: failures.append('input mutated')
    inactive = [i for i in range(LANES) if i >= N or i % 2 or not returning]
    if any(output[i] != SENTINEL for i in inactive): failures.append('predicated/tail/no-return output changed')
    bins = BINS if mode else N
    for bin_index in range(bins):
        participants = [i for i in range(bin_index, N, BINS) if i % 2 == 0] if mode else ([bin_index] if bin_index % 2 == 0 else [])
        old = original[bin_index + 1]
        values = [data[2 * i] for i in participants]
        returns = [output[i] for i in participants]
        final = state[bin_index + 1]
        if operation == 'add':
            expected = (old + sum(values)) & 0xffffffff
            if final != expected: failures.append(f'bin {bin_index}: incorrect sum')
            if returning:
                expected_returns = [(old + j) & 0xffffffff for j in range(len(values))] if mode else ([old] if participants else [])
                if Counter(returns) != Counter(expected_returns): failures.append(f'bin {bin_index}: old-value tickets')
        elif operation == 'exch':
            if values:
                if final not in values or Counter(returns + [final]) != Counter([old] + values): failures.append(f'bin {bin_index}: exchange chain')
            elif final != old: failures.append(f'bin {bin_index}: inactive exchange')
        else:
            if mode and values:
                if final not in values or Counter(returns) != Counter([old] + [final] * (len(values) - 1)): failures.append(f'bin {bin_index}: CAS winner/returns')
            else:
                expected = values[0] if participants and data[2 * participants[0] + 1] == old else old
                if final != expected or returns != ([old] if participants else []): failures.append(f'bin {bin_index}: CAS compare result')
    if not mode and state[N + 1:] != original[N + 1:]: failures.append('state tail changed')
    return {'passed': not failures, 'failures': failures[:30], 'participants': (N + 1) // 2,
            'bins': bins, 'scope': 'complete state and old-value linearization invariants; hardware scheduling order is unspecified'}

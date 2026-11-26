"""Stage the missing control-flow edge in native predicate guard readiness."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 text=(HERE/'baseline-backend.py').read_text();old='    return _branch_pred_latency(out)\n';assert text.count(old)==1;text=text.replace(old,'    return _branch_pred_latency(out, labels)\n',1)
 text=text.replace('def _branch_pred_latency(out):','def _branch_pred_latency(out, labels=None):',1)
 anchor='    return out\n\n\n# Fixed-latency pipes on sm_80'
 addition='''    return _cfg_pred_latency(out, labels) if labels else out


def _cfg_pred_latency(out, labels):
    """Keep guard predicates ready on every direct branch/fallthrough path.

    The linear pass above preserves its established schedule. A backedge or
    alternate predecessor can carry a newer predicate definition to a guard;
    inspect those paths too, using the same measured 13-cycle requirement.
    """
    import heapq
    import re

    out = list(out)
    predecessors = [[] for _ in out]
    for index, (text, _) in enumerate(out):
        operation = _opname(text)
        guarded = text.startswith('@')
        if operation == 'BRA':
            target = text.split()[-1].strip('`;()')
            if target in labels and 0 <= labels[target] < len(out):
                predecessors[labels[target]].append(index)
        if index + 1 < len(out) and (operation not in ('BRA', 'EXIT', 'RET') or guarded):
            predecessors[index + 1].append(index)
    for consumer, (text, _) in enumerate(out):
        guard = re.match(r'@!?(P[0-6]) ', text)
        if guard is None:
            continue
        predicate = guard[1]
        queue = [(out[index][1].stall, index) for index in predecessors[consumer]]
        heapq.heapify(queue)
        visited, required = {}, {}
        while queue:
            distance, producer = heapq.heappop(queue)
            if distance >= PRED_TO_GUARD:
                break
            if producer in visited and visited[producer] <= distance:
                continue
            visited[producer] = distance
            instruction, control = out[producer]
            if predicate in _pdefs(instruction):
                required[producer] = max(required.get(producer, 0),
                                         control.stall + PRED_TO_GUARD - distance)
                if not instruction.startswith('@'):
                    continue
            for previous in predecessors[producer]:
                heapq.heappush(queue, (distance + out[previous][1].stall, previous))
        for producer, stall in required.items():
            instruction, control = out[producer]
            out[producer] = (instruction, _dc_replace(control, stall=stall,
                                yield_=control.yield_ or stall > 11))
    return out


# Fixed-latency pipes on sm_80'''
 assert text.count(anchor)==1;text=text.replace(anchor,addition,1);(HERE/'candidate-backend.py').write_text(text)

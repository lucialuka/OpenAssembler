
def _converge_cta_phases(code, labels):
    """Reconverge acyclic conditional paths before an all-thread CTA barrier.

    B0 is reused only after its matching BSYNC. Every selected path reaches the
    same BAR without encountering another barrier, exit or cycle. This is an
    execution dependency, independent of the functional model's implicit joins.
    """
    import re
    labels = labels if labels is not None else {}
    count = len(code)
    successors = []
    for i, operation in enumerate(code):
        name = _opname(operation.text)
        if name == "BRA":
            target = re.search(r"`\((\S+?)\)", operation.text)
            if target is None or target[1] not in labels:
                return code
            edges = [labels[target[1]]]
            if operation.text.startswith("@"):
                edges.append(i + 1)
        elif name == "EXIT":
            edges = []
        else:
            edges = [i + 1]
        successors.append(edges)
    opens, closes = {}, {}
    previous = 0
    for end, operation in enumerate(code):
        if _opname(operation.text) != "BAR":
            continue
        memo, visiting = {}, set()
        def reaches_barrier(index):
            if index == end:
                return True
            if index < previous or index >= end or index in visiting:
                return False
            if index in memo:
                return memo[index]
            if _opname(code[index].text) in ("BAR", "EXIT") or not successors[index]:
                memo[index] = False
                return False
            visiting.add(index)
            result = all(reaches_barrier(next_index) for next_index in successors[index])
            visiting.remove(index)
            memo[index] = result
            return result
        for start in range(previous, end):
            if (code[start].text.startswith("@") and _opname(code[start].text) == "BRA"
                    and reaches_barrier(start)):
                label = f"__oa_cta_join_{end}"
                while label in labels:
                    label += "_"
                opens[start] = label
                closes[end] = label
                break
        previous = end + 1
    if not opens:
        return code
    # No token lives across another BAR, and the selected paths are acyclic.
    mapping, updated, join_labels = {}, [], {}
    for index, operation in enumerate(code):
        mapping[index] = len(updated)
        if index in closes:
            join_labels[closes[index]] = len(updated)
            updated.append(Out("BSYNC B0", False, [], []))
        if index in opens:
            updated.append(Out(f"BSSY B0, `({opens[index]})", False, [], []))
        updated.append(operation)
    mapping[count] = len(updated)
    for label in list(labels):
        labels[label] = mapping[labels[label]]
    labels.update(join_labels)
    return updated


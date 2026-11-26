
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
        reaches = {end: True}
        # Reverse dynamic programming avoids Python recursion for long scalar
        # phases. Backedges and paths crossing another phase stay outside this
        # forward-only reconvergence proof.
        for index in range(end - 1, previous - 1, -1):
            reaches[index] = (_opname(code[index].text) not in ("BAR", "EXIT")
                              and bool(successors[index])
                              and all(index < target <= end and reaches.get(target, False)
                                      for target in successors[index]))
        for start in range(previous, end):
            if (code[start].text.startswith("@") and _opname(code[start].text) == "BRA"
                    and reaches[start]):
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


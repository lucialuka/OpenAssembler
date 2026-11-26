"""All twelve new global ordered forms plus twelve unchanged relaxed controls."""
from cases import cases as relaxed_cases


def cases():
    originals = list(relaxed_cases())
    yield from originals
    for ordering in ('acquire', 'release', 'acq_rel'):
        for original in originals:
            contract = original['contract']
            if contract['family'] != 'atom' or not contract['form'].startswith('atom.global.'):
                continue
            form = contract['form'].replace('atom.global.', f'atom.{ordering}.gpu.global.')
            yield {'name': form.replace('.', '_'),
                   'source': original['source'].replace(contract['form'], form),
                   'contract': dict(contract, form=form, ordering=ordering)}

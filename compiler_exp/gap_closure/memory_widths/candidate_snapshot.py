"""Second research cycle: repair reviewer-discovered pack/unpack snapshot aliases."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


if __name__ == '__main__':
    text = (HERE / 'candidate-ra-v2.py').read_text()
    text = replace_once(text, '    def newp(self) -> PV:', '''    def snapshot_parts(self, values):
        """A pack/unpack copies values now, including components of mutable homes."""
        mutable = set(_flat(self.homes.values()))
        if any(value in mutable for value in values):
            snapshots = tuple(self.new() for _ in values)
            for destination, value in zip(snapshots, values):
                self.emit("MOV {d0}, {s0}", [destination], [value], copy=True)
            return snapshots
        return values

    def newp(self) -> PV:''')
    text = replace_once(text, 'self.bind(a[0].name, tuple(self.reg32(e) for e in a[1].elems))',
                        'self.bind(a[0].name, self.snapshot_parts(tuple(self.reg32(e) for e in a[1].elems)))')
    text = replace_once(text, '            lo, hi = self.pair(a[1].name)\n            for e, v in zip(a[0].elems, (lo, hi)):',
                        '            lo, hi = self.snapshot_parts(self.pair(a[1].name))\n            for e, v in zip(a[0].elems, (lo, hi)):')
    (HERE / 'candidate-ra.py').write_text(text)
    names = ['openassembler/backend/ra.py', 'openassembler/backend/contracts.py']
    (HERE / 'snapshot-candidate-identities.json').write_text(json.dumps({
        'baseline': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
        'candidate': {name: hashlib.sha256((HERE / ('candidate-' + Path(name).name)).read_bytes()).hexdigest() for name in names}}, indent=2) + '\n')

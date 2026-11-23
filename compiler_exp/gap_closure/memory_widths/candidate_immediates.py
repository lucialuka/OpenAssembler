"""Final review repair: split legal binary64 immediates into exact low/high words."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


if __name__ == '__main__':
    text = (HERE / 'candidate-ra-v3.py').read_text()
    text = replace_once(text, '''        if isinstance(operand, ImmI):
            return (self.reg32(ImmI(operand.value & 0xffffffff)),
                    self.reg32(ImmI((operand.value >> 32) & 0xffffffff)))
        return self.pair(operand.name)''', '''        if isinstance(operand, (ImmI, ImmF)):
            if isinstance(operand, ImmF) and operand.width != 64:
                raise NotSupported(f"64-bit operand requires a 64-bit float literal: {operand}")
            bits = operand.bits if isinstance(operand, ImmF) else operand.value
            return (self.reg32(ImmI(bits & 0xffffffff)),
                    self.reg32(ImmI((bits >> 32) & 0xffffffff)))
        if not isinstance(operand, Var):
            raise NotSupported(f"64-bit operand {operand}")
        return self.pair(operand.name)''')
    text = replace_once(text, 'v = self.operand64(a[1]) if isinstance(a[1], ImmI) else self.val64(a[1].name)',
                        'v = self.operand64(a[1]) if isinstance(a[1], (ImmI, ImmF)) else self.val64(a[1].name)')
    (HERE / 'candidate-ra.py').write_text(text)
    names = ['openassembler/backend/ra.py', 'openassembler/backend/contracts.py']
    (HERE / 'immediate-candidate-identities.json').write_text(json.dumps({
        'baseline': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
        'candidate': {name: hashlib.sha256((HERE / ('candidate-' + Path(name).name)).read_bytes()).hexdigest() for name in names}}, indent=2) + '\n')

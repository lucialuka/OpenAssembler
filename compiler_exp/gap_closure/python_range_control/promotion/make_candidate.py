"""Stage safe i32 range exhaustion and structured runtime loop else."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
def once(text,old,new):
 assert text.count(old)==1,old;return text.replace(old,new,1)
if __name__=='__main__':
 text=(HERE/'baseline-pyfront.py').read_text()
 text=once(text,"""        if node.orelse:
            raise NotSupported("'while ... else' is not supported", node.orelse[0])
        top, end = self.s.new_label("while"), self.s.new_label("wend")""","""        top, end = self.s.new_label("while"), self.s.new_label("wend")
        exhausted = self.s.new_label("whileelse") if node.orelse else end""")
 text=once(text,'        self._jump(node.test, end, when=False)\n','        self._jump(node.test, exhausted, when=False)\n')
 text=once(text,"""        self.s.emit(f" bra {top};")
        self.s.emit(f"{end}:")

    def _static_tuple_loop""","""        self.s.emit(f" bra {top};")
        if node.orelse:
            self.s.emit(f"{exhausted}:")
            self.body(node.orelse)
        self.s.emit(f"{end}:")

    def _static_tuple_loop""")
 text=once(text,"""        if node.orelse:
            raise NotSupported("'for ... else' is not supported", node.orelse[0])
""",'')
 text=once(text,'        if step == 0:\n','        step = int(step)  # Python permits a compile-time True step as integer one.\n        if step == 0:\n')
 text=once(text,'        top, cont, end = self.s.new_label("loop"), self.s.new_label("next"), self.s.new_label("loopend")\n','        top, cont, end = self.s.new_label("loop"), self.s.new_label("next"), self.s.new_label("loopend")\n        exhausted = self.s.new_label("forelse") if node.orelse else end\n')
 text=once(text,'        self.s.emit(f" @{pred.name} bra {end};")\n','        self.s.emit(f" @{pred.name} bra {exhausted};")\n')
 text=once(text,"""        self.s.emit(f"{cont}:")
        if step > 0:
""","""        self.s.emit(f"{cont}:")
        if abs(step) > 1:
            # With i32 start/stop, a next value outside i32 has crossed stop.
            # Exhaust before the increment can wrap, preserving the last target.
            overflow = self.s.new_reg("pred")
            relation, limit = ("gt", _I32_MAX - step) if step > 0 else ("lt", _I32_MIN - step)
            self.s.emit(f" setp.{relation}.s32 {overflow.name}, {ctr.name}, {limit};")
            self.s.emit(f" @{overflow.name} bra {exhausted};")
        if step > 0:
""")
 text=once(text,"""        self.s.emit(f" bra {top};")
        self.s.emit(f"{end}:")

    def visit_Break""","""        self.s.emit(f" bra {top};")
        if node.orelse:
            self.s.emit(f"{exhausted}:")
            self.body(node.orelse)
        self.s.emit(f"{end}:")

    def visit_Break""")
 text=once(text,"""            if isinstance(st, ast.For) and hasattr(st, '_oa_tuple_values'):
                a, b = self._block(st.orelse, d or le)""","""            if st.orelse:
                a, b = self._block(st.orelse, d or le or re_)""")
 (HERE/'candidate-pyfront.py').write_text(text)
 text=(HERE/'baseline-return_flow.py').read_text();text=once(text,"""                if node.orelse and not static_tuple:
                    raise c.NotSupported('device helper loop else clauses are not supported', node.orelse[0])
""",'');(HERE/'candidate-return_flow.py').write_text(text)

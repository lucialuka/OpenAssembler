    def _static_tuple_loop(self, node: ast.For) -> None:
        """Finite scalar tuple iteration with real loop exits and target homes."""
        if not isinstance(node.target, ast.Name):
            raise NotSupported('static tuple loop target must be a scalar name', node.target)
        values = node._oa_tuple_values
        if not values:
            # No target binding or body effects occur for an empty tuple.
            self.body(node.orelse)
            return
        self.s.static_tuple_values = getattr(self.s, 'static_tuple_values', 0) + len(values)
        if self.s.static_tuple_values > 1024:
            raise NotSupported('static tuple loops exceed 1024 selected values per kernel', node)
        if any(type(value) not in (int, float) for value in values):
            raise NotSupported('static tuple iteration requires numeric scalar elements', node.iter)
        prior = self.s.vars.get(node.target.id)
        if prior is not None:
            kind = prior.kind
        elif any(type(value) is float for value in values):
            kind = 'f32'
        elif all(_I32_MIN <= value <= _I32_MAX for value in values):
            kind = 's32'
        elif all(-(1 << 63) <= value < (1 << 63) for value in values):
            kind = 's64'
        elif all(0 <= value < (1 << 64) for value in values):
            kind = 'u64'
        else:
            raise NotSupported('static tuple elements do not fit a single device integer type', node.iter)
        if kind not in ('s32', 'u32', 's64', 'u64', 'f32'):
            raise NotSupported('static tuple target requires a scalar numeric type', node.target)
        target = self.s.var_reg(node.target.id, kind, node.target)
        ctr = self.s.new_reg('s32')
        self.s.emit(f' mov.u32 {ctr.name}, 0;')
        top, body, cont, exhausted, end = [self.s.new_label(tag) for tag in
                                          ('tuple', 'tuplebody', 'tuplenext', 'tupleelse', 'tupleend')]
        self.s.emit(f'{top}:')
        pred = self.s.new_reg('pred')
        self.s.emit(f' setp.ge.s32 {pred.name}, {ctr.name}, {len(values)};')
        self.s.emit(f' @{pred.name} bra {exhausted};')
        for index, value in enumerate(values):
            next_choice = self.s.new_label('tuplechoice') if index + 1 < len(values) else None
            if next_choice:
                check = self.s.new_reg('pred')
                self.s.emit(f' setp.ne.s32 {check.name}, {ctr.name}, {index};')
                self.s.emit(f' @{check.name} bra {next_choice};')
            literal, actual_kind = self._lit_as(value, kind, node.iter)
            self._mov(target, literal, actual_kind)
            if next_choice:
                self.s.emit(f' bra {body};')
                self.s.emit(f'{next_choice}:')
        self.s.emit(f'{body}:')
        self.loops.append((cont, end))
        self.depth += 1
        self.body(node.body)
        self.depth -= 1
        self.loops.pop()
        self.s.emit(f'{cont}:')
        self.s.emit(f' add.s32 {ctr.name}, {ctr.name}, 1;')
        self.s.emit(f' bra {top};')
        self.s.emit(f'{exhausted}:')
        self.body(node.orelse)
        self.s.emit(f'{end}:')


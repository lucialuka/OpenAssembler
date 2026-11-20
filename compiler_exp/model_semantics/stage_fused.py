from pathlib import Path
P=Path(__file__).parent/'candidate/openassembler/modelfront'
p=P/'kernels_semantics.py';s=p.read_text().replace("'adam_m','adam_v','adam_update'","'adam_m','adam_v','adam_update','adam_fused'")
s=s.replace("(4 if op=='adam_update' else 3)","(4 if op in ('adam_update','adam_fused') else 3)")
s=s.replace("if op=='adam_update':params", "if op in ('adam_update','adam_fused'):params")
anchor="        elif op=='adam_update':"
s=s.replace(anchor,'''        elif op=='adam_fused':
            # ABI: weight, m, v, gradient, output weight; all reads precede writes.
            for i in range(4):load(i,'%r2',f'%f{i}')
            lines += ['ld.param.f32 %f4,[rate]; ld.param.f32 %f5,[correction1]; ld.param.f32 %f6,[correction2];']
            if attrs['name']=='adam':
                lines += [f'mul.rn.f32 %f7,%f0,{_float(attrs["weight_decay"])}; add.rn.f32 %f3,%f3,%f7;']
            lines += [f'sub.rn.f32 %f7,%f3,%f1; mul.rn.f32 %f7,%f7,{_float(1-attrs["beta1"])}; add.rn.f32 %f1,%f1,%f7;',
                      f'mul.rn.f32 %f2,%f2,{_float(attrs["beta2"])}; mul.rn.f32 %f7,%f3,%f3; mul.rn.f32 %f7,%f7,{_float(1-attrs["beta2"])}; add.rn.f32 %f2,%f2,%f7;',
                      'mul.wide.u32 %d10,%r2,4; add.u64 %d11,%d1,%d10; st.global.f32 [%d11],%f1;',
                      'add.u64 %d11,%d2,%d10; st.global.f32 [%d11],%f2;',
                      'sqrt.rn.f32 %f2,%f2; div.rn.f32 %f2,%f2,%f6;',
                      f'add.rn.f32 %f2,%f2,{_float(attrs["epsilon"])}; div.rn.f32 %f1,%f1,%f2; mul.rn.f32 %f1,%f1,%f5;']
            if attrs['name']=='adamw':lines += [f'mul.rn.f32 %f7,%f4,{_float(attrs["weight_decay"])}; sub.rn.f32 %f7,{_float(1)},%f7; mul.rn.f32 %f0,%f0,%f7;']
            lines += ['sub.rn.f32 %f0,%f0,%f1;']
'''+anchor)
p.write_text(s)
p=P/'kernels.py';s=p.read_text().replace('("adam_m", "adam_v", "adam_update")','("adam_m", "adam_v", "adam_update", "adam_fused")').replace('(4 if op == "adam_update" else 3)','(4 if op in ("adam_update", "adam_fused") else 3)');p.write_text(s)
p=P/'runtime.py';s=p.read_text().replace('if kernel.op == "adam_update":','if kernel.op in ("adam_update", "adam_fused"):')
start=s.index('                    for suffix in ("m","v"):');end=s.index('                increment=',start)
s=s[:start]+'''                    correction=(lr/(1-options["beta1"]**step),math.sqrt(1-options["beta2"]**step))
                    if self.plan.get("optimizer_fusion", False):
                        self._launch(emit_kernel("adam_fused",[spec]*4,spec,options),
                                     [weight,pair["m"],pair["v"],gradient],weight,lr=lr,correction=correction)
                    else:
                        for suffix in ("m","v"):
                            self._launch(emit_kernel("adam_"+suffix,[spec]*3,spec,options),
                                         [pair[suffix],gradient,weight],pair[suffix])
                        self._launch(emit_kernel("adam_update",[spec]*4,spec,options),
                                     [weight,pair["m"],pair["v"],gradient],weight,lr=lr,correction=correction)
'''+s[end:];p.write_text(s)
print('fused candidate staged')

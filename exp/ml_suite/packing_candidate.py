"""Bounded per-value pack slots, common operands, and versioned inference weights."""
from openassembler.modelfront.runtime import Runtime
from openassembler.modelfront.kernels import canonical


class PackedRuntime(Runtime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._pack_epoch=0
        self._pack_valid={}
        self._pack_slots={}
        self._persistent_packs=False

    def _initialize(self,device):
        super()._initialize(device)
        if self._pack_slots or not self.recipes:return
        for step in self.steps:
            if not step.get('tensor'):continue
            recipe=self.recipes[step['entry']]
            for stage in recipe.stages:
                if stage.op!='pack_f16_rne':continue
                source=step['inputs'][int(stage.inputs[0][1:])]
                key=(source,canonical(stage.signature))
                if key not in self._pack_slots:
                    self._pack_slots[key]=self._allocate(recipe.workspaces[stage.output]['shape'],dtype='f16')

    def forward(self,inputs,parameters=None,*,require_tape=False):
        self._pack_epoch+=1
        self._persistent_packs=not require_tape and not self.torch.cuda.is_current_stream_capturing()
        return super().forward(inputs,parameters,require_tape=require_tape)

    def _run(self,phase,values):
        for step in self.steps:
            if step['phase']!=phase:continue
            if step.get('alias'):
                values[step['output']]=values[step['inputs'][0]].view(self.specs[step['output']]['shape'])
            elif step.get('tensor'):
                recipe=self.recipes[step['entry']]
                bindings={f'a{i}':values[name] for i,name in enumerate(step['inputs'])}
                bindings.update(self._tensor_workspaces)
                bindings['output']=values[step['output']]
                for stage in recipe.stages:
                    if stage.op=='pack_f16_rne':
                        source=step['inputs'][int(stage.inputs[0][1:])]
                        key=(source,canonical(stage.signature))
                        tensor=values[source]
                        persistent=self._persistent_packs and source in self.weights
                        identity=('weight',tensor.data_ptr(),tensor._version) if persistent else ('epoch',self._pack_epoch)
                        bindings[stage.output]=self._pack_slots[key]
                        if self._pack_valid.get(key)==identity:continue
                        self._launch(stage,[bindings[x] for x in stage.inputs],bindings[stage.output])
                        self._pack_valid[key]=identity
                    else:
                        self._launch(stage,[bindings[x] for x in stage.inputs],bindings[stage.output])
            else:
                self._launch(self.kernels[step['entry']],[values[x] for x in step['inputs']],values[step['output']])

    def poison_workspace(self):
        super().poison_workspace()
        with self._context_scope():
            for tensor in self._pack_slots.values():
                self._driver.memset(tensor.data_ptr(),0xFF,tensor.numel()*2)
        self._pack_valid.clear()

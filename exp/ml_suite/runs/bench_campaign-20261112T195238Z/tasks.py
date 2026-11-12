"""Frozen synthetic ML workloads with explicit numerical and shape contracts."""
import runpy
import torch


TASKS={
    'tiny_regression':{'factory':'native_model','shape':(17,8),'math':'fp32','training':True},
    'dense_regression_fp32':{'factory':'native_tensor_model','shape':(128,128),'math':'fp32','training':True},
    'dense_regression_fp16':{'factory':'native_tensor_model','shape':(128,128),'math':'f16_rne_f32_accum','training':True},
    'attention_training':{'factory':'native_attention_model','shape':(2,7,8),'math':'fp32','training':True},
    'projection_inference':{'factory':'projections','shape':(128,128),'math':'f16_rne_f32_accum','training':False},
}


class Projections(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q=torch.nn.Linear(128,128);self.k=torch.nn.Linear(128,128);self.v=torch.nn.Linear(128,128)
    def forward(self,x):return self.q(x)+self.k(x)+self.v(x)


def create(name):
    spec=TASKS[name]
    if spec['factory']=='projections':return Projections()
    return runpy.run_path('examples/'+spec['factory']+'/model.py')['create_model']()

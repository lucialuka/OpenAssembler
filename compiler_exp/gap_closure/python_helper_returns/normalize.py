"""Isolated early-return-to-flag hypothesis with required terminal fallback."""
import ast,copy

def normalize(text):
 tree=ast.parse(text)
 class Rewrite(ast.NodeTransformer):
  def visit_FunctionDef(self,fn):
   if not any(isinstance(d,ast.Name)and d.id=='device'for d in fn.decorator_list):return fn
   if not fn.body or not isinstance(fn.body[-1],ast.Return)or fn.body[-1].value is None:raise ValueError('prototype requires a terminal fallback; missing returns cannot become zero')
   if not isinstance(fn.returns,ast.Name)or fn.returns.id not in('i64','u64','i32','u32','f32'):raise ValueError('prototype needs explicit scalar annotation')
   used={n.id for n in ast.walk(fn)if isinstance(n,ast.Name)};prefix='__return'
   while any(name.startswith(prefix)for name in used):prefix+='x'
   done,result=prefix+'_done',prefix+'_value'
   def condition():return ast.Compare(left=ast.Name(done,ast.Load()),ops=[ast.Eq()],comparators=[ast.Constant(0)])
   def block(statements):
    output=[]; may_have_returned=False
    for st in statements:
     if isinstance(st,ast.Return):
      if st.value is None:raise ValueError('prototype scalar return requires a value')
      body=[ast.Assign([ast.Name(result,ast.Store())],st.value),ast.Assign([ast.Name(done,ast.Store())],ast.Constant(1))]
     elif isinstance(st,ast.If):body=[ast.If(test=st.test,body=block(st.body),orelse=block(st.orelse))]
     elif isinstance(st,(ast.For,ast.While)):
      if st.orelse:raise ValueError('prototype loop else return binding is unvalidated')
      new=copy.deepcopy(st);new.body=block(new.body)
      new.body.append(ast.If(test=ast.Compare(left=ast.Name(done,ast.Load()),ops=[ast.NotEq()],comparators=[ast.Constant(0)]),body=[ast.Break()],orelse=[]))
      body=[new]
     elif isinstance(st,(ast.Break,ast.Continue)):
      raise ValueError('prototype original break/continue is unvalidated')
     else:body=[st]
     if may_have_returned:output.append(ast.If(test=condition(),body=body,orelse=[]))
     else:output.extend(body)
     may_have_returned |= any(isinstance(item,ast.Return)for item in ast.walk(st))
    return output
   result_init=ast.Constant(0.0)if fn.returns.id=='f32'else ast.Call(func=ast.Name(fn.returns.id,ast.Load()),args=[ast.Constant(0)],keywords=[])
   fn.body=[ast.Assign([ast.Name(done,ast.Store())],ast.Constant(0)),ast.Assign([ast.Name(result,ast.Store())],result_init)]+block(fn.body)+[ast.Return(ast.Name(result,ast.Load()))]
   return fn
 return ast.unparse(ast.fix_missing_locations(Rewrite().visit(copy.deepcopy(tree))))+'\n'

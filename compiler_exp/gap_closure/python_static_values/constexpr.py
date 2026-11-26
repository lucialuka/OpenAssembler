"""Isolated finite literal-value hypothesis; never eval, import or execute user source."""
import ast,copy,operator
class Rejected(ValueError):pass
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
     ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.LShift:operator.lshift,
     ast.RShift:operator.rshift,ast.BitAnd:operator.and_,ast.BitOr:operator.or_,ast.BitXor:operator.xor}

def value(node,known,depth=0):
 if depth>32:raise Rejected('constant expression exceeds depth 32')
 if isinstance(node,ast.Constant)and type(node.value)in(int,float):result=node.value
 elif isinstance(node,ast.Name)and node.id in known:result=known[node.id]
 elif isinstance(node,ast.Tuple):
  if len(node.elts)>256:raise Rejected('constant tuple exceeds 256 elements')
  result=tuple(value(x,known,depth+1)for x in node.elts)
 elif isinstance(node,ast.UnaryOp)and type(node.op)in(ast.UAdd,ast.USub,ast.Invert):
  v=value(node.operand,known,depth+1);result={ast.UAdd:operator.pos,ast.USub:operator.neg,ast.Invert:operator.invert}[type(node.op)](v)
 elif isinstance(node,ast.BinOp)and type(node.op)in OPS:
  left,right=value(node.left,known,depth+1),value(node.right,known,depth+1)
  if isinstance(node.op,(ast.LShift,ast.RShift))and(type(right)is not int or not 0<=right<=2048):raise Rejected('constant shift out of bound')
  if type(left)not in(int,float)or type(right)not in(int,float):raise Rejected('constant binary values must be numeric')
  result=OPS[type(node.op)](left,right)
 elif isinstance(node,ast.Subscript):
  container,index=value(node.value,known,depth+1),value(node.slice,known,depth+1)
  if type(container)is not tuple or type(index)is not int:raise Rejected('constant index must select a tuple element')
  result=container[index]
 else:raise Rejected('constant expression must contain only declared literal values')
 if type(result)is int and result.bit_length()>2048:raise Rejected('constant integer exceeds 2048 bits')
 if type(result)not in(int,float,tuple):raise Rejected('invalid constant value')
 return result

def normalize(text):
 tree=ast.parse(text);final_names=set()
 for node in tree.body:
  if isinstance(node,ast.ImportFrom)and node.module=='typing':
   final_names.update(alias.asname or alias.name for alias in node.names if alias.name=='Final')
 known={};declared=set();declarations=set()
 for node in tree.body:
  if isinstance(node,ast.AnnAssign)and isinstance(node.target,ast.Name):
   annotation=node.annotation.value if isinstance(node.annotation,ast.Subscript)else node.annotation
   if isinstance(annotation,ast.Name)and annotation.id in final_names:
    if node.target.id in known or node.value is None:raise Rejected('constant declaration is missing or rebound')
    known[node.target.id]=value(node.value,known);declared.add(node.target.id);declarations.add(id(node))
 for node in tree.body:
  if id(node)in declarations or isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)):continue
  for child in ast.walk(node):
   if isinstance(child,ast.Name)and child.id in declared and isinstance(child.ctx,(ast.Store,ast.Del)):raise Rejected('constant is rebound')
   if isinstance(child,(ast.Subscript,ast.Attribute))and isinstance(child.ctx,(ast.Store,ast.Del)):
    head=child
    while isinstance(head,(ast.Subscript,ast.Attribute)):head=head.value
    if isinstance(head,ast.Name)and head.id in declared:raise Rejected('constant object is mutated')
 class Replace(ast.NodeTransformer):
  def visit_FunctionDef(self,node):
   local={arg.arg for arg in node.args.args}|{n.id for n in ast.walk(node)if isinstance(n,ast.Name)and isinstance(n.ctx,ast.Store)}
   replacements={k:v for k,v in known.items()if k not in local}
   class Values(ast.NodeTransformer):
    def visit_Name(self,item):
     if isinstance(item.ctx,ast.Load)and item.id in replacements:
      if type(replacements[item.id])is tuple:raise Rejected('constant tuple must be indexed at compile time')
      return ast.copy_location(ast.Constant(value=replacements[item.id]),item)
     return item
    def visit_Subscript(self,item):
     if isinstance(item.value,ast.Name)and item.value.id in replacements:
      return ast.copy_location(ast.Constant(value=value(item,replacements)),item)
     return self.generic_visit(item)
   return Values().visit(node)
 revised=ast.fix_missing_locations(Replace().visit(copy.deepcopy(tree)))
 return ast.unparse(revised)+'\n',known

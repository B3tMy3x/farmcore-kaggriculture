"""Execute pinned Environment as well as interpreter without importing other envs.

Only package-relative imports are supplied locally. Agent execution is disabled:
these tests call Environment.step with actions, never Environment.run.
"""
import ast
import json
from oracle import ROOT, module

def load_framework():
    exceptions = {name: type(name, (Exception,), {}) for name in ['InvalidArgument','NotFound','DeadlineExceeded','FailedPrecondition']}
    utils_source=ast.parse((ROOT/'reference/utils.py').read_text())
    utils_source.body=[n for n in utils_source.body if not (isinstance(n,ast.ImportFrom) and n.level)]
    utils={'__file__':str(ROOT/'reference/utils.py'), **exceptions}
    exec(compile(utils_source,'reference/utils.py','exec'),utils)
    core_source=ast.parse((ROOT/'reference/core.py').read_text())
    core_source.body=[n for n in core_source.body if not (isinstance(n,ast.ImportFrom) and n.level)]
    class Agent:
        def __init__(self,*args,**kwargs):
            raise RuntimeError('Agent execution is deliberately outside framework tests')
    core={**exceptions,'Agent':Agent,'__version__':'pinned',**{k:utils[k] for k in ['get','get_player','has','process_schema','schemas','structify']}}
    exec(compile(core_source,'reference/core.py','exec'),core)
    return core['Environment']

Environment=load_framework()

def make(config=None, seed=0):
    spec=json.loads((ROOT/'reference/kaggriculture.json').read_text())
    return Environment(specification=spec,configuration={**(config or {}),'seed':seed},interpreter=module.interpreter,renderer=lambda *a:'',html_renderer=lambda *a:'',debug=True)

def result(env):
    return dict(observations=[env._Environment__get_shared_state(i).observation for i in range(2)],rewards=[s.reward for s in env.state],statuses=[s.status for s in env.state],done=env.done)

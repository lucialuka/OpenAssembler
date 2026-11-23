"""Validate original translation-unit settings before integrating discovery/build."""
import json
from pathlib import Path
import shlex
import tempfile


def candidate(record,root):
    directory=(root/record['directory']).resolve()
    arguments=record.get('arguments') or shlex.split(record['command'])
    assert Path(arguments[0]).name=='nvcc'
    source=(directory/record['file']).resolve()
    settings=[];i=1
    while i<len(arguments):
        token=arguments[i]
        if token in ('-c','--compile'):i+=1;continue
        if token in ('-o','--output-file'):
            i+=2;continue
        if (directory/token).resolve()==source:i+=1;continue
        if token in ('-I','-isystem','-include'):
            settings.extend((token,str((directory/arguments[i+1]).resolve())));i+=2;continue
        settings.append(token);i+=1
    return source,settings


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary).resolve();(root/'build').mkdir();(root/'include').mkdir()
        (root/'kernel.cu').write_text('#ifdef OA_ENABLE\n__global__ void selected() {}\n#endif\n')
        record={'directory':'build','file':'../kernel.cu',
                'arguments':['nvcc','-DOA_ENABLE=1','-I','../include','-std=c++17','-arch=sm_80','-c','../kernel.cu','-o','kernel.o']}
        source,flags=candidate(record,root)
        assert source==root/'kernel.cu'
        assert flags==['-DOA_ENABLE=1','-I',str(root/'include'),'-std=c++17','-arch=sm_80']
        result={'schema':1,'passed':True,'gap':'G42 loses build-generated per-source settings and generated translation-unit discovery',
            'solution':'consume static compile_commands records, preserve per-unit preprocessor/precision settings and cwd, resolve actual dependencies with nvcc and retain all identities',
            'scope':'pre-integration argument/relative-path prototype, no tool invocation or hardware proof',
            'input':record,'expected_preserved_flags':['-DOA_ENABLE=1','-I','<root>/include','-std=c++17','-arch=sm_80']}
        path=Path(__file__).with_name('candidate-results.json')
        with path.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print('candidate per-translation-unit setting contract passed')

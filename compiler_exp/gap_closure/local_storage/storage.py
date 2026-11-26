"""Static per-thread local frame layout and observed SM80 donor metadata."""
from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class LocalFrame:
    offsets: dict[str,int]
    sizes: dict[str,int]
    bytes: int


def local_frame(kernel):
    offsets,sizes={},{}
    extent=0
    for declaration in kernel.decls:
        if declaration.space!='local':continue
        if declaration.name in offsets:
            raise ValueError('duplicate local symbol '+declaration.name)
        if (type(declaration.align)is not int or declaration.align not in(1,2,4,8,16)
                or type(declaration.size)is not int or declaration.size<=0):
            raise ValueError('local storage requires positive static sizes and alignment 1/2/4/8/16')
        extent=(extent+declaration.align-1)//declaration.align*declaration.align
        offsets[declaration.name]=extent;sizes[declaration.name]=declaration.size
        extent+=declaration.size
    # The ABI stack top is 16-byte aligned; choose a 16-byte frame multiple
    # independent of donor optimization. Bound immediate offsets and resources.
    extent=(extent+15)//16*16
    if extent>65536:raise ValueError('local frame exceeds the admitted 65536-byte bound')
    return LocalFrame(offsets,sizes,extent)


def set_local_frames(binary,frames):
    """Patch exact observed function-id/u32-size records, retaining symbol ids."""
    from openassembler.container.elf import Elf
    from openassembler.container import nvinfo
    frames={name:size for name,size in frames.items()if size}
    if not frames:return binary
    elf=Elf.parse(binary)
    symbols={index:symbol.name for index,symbol in enumerate(elf.symbols)}
    records=nvinfo.parse(elf.section('.nv.info').data)
    seen={name:{0x11:0,0x12:0}for name in frames}
    for record in records:
        # Observed in the retained CUDA12.8 frame sweeps: both records have
        # (function symbol index, reserved bytes). 0x12 is FRAME_SIZE; the
        # companion 0x11 is kept explicit until a broader metadata contract.
        if record.attr not in(0x11,0x12):continue
        if record.fmt!=4 or len(record.data)!=8:
            raise ValueError('unsupported local frame metadata layout')
        symbol,_=struct.unpack('<II',record.data);name=symbols.get(symbol)
        if name not in frames:continue
        if type(frames[name])is not int or not 0<frames[name]<=65536 or frames[name]%16:
            raise ValueError('invalid native local frame reservation')
        seen[name][record.attr]+=1
        record.data=struct.pack('<II',symbol,frames[name])
    if any(count!=1 for attributes in seen.values()for count in attributes.values()):
        raise ValueError('donor must contain one observed local frame record pair per kernel')
    elf.replace_data('.nv.info',nvinfo.serialize(records))
    return elf.serialize()


def local_frame_inventory(binary):
    from openassembler.container.elf import Elf
    from openassembler.container import nvinfo
    elf=Elf.parse(binary);sizes={};companion={}
    for record in nvinfo.parse(elf.section('.nv.info').data):
        if record.attr not in(0x11,0x12):continue
        if record.fmt!=4 or len(record.data)!=8:raise ValueError('unsupported FRAME_SIZE metadata')
        symbol,size=struct.unpack('<II',record.data)
        if symbol>=len(elf.symbols):raise ValueError('invalid local frame function symbol')
        name=elf.symbols[symbol].name
        table=sizes if record.attr==0x12 else companion
        if name in table:raise ValueError('duplicate local frame function metadata')
        table[name]=size
    if any(size and companion.get(name)!=size for name,size in sizes.items()):
        raise ValueError('local frame metadata record pair disagrees')
    return sizes

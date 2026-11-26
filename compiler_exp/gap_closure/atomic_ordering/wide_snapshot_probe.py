"""A deferred wide product must retain the operands at its PTX definition."""
SOURCE = '''.version 8.7
.target sm_80
.address_size 64
.entry wide_snapshot(.param .u64 output) {
.reg .u64 %out,%product,%offset,%dst;
.reg .u32 %index,%destination; .reg .pred %branch;
ld.param.u64 %out,[output]; mov.u32 %index,%tid.x;
setp.ge.u32 %branch,%index,32; @%branch bra END;
mul.wide.u32 %product,%index,7;
add.u32 %index,%index,100;
mov.u32 %destination,%tid.x;
mul.wide.u32 %offset,%destination,8; add.u64 %dst,%out,%offset;
st.global.u64 [%dst],%product;
END: ret; }
'''

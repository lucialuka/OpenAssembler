"""Minimal independent branch-first pointer materialization counterexample."""
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SOURCE=HEADER+'''.visible .entry pointer_branch(.param .u64 output,.param .u64 input) {
.reg .u64 %output,%input,%offset,%src,%dst;
.reg .u32 %t,%value; .reg .pred %branch;
ld.param.u64 %output,[output]; ld.param.u64 %input,[input];
mov.u32 %t,%tid.x; setp.ge.u32 %branch,%t,16; @%branch bra SECOND;
mul.wide.u32 %offset,%t,4; add.u64 %src,%input,%offset;
ld.global.u32 %value,[%src]; bra STORE;
SECOND:
mul.wide.u32 %offset,%t,4; add.u64 %src,%input,%offset;
ld.global.u32 %value,[%src];
STORE:
mul.wide.u32 %offset,%t,4; add.u64 %dst,%output,%offset;
st.global.u32 [%dst],%value; ret; }
'''

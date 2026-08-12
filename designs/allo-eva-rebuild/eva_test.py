#!/usr/bin/env python3
# eva_test.py — eva_elastic_leanalu.py + a self-contained 1x1 build in __main__.
# Run:  /home/zsm9/miniconda3/envs/allo/bin/python eva_test.py
import sys, os
# --- portability: point these at YOUR Allo install ---
ALLO_HOME = os.environ.get("ALLO_HOME", os.path.expanduser("~/allo_sup"))
sys.path.insert(0, ALLO_HOME)                       # the dir containing the `allo` package
os.environ.setdefault("LLVM_BUILD_DIR",
                      os.environ.get("LLVM_BUILD_DIR", os.path.join(ALLO_HOME, "mlir/build_xcel")))

# ============================================================================
# eva_elastic_leanalu.py — LATENCY-INSENSITIVE (elastic) variant of the EVA
# mesh, derived from eva_sb_syscredit_rtprime_leanalu.py (base region).
#
# WHAT CHANGED vs the credit-exact BSP model (design fork, second design point;
# the credit-exact model remains the RTL-fidelity anchor):
#   * Streams carry ONLY real words. No bubbles, no one-token-per-t discipline.
#     Backpressure = the stream FIFO itself (full/empty), Connections-style.
#   * cr_*/scr_* credit planes: DELETED (all of them). The information they
#     carried now lives in FIFO occupancy.
#   * rbuf/rbcnt/rcred (2-deep receive buffer + credits): DELETED. Replaced by
#     one 1-deep ingress register per direction (ig_p/ig_v); a packet that
#     cannot progress stays in the FIFO — backpressure propagates upstream
#     for free, and the drop-on-overflow hazard is gone by construction.
#   * Router egress regs oe_r/... -> holding regs oh_p/oh_v drained on !full
#     (the ravenoc_lite holding-register idiom). Systolic egress REUSES the
#     existing txp_v/txp_d pending regs — they already were holding registers;
#     only the drain gate changed (scred>0 -> !full). txp_r dropped (vestigial).
#   * All priming machinery DELETED: prime_cfg arg, prologue loops, END-PUT
#     first-token block, gen_kernel PRIME_TOKENS/STREAM_DEPTH overrides.
#     Feedback edges cannot deadlock at init because NOTHING BLOCKS — every
#     stream op in this file is guarded by empty()/full().
#   * Loop counter is a RUN BUDGET, not modeled time. One iteration = one
#     progress attempt. Termination: every kernel runs RUN_BUDGET iterations
#     of pure non-blocking ops and exits — the region terminates
#     deterministically, deadlock is impossible by construction (no blocking
#     calls exist). If RUN_BUDGET is too small the symptom is SHORT OUTPUT
#     (collector k[] < expected count), never a hang.
#
# WHAT IS REUSED VERBATIM (i.e. the PE core is the same machine):
#   packet format/offsets, opcodes, IRF/DRF/dsmask, hold_v/hold_cnt operand
#   buffers, csd_* core-send regs, the ENTIRE retire -> fetch -> operand ->
#   forwarding-scan -> grant -> LEAN ALU -> issue pipeline including the
#   scoreboard shift, resq/cmpq rings, retire_ok structural stall (txp
#   occupancy — unchanged and still correct here), and the crv writeback
#   decode. Drivers/collectors keep their skeletons minus credits/eligibility.
#
# SEMANTICS CAVEATS (by design — this is the elastic machine, not the RTL):
#   * NOT cycle-exact vs the golden RTL. Arbitration outcomes depend on
#     arrival order, which depends on FIFO occupancy / the HLS schedule.
#     Verification = per-flow flit-sequence + final-state scoreboard, NOT
#     cycle trace diff. (Streams preserve per-flow order; cross-flow order
#     is free.)
#   * Driver "eligibility" (never send before slot t) is gone with global t.
#     Drivers emit as fast as accepted; iv_*==0 slots are skipped as before.
#   * Deadlock-freedom argument: routing is straight-through only
#     (W->E, E->W, N->S, S->N + local ejection) — the channel dependency
#     graph is a set of disjoint straight lines terminating in always-
#     draining collectors: acyclic, hence deadlock-free even under
#     hold-and-wait. Core injection holds in csd_pkt (its own slot) and only
#     waits on a downstream chain that drains. On top of that, no operation
#     in the region ever blocks, so the worst case of any bug is
#     non-progress of one flow, never a wedged region.
#
# TOOLING REQUIREMENT: needs the non-blocking Stream probe extension —
#   Stream.empty() on the consumer side and Stream.full() on the producer
#   side (the Connections::In::Empty()-sideband class of support). If the
#   current build lacks .full(), that is a concrete emitter gap to extend
#   (there is no sound sender-side emulation: a producer cannot observe
#   consumer pops). NOTE for Vitis: non-blocking behavior is exactly where
#   csim and cosim diverge (occupancy at call time differs between untimed
#   and scheduled worlds) — debug through cosim, and treat csim timing as
#   meaningless here.
#
# PERFORMANCE NOTES / II RECIPE (unchanged from the credit-exact analysis):
#   The BSP coupling term (L/k cyc/timestep) does not exist here — there are
#   no per-timestep tokens. Steady-state cyc/output ~= node loop II x
#   instructions + congestion stalls, so the loop II now pays off directly:
#     * FP_LAT=2 in this file + latency-2 hadd/hmul binds +
#       set_directive_dependence -variable resq -type inter -direction RAW
#         -distance 2 -dependent true  (same for cmpq; use emitted names)
#     -> II=1, MMM ~= 4 cyc/output at the achieved clock.
#   STREAM_DEPTH is now a real knob (elasticity/burst absorption), not a
#   priming requirement. 8 is a reasonable start.
# ============================================================================

import allo
from allo.ir.types import float16, int16, int32, UInt, AlloType, Stream, float32
import allo.dataflow as df
import numpy as np

M, N = 2, 2          # Mesh dimensions
LANELEN = 10         # workload length per lane (was tied to NSTEP; now independent)
RUN_BUDGET = 4 * LANELEN + 64   # progress-attempt budget per kernel (drain margin);
                                # too small -> short outputs, never a hang

DRF_DEPTH, IRF_DEPTH = 8, 8
# (BUF_DEPTH gone: the receive buffer is now the stream FIFO + 1 ingress reg)

# SCOREBOARD: issue-to-retire (verbatim semantics from the credit-exact model)
SB_DEPTH, RESQ_DEPTH = 5, 8
FWD = 1
FP_LAT = 2   # forward-ready threshold. 2 = matches latency-2 FP binds and the
             # distance-2 dependence directive for II=1 (see header). For MMM
             # this is bit-identical to FP_LAT=1 (its one FP consumer is
             # already at distance 2).
MOV_LAT = 1   # FIX: distance-2 dependence is unsound for MOV_LAT=0 (stale-by-RESQ_DEPTH read); MOV_LAT=1 makes every producer need dist>=2
DATADRIVEN = 1

STREAM_DEPTH = 8     # elasticity per link (perf knob, no priming role)

# OPCODES (verbatim)
OP_ADD, OP_SUB, OP_MULT, OP_MOV = 0x0, 0x1, 0x2, 0x3
OP_RTR0 = 0x4                     # 0x4..0x7 = router send, dir = opcode[1:0]
OP_GEQ, OP_LT = 0x8, 0x9
OP_CRTR0 = 0xC                    # 0xC..0xF = CONDITIONAL router send


def get_eva_top_elastic(Ty: AlloType = float16):
    DATA_W = Ty.bits
    ID_W, MODE_W, ADDR_W, RQ_W = 4, 1, 4, 1

    # router packet = {rq,id,mode,addr,data} packed LSB-first (verbatim layout;
    # RQ bit is kept for rin/rout array compatibility and decode reuse — on
    # the wire it is now always 1, since only real packets travel)
    D_OFF  = 0
    A_OFF  = D_OFF + DATA_W
    MD_OFF = A_OFF + ADDR_W
    ID_OFF = MD_OFF + MODE_W
    RQ_OFF = ID_OFF + ID_W
    PKT_W  = RQ_OFF + RQ_W        # 26 bits

    # sign-extension guard for int32 stores (same Allo vhls emission bug class)
    PMASK  = (1 << PKT_W) - 1 if PKT_W < 32 else 0x7FFFFFFF
    Pkt    = UInt(PKT_W)

    # systolic payload: RAW DATA ONLY — the vld bit is gone with the bubbles.
    # UInt carrier + bitcast at the ends, matching the existing bit-plumbing
    # style (avoids relying on float-typed streams).
    SDATA  = UInt(DATA_W)

    @df.region()
    def top(
        # systolic in-/outputs on all edges (unchanged shapes)
        in_w: Ty[M, LANELEN], in_e: Ty[M, LANELEN],
        in_n: Ty[N, LANELEN], in_s: Ty[N, LANELEN],
        out_w: Ty[M, LANELEN], out_e: Ty[M, LANELEN],
        out_n: Ty[N, LANELEN], out_s: Ty[N, LANELEN],
        # router in-/outputs on all edges (unchanged shapes)
        rin_w: int32[M, LANELEN], rin_e: int32[M, LANELEN],
        rin_n: int32[N, LANELEN], rin_s: int32[N, LANELEN],
        rout_w: int32[M, LANELEN], rout_e: int32[M, LANELEN],
        rout_n: int32[N, LANELEN], rout_s: int32[N, LANELEN],
        # input-valid flags (systolic) — still used to SKIP invalid slots
        iv_w: int32[M, LANELEN], iv_e: int32[M, LANELEN],
        iv_n: int32[N, LANELEN], iv_s: int32[N, LANELEN],
        # NOTE: prime_cfg is GONE — no priming in the elastic machine
    ):
        # Data planes only. cr_*/scr_* credit planes deleted.
        sys_e: Stream[SDATA, STREAM_DEPTH][M, N + 1]
        sys_w: Stream[SDATA, STREAM_DEPTH][M, N + 1]
        sys_s: Stream[SDATA, STREAM_DEPTH][M + 1, N]
        sys_n: Stream[SDATA, STREAM_DEPTH][M + 1, N]
        rtr_e: Stream[Pkt, STREAM_DEPTH][M, N + 1]
        rtr_w: Stream[Pkt, STREAM_DEPTH][M, N + 1]
        rtr_s: Stream[Pkt, STREAM_DEPTH][M + 1, N]
        rtr_n: Stream[Pkt, STREAM_DEPTH][M + 1, N]

        @df.kernel(mapping=[M, N])
        def node():
            i, j = df.get_pid()
            # ---- PE core state: verbatim from the credit-exact model ----
            irf: int32[IRF_DEPTH] = 0
            drf: Ty[DRF_DEPTH] = 0
            drf_full: int32[DRF_DEPTH] = 0
            dsmask: int32 = 0

            crv_vld: int32 = 0
            crv_data: Ty = 0
            crv_addr: int32 = 0
            crv_mode: int32 = 0
            crv_raw:  int32 = 0

            csd_pkt: Pkt = 0
            csd_dir: int32 = 0
            row_id: int32 = i
            col_id: int32 = j

            hold_v: Ty[4, 2] = 0; hold_cnt: UInt(8)[4] = 0

            # ---- ELASTIC interconnect state (replaces rbuf/rbcnt/rcred +
            #      credit regs + egress latch regs) ----
            ig_p: Pkt[4] = 0;   ig_v: int32[4] = 0     # 1-deep ingress reg per dir
            oh_p: Pkt[4] = 0;   oh_v: int32[4] = 0     # router egress holding regs
            txp_v: int32[4] = 0                        # systolic egress holding
            txp_d: Ty[4] = 0                           #   (REUSED; drain gate is
                                                       #    now !full, was scred)

            # Config / program counters (verbatim)
            cfg_isz: int32 = 0
            cfg_itsz: int32 = 0
            fetch_en: UInt(8) = 0
            instr_cnt: UInt(8) = 0
            iter_cnt: UInt(8) = 0
            condition_reg: UInt(8) = 0

            # Scoreboard + result rings (verbatim)
            sb_v: UInt(8)[SB_DEPTH] = 0;  sb_dst: UInt(8)[SB_DEPTH] = 0
            sb_cmp: UInt(8)[SB_DEPTH] = 0; sb_rtr: UInt(8)[SB_DEPTH] = 0
            sb_inj: UInt(8)[SB_DEPTH] = 0; sb_dir: UInt(8)[SB_DEPTH] = 0
            sb_id: UInt(8)[SB_DEPTH] = 0; sb_rvld: UInt(8)[SB_DEPTH] = 0
            sb_ix: UInt(8)[SB_DEPTH] = 0
            sb_long: UInt(8)[SB_DEPTH] = 0
            resq: Ty[RESQ_DEPTH] = 0
            cmpq: UInt(8)[RESQ_DEPTH] = 0
            resq_wr: UInt(8) = 0

            # NO PRIMING. Nothing below blocks, so nothing needs a first token.

            for it in range(RUN_BUDGET):
                # ============ (1) DRAIN router egress holding regs ============
                # (drain-before-fill: a packet spends >=1 iteration in the
                #  holding reg — the registered-output behavior of the
                #  original latch-then-put-next-t idiom)
                if oh_v[0] == 1 and rtr_e[i, j + 1].full() == 0:
                    rtr_e[i, j + 1].put(oh_p[0]); oh_v[0] = 0
                if oh_v[1] == 1 and rtr_w[i, j].full() == 0:
                    rtr_w[i, j].put(oh_p[1]); oh_v[1] = 0
                if oh_v[2] == 1 and rtr_s[i + 1, j].full() == 0:
                    rtr_s[i + 1, j].put(oh_p[2]); oh_v[2] = 0
                if oh_v[3] == 1 and rtr_n[i, j].full() == 0:
                    rtr_n[i, j].put(oh_p[3]); oh_v[3] = 0

                # ============ (2) DRAIN systolic egress (txp regs) ============
                # dir map preserved from the credit-exact drain: 0=N,1=S,2=W,3=E
                if txp_v[0] == 1 and sys_n[i, j].full() == 0:
                    sys_n[i, j].put(txp_d[0].bitcast()); txp_v[0] = 0
                if txp_v[1] == 1 and sys_s[i + 1, j].full() == 0:
                    sys_s[i + 1, j].put(txp_d[1].bitcast()); txp_v[1] = 0
                if txp_v[2] == 1 and sys_w[i, j].full() == 0:
                    sys_w[i, j].put(txp_d[2].bitcast()); txp_v[2] = 0
                if txp_v[3] == 1 and sys_e[i, j + 1].full() == 0:
                    sys_e[i, j + 1].put(txp_d[3].bitcast()); txp_v[3] = 0

                # ============ (3) FILL router ingress regs ============
                # A packet is popped only when the ingress reg is free; an
                # unconsumable packet stays in the FIFO -> upstream backpressure.
                if ig_v[0] == 0 and rtr_e[i, j].empty() == 0:      # from West
                    ig_p[0] = rtr_e[i, j].get(); ig_v[0] = 1
                if ig_v[1] == 0 and rtr_w[i, j + 1].empty() == 0:  # from East
                    ig_p[1] = rtr_w[i, j + 1].get(); ig_v[1] = 1
                if ig_v[2] == 0 and rtr_s[i, j].empty() == 0:      # from North
                    ig_p[2] = rtr_s[i, j].get(); ig_v[2] = 1
                if ig_v[3] == 0 and rtr_n[i + 1, j].empty() == 0:  # from South
                    ig_p[3] = rtr_n[i + 1, j].get(); ig_v[3] = 1

                # ============ (4) FILL systolic hold buffers ============
                # hold_v/hold_cnt are the SAME operand buffers as before; the
                # <2 guard is the backpressure (word stays in FIFO when full).
                if hold_cnt[0] < 2 and sys_s[i, j].empty() == 0:      # from N
                    wN: SDATA = sys_s[i, j].get()
                    hold_v[0, hold_cnt[0]] = wN.bitcast(); hold_cnt[0] += 1
                if hold_cnt[1] < 2 and sys_n[i + 1, j].empty() == 0:  # from S
                    wS: SDATA = sys_n[i + 1, j].get()
                    hold_v[1, hold_cnt[1]] = wS.bitcast(); hold_cnt[1] += 1
                if hold_cnt[2] < 2 and sys_e[i, j].empty() == 0:      # from W
                    wW: SDATA = sys_e[i, j].get()
                    hold_v[2, hold_cnt[2]] = wW.bitcast(); hold_cnt[2] += 1
                if hold_cnt[3] < 2 and sys_w[i, j + 1].empty() == 0:  # from E
                    wE: SDATA = sys_w[i, j + 1].get()
                    hold_v[3, hold_cnt[3]] = wE.bitcast(); hold_cnt[3] += 1

                # ============ (5) LOCAL DELIVERY (priority S>N>E>W, verbatim) ==
                hd: Pkt[4] = 0; hvld: int32[4] = 0; hit: int32[4] = 0
                axis: int32[4] = 0
                axis[0] = col_id; axis[1] = col_id; axis[2] = row_id; axis[3] = row_id
                for d in range(4):
                    if ig_v[d] == 1:
                        hd[d] = ig_p[d]; hvld[d] = 1
                        if hd[d][Ty.bits + 5 : Ty.bits + 9] == axis[d]: hit[d] = 1
                o_crv: Pkt = 0; crv_in: int32 = -1
                if   hit[3] == 1: o_crv = hd[3]; crv_in = 3
                elif hit[2] == 1: o_crv = hd[2]; crv_in = 2
                elif hit[1] == 1: o_crv = hd[1]; crv_in = 1
                elif hit[0] == 1: o_crv = hd[0]; crv_in = 0
                if crv_in >= 0: ig_v[crv_in] = 0   # consumed (was: pop+credit)

                # ============ (6) EGRESS ARBITRATION into holding regs ========
                # Straight-through transit (input d -> output d), core send has
                # priority per port — same arbitration as the credit model,
                # with "credit available" replaced by "holding reg free".
                idir: int32 = -1
                if csd_pkt[RQ_OFF] == 1: idir = 3 - csd_dir
                inj_done: int32 = 0
                for o in range(4):
                    if oh_v[o] == 0:
                        if idir == o:
                            oh_p[o] = csd_pkt; oh_v[o] = 1; inj_done = 1
                        elif ig_v[o] == 1 and hit[o] == 0:
                            oh_p[o] = ig_p[o]; oh_v[o] = 1; ig_v[o] = 0
                if inj_done == 1: csd_pkt = 0

                crv_vld = o_crv[RQ_OFF]
                crv_data = o_crv[0 : Ty.bits].bitcast()
                crv_addr = o_crv[Ty.bits : Ty.bits + 4]
                crv_mode = o_crv[Ty.bits + 4]
                crv_raw  = o_crv[0 : Ty.bits]

                # ============ (7) PE CORE — verbatim from here down ============
                # (4b) RETIRE sb[0] BEFORE fetch. A TX retire whose pending
                # slot is still occupied STALLS (sb frozen, no issue) — the
                # structural stall is UNCHANGED: txp occupancy now clears on
                # !full instead of on credit, but the stall condition is
                # identical.
                retire_ok: int32 = 1
                if sb_v[0] == 1 and sb_rtr[0] == 0 and sb_dst[0] >= 12:
                    if sb_rvld[0] == 1 and txp_v[sb_dst[0] & 3] == 1: retire_ok = 0
                if sb_v[0] == 1 and retire_ok == 1:
                    wb: Ty = resq[sb_ix[0]]
                    if sb_cmp[0] == 1: condition_reg = cmpq[sb_ix[0]]
                    if sb_rtr[0] == 1:
                        if sb_inj[0] == 1 and csd_pkt[RQ_OFF] == 0:
                            csd_pkt[0 : Ty.bits] = wb.bitcast()
                            csd_pkt[Ty.bits : Ty.bits + 4] = sb_dst[0]
                            csd_pkt[Ty.bits + 5 : Ty.bits + 9] = sb_id[0]
                            csd_pkt[RQ_OFF] = sb_rvld[0]
                            csd_dir = sb_dir[0]
                    elif sb_dst[0] >= 12:
                        if sb_rvld[0] == 1:       # bubble result = no-op
                            txp_v[sb_dst[0] & 3] = 1
                            txp_d[sb_dst[0] & 3] = wb
                    else:
                        if sb_rvld[0] == 1:
                            if sb_dst[0] < DRF_DEPTH and ((dsmask >> sb_dst[0]) & 1) == 1:
                                if drf_full[sb_dst[0]] == 0:
                                    drf[sb_dst[0]] = wb
                                    drf_full[sb_dst[0]] = 1
                            else:
                                drf[sb_dst[0] & 7] = wb

                pc: int32 = -1
                if fetch_en == 1: pc = instr_cnt
                instr: int32 = 0
                if pc >= 0: instr = irf[pc]
                op: int32 = instr & 0xF
                dst: int32 = (instr >> 4) & 0xF
                s1: int32 = (instr >> 8) & 0xF
                s2: int32 = (instr >> 12) & 0xF
                a: Ty = 0; b: Ty = 0
                if s1 >= 12: a = hold_v[s1 & 3, 0]
                else:        a = drf[s1]
                if s2 >= 12: b = hold_v[s2 & 3, 0]
                else:        b = drf[s2]

                a_vld: int32 = 1; b_vld: int32 = 1
                if s1 >= 12:
                    a_vld = 0
                    if hold_cnt[s1 & 3] > 0: a_vld = 1
                if s2 >= 12:
                    b_vld = 0
                    if hold_cnt[s2 & 3] > 0: b_vld = 1
                if s1 < DRF_DEPTH and ((dsmask >> s1) & 1) == 1:
                    if drf_full[s1] == 0: a_vld = 0
                if s2 < DRF_DEPTH and ((dsmask >> s2) & 1) == 1:
                    if drf_full[s2] == 0: b_vld = 0
                binop: int32 = 0
                if op == OP_ADD or op == OP_SUB or op == OP_MULT or op == OP_GEQ or op == OP_LT: binop = 1
                # FORWARDING scan: youngest ready producer (verbatim)
                raw: int32 = 0; cmp_busy: int32 = 0
                fwd_a: int32 = 0; fwd_a_ix: int32 = 0; raw_a: int32 = 0
                fwd_b: int32 = 0; fwd_b_ix: int32 = 0; raw_b: int32 = 0
                for k in range(SB_DEPTH - 1):
                    kk: int32 = k + 1
                    inflight: int32 = SB_DEPTH - 1 - kk
                    need: int32 = MOV_LAT
                    if sb_long[kk] == 1: need = FP_LAT
                    rdy: int32 = 0
                    if inflight >= need: rdy = 1
                    if sb_v[kk] == 1 and sb_rtr[kk] == 0 and sb_dst[kk] < 12:
                        if s1 < 12 and (sb_dst[kk] & 7) == (s1 & 7):
                            if FWD == 1 and rdy == 1: fwd_a = 1; fwd_a_ix = sb_ix[kk]; raw_a = 0
                            else:                     fwd_a = 0; raw_a = 1
                        if binop == 1 and s2 < 12 and (sb_dst[kk] & 7) == (s2 & 7):
                            if FWD == 1 and rdy == 1: fwd_b = 1; fwd_b_ix = sb_ix[kk]; raw_b = 0
                            else:                     fwd_b = 0; raw_b = 1
                    if sb_v[kk] == 1 and sb_cmp[kk] == 1: cmp_busy = 1
                raw = raw_a
                if binop == 1 and raw_b == 1: raw = 1
                if fwd_a == 1: a = resq[fwd_a_ix]; a_vld = 1
                if fwd_b == 1: b = resq[fwd_b_ix]; b_vld = 1
                is_cond: int32 = 0
                if op >= OP_CRTR0 and op <= OP_CRTR0 + 3: is_cond = 1

                grant: int32 = 0
                if pc >= 0: grant = 1
                if pc >= 0 and DATADRIVEN == 1 and (a_vld == 0 or (binop == 1 and b_vld == 0)): grant = 0
                if pc >= 0 and (raw == 1 or (is_cond == 1 and cmp_busy == 1)): grant = 0
                if retire_ok == 0: grant = 0     # structural stall (unchanged)
                if grant == 1:
                    if instr_cnt == cfg_isz:
                        instr_cnt = 0
                        if iter_cnt == cfg_itsz - 1: fetch_en = 0
                        else: iter_cnt += 1
                    else: instr_cnt += 1

                c1: int32 = -1; c2: int32 = -1
                if grant == 1 and s1 >= 12: c1 = s1 & 3
                if grant == 1 and s2 >= 12: c2 = s2 & 3
                if c1 >= 0:
                    hold_v[c1, 0] = hold_v[c1, 1]; hold_cnt[c1] -= 1
                if c2 >= 0 and c2 != c1:
                    hold_v[c2, 0] = hold_v[c2, 1]; hold_cnt[c2] -= 1
                # (credit return per consumed hold entry: DELETED — the freed
                #  hold slot re-enables the fill in step (4) next iteration,
                #  which is the same information flow without a return stream)

                if grant == 1 and s1 < DRF_DEPTH and ((dsmask >> s1) & 1) == 1: drf_full[s1] = 0
                if grant == 1 and s2 < DRF_DEPTH and ((dsmask >> s2) & 1) == 1: drf_full[s2] = 0

                res: Ty = 0
                # LEAN #1: SUB == ADD with subtrahend sign flipped (verbatim)
                bbits: UInt(16) = b.bitcast()
                if op == OP_SUB: bbits = bbits ^ 0x8000
                b_eff: Ty = bbits.bitcast()
                if op == OP_ADD or op == OP_SUB: res = a + b_eff
                elif op == OP_MULT: res = a * b
                elif op == OP_GEQ:
                    if a >= b: res = 1.0
                    else:      res = -1.0
                elif op == OP_LT:
                    if a < b:  res = 1.0
                    else:      res = -1.0
                else:               res = a

                res_vld: int32 = a_vld
                if op == OP_ADD or op == OP_SUB or op == OP_MULT or op == OP_GEQ or op == OP_LT:
                    res_vld = a_vld * b_vld
                if grant == 0: res_vld = 0
                is_rtr: int32 = 0
                if op >= OP_RTR0 and op <= OP_RTR0 + 3: is_rtr = 1
                # scoreboard shift (FROZEN during a retire stall) — verbatim
                if retire_ok == 1:
                    for k in range(SB_DEPTH - 1):
                        sb_v[k] = sb_v[k + 1];     sb_dst[k] = sb_dst[k + 1]
                        sb_cmp[k] = sb_cmp[k + 1]; sb_rtr[k] = sb_rtr[k + 1]
                        sb_inj[k] = sb_inj[k + 1]; sb_dir[k] = sb_dir[k + 1]
                        sb_id[k] = sb_id[k + 1];   sb_rvld[k] = sb_rvld[k + 1]
                        sb_ix[k] = sb_ix[k + 1]; sb_long[k] = sb_long[k + 1]
                    sb_v[SB_DEPTH - 1] = 0
                if grant == 1:
                    resq[resq_wr] = res
                    cq: int32 = 0
                    if op == OP_GEQ:
                        if a >= b: cq = 1
                    if op == OP_LT:
                        if a < b: cq = 1
                    cmpq[resq_wr] = cq
                    sb_v[SB_DEPTH - 1] = 1
                    sb_dst[SB_DEPTH - 1] = dst
                    sb_ix[SB_DEPTH - 1] = resq_wr
                    sb_long[SB_DEPTH - 1] = binop
                    sb_cmp[SB_DEPTH - 1] = 0
                    if op == OP_GEQ or op == OP_LT: sb_cmp[SB_DEPTH - 1] = 1
                    rtrf: int32 = is_rtr
                    if is_cond == 1: rtrf = 1
                    sb_rtr[SB_DEPTH - 1] = rtrf
                    inj: int32 = is_rtr
                    if is_cond == 1 and condition_reg == 1: inj = 1
                    sb_inj[SB_DEPTH - 1] = inj
                    sb_dir[SB_DEPTH - 1] = op & 3
                    sb_id[SB_DEPTH - 1] = s2
                    sb_rvld[SB_DEPTH - 1] = res_vld
                    resq_wr = (resq_wr + 1) & (RESQ_DEPTH - 1)

                # (systolic drain moved to step (2); credit gate deleted)

                # crv WRITEBACK — verbatim
                if crv_vld == 1:
                    if crv_mode == 1:
                        # & 1: signedness-immune (Allo emits UInt slices as
                        # SIGNED intN)
                        if ((crv_addr >> 3) & 1) == 1: irf[crv_addr & 7] = crv_raw
                        elif crv_addr == 0:
                            dsmask = crv_raw & 0xFF
                            cfg_isz = (crv_raw >> 8) & 0x7
                            if ((crv_raw >> 15) & 1) == 1: fetch_en = 1; instr_cnt = 0; iter_cnt = 0
                        elif crv_addr == 1: cfg_itsz = crv_raw & 0xFF
                    elif ((crv_addr >> 2) & 3) == 3:
                        # ROUTER DATA pkt to sys addr 0xC..0xF writes the
                        # SYSTOLIC TX via txp (drains on !full, was next-t
                        # credit gate). dir = crv_addr&3: 0=N,1=S,2=W,3=E.
                        txp_v[crv_addr & 3] = 1
                        txp_d[crv_addr & 3] = crv_data
                    elif crv_addr < DRF_DEPTH and ((dsmask >> crv_addr) & 1) == 1:
                        if drf_full[crv_addr] == 0:
                            drf[crv_addr] = crv_data; drf_full[crv_addr] = 1
                    else:
                        drf[crv_addr] = crv_data

                # NO END-PUT BLOCK — all emission happens in steps (1)/(2)
                # gated on !full. Nothing is sent when there is nothing to say.

        # ================= DRIVERS (elastic: push when accepted) =============
        # Credits and slot-eligibility deleted; iv==0 slots still skipped.
        @df.kernel(mapping=[1], args=[in_w, iv_w])
        def drv_w(din_w: Ty[M, LANELEN], vd_w: int32[M, LANELEN]):
            sp: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if sp[r] < LANELEN:
                        if vd_w[r, sp[r]] == 0:
                            sp[r] += 1
                        elif sys_e[r, 0].full() == 0:
                            sys_e[r, 0].put(din_w[r, sp[r]].bitcast())
                            sp[r] += 1

        @df.kernel(mapping=[1], args=[in_e, iv_e])
        def drv_e(din_e: Ty[M, LANELEN], vd_e: int32[M, LANELEN]):
            sp: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if sp[r] < LANELEN:
                        if vd_e[r, sp[r]] == 0:
                            sp[r] += 1
                        elif sys_w[r, N].full() == 0:
                            sys_w[r, N].put(din_e[r, sp[r]].bitcast())
                            sp[r] += 1

        @df.kernel(mapping=[1], args=[in_n, iv_n])
        def drv_n(din_n: Ty[N, LANELEN], vd_n: int32[N, LANELEN]):
            sp: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if sp[c] < LANELEN:
                        if vd_n[c, sp[c]] == 0:
                            sp[c] += 1
                        elif sys_s[0, c].full() == 0:
                            sys_s[0, c].put(din_n[c, sp[c]].bitcast())
                            sp[c] += 1

        @df.kernel(mapping=[1], args=[in_s, iv_s])
        def drv_s(din_s: Ty[N, LANELEN], vd_s: int32[N, LANELEN]):
            sp: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if sp[c] < LANELEN:
                        if vd_s[c, sp[c]] == 0:
                            sp[c] += 1
                        elif sys_n[M, c].full() == 0:
                            sys_n[M, c].put(din_s[c, sp[c]].bitcast())
                            sp[c] += 1

        # ================= COLLECTORS (elastic: pop when present) ============
        # Note: no credit returns and no cyc stamps — the loop index is not
        # time here; real-cycle measurement is the Verilog boundary monitor.
        @df.kernel(mapping=[1], args=[out_w])
        def col_w(dout_w: Ty[M, LANELEN]):
            k: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if sys_w[r, 0].empty() == 0:
                        w: SDATA = sys_w[r, 0].get()
                        if k[r] < LANELEN:
                            dout_w[r, k[r]] = w.bitcast()
                            k[r] += 1

        @df.kernel(mapping=[1], args=[out_e])
        def col_e(dout_e: Ty[M, LANELEN]):
            k: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if sys_e[r, N].empty() == 0:
                        w: SDATA = sys_e[r, N].get()
                        if k[r] < LANELEN:
                            dout_e[r, k[r]] = w.bitcast()
                            k[r] += 1

        @df.kernel(mapping=[1], args=[out_n])
        def col_n(dout_n: Ty[N, LANELEN]):
            k: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if sys_n[0, c].empty() == 0:
                        w: SDATA = sys_n[0, c].get()
                        if k[c] < LANELEN:
                            dout_n[c, k[c]] = w.bitcast()
                            k[c] += 1

        @df.kernel(mapping=[1], args=[out_s])
        def col_s(dout_s: Ty[N, LANELEN]):
            k: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if sys_s[M, c].empty() == 0:
                        w: SDATA = sys_s[M, c].get()
                        if k[c] < LANELEN:
                            dout_s[c, k[c]] = w.bitcast()
                            k[c] += 1

        # ================= ROUTER DRIVERS ====================================
        # rq==0 entries in rin are still skipped (they were bubbles in the
        # stimulus arrays; on the elastic wire they simply don't travel).
        @df.kernel(mapping=[1], args=[rin_w])
        def rdrv_w(rdin_w: int32[M, LANELEN]):
            sp: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if sp[r] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_w[r, sp[r]]
                        if cand[RQ_OFF] == 0:
                            sp[r] += 1
                        elif rtr_e[r, 0].full() == 0:
                            rtr_e[r, 0].put(cand); sp[r] += 1

        @df.kernel(mapping=[1], args=[rin_e])
        def rdrv_e(rdin_e: int32[M, LANELEN]):
            sp: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if sp[r] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_e[r, sp[r]]
                        if cand[RQ_OFF] == 0:
                            sp[r] += 1
                        elif rtr_w[r, N].full() == 0:
                            rtr_w[r, N].put(cand); sp[r] += 1

        @df.kernel(mapping=[1], args=[rin_n])
        def rdrv_n(rdin_n: int32[N, LANELEN]):
            sp: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if sp[c] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_n[c, sp[c]]
                        if cand[RQ_OFF] == 0:
                            sp[c] += 1
                        elif rtr_s[0, c].full() == 0:
                            rtr_s[0, c].put(cand); sp[c] += 1

        @df.kernel(mapping=[1], args=[rin_s])
        def rdrv_s(rdin_s: int32[N, LANELEN]):
            sp: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if sp[c] < LANELEN:
                        cand: Pkt = 0
                        cand[0 : Ty.bits + 10] = rdin_s[c, sp[c]]
                        if cand[RQ_OFF] == 0:
                            sp[c] += 1
                        elif rtr_n[M, c].full() == 0:
                            rtr_n[M, c].put(cand); sp[c] += 1

        # ================= ROUTER COLLECTORS =================================
        @df.kernel(mapping=[1], args=[rout_w])
        def rclc_w(rdout_w: int32[M, LANELEN]):
            k: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if rtr_w[r, 0].empty() == 0:
                        pw: Pkt = rtr_w[r, 0].get()
                        if pw[RQ_OFF] == 1 and k[r] < LANELEN:
                            rdout_w[r, k[r]] = pw & PMASK
                            k[r] += 1

        @df.kernel(mapping=[1], args=[rout_e])
        def rclc_e(rdout_e: int32[M, LANELEN]):
            k: int32[M] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, M) as r:
                    if rtr_e[r, N].empty() == 0:
                        pw: Pkt = rtr_e[r, N].get()
                        if pw[RQ_OFF] == 1 and k[r] < LANELEN:
                            rdout_e[r, k[r]] = pw & PMASK
                            k[r] += 1

        @df.kernel(mapping=[1], args=[rout_n])
        def rclc_n(rdout_n: int32[N, LANELEN]):
            k: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if rtr_n[0, c].empty() == 0:
                        pw: Pkt = rtr_n[0, c].get()
                        if pw[RQ_OFF] == 1 and k[c] < LANELEN:
                            rdout_n[c, k[c]] = pw & PMASK
                            k[c] += 1

        @df.kernel(mapping=[1], args=[rout_s])
        def rclc_s(rdout_s: int32[N, LANELEN]):
            k: int32[N] = 0
            for it in range(RUN_BUDGET):
                with allo.meta_for(0, N) as c:
                    if rtr_s[M, c].empty() == 0:
                        pw: Pkt = rtr_s[M, c].get()
                        if pw[RQ_OFF] == 1 and k[c] < LANELEN:
                            rdout_s[c, k[c]] = pw & PMASK
                            k[c] += 1

    return top


# ============================================================================
# 1x1 BUILD (pure Allo: pipeline + partition on node AND all drivers/collectors)
# ============================================================================
if __name__ == "__main__":
    import re
    M = N = 1                          # <- 1x1 (reassigns module globals used by the region)
    LANELEN = 120
    RUN_BUDGET = 4 * LANELEN + 64

    s = df.customize(get_eva_top_elastic(float16))

    # partition node register state (scoreboard/rings must be registers)
    NODE_BUFS = ("irf","drf","drf_full","hold_v","hold_cnt","ig_p","ig_v","oh_p","oh_v",
                 "txp_v","txp_d","resq","cmpq","sb_v","sb_dst","sb_cmp","sb_rtr",
                 "sb_inj","sb_dir","sb_id","sb_rvld","sb_ix","sb_long")
    for buf in NODE_BUFS:
        try: s.partition(f"node_0_0:{buf}")
        except Exception as e: print("partition skip", buf, "->", str(e)[:50])

    # pipeline II=1 on EVERY per-iteration loop: node + sys drv/col + rtr drv/col
    PIPE = ["node_0_0"] + [f"{k}_0" for k in
            ("drv_w","drv_e","drv_n","drv_s","col_w","col_e","col_n","col_s",
             "rdrv_w","rdrv_e","rdrv_n","rdrv_s","rclc_w","rclc_e","rclc_n","rclc_s")]
    for inst in PIPE:
        s.pipeline(f"{inst}:it", initiation_interval=1)
        print("pipelined", inst)

    P = os.environ.get("EVA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prj_eva_test"))
    s.build(target="vhls", mode="csyn", project=P)
    kp = os.path.join(P, "kernel.cpp"); src = open(kp).read()
    src = re.sub(r"(union \{[^}]*\}\s*_converter\w*)\s*;", r"\1 = {};", src)  # union-fix
    open(kp, "w").write(src)
    print(f"DONE: {kp}  ({sum(1 for _ in open(kp))} lines, "
          f"{src.count('#pragma HLS pipeline')} pipeline pragmas)")

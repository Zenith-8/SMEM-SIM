#!/usr/bin/env python3
"""
Combined multi-instruction warp workload for the SMEM functional simulator.

This test exercises a realistic "kernel-shaped" traffic pattern that mixes every
transaction type the simulator supports across two full 32-thread warps issued
back-to-back.  The goal is to make steady-state throughput and DCache-vs-SMEM
cycle cost visible over a long workload (not a single micro-test).

Kernel shape (per warp, 32 threads, one warp-wide instruction at a time):

    1.  global.ld.dram2sram  (stream an input tile from DRAM into SMEM)
    2.  sh.st                (write a derived/local value into the tile)
    3.  sh.ld                (read the derived value back, e.g. reduction feed)
    4.  global.st.smem2dram  (stream the final tile back out to DRAM)

Warp 0 uses resident thread_block_id=0 and thread_ids 0..31.
Warp 1 uses resident thread_block_id=1 and thread_ids 32..63; with
``thread_block_size_bytes = 0x80`` this preserves the same effective SMEM
spacing as the old 0x000 / 0x080 offset layout while moving the workload onto
the new resident-TBID path.

The same traffic is also driven through ``LockupFreeCacheStage`` (the real
DCache) in parallel, so we can report both cycle counts side-by-side, which is
the whole point of a longer combined workload.

The DCache-vs-SMEM comparison path uses 0-cycle DRAM latency on both sides so
the cycle gap reflects only the on-chip pipeline and queuing behavior.

Run:
    python3 test_combined_warp_workload.py

Output is appended to ``output_extended.txt`` via
``test_output.capture_to_extended_log`` so every test script composes into a
single consolidated report.
"""

from __future__ import annotations

from collections import deque
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

from main import (
    ShmemFunctionalSimulator,
    SmemArbiter,
    Transaction,
    TxnType,
    load_smem_config,
)
from test_dcache_and_smem import _load_dcache_symbols


DCACHE = _load_dcache_symbols()

from simulator.mem_types import (
    BANK_ID_BIT_LEN as DCACHE_BANK_ID_BIT_LEN,
    BLOCK_OFF_BIT_LEN as DCACHE_BLOCK_OFF_BIT_LEN,
    BLOCK_SIZE_WORDS as DCACHE_BLOCK_SIZE_WORDS,
    BYTE_OFF_BIT_LEN as DCACHE_BYTE_OFF_BIT_LEN,
    NUM_BANKS as DCACHE_NUM_BANKS,
    NUM_SETS_PER_BANK as DCACHE_NUM_SETS,
    NUM_WAYS as DCACHE_NUM_WAYS,
    SET_INDEX_BIT_LEN as DCACHE_SET_INDEX_BIT_LEN,
)


NUM_THREADS_PER_WARP: int = 32
WARP_COUNT: int = 2
TOTAL_THREADS: int = NUM_THREADS_PER_WARP * WARP_COUNT

WORD_BYTES: int = 4
THREAD_BLOCK_SIZE_BYTES: int = NUM_THREADS_PER_WARP * WORD_BYTES
WARP0_TBID: int = 0
WARP1_TBID: int = 1
SMEM_SLOT_STRIDE_BYTES: int = WORD_BYTES
DRAM_A_BASE: int = 0x0001_0000
DRAM_B_BASE: int = 0x0002_0000
DRAM_OUT_A_BASE: int = 0x0003_0000
DRAM_OUT_B_BASE: int = 0x0004_0000
DCACHE_LINE_STRIDE_SLOTS: int = 1
COMPARISON_DRAM_LATENCY_CYCLES: int = 0

DRAM_SEED_WARP0: int = 0xA000_0000
DRAM_SEED_WARP1: int = 0xB000_0000
COMPUTE_XOR_MASK: int = 0x0000_ABCD
RESIDENT_THREAD_BLOCK_IDS: Tuple[Optional[int], ...] = (
    WARP0_TBID,
    WARP1_TBID,
    None,
    None,
)

_DCACHE_NUM_BANKS = int(DCACHE_NUM_BANKS)
_DCACHE_NUM_SETS = int(DCACHE_NUM_SETS)
_DCACHE_NUM_WAYS = int(DCACHE_NUM_WAYS)
_DCACHE_BLOCK_WORDS = int(DCACHE_BLOCK_SIZE_WORDS)
_DCACHE_SET_INDEX_BITS = int(DCACHE_SET_INDEX_BIT_LEN)
_DCACHE_BANK_ID_BITS = int(DCACHE_BANK_ID_BIT_LEN)
_DCACHE_BLOCK_OFF_BITS = int(DCACHE_BLOCK_OFF_BIT_LEN)
_DCACHE_BYTE_OFF_BITS = int(DCACHE_BYTE_OFF_BIT_LEN)


@dataclass
class WarpInstruction:
    """A single warp-wide instruction: 32 per-thread transactions issued together."""

    name: str
    txn_type: TxnType
    transactions: List[Transaction]


@dataclass
class WarpProgram:
    """The full back-to-back instruction stream for one warp."""

    warp_id: int
    thread_id_base: int
    thread_block_id: int
    instructions: List[WarpInstruction] = field(default_factory=list)


@dataclass
class PhaseMetrics:
    """Per-warp-instruction cycle accounting for the SMEM run."""

    warp_id: int
    instruction_name: str
    txn_type: str
    cycle_started: int
    cycle_completed: int
    sub_batches: int
    sub_batch_sizes: List[int]
    completed_count: int

    @property
    def cycles(self) -> int:
        return int(self.cycle_completed) - int(self.cycle_started)

    @property
    def throughput_txns_per_cycle(self) -> float:
        span = max(self.cycles, 1)
        return float(self.completed_count) / float(span)


@dataclass
class DcachePhaseMetrics:
    """Per-warp-instruction cycle accounting for the DCache run."""

    warp_id: int
    instruction_name: str
    txn_type: str
    cycle_started: int
    cycle_completed: int
    completed_count: int
    hits: int
    misses: int

    @property
    def cycles(self) -> int:
        return int(self.cycle_completed) - int(self.cycle_started)

    @property
    def throughput_txns_per_cycle(self) -> float:
        span = max(self.cycles, 1)
        return float(self.completed_count) / float(span)


def _build_warp_program(
    *,
    warp_id: int,
    thread_id_base: int,
    thread_block_id: int,
    dram_in_base: int,
    dram_out_base: int,
) -> WarpProgram:
    """
    Build the 4-instruction program for a single 32-thread warp.

    Each per-thread transaction uses a per-thread relative SMEM offset of
    ``lane * 4`` within the resident SMEM slot assigned to ``thread_block_id``,
    giving no intra-warp bank conflicts at the XOR-mapped crossbar. DRAM slots
    are per-lane too, so global-memory traffic pipelines without aliasing.
    """
    program = WarpProgram(
        warp_id=warp_id,
        thread_id_base=thread_id_base,
        thread_block_id=thread_block_id,
    )

    global_load = WarpInstruction(
        name=f"warp{warp_id}.global.ld.dram2sram",
        txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
        transactions=[
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                thread_id=thread_id_base + lane,
                thread_block_id=thread_block_id,
                resident_thread_block_ids=RESIDENT_THREAD_BLOCK_IDS,
                thread_block_done_bits=[0],
                shmem_addr=lane * SMEM_SLOT_STRIDE_BYTES,
                dram_addr=dram_in_base + (lane * WORD_BYTES),
            )
            for lane in range(NUM_THREADS_PER_WARP)
        ],
    )

    compute_store = WarpInstruction(
        name=f"warp{warp_id}.sh.st",
        txn_type=TxnType.SH_ST,
        transactions=[
            Transaction(
                txn_type=TxnType.SH_ST,
                thread_id=thread_id_base + lane,
                thread_block_id=thread_block_id,
                resident_thread_block_ids=RESIDENT_THREAD_BLOCK_IDS,
                thread_block_done_bits=[0],
                shmem_addr=lane * SMEM_SLOT_STRIDE_BYTES,
                write_data=(thread_id_base + lane) ^ COMPUTE_XOR_MASK,
            )
            for lane in range(NUM_THREADS_PER_WARP)
        ],
    )

    readback_load = WarpInstruction(
        name=f"warp{warp_id}.sh.ld",
        txn_type=TxnType.SH_LD,
        transactions=[
            Transaction(
                txn_type=TxnType.SH_LD,
                thread_id=thread_id_base + lane,
                thread_block_id=thread_block_id,
                resident_thread_block_ids=RESIDENT_THREAD_BLOCK_IDS,
                thread_block_done_bits=[0],
                shmem_addr=lane * SMEM_SLOT_STRIDE_BYTES,
            )
            for lane in range(NUM_THREADS_PER_WARP)
        ],
    )

    global_store = WarpInstruction(
        name=f"warp{warp_id}.global.st.smem2dram",
        txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
        transactions=[
            Transaction(
                txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                thread_id=thread_id_base + lane,
                thread_block_id=thread_block_id,
                resident_thread_block_ids=RESIDENT_THREAD_BLOCK_IDS,
                thread_block_done_bits=[0],
                shmem_addr=lane * SMEM_SLOT_STRIDE_BYTES,
                dram_addr=dram_out_base + (lane * WORD_BYTES),
            )
            for lane in range(NUM_THREADS_PER_WARP)
        ],
    )

    program.instructions.extend([global_load, compute_store, readback_load, global_store])
    return program


def _build_both_warp_programs() -> Tuple[WarpProgram, WarpProgram]:
    warp0 = _build_warp_program(
        warp_id=0,
        thread_id_base=0,
        thread_block_id=WARP0_TBID,
        dram_in_base=DRAM_A_BASE,
        dram_out_base=DRAM_OUT_A_BASE,
    )
    warp1 = _build_warp_program(
        warp_id=1,
        thread_id_base=NUM_THREADS_PER_WARP,
        thread_block_id=WARP1_TBID,
        dram_in_base=DRAM_B_BASE,
        dram_out_base=DRAM_OUT_B_BASE,
    )
    return warp0, warp1


def _build_dram_init(
    warp_programs: Sequence[WarpProgram],
) -> Dict[int, int]:
    """Populate DRAM with distinct per-thread seed values for global.ld paths."""
    dram: Dict[int, int] = {}
    for program in warp_programs:
        seed = DRAM_SEED_WARP0 if program.warp_id == 0 else DRAM_SEED_WARP1
        for instr in program.instructions:
            if instr.txn_type is not TxnType.GLOBAL_LD_DRAM_TO_SRAM:
                continue
            for txn in instr.transactions:
                assert txn.dram_addr is not None
                dram[int(txn.dram_addr)] = int(seed + int(txn.thread_id)) & 0xFFFF_FFFF
    return dram


def _build_smem_simulator(dram_init: Dict[int, int]) -> ShmemFunctionalSimulator:
    """Build the SMEM simulator using .config defaults with resident TBIDs."""
    cfg = load_smem_config()
    kwargs = cfg.to_sim_kwargs()
    kwargs["num_threads"] = TOTAL_THREADS
    kwargs["dram_latency_cycles"] = COMPARISON_DRAM_LATENCY_CYCLES
    kwargs.pop("thread_block_offsets", None)
    kwargs["thread_block_size_bytes"] = int(
        kwargs.get("thread_block_size_bytes") or THREAD_BLOCK_SIZE_BYTES
    )
    kwargs["verbose"] = True
    return ShmemFunctionalSimulator(dram_init=dram_init, **kwargs)


def _effective_tbo_for_thread_block(thread_block_id: int) -> int:
    """
    Ask the simulator to derive the effective TBO for the given resident TBID.

    This keeps the test report aligned with the simulator's slot-based TBID
    residency logic instead of reconstructing the offset manually.
    """
    sim = _build_smem_simulator({})
    probe = Transaction(
        txn_type=TxnType.SH_LD,
        thread_id=0,
        thread_block_id=int(thread_block_id),
        resident_thread_block_ids=RESIDENT_THREAD_BLOCK_IDS,
        thread_block_done_bits=[0],
        shmem_addr=0,
    )
    return int(sim._effective_thread_block_offset(probe))


def _run_smem_workload(
    warp_programs: Sequence[WarpProgram],
    dram_init: Dict[int, int],
) -> Tuple[int, List[PhaseMetrics], ShmemFunctionalSimulator]:
    """
    Issue every warp-instruction back-to-back through the SMEM arbiter and
    drain to completion, recording per-instruction cycle cost + throughput.
    """
    sim = _build_smem_simulator(dram_init)
    arbiter = SmemArbiter(sim)

    metrics: List[PhaseMetrics] = []
    completion_cursor = 0

    for program in warp_programs:
        for instr in program.instructions:
            cycle_started = sim.cycle
            partition_info = arbiter.process_batch(instr.transactions)

            expected_completions = completion_cursor + len(instr.transactions)
            max_cycles = max(1024, len(instr.transactions) * 64)
            steps_taken = 0
            while len(sim.completions) < expected_completions:
                steps_taken += 1
                if steps_taken > max_cycles:
                    raise TimeoutError(
                        f"SMEM instruction {instr.name} did not complete within "
                        f"{max_cycles} cycles."
                    )
                sim.step()

            completion_cursor = expected_completions
            metrics.append(
                PhaseMetrics(
                    warp_id=program.warp_id,
                    instruction_name=instr.name,
                    txn_type=instr.txn_type.value,
                    cycle_started=int(cycle_started),
                    cycle_completed=int(sim.cycle),
                    sub_batches=int(partition_info["num_sub_batches"]),
                    sub_batch_sizes=list(partition_info["sub_batch_sizes"]),
                    completed_count=len(instr.transactions),
                )
            )

    while sim._has_pending_work():
        sim.step()

    return int(sim.cycle), metrics, sim


def _build_dcache_address(
    *,
    bank_id: int,
    set_index: int,
    tag: int,
    block_offset: int = 0,
    byte_offset: int = 0,
) -> int:
    """Compose a dcache address from its bit-slices (matches `simulator.mem_types`)."""
    return (
        (int(tag) << (
            _DCACHE_SET_INDEX_BITS
            + _DCACHE_BANK_ID_BITS
            + _DCACHE_BLOCK_OFF_BITS
            + _DCACHE_BYTE_OFF_BITS
        ))
        | (int(set_index) << (
            _DCACHE_BANK_ID_BITS + _DCACHE_BLOCK_OFF_BITS + _DCACHE_BYTE_OFF_BITS
        ))
        | (int(bank_id) << (_DCACHE_BLOCK_OFF_BITS + _DCACHE_BYTE_OFF_BITS))
        | (int(block_offset) << _DCACHE_BYTE_OFF_BITS)
        | int(byte_offset)
    )


def _dcache_slot_address(slot_index: int, *, warp_id: int) -> int:
    """
    Map a per-warp (lane, warp_id) pair onto a unique, bank-balanced dcache address.

    Warp 0 and warp 1 use disjoint set_index ranges so they address different
    cachelines (analogous to the per-warp resident thread-block slot in the SMEM model).
    """
    bank_id = slot_index % _DCACHE_NUM_BANKS
    set_index_offset = (slot_index // _DCACHE_NUM_BANKS) % _DCACHE_NUM_SETS
    set_index = (int(warp_id) * (_DCACHE_NUM_SETS // 2) + set_index_offset) % _DCACHE_NUM_SETS
    tag = (int(warp_id) << 8) | (slot_index // (_DCACHE_NUM_BANKS * _DCACHE_NUM_SETS))
    return _build_dcache_address(
        bank_id=bank_id,
        set_index=set_index,
        tag=tag,
        block_offset=0,
        byte_offset=0,
    )


@dataclass
class _DcacheMemResp:
    """Minimal memory-response adapter that mirrors what the dcache expects."""

    warp_id: int
    packet: Any = None
    status: Any = None


class _DcachePacketWords:
    """Little-endian bytes adapter for cacheline read responses."""

    def __init__(self, words: Sequence[int]) -> None:
        self._bytes = b"".join(
            int(word & 0xFFFF_FFFF).to_bytes(4, byteorder="little", signed=False)
            for word in words
        )

    def tobytes(self) -> bytes:
        return self._bytes


def _preload_dcache_line(
    stage: Any,
    req: Any,
    value: int,
) -> None:
    """Seed a dcache way so ``req`` hits with ``value`` at its block offset."""
    frame = DCACHE["dCacheFrame"](
        valid=True,
        dirty=False,
        tag=req.addr.tag,
        block=[0] * _DCACHE_BLOCK_WORDS,
    )
    frame.block[req.addr.block_offset] = int(value) & 0xFFFF_FFFF
    stage.banks[req.addr.bank_id].sets[req.addr.set_index][0] = frame


def _build_dcache_request_for_txn(
    txn: Transaction,
    *,
    warp_id: int,
) -> Tuple[Any, int]:
    """
    Translate a single SMEM transaction into the equivalent dcache request.

    global ops become read/write MISS-style requests to a fresh tag so the
    dcache has to actually talk to memory (mirroring DRAM latency).
    """
    lane = int(txn.thread_id) - (warp_id * NUM_THREADS_PER_WARP)
    hit_addr = _dcache_slot_address(lane, warp_id=warp_id)
    miss_addr = _dcache_slot_address(
        lane + (NUM_THREADS_PER_WARP * (warp_id + 1) * 2),
        warp_id=warp_id,
    )

    if txn.txn_type is TxnType.SH_LD:
        return (
            DCACHE["dCacheRequest"](
                addr_val=hit_addr,
                rw_mode="read",
                size="word",
            ),
            hit_addr,
        )

    if txn.txn_type is TxnType.SH_ST:
        return (
            DCACHE["dCacheRequest"](
                addr_val=hit_addr,
                rw_mode="write",
                size="word",
                store_value=int(txn.write_data) & 0xFFFF_FFFF,
            ),
            hit_addr,
        )

    if txn.txn_type is TxnType.GLOBAL_LD_DRAM_TO_SRAM:
        return (
            DCACHE["dCacheRequest"](
                addr_val=miss_addr,
                rw_mode="read",
                size="word",
            ),
            miss_addr,
        )

    if txn.txn_type is TxnType.GLOBAL_ST_SMEM_TO_DRAM:
        return (
            DCACHE["dCacheRequest"](
                addr_val=miss_addr,
                rw_mode="write",
                size="word",
                store_value=int(
                    DRAM_SEED_WARP0 if warp_id == 0 else DRAM_SEED_WARP1
                )
                + lane,
            ),
            miss_addr,
        )

    raise ValueError(f"Unsupported txn_type for dcache mapping: {txn.txn_type}")


def _run_dcache_workload(
    warp_programs: Sequence[WarpProgram],
    *,
    dram_latency_cycles: int,
) -> Tuple[int, List[DcachePhaseMetrics]]:
    """
    Drive the equivalent warp-instruction stream through the real dcache stage.

    Each sh.* lane is preloaded as a cache hit; each global.* lane is routed
    to a fresh tag so the dcache incurs a miss + memory round-trip at the
    same 0-cycle synthetic memory budget the SMEM comparison path uses.
    """
    behind = DCACHE["LatchIF"](name="combined_lsu_to_dcache")
    mem_req_if = DCACHE["LatchIF"](name="combined_dcache_to_mem")
    mem_resp_if = DCACHE["LatchIF"](name="combined_mem_to_dcache")
    fwd = DCACHE["ForwardingIF"](name="combined_dcache_to_lsu")

    stage = DCACHE["LockupFreeCacheStage"](
        name="CombinedWarpDCache",
        behind_latch=behind,
        forward_ifs_write={"DCache_LSU_Resp": fwd},
        mem_req_if=mem_req_if,
        mem_resp_if=mem_resp_if,
    )

    mem_image: Dict[int, int] = {}
    pending_responses: List[Tuple[int, _DcacheMemResp]] = []
    metrics: List[DcachePhaseMetrics] = []

    for program in warp_programs:
        for instr in program.instructions:
            reqs: List[Any] = []
            for txn in instr.transactions:
                req, miss_base = _build_dcache_request_for_txn(
                    txn, warp_id=program.warp_id
                )
                if instr.txn_type in (TxnType.SH_LD, TxnType.SH_ST):
                    _preload_dcache_line(
                        stage,
                        req,
                        value=0xC000_0000
                        + (program.warp_id << 20)
                        + int(txn.thread_id),
                    )
                elif instr.txn_type is TxnType.GLOBAL_LD_DRAM_TO_SRAM:
                    mem_image[int(req.addr_val)] = (
                        (DRAM_SEED_WARP0 if program.warp_id == 0 else DRAM_SEED_WARP1)
                        + int(txn.thread_id)
                    )
                reqs.append(req)

            pending_queue = deque(reqs)
            phase_target = len(reqs)
            phase_completions = 0
            hits = 0
            misses = 0
            cycle_started = int(stage.get_cycle_count())
            max_cycles = max(2048, phase_target * 128)
            phase_steps = 0

            while phase_completions < phase_target:
                phase_steps += 1
                if phase_steps > max_cycles:
                    raise TimeoutError(
                        f"DCache instruction {instr.name} did not complete within "
                        f"{max_cycles} cycles."
                    )

                if pending_queue and behind.ready_for_push():
                    behind.push(pending_queue.popleft())

                if pending_responses and mem_resp_if.ready_for_push():
                    ready_cycle, resp = pending_responses[0]
                    if stage.get_cycle_count() >= (ready_cycle - 1):
                        mem_resp_if.push(resp)
                        pending_responses.pop(0)

                with redirect_stdout(io.StringIO()):
                    stage.compute()

                if mem_req_if.valid:
                    payload = mem_req_if.pop()
                    issue_cycle = stage.get_cycle_count()
                    warp_id_field = int(
                        payload.get("warp", payload.get("warp_id", program.warp_id))
                    )
                    rw_mode = str(payload.get("rw_mode", "read")).lower()
                    base_addr = int(payload.get("addr", 0))
                    ready_cycle = int(issue_cycle) + int(dram_latency_cycles)

                    if rw_mode == "read":
                        words = [
                            int(
                                mem_image.get(base_addr + (i * WORD_BYTES), 0)
                            )
                            & 0xFFFF_FFFF
                            for i in range(_DCACHE_BLOCK_WORDS)
                        ]
                        pending_responses.append(
                            (
                                ready_cycle,
                                _DcacheMemResp(
                                    warp_id=warp_id_field,
                                    packet=_DcachePacketWords(words),
                                ),
                            )
                        )
                    else:
                        data_words = payload.get("data", [])
                        if isinstance(data_words, list):
                            for i, word in enumerate(data_words):
                                mem_image[base_addr + (i * WORD_BYTES)] = (
                                    int(word) & 0xFFFF_FFFF
                                )
                        pending_responses.append(
                            (
                                ready_cycle,
                                _DcacheMemResp(
                                    warp_id=warp_id_field,
                                    status="WRITE_DONE",
                                ),
                            )
                        )

                payload = fwd.pop()
                if payload is None:
                    continue

                payload_type = str(getattr(payload, "type", ""))
                if payload_type == "MISS_ACCEPTED":
                    continue
                if payload_type in {"HIT_COMPLETE", "MISS_COMPLETE", "FLUSH_COMPLETE"}:
                    phase_completions += 1
                    if payload_type == "HIT_COMPLETE":
                        hits += 1
                    elif payload_type == "MISS_COMPLETE":
                        misses += 1

            metrics.append(
                DcachePhaseMetrics(
                    warp_id=program.warp_id,
                    instruction_name=instr.name,
                    txn_type=instr.txn_type.value,
                    cycle_started=cycle_started,
                    cycle_completed=int(stage.get_cycle_count()),
                    completed_count=phase_target,
                    hits=hits,
                    misses=misses,
                )
            )

    return int(stage.get_cycle_count()), metrics


def _dump_completions(sim: ShmemFunctionalSimulator) -> None:
    """Mirror the ``Completions:`` + ``Traceback:`` dump used by ``main.py``."""
    print("Completions:")
    for completion in sim.snapshot()["completions"]:
        print(completion)
        print("  Traceback:")
        for trace_line in completion["trace"]:
            print(f"    {trace_line}")


def _print_smem_phase_table(metrics: Sequence[PhaseMetrics]) -> None:
    print()
    print("=" * 100)
    print("SMEM per-instruction cycle / throughput breakdown")
    print("=" * 100)
    header = (
        f"{'Warp':>4} | {'Instruction':<32} | {'TxnType':<22} | "
        f"{'Start':>6} | {'End':>6} | {'dC':>4} | {'SubB':>5} | "
        f"{'Txn/Cyc':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m.warp_id:>4} | {m.instruction_name:<32} | {m.txn_type:<22} | "
            f"{m.cycle_started:>6} | {m.cycle_completed:>6} | "
            f"{m.cycles:>4} | {m.sub_batches:>5} | "
            f"{m.throughput_txns_per_cycle:>8.3f}"
        )


def _print_dcache_phase_table(metrics: Sequence[DcachePhaseMetrics]) -> None:
    print()
    print("=" * 100)
    print("DCache per-instruction cycle / throughput breakdown")
    print("=" * 100)
    header = (
        f"{'Warp':>4} | {'Instruction':<32} | {'TxnType':<22} | "
        f"{'Start':>6} | {'End':>6} | {'dC':>4} | "
        f"{'Hits':>5} | {'Miss':>5} | {'Txn/Cyc':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m.warp_id:>4} | {m.instruction_name:<32} | {m.txn_type:<22} | "
            f"{m.cycle_started:>6} | {m.cycle_completed:>6} | "
            f"{m.cycles:>4} | {m.hits:>5} | {m.misses:>5} | "
            f"{m.throughput_txns_per_cycle:>8.3f}"
        )


def _print_side_by_side_summary(
    smem_cycles: int,
    smem_metrics: Sequence[PhaseMetrics],
    dcache_cycles: int,
    dcache_metrics: Sequence[DcachePhaseMetrics],
) -> None:
    print()
    print("=" * 100)
    print("Side-by-side SMEM vs DCache -- combined warp workload")
    print("=" * 100)
    total_txns = sum(m.completed_count for m in smem_metrics)
    smem_throughput = float(total_txns) / float(max(smem_cycles, 1))
    dcache_throughput = float(total_txns) / float(max(dcache_cycles, 1))
    print(
        f"Total warp-wide instructions : {len(smem_metrics):>4} "
        f"({WARP_COUNT} warps x {len(smem_metrics) // max(WARP_COUNT, 1)} instr/warp)"
    )
    print(f"Total per-thread transactions: {total_txns:>4}")
    print(f"SMEM   total cycles: {smem_cycles:>6}  "
          f"throughput: {smem_throughput:>6.3f} txn/cyc")
    print(f"DCache total cycles: {dcache_cycles:>6}  "
          f"throughput: {dcache_throughput:>6.3f} txn/cyc")

    header = (
        f"{'Warp':>4} | {'Instruction':<32} | {'TxnType':<22} | "
        f"{'SMEM dC':>7} | {'D$ dC':>6} | {'D$ Hits':>7} | {'D$ Miss':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for smem_m, dcache_m in zip(smem_metrics, dcache_metrics):
        print(
            f"{smem_m.warp_id:>4} | {smem_m.instruction_name:<32} | "
            f"{smem_m.txn_type:<22} | "
            f"{smem_m.cycles:>7} | {dcache_m.cycles:>6} | "
            f"{dcache_m.hits:>7} | {dcache_m.misses:>7}"
        )


def demo_combined_warp_workload() -> None:
    """
    End-to-end demo: two full warps issuing four back-to-back instructions each
    (global.ld -> sh.st -> sh.ld -> global.st) across 64 threads.

    Produces:
      * Verbose per-cycle queue dumps from the SMEM simulator (same format as
        the existing demos in ``main.py``).
      * Arbiter sub-batch debug lines per warp-instruction.
      * Final ``Completions:`` + ``Traceback:`` dump for all 256 transactions.
      * Per-instruction cycle/throughput tables for SMEM and DCache.
      * A side-by-side SMEM vs DCache summary.
    """
    print("\n=== TEST: Combined Warp Workload (2 warps x 4 back-to-back instructions) ===")
    print(
        f"Warp 0: threads {0}..{NUM_THREADS_PER_WARP - 1}, "
        f"tbid={WARP0_TBID}, calculated_tbo=0x{_effective_tbo_for_thread_block(WARP0_TBID):04x}"
    )
    print(
        f"Warp 1: threads {NUM_THREADS_PER_WARP}..{TOTAL_THREADS - 1}, "
        f"tbid={WARP1_TBID}, calculated_tbo=0x{_effective_tbo_for_thread_block(WARP1_TBID):04x}"
    )
    print(
        f"Per-warp program: global.ld.dram2sram -> sh.st -> sh.ld -> global.st.smem2dram"
    )
    print(
        f"Comparison DRAM latency: {COMPARISON_DRAM_LATENCY_CYCLES} cycles on both SMEM and DCache."
    )

    warp0, warp1 = _build_both_warp_programs()
    warp_programs = (warp0, warp1)
    dram_init = _build_dram_init(warp_programs)

    smem_cycles, smem_metrics, sim = _run_smem_workload(warp_programs, dram_init)

    _dump_completions(sim)
    _print_smem_phase_table(smem_metrics)

    dcache_cycles, dcache_metrics = _run_dcache_workload(
        warp_programs,
        dram_latency_cycles=COMPARISON_DRAM_LATENCY_CYCLES,
    )

    _print_dcache_phase_table(dcache_metrics)
    _print_side_by_side_summary(
        smem_cycles=smem_cycles,
        smem_metrics=smem_metrics,
        dcache_cycles=dcache_cycles,
        dcache_metrics=dcache_metrics,
    )

    total_txns = sum(m.completed_count for m in smem_metrics)
    print(
        f"\nCompleted in {smem_cycles} SMEM cycles / {dcache_cycles} DCache cycles "
        f"over {total_txns} per-thread transactions "
        f"({len(smem_metrics)} warp-wide instructions)."
    )


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        demo_combined_warp_workload()

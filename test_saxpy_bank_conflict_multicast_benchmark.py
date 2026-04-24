#!/usr/bin/env python3
"""
SAXPY-like bank-conflict benchmark with multicast and non-multicast variants.

This benchmark sweeps conflict depth from 1 to 32 active threads and measures
how many cycles the DCache and SMEM models take to resolve a deliberately
same-bank workload.

For each conflict depth, two SAXPY-like streams are measured:

1. Non-multicast
   Every active thread uses a distinct ``x[i]`` source and a distinct ``y[i]``.

2. Multicast
   Every active thread shares the same read-only ``x`` source address, reuses
   it multiple times per round, and still reads, writes, and reads back a
   distinct ``y[i]``.

The graph focuses on cycle cost, not arithmetic correctness. The access stream
is intentionally kernel-shaped:

    load x
    load y
    store y = a * x + y
    read back y

The conflict-only sweep runs multiple back-to-back same-bank SAXPY rounds so
the comparison reflects a longer stream instead of a single isolated conflict.
The separate full-stream graph keeps a full 32-lane kernel stream active at
every point and changes how many lanes pile onto each bank within the warp, so
the curve reflects true k-way serialization over the same total SAXPY work.

Outputs:
- saxpy_bank_conflict_multicast.csv
- saxpy_bank_conflict_multicast_report.txt
- saxpy_bank_conflict_non_multicast_plot.png
- saxpy_bank_conflict_multicast_plot.png
- saxpy_full_stream_kernel_conflict.csv
- saxpy_full_stream_kernel_conflict_report.txt
- saxpy_full_stream_kernel_conflict_plot.png

Run:
    ./.venv/bin/python test_saxpy_bank_conflict_multicast_benchmark.py
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import redirect_stdout
from dataclasses import dataclass
import csv
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from main import ShmemFunctionalSimulator, Transaction, TxnType, load_smem_config
from test_dcache_and_smem import _load_dcache_symbols


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "saxpy_bank_conflict_multicast.csv"
REPORT_PATH = ROOT / "saxpy_bank_conflict_multicast_report.txt"
NON_MULTICAST_PLOT_PATH = ROOT / "saxpy_bank_conflict_non_multicast_plot.png"
MULTICAST_PLOT_PATH = ROOT / "saxpy_bank_conflict_multicast_plot.png"
FULL_STREAM_CSV_PATH = ROOT / "saxpy_full_stream_kernel_conflict.csv"
FULL_STREAM_REPORT_PATH = ROOT / "saxpy_full_stream_kernel_conflict_report.txt"
FULL_STREAM_PLOT_PATH = ROOT / "saxpy_full_stream_kernel_conflict_plot.png"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/smem-sim-mpl-cache")

DCACHE = _load_dcache_symbols()
from simulator.mem_types import (
    BANK_ID_BIT_LEN as DCACHE_BANK_ID_BIT_LEN,
    BLOCK_OFF_BIT_LEN as DCACHE_BLOCK_OFF_BIT_LEN,
    BLOCK_SIZE_WORDS as DCACHE_BLOCK_SIZE_WORDS,
    BYTE_OFF_BIT_LEN as DCACHE_BYTE_OFF_BIT_LEN,
    NUM_BANKS as DCACHE_NUM_BANKS,
    NUM_SETS_PER_BANK as DCACHE_NUM_SETS,
    SET_INDEX_BIT_LEN as DCACHE_SET_INDEX_BIT_LEN,
)


DCACHE_NUM_BANKS = int(DCACHE_NUM_BANKS)
DCACHE_NUM_SETS = int(DCACHE_NUM_SETS)
DCACHE_BLOCK_WORDS = int(DCACHE_BLOCK_SIZE_WORDS)

_SMEM_CFG = load_smem_config()
SMEM_NUM_BANKS = int(_SMEM_CFG.num_banks)
WORD_BYTES = int(_SMEM_CFG.word_bytes)
MISS_LATENCY_CYCLES = 0
SMEM_ARBITER_ISSUE_WIDTH = int(_SMEM_CFG.arbiter_issue_width)
SMEM_READ_CROSSBAR_PIPELINE = int(_SMEM_CFG.read_crossbar_pipeline_cycles)

SAXPY_A = 3
SHARED_X_SLOT = 0x400
UNIQUE_X_SLOT_BASE = 0x000
Y_SLOT_BASE = 0x800
DRAM_BASE_ADDR = 0x200000
SHARED_X_VALUE = 0x1F00
MAX_CONFLICT_DEPTH = 32
BENCHMARK_NUM_THREADS = MAX_CONFLICT_DEPTH
DCACHE_SET_GROUP_SIZE = max(1, DCACHE_NUM_SETS // 2)
DCACHE_X_SET_BASE = 0
DCACHE_Y_SET_BASE = min(DCACHE_SET_GROUP_SIZE, max(DCACHE_NUM_SETS - 1, 0))
# Reuse the Y working set across rounds so we can stream a longer same-bank
# sequence without turning the dcache side into a preload-way-capacity test.
CONFLICT_STREAM_ROUNDS = 5
MULTICAST_SHARED_X_READS_PER_ROUND = 2
# Long streamed SAXPY benchmark: repeat the memory-only interaction pattern
# many times so each conflict depth is measured over a larger access stream.
FULL_STREAM_INTERACTION_ROUNDS = 100


@dataclass
class ConflictRow:
    conflict_depth: int
    non_multicast_divergent_access_pct: float
    multicast_divergent_access_pct: float
    dcache_non_multicast_cycles: int
    smem_non_multicast_cycles: int
    dcache_multicast_cycles: int
    smem_multicast_cycles: int


@dataclass
class FullStreamRow:
    conflict_depth: int
    dcache_cycles: int
    scratchpad_cycles: int


@dataclass
class DCacheScenario:
    phases: List[List[Any]]
    preload_hits: List[Tuple[Any, int]]
    memory_words: Dict[int, int]


@dataclass
class SmemScenario:
    phases: List[List[Transaction]]
    preload_words: List[Tuple[int, int]]
    dram_init: Dict[int, int]


@dataclass
class _MemResp:
    warp_id: int
    packet: Any = None
    status: Any = None


class _PacketWords:
    def __init__(self, words: Sequence[int]):
        self._bytes = b"".join(
            int(word & 0xFFFF_FFFF).to_bytes(4, byteorder="little", signed=False)
            for word in words
        )

    def tobytes(self) -> bytes:
        return self._bytes


def _build_dcache_addr(
    *,
    bank_id: int,
    set_index: int,
    tag: int = 0,
    block_offset: int = 0,
    byte_offset: int = 0,
) -> int:
    return (
        (int(tag) << (
            int(DCACHE_SET_INDEX_BIT_LEN)
            + int(DCACHE_BANK_ID_BIT_LEN)
            + int(DCACHE_BLOCK_OFF_BIT_LEN)
            + int(DCACHE_BYTE_OFF_BIT_LEN)
        ))
        | (
            int(set_index)
            << (
                int(DCACHE_BANK_ID_BIT_LEN)
                + int(DCACHE_BLOCK_OFF_BIT_LEN)
                + int(DCACHE_BYTE_OFF_BIT_LEN)
            )
        )
        | (
            int(bank_id)
            << (int(DCACHE_BLOCK_OFF_BIT_LEN) + int(DCACHE_BYTE_OFF_BIT_LEN))
        )
        | (int(block_offset) << int(DCACHE_BYTE_OFF_BIT_LEN))
        | int(byte_offset)
    )


def _dcache_same_bank_addr(slot_index: int) -> int:
    return _build_dcache_addr(
        bank_id=0,
        set_index=int(slot_index) % DCACHE_NUM_SETS,
        tag=int(slot_index) // DCACHE_NUM_SETS,
        block_offset=0,
        byte_offset=0,
    )


def _dcache_same_bank_addr_in_set_group(
    slot_index: int,
    *,
    set_base: int,
    set_group_size: int = DCACHE_SET_GROUP_SIZE,
) -> int:
    set_span = max(1, int(set_group_size))
    logical_slot = int(slot_index)
    return _build_dcache_addr(
        bank_id=0,
        set_index=(int(set_base) + (logical_slot % set_span)) % DCACHE_NUM_SETS,
        tag=logical_slot // set_span,
        block_offset=0,
        byte_offset=0,
    )


def _smem_same_bank_addr(slot_index: int) -> int:
    return int(slot_index) * SMEM_NUM_BANKS * WORD_BYTES


def _dram_addr_for_slot(slot_index: int) -> int:
    return int(DRAM_BASE_ADDR + (slot_index * WORD_BYTES))


def _conflict_phase_mix(*, multicast: bool) -> Tuple[int, int]:
    shared_x_phase_count = MULTICAST_SHARED_X_READS_PER_ROUND if multicast else 0
    divergent_phase_count = 3 + (0 if multicast else 1)
    total_phase_count = divergent_phase_count + shared_x_phase_count
    return divergent_phase_count, total_phase_count


def _conflict_divergent_access_pct(
    conflict_depth: int,
    *,
    multicast: bool,
) -> float:
    """
    Panel-normalized divergent-access percentage.

    The conflict-only sweep models a 32-thread warp footprint, but only the
    first ``conflict_depth`` lanes participate in the same-bank access cohort.
    We normalize to the 32-way maximum divergent-access count for the selected
    panel so both multicast and non-multicast traces reach 100% at the right
    edge even though multicast contains shared, non-divergent ``x`` phases.
    """
    divergent_phase_count, _ = _conflict_phase_mix(multicast=multicast)
    max_divergent_lane_accesses = BENCHMARK_NUM_THREADS * divergent_phase_count
    divergent_lane_accesses = int(conflict_depth) * divergent_phase_count
    return 100.0 * divergent_lane_accesses / max(1, max_divergent_lane_accesses)


def _validate_full_stream_smem_conflict_profile(
    conflict_depth: int,
    scenario: SmemScenario,
) -> None:
    """
    Guard against accidentally benchmarking multicast instead of bank conflict.

    For the non-multicast full-stream SAXPY benchmark we expect:

    * a 32-thread simulator configuration, but only ``conflict_depth`` active
      lanes in the conflict cohort,
    * every ``sh.ld`` recipient to have a unique multicast key
      ``(bank, absolute_addr)``, and
    * every active lane in the first streamed batch to hit the same bank.
    """
    cfg = load_smem_config()
    sim = ShmemFunctionalSimulator(
        dram_init=scenario.dram_init,
        num_banks=SMEM_NUM_BANKS,
        word_bytes=WORD_BYTES,
        dram_latency_cycles=MISS_LATENCY_CYCLES,
        arbiter_issue_width=int(cfg.arbiter_issue_width),
        num_threads=BENCHMARK_NUM_THREADS,
        read_crossbar_pipeline_cycles=int(cfg.read_crossbar_pipeline_cycles),
    )

    for phase_name, phase in (("x_read", scenario.phases[2]), ("y_read", scenario.phases[3])):
        if len(phase) != int(conflict_depth):
            raise AssertionError(
                f"{phase_name} phase should contain {conflict_depth} active lanes, "
                f"got {len(phase)}."
            )

        bank_counts: Counter[int] = Counter()
        multicast_keys: Counter[Tuple[int, int]] = Counter()

        for txn in phase:
            bank_counts[sim._bank_for_transaction(txn)] += 1
            key = sim._multicast_read_key(txn)
            if key is None:
                raise AssertionError(f"{phase_name} produced a non-read transaction.")
            multicast_keys[key] += 1

        if any(count != 1 for count in multicast_keys.values()):
            raise AssertionError(
                f"{phase_name} depth {conflict_depth} accidentally collapsed into multicast."
            )

        hottest_bank = max(bank_counts.values(), default=0)
        if hottest_bank != int(conflict_depth):
            raise AssertionError(
                f"{phase_name} depth {conflict_depth} expected hottest bank occupancy "
                f"{conflict_depth}, got {hottest_bank}."
            )
        if len(bank_counts) != 1:
            raise AssertionError(
                f"{phase_name} depth {conflict_depth} should hit exactly one bank, "
                f"got {len(bank_counts)} banks."
            )


def _preload_dcache_hits(stage: Any, preload_hits: Iterable[Tuple[Any, int]]) -> None:
    used_ways = defaultdict(int)

    for req, value in preload_hits:
        key = (req.addr.bank_id, req.addr.set_index)
        way = used_ways[key]
        frame = DCACHE["dCacheFrame"](
            valid=True,
            dirty=False,
            tag=req.addr.tag,
            block=[0] * DCACHE_BLOCK_WORDS,
        )
        frame.block[req.addr.block_offset] = int(value) & 0xFFFF_FFFF
        stage.banks[req.addr.bank_id].sets[req.addr.set_index][way] = frame
        used_ways[key] += 1


def _preload_smem_words(
    sim: ShmemFunctionalSimulator,
    preload_words: Iterable[Tuple[int, int]],
) -> None:
    for shmem_addr, value in preload_words:
        probe = Transaction(txn_type=TxnType.SH_LD, shmem_addr=int(shmem_addr))
        absolute = sim._absolute_smem_addr(probe)
        bank, slot = sim._address_crossbar(
            absolute, sim._effective_thread_block_offset(probe)
        )
        masked = int(value) & sim.word_mask
        sim.banks[bank][slot] = masked
        sim.sram_linear[absolute] = masked


def _run_dcache_scenario(
    scenario: DCacheScenario,
    *,
    mem_latency_cycles: int = MISS_LATENCY_CYCLES,
) -> int:
    behind = DCACHE["LatchIF"](name="saxpy_conflict_lsu_to_dcache")
    mem_req_if = DCACHE["LatchIF"](name="saxpy_conflict_dcache_to_mem")
    mem_resp_if = DCACHE["LatchIF"](name="saxpy_conflict_mem_to_dcache")
    response_if = DCACHE["ForwardingIF"](name="saxpy_conflict_dcache_to_lsu")

    stage = DCACHE["LockupFreeCacheStage"](
        name="SaxpyConflictDCache",
        behind_latch=behind,
        forward_ifs_write={"DCache_LSU_Resp": response_if},
        mem_req_if=mem_req_if,
        mem_resp_if=mem_resp_if,
    )
    _preload_dcache_hits(stage, scenario.preload_hits)

    mem_image = {
        int(addr): int(word) & 0xFFFF_FFFF
        for addr, word in scenario.memory_words.items()
    }
    pending_responses: List[Tuple[int, _MemResp]] = []

    def _stage_quiescent() -> bool:
        if pending_responses:
            return False
        if getattr(mem_req_if, "valid", False) or getattr(mem_resp_if, "valid", False):
            return False
        if getattr(stage, "pending_request", None) is not None:
            return False
        if getattr(stage, "output_buffer", None):
            return False
        if getattr(stage, "active_misses", None):
            return False
        for mshr in getattr(stage, "mshrs", []):
            if not mshr.is_empty():
                return False
        for bank in getattr(stage, "banks", []):
            if getattr(bank, "state", None) == "HALT":
                continue
            if bank.busy or bank.hit_pipeline_busy or bank.waiting_for_mem:
                return False
            if bank.incoming_mem_data is not None:
                return False
            if any(entry is not None for entry in bank.hit_pipeline):
                return False
        return True

    for phase in scenario.phases:
        pending = deque(phase)
        phase_accounted = 0
        phase_target = len(phase)
        max_cycles = max(200, phase_target * 80)
        if any(getattr(req, "halt", False) for req in phase):
            max_cycles = max(max_cycles, len(scenario.memory_words) * 40)
        phase_steps = 0

        while pending or phase_accounted < phase_target or not _stage_quiescent():
            phase_steps += 1
            if phase_steps > max_cycles:
                raise TimeoutError(
                    "DCache phase did not complete within the expected cycle budget."
                )

            if pending and behind.ready_for_push():
                behind.push(pending.popleft())

            if pending_responses and mem_resp_if.ready_for_push():
                ready_cycle, resp = pending_responses[0]
                if stage.get_cycle_count() >= (ready_cycle - 1):
                    mem_resp_if.push(resp)
                    pending_responses.pop(0)

            with redirect_stdout(io.StringIO()):
                stage.compute()

            if mem_req_if.valid:
                req_payload = mem_req_if.pop()
                issue_cycle = stage.get_cycle_count()
                warp_id = int(req_payload.get("warp", req_payload.get("warp_id", 0)))
                rw_mode = str(req_payload.get("rw_mode", "read")).lower()
                base_addr = int(req_payload.get("addr", 0))
                ready_cycle = int(issue_cycle) + int(mem_latency_cycles)

                if rw_mode == "read":
                    words = [
                        int(mem_image.get(base_addr + (i * WORD_BYTES), 0)) & 0xFFFF_FFFF
                        for i in range(DCACHE_BLOCK_WORDS)
                    ]
                    pending_responses.append(
                        (ready_cycle, _MemResp(warp_id=warp_id, packet=_PacketWords(words)))
                    )
                else:
                    data_words = req_payload.get("data", [])
                    if isinstance(data_words, list):
                        for i, word in enumerate(data_words):
                            mem_image[base_addr + (i * WORD_BYTES)] = (
                                int(word) & 0xFFFF_FFFF
                            )
                    pending_responses.append(
                        (ready_cycle, _MemResp(warp_id=warp_id, status="WRITE_DONE"))
                    )

            payload = response_if.pop()
            if payload is None:
                continue

            payload_type = str(getattr(payload, "type", ""))
            if payload_type in {"MISS_ACCEPTED", "HIT_COMPLETE", "FLUSH_COMPLETE"}:
                phase_accounted += 1

    return int(stage.get_cycle_count())


def _run_smem_scenario(
    scenario: SmemScenario,
    *,
    num_threads: int,
    dram_latency_cycles: int = MISS_LATENCY_CYCLES,
) -> int:
    sim = ShmemFunctionalSimulator(
        dram_init=scenario.dram_init,
        num_banks=SMEM_NUM_BANKS,
        word_bytes=WORD_BYTES,
        dram_latency_cycles=int(dram_latency_cycles),
        arbiter_issue_width=SMEM_ARBITER_ISSUE_WIDTH,
        num_threads=int(num_threads),
        read_crossbar_pipeline_cycles=SMEM_READ_CROSSBAR_PIPELINE,
    )
    _preload_smem_words(sim, scenario.preload_words)

    completion_cursor = 0
    for phase in scenario.phases:
        for txn in phase:
            sim.issue(txn)

        phase_target = len(phase)
        max_cycles = max(200, phase_target * 80)
        phase_steps = 0

        while (len(sim.completions) - completion_cursor) < phase_target:
            phase_steps += 1
            if phase_steps > max_cycles:
                raise TimeoutError(
                    "SMEM phase did not complete within the expected cycle budget."
                )
            sim.step()

        completion_cursor += phase_target

    return int(sim.get_cycle_count())


def _build_dcache_saxpy_conflict(
    conflict_depth: int,
    *,
    multicast: bool,
) -> DCacheScenario:
    phases: List[List[Any]] = []
    preload_hits: List[Tuple[Any, int]] = []
    memory_words: Dict[int, int] = {}

    for tid in range(int(conflict_depth)):
        y_addr = _dcache_same_bank_addr_in_set_group(
            Y_SLOT_BASE + tid,
            set_base=DCACHE_Y_SET_BASE,
        )
        y_val = 0x0100 + tid
        preload_hits.append(
            (
                DCACHE["dCacheRequest"](addr_val=y_addr, rw_mode="read", size="word"),
                y_val,
            )
        )

    for round_idx in range(CONFLICT_STREAM_ROUNDS):
        x_phase_count = MULTICAST_SHARED_X_READS_PER_ROUND if multicast else 1
        x_phases: List[List[Any]] = [[] for _ in range(x_phase_count)]
        y_read_phase: List[Any] = []
        y_write_phase: List[Any] = []
        y_readback_phase: List[Any] = []

        shared_x_addr = _dcache_same_bank_addr_in_set_group(
            SHARED_X_SLOT + round_idx,
            set_base=DCACHE_X_SET_BASE,
        )
        memory_words[shared_x_addr] = SHARED_X_VALUE

        for tid in range(int(conflict_depth)):
            stream_slot = (round_idx * MAX_CONFLICT_DEPTH) + tid
            x_addr = (
                shared_x_addr
                if multicast
                else _dcache_same_bank_addr_in_set_group(
                    UNIQUE_X_SLOT_BASE + stream_slot,
                    set_base=DCACHE_X_SET_BASE,
                )
            )
            y_addr = _dcache_same_bank_addr_in_set_group(
                Y_SLOT_BASE + tid,
                set_base=DCACHE_Y_SET_BASE,
            )

            x_val = SHARED_X_VALUE if multicast else (tid + 1 + (round_idx * 0x40))
            y_seed = 0x0100 + tid
            result = (SAXPY_A * x_val) + y_seed

            y_read_req = DCACHE["dCacheRequest"](addr_val=y_addr, rw_mode="read", size="word")
            y_write_req = DCACHE["dCacheRequest"](
                addr_val=y_addr,
                rw_mode="write",
                size="word",
                store_value=result,
            )
            y_readback_req = DCACHE["dCacheRequest"](addr_val=y_addr, rw_mode="read", size="word")

            for x_phase in x_phases:
                x_phase.append(
                    DCACHE["dCacheRequest"](addr_val=x_addr, rw_mode="read", size="word")
                )
            y_read_phase.append(y_read_req)
            y_write_phase.append(y_write_req)
            y_readback_phase.append(y_readback_req)

            if not multicast:
                memory_words[x_addr] = x_val

        phases.extend([*x_phases, y_read_phase, y_write_phase, y_readback_phase])

    return DCacheScenario(
        phases=phases,
        preload_hits=preload_hits,
        memory_words=memory_words,
    )


def _build_smem_saxpy_conflict(
    conflict_depth: int,
    *,
    multicast: bool,
) -> SmemScenario:
    phases: List[List[Transaction]] = []
    preload_words: List[Tuple[int, int]] = []
    dram_init: Dict[int, int] = {}

    for tid in range(int(conflict_depth)):
        preload_words.append((_smem_same_bank_addr(Y_SLOT_BASE + tid), 0x0100 + tid))

    for round_idx in range(CONFLICT_STREAM_ROUNDS):
        global_ld_phase: List[Transaction] = []
        x_phase_count = MULTICAST_SHARED_X_READS_PER_ROUND if multicast else 1
        x_read_phases: List[List[Transaction]] = [[] for _ in range(x_phase_count)]
        y_read_phase: List[Transaction] = []
        y_write_phase: List[Transaction] = []
        y_readback_phase: List[Transaction] = []

        shared_x_shmem_addr = _smem_same_bank_addr(SHARED_X_SLOT + round_idx)
        shared_x_dram_addr = _dram_addr_for_slot(SHARED_X_SLOT + round_idx)
        dram_init[shared_x_dram_addr] = SHARED_X_VALUE

        for tid in range(int(conflict_depth)):
            stream_slot = (round_idx * MAX_CONFLICT_DEPTH) + tid
            x_shmem_addr = (
                shared_x_shmem_addr
                if multicast
                else _smem_same_bank_addr(UNIQUE_X_SLOT_BASE + stream_slot)
            )
            x_dram_addr = (
                shared_x_dram_addr
                if multicast
                else _dram_addr_for_slot(UNIQUE_X_SLOT_BASE + stream_slot)
            )
            y_shmem_addr = _smem_same_bank_addr(Y_SLOT_BASE + tid)

            x_val = SHARED_X_VALUE if multicast else (tid + 1 + (round_idx * 0x40))
            y_seed = 0x0100 + tid
            result = (SAXPY_A * x_val) + y_seed

            global_ld_phase.append(
                Transaction(
                    txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=x_dram_addr,
                    shmem_addr=x_shmem_addr,
                    thread_id=tid,
                )
            )
            for x_read_phase in x_read_phases:
                x_read_phase.append(
                    Transaction(
                        txn_type=TxnType.SH_LD,
                        shmem_addr=x_shmem_addr,
                        thread_id=tid,
                    )
                )
            y_read_phase.append(
                Transaction(
                    txn_type=TxnType.SH_LD,
                    shmem_addr=y_shmem_addr,
                    thread_id=tid,
                )
            )
            y_write_phase.append(
                Transaction(
                    txn_type=TxnType.SH_ST,
                    shmem_addr=y_shmem_addr,
                    write_data=result,
                    thread_id=tid,
                )
            )
            y_readback_phase.append(
                Transaction(
                    txn_type=TxnType.SH_LD,
                    shmem_addr=y_shmem_addr,
                    thread_id=tid,
                )
            )

            if not multicast:
                dram_init[x_dram_addr] = x_val

        phases.extend(
            [
                global_ld_phase,
                *x_read_phases,
                y_read_phase,
                y_write_phase,
                y_readback_phase,
            ]
        )

    return SmemScenario(
        phases=phases,
        preload_words=preload_words,
        dram_init=dram_init,
    )


def _build_dcache_full_stream_kernel(conflict_depth: int) -> DCacheScenario:
    """
    Model the full SAXPY kernel memory stream on the DCache path:

        y[i] = a * x[i] + y[i]

    as:
        load x, load y, store y, flush dirty output lines.
    """
    phases: List[List[Any]] = []
    memory_words: Dict[int, int] = {}

    for round_idx in range(FULL_STREAM_INTERACTION_ROUNDS):
        x_phase: List[Any] = []
        y_phase: List[Any] = []
        y_store_phase: List[Any] = []
        active_lanes = int(conflict_depth)

        for lane in range(active_lanes):
            stream_slot = (round_idx * MAX_CONFLICT_DEPTH) + lane
            x_addr = _dcache_same_bank_addr_in_set_group(
                UNIQUE_X_SLOT_BASE + stream_slot,
                set_base=DCACHE_X_SET_BASE,
                set_group_size=DCACHE_NUM_SETS,
            )
            y_addr = _dcache_same_bank_addr_in_set_group(
                Y_SLOT_BASE + stream_slot,
                set_base=DCACHE_Y_SET_BASE,
                set_group_size=DCACHE_NUM_SETS,
            )

            x_val = lane + 1 + (round_idx * 0x40)
            y_val = 0x0100 + lane + (round_idx * 0x100)
            result = (SAXPY_A * x_val) + y_val

            x_phase.append(
                DCACHE["dCacheRequest"](addr_val=x_addr, rw_mode="read", size="word")
            )
            y_phase.append(
                DCACHE["dCacheRequest"](addr_val=y_addr, rw_mode="read", size="word")
            )
            y_store_phase.append(
                DCACHE["dCacheRequest"](
                    addr_val=y_addr,
                    rw_mode="write",
                    size="word",
                    store_value=result,
                )
            )

            memory_words[x_addr] = x_val
            memory_words[y_addr] = y_val

        phases.extend([x_phase, y_phase, y_store_phase])

    phases.append(
        [DCACHE["dCacheRequest"](addr_val=0, rw_mode="read", size="word", halt=True)]
    )

    return DCacheScenario(phases=phases, preload_hits=[], memory_words=memory_words)


def _build_smem_full_stream_kernel(conflict_depth: int) -> SmemScenario:
    """
    Model the full SAXPY kernel memory stream on the Scratchpad path:

        load x -> scratchpad
        load y -> scratchpad
        read x from scratchpad
        read y from scratchpad
        update y in scratchpad
        store y back to DRAM
    """
    phases: List[List[Transaction]] = []
    dram_init: Dict[int, int] = {}

    for round_idx in range(FULL_STREAM_INTERACTION_ROUNDS):
        x_load_phase: List[Transaction] = []
        y_load_phase: List[Transaction] = []
        x_read_phase: List[Transaction] = []
        y_read_phase: List[Transaction] = []
        y_write_phase: List[Transaction] = []
        y_store_phase: List[Transaction] = []
        active_lanes = int(conflict_depth)

        for lane in range(active_lanes):
            stream_slot = (round_idx * MAX_CONFLICT_DEPTH) + lane
            x_shmem_addr = _smem_same_bank_addr(UNIQUE_X_SLOT_BASE + stream_slot)
            y_shmem_addr = _smem_same_bank_addr(Y_SLOT_BASE + stream_slot)
            x_dram_addr = _dram_addr_for_slot(UNIQUE_X_SLOT_BASE + stream_slot)
            y_dram_addr = _dram_addr_for_slot(Y_SLOT_BASE + stream_slot)

            x_val = lane + 1 + (round_idx * 0x40)
            y_val = 0x0100 + lane + (round_idx * 0x100)
            result = (SAXPY_A * x_val) + y_val

            x_load_phase.append(
                Transaction(
                    txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=x_dram_addr,
                    shmem_addr=x_shmem_addr,
                    thread_id=lane,
                )
            )
            y_load_phase.append(
                Transaction(
                    txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=y_dram_addr,
                    shmem_addr=y_shmem_addr,
                    thread_id=lane,
                )
            )
            x_read_phase.append(
                Transaction(
                    txn_type=TxnType.SH_LD,
                    shmem_addr=x_shmem_addr,
                    thread_id=lane,
                )
            )
            y_read_phase.append(
                Transaction(
                    txn_type=TxnType.SH_LD,
                    shmem_addr=y_shmem_addr,
                    thread_id=lane,
                )
            )
            y_write_phase.append(
                Transaction(
                    txn_type=TxnType.SH_ST,
                    shmem_addr=y_shmem_addr,
                    write_data=result,
                    thread_id=lane,
                )
            )
            y_store_phase.append(
                Transaction(
                    txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                    dram_addr=y_dram_addr,
                    shmem_addr=y_shmem_addr,
                    thread_id=lane,
                )
            )

            dram_init[x_dram_addr] = x_val
            dram_init[y_dram_addr] = y_val

        phases.extend(
            [
                x_load_phase,
                y_load_phase,
                x_read_phase,
                y_read_phase,
                y_write_phase,
                y_store_phase,
            ]
        )

    return SmemScenario(phases=phases, preload_words=[], dram_init=dram_init)


def run_conflict_benchmark(max_conflict_depth: int = 32) -> List[ConflictRow]:
    rows: List[ConflictRow] = []

    for conflict_depth in range(1, int(max_conflict_depth) + 1):
        dcache_non = _run_dcache_scenario(
            _build_dcache_saxpy_conflict(conflict_depth, multicast=False),
            mem_latency_cycles=MISS_LATENCY_CYCLES,
        )
        smem_non = _run_smem_scenario(
            _build_smem_saxpy_conflict(conflict_depth, multicast=False),
            num_threads=BENCHMARK_NUM_THREADS,
            dram_latency_cycles=MISS_LATENCY_CYCLES,
        )
        dcache_multi = _run_dcache_scenario(
            _build_dcache_saxpy_conflict(conflict_depth, multicast=True),
            mem_latency_cycles=MISS_LATENCY_CYCLES,
        )
        smem_multi = _run_smem_scenario(
            _build_smem_saxpy_conflict(conflict_depth, multicast=True),
            num_threads=BENCHMARK_NUM_THREADS,
            dram_latency_cycles=MISS_LATENCY_CYCLES,
        )

        rows.append(
            ConflictRow(
                conflict_depth=conflict_depth,
                non_multicast_divergent_access_pct=_conflict_divergent_access_pct(
                    conflict_depth,
                    multicast=False,
                ),
                multicast_divergent_access_pct=_conflict_divergent_access_pct(
                    conflict_depth,
                    multicast=True,
                ),
                dcache_non_multicast_cycles=dcache_non,
                smem_non_multicast_cycles=smem_non,
                dcache_multicast_cycles=dcache_multi,
                smem_multicast_cycles=smem_multi,
            )
        )

    return rows


def run_full_stream_kernel_benchmark(max_conflict_depth: int = 32) -> List[FullStreamRow]:
    rows: List[FullStreamRow] = []

    for conflict_depth in range(1, int(max_conflict_depth) + 1):
        scenario = _build_smem_full_stream_kernel(conflict_depth)
        _validate_full_stream_smem_conflict_profile(conflict_depth, scenario)
        dcache_cycles = _run_dcache_scenario(
            _build_dcache_full_stream_kernel(conflict_depth),
            mem_latency_cycles=MISS_LATENCY_CYCLES,
        )
        scratchpad_cycles = _run_smem_scenario(
            scenario,
            num_threads=BENCHMARK_NUM_THREADS,
            dram_latency_cycles=MISS_LATENCY_CYCLES,
        )
        rows.append(
            FullStreamRow(
                conflict_depth=conflict_depth,
                dcache_cycles=dcache_cycles,
                scratchpad_cycles=scratchpad_cycles,
            )
        )

    return rows


def _write_csv(rows: Iterable[ConflictRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "conflict_depth",
                "non_multicast_divergent_access_pct",
                "multicast_divergent_access_pct",
                "dcache_non_multicast_cycles",
                "smem_non_multicast_cycles",
                "dcache_multicast_cycles",
                "smem_multicast_cycles",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.conflict_depth,
                    f"{row.non_multicast_divergent_access_pct:.6f}",
                    f"{row.multicast_divergent_access_pct:.6f}",
                    row.dcache_non_multicast_cycles,
                    row.smem_non_multicast_cycles,
                    row.dcache_multicast_cycles,
                    row.smem_multicast_cycles,
                ]
            )


def _write_full_stream_csv(rows: Iterable[FullStreamRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "conflict_depth",
                "dcache_cycles",
                "scratchpad_cycles",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.conflict_depth,
                    row.dcache_cycles,
                    row.scratchpad_cycles,
                ]
            )


def _plot_rows(rows: List[ConflictRow]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the benchmark plot. "
            "Install it in the active Python environment and rerun the script."
        ) from exc

    palette = {
        "dcache": "#c05621",
        "smem": "#1f6f8b",
        "grid": "#d8d2c8",
        "text": "#2f2a24",
    }

    def _series_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
        if len(xs) < 2 or len(ys) < 2:
            return 0.0
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    panels = [
        (
            NON_MULTICAST_PLOT_PATH,
            "Non-Multicast",
            [row.non_multicast_divergent_access_pct for row in rows],
            [row.dcache_non_multicast_cycles for row in rows],
            [row.smem_non_multicast_cycles for row in rows],
        ),
        (
            MULTICAST_PLOT_PATH,
            "Multicast",
            [row.multicast_divergent_access_pct for row in rows],
            [row.dcache_multicast_cycles for row in rows],
            [row.smem_multicast_cycles for row in rows],
        ),
    ]

    for output_path, title, divergent_pcts, dcache_cycles, smem_cycles in panels:
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.2))
        fig.patch.set_facecolor("#f6f3ee")
        ax.set_facecolor("#fffdf9")
        ax.plot(
            divergent_pcts,
            dcache_cycles,
            color=palette["dcache"],
            linewidth=2.8,
            marker="o",
            markersize=5.2,
            label="DCache",
        )
        ax.plot(
            divergent_pcts,
            smem_cycles,
            color=palette["smem"],
            linewidth=2.8,
            marker="s",
            markersize=5.0,
            label="Scratchpad",
        )
        ax.fill_between(divergent_pcts, dcache_cycles, color=palette["dcache"], alpha=0.08)
        ax.fill_between(divergent_pcts, smem_cycles, color=palette["smem"], alpha=0.08)
        scratchpad_slope = _series_slope(divergent_pcts, smem_cycles)
        if divergent_pcts and smem_cycles:
            ax.annotate(
                f"Scratchpad slope: {scratchpad_slope:+.2f} cyc/%",
                xy=(divergent_pcts[-1], smem_cycles[-1]),
                xytext=(-14, 16),
                textcoords="offset points",
                ha="right",
                color=palette["smem"],
                fontsize=9.6,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "#eef7fa",
                    "edgecolor": "#9cc7d4",
                    "linewidth": 0.8,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": palette["smem"],
                    "linewidth": 0.9,
                },
            )
        ax.set_title(title, fontsize=13, weight="bold", color=palette["text"], pad=8)
        ax.grid(True, axis="y", alpha=0.55, color=palette["grid"], linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.18, color=palette["grid"], linewidth=0.6)
        max_pct = max(divergent_pcts, default=0.0)
        tick_step = 10 if max_pct > 40.0 else 5
        tick_limit = tick_step * max(1, int((max_pct + tick_step - 1e-9) // tick_step))
        ax.set_xlim(0, float(tick_limit))
        ax.set_xticks(list(range(0, tick_limit + tick_step, tick_step)))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8e8579")
        ax.spines["bottom"].set_color("#8e8579")
        ax.tick_params(colors=palette["text"])
        ax.legend(loc="upper left", frameon=False)
        ax.set_xlabel(
            "Normalized Divergent Access Share (% of Panel Max)",
            fontsize=10.5,
            color=palette["text"],
        )
        ax.set_ylabel("Cycles to Resolve Stream", fontsize=11, color=palette["text"])
        fig.suptitle(
            f"{title} SAXPY Same-Bank Conflict Sweep",
            fontsize=17,
            weight="bold",
            color=palette["text"],
            y=0.97,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def _plot_full_stream_rows(rows: List[FullStreamRow], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the benchmark plot. "
            "Install it in the active Python environment and rerun the script."
        ) from exc

    depths = [row.conflict_depth for row in rows]

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 6.4))
    fig.patch.set_facecolor("#f6f3ee")
    ax.set_facecolor("#fffdf9")

    palette = {
        "dcache": "#c05621",
        "scratchpad": "#1f6f8b",
        "grid": "#d8d2c8",
        "text": "#2f2a24",
    }

    dcache_cycles = [row.dcache_cycles for row in rows]
    scratchpad_cycles = [row.scratchpad_cycles for row in rows]

    ax.plot(
        depths,
        dcache_cycles,
        color=palette["dcache"],
        linewidth=2.9,
        marker="o",
        markersize=5.4,
        label="DCache",
    )
    ax.plot(
        depths,
        scratchpad_cycles,
        color=palette["scratchpad"],
        linewidth=2.9,
        marker="s",
        markersize=5.1,
        label="Scratchpad",
    )
    ax.fill_between(depths, dcache_cycles, color=palette["dcache"], alpha=0.08)
    ax.fill_between(depths, scratchpad_cycles, color=palette["scratchpad"], alpha=0.08)
    ax.set_title(
        "Full Streamed SAXPY Kernel",
        fontsize=13,
        weight="bold",
        color=palette["text"],
        pad=8,
    )
    ax.set_xlim(1, 32)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.set_xlabel("Conflict Depth (1-32 active lanes on one bank)", fontsize=10.5, color=palette["text"])
    ax.set_ylabel("Cycles to Resolve Stream", fontsize=11, color=palette["text"])
    ax.grid(True, axis="y", alpha=0.55, color=palette["grid"], linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.18, color=palette["grid"], linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8e8579")
    ax.spines["bottom"].set_color("#8e8579")
    ax.tick_params(colors=palette["text"])
    ax.legend(loc="upper left", frameon=False)

    fig.suptitle(
        "Full SAXPY Kernel Streamed In And Out",
        fontsize=17,
        weight="bold",
        color=palette["text"],
        y=0.95,
    )
    fig.text(
        0.5,
        0.03,
        f"Each point runs {FULL_STREAM_INTERACTION_ROUNDS} streamed SAXPY dummy memory interactions with the "
        f"simulator configured for {BENCHMARK_NUM_THREADS} threads; only the first k lanes "
        "issue the same-bank conflict cohort. The stream is load x, load y, "
        "compute y = a*x + y, then output y. "
        "Scratchpad streams x and y in from DRAM and streams y back out; "
        "DCache performs global x/y reads, y writes, then flushes dirty output lines.",
        ha="center",
        fontsize=9.2,
        color="#5c5348",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_table(rows: Iterable[ConflictRow]) -> str:
    lines = [
        f"{'Dep':>3} | {'Non Div%':>8} | {'D$ Non':>7} | {'Scratch Non':>11} | {'Multi Div%':>10} | {'D$ Multi':>8} | {'Scratch Multi':>13}"
    ]
    lines.append("-" * 83)
    for row in rows:
        lines.append(
            f"{row.conflict_depth:>3} | "
            f"{row.non_multicast_divergent_access_pct:>8.2f} | "
            f"{row.dcache_non_multicast_cycles:>7} | "
            f"{row.smem_non_multicast_cycles:>11} | "
            f"{row.multicast_divergent_access_pct:>10.2f} | "
            f"{row.dcache_multicast_cycles:>8} | "
            f"{row.smem_multicast_cycles:>13}"
        )
    return "\n".join(lines)


def _build_full_stream_table(rows: Iterable[FullStreamRow]) -> str:
    lines = [
        f"{'Dep':>3} | {'D$ Full':>8} | {'Scratchpad Full':>16}"
    ]
    lines.append("-" * 34)
    for row in rows:
        lines.append(
            f"{row.conflict_depth:>3} | "
            f"{row.dcache_cycles:>8} | "
            f"{row.scratchpad_cycles:>16}"
        )
    return "\n".join(lines)


def _write_report(rows: List[ConflictRow], path: Path) -> None:
    last = rows[-1]
    sections = [
        "SAXPY-Like Bank-Conflict Multicast Benchmark",
        "===========================================",
        "",
        "Configuration",
        "-------------",
        "- Conflict depth swept from 1 to 32 active threads",
        "- Every run forces the active threads onto the same bank",
        f"- Each depth runs {CONFLICT_STREAM_ROUNDS} back-to-back SAXPY conflict rounds",
        f"- Plot x-axis is divergent-access share normalized to each panel's {BENCHMARK_NUM_THREADS}-lane, 32-way maximum",
        f"- DCache/SMEM comparison latency: {MISS_LATENCY_CYCLES} cycles",
        f"- SMEM read crossbar latency: {SMEM_READ_CROSSBAR_PIPELINE} cycles",
        f"- DCache banks: {DCACHE_NUM_BANKS}",
        f"- SMEM banks: {SMEM_NUM_BANKS}",
        "",
        "Case Definitions",
        "----------------",
        "- Non-multicast: each active thread loads a distinct x[i] and updates a distinct y[i].",
        f"- Multicast: all active threads share the same read-only x address across {MULTICAST_SHARED_X_READS_PER_ROUND} repeated x-read phases while still updating distinct y[i].",
        "- Divergent-access percentage counts lane-addressed operations that are not multicast, then normalizes to each panel's own 32-way maximum so both traces end at 100%.",
        f"- Stream shape per round: x-read phase(s), load y, store y = a*x + y, read back y; repeated {CONFLICT_STREAM_ROUNDS} times.",
        "",
        "32-way Summary",
        "--------------",
        f"- Non-multicast x-axis endpoint: {last.non_multicast_divergent_access_pct:.2f}% divergent accesses",
        f"- Multicast x-axis endpoint: {last.multicast_divergent_access_pct:.2f}% divergent accesses",
        f"- Non-multicast: DCache = {last.dcache_non_multicast_cycles} cycles, SMEM = {last.smem_non_multicast_cycles} cycles",
        f"- Multicast: DCache = {last.dcache_multicast_cycles} cycles, SMEM = {last.smem_multicast_cycles} cycles",
        "",
        _build_table(rows),
        "",
        f"CSV: {CSV_PATH.name}",
        f"Report: {REPORT_PATH.name}",
        f"Plot (Non-Multicast): {NON_MULTICAST_PLOT_PATH.name}",
        f"Plot (Multicast): {MULTICAST_PLOT_PATH.name}",
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def _write_full_stream_report(rows: List[FullStreamRow], path: Path) -> None:
    last = rows[-1]
    sections = [
        "Full Streamed SAXPY Kernel Bank-Conflict Benchmark",
        "==================================================",
        "",
        "Kernel Shape",
        "------------",
        "if (i < n) y[i] = a * x[i] + y[i];",
        "",
        "Configuration",
        "-------------",
        "- Conflict depth swept from 1-way to 32-way bank contention",
        f"- Simulator thread count is fixed at {BENCHMARK_NUM_THREADS} threads for every point",
        f"- Each point runs {FULL_STREAM_INTERACTION_ROUNDS} streamed SAXPY dummy memory interactions",
        "- Conflict depth k means the first k lanes form the active same-bank conflict cohort in every streamed interaction",
        f"- DCache/Scratchpad comparison latency: {MISS_LATENCY_CYCLES} cycles",
        f"- Scratchpad read crossbar latency: {SMEM_READ_CROSSBAR_PIPELINE} cycles",
        "",
        "Modeled Memory Stream",
        "---------------------",
        "- DCache: global load x, global load y, store y, then flush dirty output lines.",
        "- Scratchpad: global load x->scratchpad, global load y->scratchpad, sh.ld x, sh.ld y, sh.st y, global.st y->dram.",
        "- Only the active conflict cohort changes across the sweep; the 32-thread simulator configuration stays fixed.",
        "- Validation guard: every non-multicast sh.ld in the first warp has a unique (bank, absolute_addr) key, so these cases are real bank conflicts rather than multicast merges.",
        "- Queue cadence model: the Scratchpad read queue dequeues one request per cycle into a 3-stage pipelined Clos return path, so at most 3 read responses are overlaid at once; the write side also dequeues one request per cycle.",
        "- Therefore increasing k increases the number of serialized logical requests driven through the one-per-cycle queues across the 100-interaction stream.",
        "",
        "32-way Summary",
        "--------------",
        f"- Full streamed kernel: DCache = {last.dcache_cycles} cycles, Scratchpad = {last.scratchpad_cycles} cycles",
        "",
        _build_full_stream_table(rows),
        "",
        f"CSV: {FULL_STREAM_CSV_PATH.name}",
        f"Report: {FULL_STREAM_REPORT_PATH.name}",
        f"Plot: {FULL_STREAM_PLOT_PATH.name}",
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def _print_summary(rows: List[ConflictRow]) -> None:
    print("SAXPY-Like Same-Bank Conflict Sweep by Divergent Access Share")
    print("-------------------------------------------------------------")
    print(_build_table(rows))
    print()
    print(f"CSV saved to:   {CSV_PATH}")
    print(f"Report saved to:{REPORT_PATH}")
    print(f"Plot saved to:  {NON_MULTICAST_PLOT_PATH}")
    print(f"Plot saved to:  {MULTICAST_PLOT_PATH}")


def _print_full_stream_summary(rows: List[FullStreamRow]) -> None:
    print("Full Streamed SAXPY Kernel")
    print("-------------------------")
    print(_build_full_stream_table(rows))
    print()
    print(f"CSV saved to:   {FULL_STREAM_CSV_PATH}")
    print(f"Report saved to:{FULL_STREAM_REPORT_PATH}")
    print(f"Plot saved to:  {FULL_STREAM_PLOT_PATH}")


def main() -> None:
    rows = run_conflict_benchmark(max_conflict_depth=32)
    full_stream_rows = run_full_stream_kernel_benchmark(max_conflict_depth=32)
    _write_csv(rows, CSV_PATH)
    _write_report(rows, REPORT_PATH)
    _plot_rows(rows)
    _write_full_stream_csv(full_stream_rows, FULL_STREAM_CSV_PATH)
    _write_full_stream_report(full_stream_rows, FULL_STREAM_REPORT_PATH)
    _plot_full_stream_rows(full_stream_rows, FULL_STREAM_PLOT_PATH)
    _print_summary(rows)
    print()
    _print_full_stream_summary(full_stream_rows)


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        main()

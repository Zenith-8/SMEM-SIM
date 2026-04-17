#!/usr/bin/env python3
"""
Thread-scaling benchmark for:
- simulator/mem/dcache.py
- main.py shared-memory simulator

This benchmark reports hit/miss workload families across 1-32 threads:
- mixed hot/cold accesses with both hits and misses
- SAXPY-like streaming access pattern

Each workload is measured twice:
- bank-balanced: requests are spread across banks as evenly as possible
- same-bank-conflict: requests are forced onto one bank

Outputs:
- thread_scaling_benchmark.csv
- results.txt
- results.jpg

Run:
    python3 test_thread_scaling_benchmark.py
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import redirect_stdout
from dataclasses import dataclass
import csv
import io
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from main import ShmemFunctionalSimulator, Transaction, TxnType, load_smem_config
from test_dcache_and_smem import _load_dcache_symbols


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "thread_scaling_benchmark.csv"
REPORT_PATH = ROOT / "results.txt"
PLOT_PATH = ROOT / "results.jpg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/smem-sim-mpl-cache")

DCACHE = _load_dcache_symbols()
from simulator.mem_types import (
    BLOCK_SIZE_WORDS as DCACHE_BLOCK_SIZE_WORDS,
    BANK_ID_BIT_LEN as DCACHE_BANK_ID_BIT_LEN,
    BLOCK_OFF_BIT_LEN as DCACHE_BLOCK_OFF_BIT_LEN,
    BYTE_OFF_BIT_LEN as DCACHE_BYTE_OFF_BIT_LEN,
    NUM_BANKS as DCACHE_NUM_BANKS,
    NUM_SETS_PER_BANK as DCACHE_NUM_SETS,
    NUM_WAYS as DCACHE_NUM_WAYS,
    SET_INDEX_BIT_LEN as DCACHE_SET_INDEX_BIT_LEN,
)

DCACHE_NUM_BANKS = int(DCACHE_NUM_BANKS)
DCACHE_NUM_SETS = int(DCACHE_NUM_SETS)
DCACHE_NUM_WAYS = int(DCACHE_NUM_WAYS)
DCACHE_BLOCK_WORDS = int(DCACHE_BLOCK_SIZE_WORDS)

_SMEM_CFG = load_smem_config()
SMEM_NUM_BANKS = int(_SMEM_CFG.num_banks)
WORD_BYTES = int(_SMEM_CFG.word_bytes)
MISS_LATENCY_CYCLES = int(_SMEM_CFG.dram_latency_cycles)
SMEM_ARBITER_ISSUE_WIDTH = int(_SMEM_CFG.arbiter_issue_width)

SAXPY_A = 3
DRAM_BASE_ADDR = 0x100000


@dataclass
class BenchmarkRow:
    workload: str
    threads: int
    dcache_balanced_cycles: int
    smem_balanced_cycles: int
    dcache_conflict_cycles: int
    smem_conflict_cycles: int


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


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    title: str
    description: str
    dcache_builder: Callable[[int, bool], DCacheScenario]
    smem_builder: Callable[[int, bool], SmemScenario]


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


def _dcache_addr_for_slot(slot_index: int, same_bank_conflict: bool) -> int:
    if same_bank_conflict:
        bank_id = 0
        set_index = slot_index % DCACHE_NUM_SETS
        tag = slot_index // DCACHE_NUM_SETS
    else:
        bank_id = slot_index % DCACHE_NUM_BANKS
        set_index = (slot_index // DCACHE_NUM_BANKS) % DCACHE_NUM_SETS
        tag = slot_index // (DCACHE_NUM_BANKS * DCACHE_NUM_SETS)

    return _build_dcache_addr(
        bank_id=bank_id,
        set_index=set_index,
        tag=tag,
        block_offset=0,
        byte_offset=0,
    )


def _smem_addr_for_slot(slot_index: int, same_bank_conflict: bool) -> int:
    if same_bank_conflict:
        return int(slot_index) * SMEM_NUM_BANKS * WORD_BYTES
    return int(slot_index) * WORD_BYTES


def _dram_addr_for_slot(slot_index: int) -> int:
    return int(DRAM_BASE_ADDR + (slot_index * WORD_BYTES))


def _preload_dcache_hits(stage: Any, preload_hits: Iterable[Tuple[Any, int]]) -> None:
    used_ways = defaultdict(int)

    for req, value in preload_hits:
        key = (req.addr.bank_id, req.addr.set_index)
        way = used_ways[key]
        if way >= DCACHE_NUM_WAYS:
            raise ValueError(
                "Benchmark preload pattern overflowed the available dcache ways."
            )

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
    sim: ShmemFunctionalSimulator, preload_words: Iterable[Tuple[int, int]]
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
    behind = DCACHE["LatchIF"](name="bench_lsu_to_dcache")
    mem_req_if = DCACHE["LatchIF"](name="bench_dcache_to_mem")
    mem_resp_if = DCACHE["LatchIF"](name="bench_mem_to_dcache")
    response_if = DCACHE["ForwardingIF"](name="bench_dcache_to_lsu")

    stage = DCACHE["LockupFreeCacheStage"](
        name="ThreadScalingDCache",
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

    for phase in scenario.phases:
        pending = deque(phase)
        phase_completions = 0
        phase_target = len(phase)
        max_cycles = max(200, phase_target * 80)
        phase_steps = 0

        while phase_completions < phase_target:
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
            if payload_type == "MISS_ACCEPTED":
                continue
            if payload_type in {"HIT_COMPLETE", "MISS_COMPLETE", "FLUSH_COMPLETE"}:
                phase_completions += 1

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


def _build_hit_only_dcache(num_threads: int, same_bank_conflict: bool) -> DCacheScenario:
    phases: List[List[Any]] = [[], []]
    preload_hits: List[Tuple[Any, int]] = []

    for tid in range(int(num_threads)):
        base = tid * 2
        read_addr = _dcache_addr_for_slot(base, same_bank_conflict)
        write_addr = _dcache_addr_for_slot(base + 1, same_bank_conflict)

        read_req = DCACHE["dCacheRequest"](addr_val=read_addr, rw_mode="read", size="word")
        write_req = DCACHE["dCacheRequest"](
            addr_val=write_addr,
            rw_mode="write",
            size="word",
            store_value=0x2000_0000 + tid,
        )

        phases[0].append(read_req)
        phases[1].append(write_req)

        preload_hits.append((read_req, 0x1000_0000 + tid))
        preload_hits.append((write_req, 0x1800_0000 + tid))

    return DCacheScenario(phases=phases, preload_hits=preload_hits, memory_words={})


def _build_hit_only_smem(num_threads: int, same_bank_conflict: bool) -> SmemScenario:
    phases: List[List[Transaction]] = [[], []]
    preload_words: List[Tuple[int, int]] = []

    for tid in range(int(num_threads)):
        base = tid * 2
        read_addr = _smem_addr_for_slot(base, same_bank_conflict)
        write_addr = _smem_addr_for_slot(base + 1, same_bank_conflict)

        phases[0].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=read_addr, thread_id=tid)
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=write_addr,
                write_data=0x2000_0000 + tid,
                thread_id=tid,
            )
        )

        preload_words.append((read_addr, 0x1000_0000 + tid))
        preload_words.append((write_addr, 0x1800_0000 + tid))

    return SmemScenario(phases=phases, preload_words=preload_words, dram_init={})


def _build_mixed_dcache(num_threads: int, same_bank_conflict: bool) -> DCacheScenario:
    phases: List[List[Any]] = [[] for _ in range(4)]
    preload_hits: List[Tuple[Any, int]] = []
    memory_words: Dict[int, int] = {}

    for tid in range(int(num_threads)):
        base = tid * 4
        hot_read_addr = _dcache_addr_for_slot(base, same_bank_conflict)
        hot_write_addr = _dcache_addr_for_slot(base + 1, same_bank_conflict)
        cold_a_addr = _dcache_addr_for_slot(base + 2, same_bank_conflict)
        cold_b_addr = _dcache_addr_for_slot(base + 3, same_bank_conflict)

        hot_read_req = DCACHE["dCacheRequest"](
            addr_val=hot_read_addr, rw_mode="read", size="word"
        )
        hot_write_req = DCACHE["dCacheRequest"](
            addr_val=hot_write_addr,
            rw_mode="write",
            size="word",
            store_value=0x3300_0000 + tid,
        )
        hot_write_req_a = DCACHE["dCacheRequest"](
            addr_val=hot_write_addr,
            rw_mode="write",
            size="word",
            store_value=0x3300_0000 + tid,
        )
        cold_a_miss_req = DCACHE["dCacheRequest"](
            addr_val=cold_a_addr, rw_mode="read", size="word"
        )
        hot_write_req_b = DCACHE["dCacheRequest"](
            addr_val=hot_write_addr,
            rw_mode="write",
            size="word",
            store_value=0x3301_0000 + tid,
        )
        cold_a_reuse_req = DCACHE["dCacheRequest"](
            addr_val=cold_a_addr, rw_mode="read", size="word"
        )
        cold_b_miss_req = DCACHE["dCacheRequest"](
            addr_val=cold_b_addr, rw_mode="read", size="word"
        )
        cold_b_reuse_req = DCACHE["dCacheRequest"](
            addr_val=cold_b_addr, rw_mode="read", size="word"
        )
        hot_write_req_c = DCACHE["dCacheRequest"](
            addr_val=hot_write_addr,
            rw_mode="write",
            size="word",
            store_value=0x3302_0000 + tid,
        )

        phases[0].append(hot_read_req)
        phases[1].append(hot_write_req)
        phases[0].append(hot_write_req_a)
        phases[1].append(cold_a_miss_req)
        phases[1].append(hot_write_req_b)
        phases[2].append(cold_a_reuse_req)
        phases[2].append(cold_b_miss_req)
        phases[3].append(cold_b_reuse_req)
        phases[3].append(hot_write_req_c)

        preload_hits.append((hot_read_req, 0x1100_0000 + tid))
        preload_hits.append((hot_write_req, 0x2200_0000 + tid))
        memory_words[cold_a_addr] = 0x4400_0000 + tid
        memory_words[cold_b_addr] = 0x5500_0000 + tid

    return DCacheScenario(
        phases=phases,
        preload_hits=preload_hits,
        memory_words=memory_words,
    )


def _build_mixed_smem(num_threads: int, same_bank_conflict: bool) -> SmemScenario:
    phases: List[List[Transaction]] = [[] for _ in range(4)]
    preload_words: List[Tuple[int, int]] = []
    dram_init: Dict[int, int] = {}

    for tid in range(int(num_threads)):
        base = tid * 4
        hot_read_addr = _smem_addr_for_slot(base, same_bank_conflict)
        hot_write_addr = _smem_addr_for_slot(base + 1, same_bank_conflict)
        cold_a_addr = _smem_addr_for_slot(base + 2, same_bank_conflict)
        cold_b_addr = _smem_addr_for_slot(base + 3, same_bank_conflict)

        cold_a_dram = _dram_addr_for_slot(base + 2)
        cold_b_dram = _dram_addr_for_slot(base + 3)

        phases[0].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=hot_read_addr, thread_id=tid)
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=hot_write_addr,
                write_data=0x3300_0000 + tid,
                thread_id=tid,
            )
        )
        phases[0].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=hot_write_addr,
                write_data=0x3300_0000 + tid,
                thread_id=tid,
            )
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=cold_a_dram,
                shmem_addr=cold_a_addr,
                thread_id=tid,
            )
        )
        phases[2].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=hot_write_addr,
                write_data=0x3301_0000 + tid,
                thread_id=tid,
            )
        )
        phases[2].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=cold_a_addr, thread_id=tid)
        )
        phases[2].append(
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=cold_b_dram,
                shmem_addr=cold_b_addr,
                thread_id=tid,
            )
        )
        phases[3].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=cold_b_addr, thread_id=tid)
        )
        phases[3].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=hot_write_addr,
                write_data=0x3302_0000 + tid,
                thread_id=tid,
            )
        )

        preload_words.append((hot_read_addr, 0x1100_0000 + tid))
        preload_words.append((hot_write_addr, 0x2200_0000 + tid))
        dram_init[cold_a_dram] = 0x4400_0000 + tid
        dram_init[cold_b_dram] = 0x5500_0000 + tid

    return SmemScenario(
        phases=phases,
        preload_words=preload_words,
        dram_init=dram_init,
    )


def _build_saxpy_dcache(num_threads: int, same_bank_conflict: bool) -> DCacheScenario:
    phases: List[List[Any]] = [[] for _ in range(4)]
    memory_words: Dict[int, int] = {}

    for tid in range(int(num_threads)):
        base = tid * 4
        x0_addr = _dcache_addr_for_slot(base, same_bank_conflict)
        y0_addr = _dcache_addr_for_slot(base + 1, same_bank_conflict)
        x1_addr = _dcache_addr_for_slot(base + 2, same_bank_conflict)
        y1_addr = _dcache_addr_for_slot(base + 3, same_bank_conflict)

        x0_val = tid + 1
        y0_val = 0x100 + tid
        x1_val = (tid + 1) * 2
        y1_val = 0x200 + tid
        result0 = (SAXPY_A * x0_val) + y0_val
        result1 = (SAXPY_A * x1_val) + y1_val

        phases[0].append(DCACHE["dCacheRequest"](addr_val=x0_addr, rw_mode="read", size="word"))
        phases[0].append(DCACHE["dCacheRequest"](addr_val=y0_addr, rw_mode="read", size="word"))
        phases[1].append(DCACHE["dCacheRequest"](addr_val=x0_addr, rw_mode="read", size="word"))
        phases[1].append(DCACHE["dCacheRequest"](addr_val=y0_addr, rw_mode="read", size="word"))
        phases[1].append(
            DCACHE["dCacheRequest"](
                addr_val=y0_addr,
                rw_mode="write",
                size="word",
                store_value=result0,
            )
        )
        phases[1].append(DCACHE["dCacheRequest"](addr_val=y0_addr, rw_mode="read", size="word"))
        phases[2].append(DCACHE["dCacheRequest"](addr_val=x1_addr, rw_mode="read", size="word"))
        phases[2].append(DCACHE["dCacheRequest"](addr_val=y1_addr, rw_mode="read", size="word"))
        phases[3].append(DCACHE["dCacheRequest"](addr_val=x1_addr, rw_mode="read", size="word"))
        phases[3].append(DCACHE["dCacheRequest"](addr_val=y1_addr, rw_mode="read", size="word"))
        phases[3].append(
            DCACHE["dCacheRequest"](
                addr_val=y1_addr,
                rw_mode="write",
                size="word",
                store_value=result1,
            )
        )
        phases[3].append(DCACHE["dCacheRequest"](addr_val=y1_addr, rw_mode="read", size="word"))

        memory_words[x0_addr] = x0_val
        memory_words[y0_addr] = y0_val
        memory_words[x1_addr] = x1_val
        memory_words[y1_addr] = y1_val

    return DCacheScenario(phases=phases, preload_hits=[], memory_words=memory_words)


def _build_saxpy_smem(num_threads: int, same_bank_conflict: bool) -> SmemScenario:
    phases: List[List[Transaction]] = [[] for _ in range(4)]
    dram_init: Dict[int, int] = {}

    for tid in range(int(num_threads)):
        base = tid * 4
        x0_addr = _smem_addr_for_slot(base, same_bank_conflict)
        y0_addr = _smem_addr_for_slot(base + 1, same_bank_conflict)
        x1_addr = _smem_addr_for_slot(base + 2, same_bank_conflict)
        y1_addr = _smem_addr_for_slot(base + 3, same_bank_conflict)
        x0_dram = _dram_addr_for_slot(base)
        y0_dram = _dram_addr_for_slot(base + 1)
        x1_dram = _dram_addr_for_slot(base + 2)
        y1_dram = _dram_addr_for_slot(base + 3)

        x0_val = tid + 1
        y0_val = 0x100 + tid
        x1_val = (tid + 1) * 2
        y1_val = 0x200 + tid
        result0 = (SAXPY_A * x0_val) + y0_val
        result1 = (SAXPY_A * x1_val) + y1_val

        phases[0].append(
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=x0_dram,
                shmem_addr=x0_addr,
                thread_id=tid,
            )
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=y0_dram,
                shmem_addr=y0_addr,
                thread_id=tid,
            )
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.SH_LD,
                shmem_addr=x0_addr,
                thread_id=tid,
            )
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.SH_LD,
                shmem_addr=y0_addr,
                thread_id=tid,
            )
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=y0_addr,
                write_data=result0,
                thread_id=tid,
            )
        )
        phases[1].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=y0_addr, thread_id=tid)
        )
        phases[2].append(
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=x1_dram,
                shmem_addr=x1_addr,
                thread_id=tid,
            )
        )
        phases[2].append(
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=y1_dram,
                shmem_addr=y1_addr,
                thread_id=tid,
            )
        )
        phases[3].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=x1_addr, thread_id=tid)
        )
        phases[3].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=y1_addr, thread_id=tid)
        )
        phases[3].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=y1_addr,
                write_data=result1,
                thread_id=tid,
            )
        )
        phases[3].append(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=y1_addr, thread_id=tid)
        )

        dram_init[x0_dram] = x0_val
        dram_init[y0_dram] = y0_val
        dram_init[x1_dram] = x1_val
        dram_init[y1_dram] = y1_val

    return SmemScenario(phases=phases, preload_words=[], dram_init=dram_init)


WORKLOADS: List[WorkloadSpec] = [
    WorkloadSpec(
        name="mixed_hit_miss",
        title="Mixed Hit/Miss",
        description=(
            "Each thread mixes hot read/write hits with two cold fetches and later "
            "reuses so every scenario includes both hits and misses."
        ),
        dcache_builder=_build_mixed_dcache,
        smem_builder=_build_mixed_smem,
    ),
    WorkloadSpec(
        name="saxpy_like",
        title="SAXPY-Like Stream",
        description=(
            "Approximate SAXPY memory behavior over two elements per thread: fetch x[i] "
            "and y[i], update y[i], then read back y[i]."
        ),
        dcache_builder=_build_saxpy_dcache,
        smem_builder=_build_saxpy_smem,
    ),
]


def run_thread_scaling_benchmark(max_threads: int = 32) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []

    for workload in WORKLOADS:
        for threads in range(1, int(max_threads) + 1):
            dcache_balanced = _run_dcache_scenario(
                workload.dcache_builder(threads, False),
                mem_latency_cycles=MISS_LATENCY_CYCLES,
            )
            smem_balanced = _run_smem_scenario(
                workload.smem_builder(threads, False),
                num_threads=threads,
                dram_latency_cycles=MISS_LATENCY_CYCLES,
            )
            dcache_conflict = _run_dcache_scenario(
                workload.dcache_builder(threads, True),
                mem_latency_cycles=MISS_LATENCY_CYCLES,
            )
            smem_conflict = _run_smem_scenario(
                workload.smem_builder(threads, True),
                num_threads=threads,
                dram_latency_cycles=MISS_LATENCY_CYCLES,
            )

            rows.append(
                BenchmarkRow(
                    workload=workload.name,
                    threads=threads,
                    dcache_balanced_cycles=dcache_balanced,
                    smem_balanced_cycles=smem_balanced,
                    dcache_conflict_cycles=dcache_conflict,
                    smem_conflict_cycles=smem_conflict,
                )
            )

    return rows


def _rows_for_workload(rows: Iterable[BenchmarkRow], workload_name: str) -> List[BenchmarkRow]:
    return [row for row in rows if row.workload == workload_name]


def _write_csv(rows: Iterable[BenchmarkRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "workload",
                "threads",
                "dcache_balanced_cycles",
                "smem_balanced_cycles",
                "dcache_conflict_cycles",
                "smem_conflict_cycles",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.workload,
                    row.threads,
                    row.dcache_balanced_cycles,
                    row.smem_balanced_cycles,
                    row.dcache_conflict_cycles,
                    row.smem_conflict_cycles,
                ]
            )


def _plot_rows(rows: List[BenchmarkRow], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the benchmark plot. "
            "Install it in the active Python environment and rerun the script."
        ) from exc

    colors = {
        "dcache_balanced": "#D55E00",
        "dcache_conflict": "#8B1E3F",
        "smem_balanced": "#0072B2",
        "smem_conflict": "#009E73",
    }

    saxpy = next(workload for workload in WORKLOADS if workload.name == "saxpy_like")
    workload_rows = _rows_for_workload(rows, saxpy.name)
    threads = [row.threads for row in workload_rows]

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.2))

    ax.plot(
        threads,
        [row.dcache_balanced_cycles for row in workload_rows],
        color=colors["dcache_balanced"],
        marker="o",
        linewidth=2.4,
        label="DCache balanced",
    )
    ax.plot(
        threads,
        [row.dcache_conflict_cycles for row in workload_rows],
        color=colors["dcache_conflict"],
        marker="o",
        linestyle="--",
        linewidth=2.4,
        label="DCache conflict",
    )
    ax.plot(
        threads,
        [row.smem_balanced_cycles for row in workload_rows],
        color=colors["smem_balanced"],
        marker="s",
        linewidth=2.4,
        label="SMEM balanced",
    )
    ax.plot(
        threads,
        [row.smem_conflict_cycles for row in workload_rows],
        color=colors["smem_conflict"],
        marker="s",
        linestyle="--",
        linewidth=2.4,
        label="SMEM conflict",
    )

    ax.set_title("SAXPY-Like Stream")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Cycles")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=10)

    fig.suptitle(
        "SMEM vs DCache Thread Scaling",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        "Balanced and same-bank-conflict runs are overlaid. "
        f"The dcache model has {DCACHE_NUM_BANKS} banks; the shared-memory model has {SMEM_NUM_BANKS} banks.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight", format="jpg")
    plt.close(fig)


def _build_summary_table(rows: Iterable[BenchmarkRow]) -> str:
    lines = [
        f"{'Thr':>3} | {'D$ Bal':>7} | {'SMEM Bal':>8} | {'D$ Conf':>7} | {'SMEM Conf':>9}"
    ]
    lines.append("-" * 49)
    for row in rows:
        lines.append(
            f"{row.threads:>3} | "
            f"{row.dcache_balanced_cycles:>7} | "
            f"{row.smem_balanced_cycles:>8} | "
            f"{row.dcache_conflict_cycles:>7} | "
            f"{row.smem_conflict_cycles:>9}"
        )
    return "\n".join(lines)


def _write_report(rows: List[BenchmarkRow], path: Path) -> None:
    sections: List[str] = [
        "SMEM vs DCache Thread Scaling Benchmark",
        "=======================================",
        "",
        "Configuration",
        "-------------",
        "- Threads swept from 1 to 32",
        f"- Synthetic miss latency used in this benchmark: {MISS_LATENCY_CYCLES} cycles",
        "- Balanced runs spread requests across banks as evenly as possible",
        "- Conflict runs force requests onto one bank",
        f"- DCache has {DCACHE_NUM_BANKS} banks; SMEM has {SMEM_NUM_BANKS} banks",
        "",
    ]

    for workload in WORKLOADS:
        workload_rows = _rows_for_workload(rows, workload.name)
        last_row = workload_rows[-1]
        sections.extend(
            [
                workload.title,
                "-" * len(workload.title),
                workload.description,
                "",
                "32-thread summary",
                "-----------------",
                f"- Balanced: DCache = {last_row.dcache_balanced_cycles} cycles, "
                f"SMEM = {last_row.smem_balanced_cycles} cycles",
                f"- Same-bank conflict: DCache = {last_row.dcache_conflict_cycles} cycles, "
                f"SMEM = {last_row.smem_conflict_cycles} cycles",
                "",
                _build_summary_table(workload_rows),
                "",
            ]
        )

    sections.extend(
        [
            f"CSV: {CSV_PATH.name}",
            f"Plot: {PLOT_PATH.name}",
            "",
        ]
    )

    path.write_text("\n".join(sections), encoding="utf-8")


def _print_terminal_summary(rows: Iterable[BenchmarkRow]) -> None:
    for workload in WORKLOADS:
        workload_rows = _rows_for_workload(list(rows), workload.name)
        print(workload.title)
        print("-" * len(workload.title))
        print(_build_summary_table(workload_rows))
        print()


def main() -> None:
    rows = run_thread_scaling_benchmark(max_threads=32)
    _write_csv(rows, CSV_PATH)
    _write_report(rows, REPORT_PATH)
    _plot_rows(rows, PLOT_PATH)
    _print_terminal_summary(rows)
    print(f"Report saved to: {REPORT_PATH}")
    print(f"CSV saved to:  {CSV_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        main()

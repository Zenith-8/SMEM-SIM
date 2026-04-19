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
   Every active thread shares the same read-only ``x`` source address while
   still reading, writing, and reading back a distinct ``y[i]``.

The graph focuses on cycle cost, not arithmetic correctness. The access stream
is intentionally kernel-shaped:

    load x
    load y
    store y = a * x + y
    read back y

Outputs:
- saxpy_bank_conflict_multicast.csv
- saxpy_bank_conflict_multicast_report.txt
- saxpy_bank_conflict_multicast_plot.png

Run:
    ./.venv/bin/python test_saxpy_bank_conflict_multicast_benchmark.py
"""

from __future__ import annotations

from collections import defaultdict, deque
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
PLOT_PATH = ROOT / "saxpy_bank_conflict_multicast_plot.png"
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
MISS_LATENCY_CYCLES = int(_SMEM_CFG.dram_latency_cycles)
SMEM_ARBITER_ISSUE_WIDTH = int(_SMEM_CFG.arbiter_issue_width)
SMEM_READ_CROSSBAR_PIPELINE = int(_SMEM_CFG.read_crossbar_pipeline_cycles)

SAXPY_A = 3
SHARED_X_SLOT = 0x400
UNIQUE_X_SLOT_BASE = 0x000
Y_SLOT_BASE = 0x800
DRAM_BASE_ADDR = 0x200000
SHARED_X_VALUE = 0x1F00


@dataclass
class ConflictRow:
    conflict_depth: int
    dcache_non_multicast_cycles: int
    smem_non_multicast_cycles: int
    dcache_multicast_cycles: int
    smem_multicast_cycles: int


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


def _smem_same_bank_addr(slot_index: int) -> int:
    return int(slot_index) * SMEM_NUM_BANKS * WORD_BYTES


def _dram_addr_for_slot(slot_index: int) -> int:
    return int(DRAM_BASE_ADDR + (slot_index * WORD_BYTES))


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
    phases: List[List[Any]] = [[] for _ in range(4)]
    preload_hits: List[Tuple[Any, int]] = []
    memory_words: Dict[int, int] = {}

    shared_x_addr = _dcache_same_bank_addr(SHARED_X_SLOT)
    memory_words[shared_x_addr] = SHARED_X_VALUE

    for tid in range(int(conflict_depth)):
        x_addr = (
            shared_x_addr
            if multicast
            else _dcache_same_bank_addr(UNIQUE_X_SLOT_BASE + tid)
        )
        y_addr = _dcache_same_bank_addr(Y_SLOT_BASE + tid)

        x_val = SHARED_X_VALUE if multicast else (tid + 1)
        y_val = 0x0100 + tid
        result = (SAXPY_A * x_val) + y_val

        x_req = DCACHE["dCacheRequest"](addr_val=x_addr, rw_mode="read", size="word")
        y_read_req = DCACHE["dCacheRequest"](addr_val=y_addr, rw_mode="read", size="word")
        y_write_req = DCACHE["dCacheRequest"](
            addr_val=y_addr,
            rw_mode="write",
            size="word",
            store_value=result,
        )
        y_readback_req = DCACHE["dCacheRequest"](addr_val=y_addr, rw_mode="read", size="word")

        phases[0].append(x_req)
        phases[1].append(y_read_req)
        phases[2].append(y_write_req)
        phases[3].append(y_readback_req)

        preload_hits.append((y_read_req, y_val))
        if not multicast:
            memory_words[x_addr] = x_val

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
    phases: List[List[Transaction]] = [[] for _ in range(5)]
    preload_words: List[Tuple[int, int]] = []
    dram_init: Dict[int, int] = {}

    shared_x_shmem_addr = _smem_same_bank_addr(SHARED_X_SLOT)
    shared_x_dram_addr = _dram_addr_for_slot(SHARED_X_SLOT)
    dram_init[shared_x_dram_addr] = SHARED_X_VALUE

    for tid in range(int(conflict_depth)):
        x_shmem_addr = (
            shared_x_shmem_addr
            if multicast
            else _smem_same_bank_addr(UNIQUE_X_SLOT_BASE + tid)
        )
        x_dram_addr = (
            shared_x_dram_addr
            if multicast
            else _dram_addr_for_slot(UNIQUE_X_SLOT_BASE + tid)
        )
        y_shmem_addr = _smem_same_bank_addr(Y_SLOT_BASE + tid)

        x_val = SHARED_X_VALUE if multicast else (tid + 1)
        y_val = 0x0100 + tid
        result = (SAXPY_A * x_val) + y_val

        phases[0].append(
            Transaction(
                txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=x_dram_addr,
                shmem_addr=x_shmem_addr,
                thread_id=tid,
            )
        )
        phases[1].append(
            Transaction(
                txn_type=TxnType.SH_LD,
                shmem_addr=x_shmem_addr,
                thread_id=tid,
            )
        )
        phases[2].append(
            Transaction(
                txn_type=TxnType.SH_LD,
                shmem_addr=y_shmem_addr,
                thread_id=tid,
            )
        )
        phases[3].append(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=y_shmem_addr,
                write_data=result,
                thread_id=tid,
            )
        )
        phases[4].append(
            Transaction(
                txn_type=TxnType.SH_LD,
                shmem_addr=y_shmem_addr,
                thread_id=tid,
            )
        )

        preload_words.append((y_shmem_addr, y_val))
        if not multicast:
            dram_init[x_dram_addr] = x_val

    return SmemScenario(
        phases=phases,
        preload_words=preload_words,
        dram_init=dram_init,
    )


def run_conflict_benchmark(max_conflict_depth: int = 32) -> List[ConflictRow]:
    rows: List[ConflictRow] = []

    for conflict_depth in range(1, int(max_conflict_depth) + 1):
        dcache_non = _run_dcache_scenario(
            _build_dcache_saxpy_conflict(conflict_depth, multicast=False),
            mem_latency_cycles=MISS_LATENCY_CYCLES,
        )
        smem_non = _run_smem_scenario(
            _build_smem_saxpy_conflict(conflict_depth, multicast=False),
            num_threads=conflict_depth,
            dram_latency_cycles=MISS_LATENCY_CYCLES,
        )
        dcache_multi = _run_dcache_scenario(
            _build_dcache_saxpy_conflict(conflict_depth, multicast=True),
            mem_latency_cycles=MISS_LATENCY_CYCLES,
        )
        smem_multi = _run_smem_scenario(
            _build_smem_saxpy_conflict(conflict_depth, multicast=True),
            num_threads=conflict_depth,
            dram_latency_cycles=MISS_LATENCY_CYCLES,
        )

        rows.append(
            ConflictRow(
                conflict_depth=conflict_depth,
                dcache_non_multicast_cycles=dcache_non,
                smem_non_multicast_cycles=smem_non,
                dcache_multicast_cycles=dcache_multi,
                smem_multicast_cycles=smem_multi,
            )
        )

    return rows


def _write_csv(rows: Iterable[ConflictRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "conflict_depth",
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
                    row.dcache_non_multicast_cycles,
                    row.smem_non_multicast_cycles,
                    row.dcache_multicast_cycles,
                    row.smem_multicast_cycles,
                ]
            )


def _plot_rows(rows: List[ConflictRow], path: Path) -> None:
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

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharey=True)
    fig.patch.set_facecolor("#f6f3ee")

    palette = {
        "dcache": "#c05621",
        "smem": "#1f6f8b",
        "grid": "#d8d2c8",
        "text": "#2f2a24",
    }

    panels = [
        (
            axes[0],
            "Non-Multicast",
            "Distinct x[i], distinct y[i], all forced onto one bank.",
            [row.dcache_non_multicast_cycles for row in rows],
            [row.smem_non_multicast_cycles for row in rows],
        ),
        (
            axes[1],
            "Multicast",
            "Shared x across the active conflict group, distinct y[i].",
            [row.dcache_multicast_cycles for row in rows],
            [row.smem_multicast_cycles for row in rows],
        ),
    ]

    for ax, title, subtitle, dcache_cycles, smem_cycles in panels:
        ax.set_facecolor("#fffdf9")
        ax.plot(
            depths,
            dcache_cycles,
            color=palette["dcache"],
            linewidth=2.8,
            marker="o",
            markersize=5.2,
            label="DCache",
        )
        ax.plot(
            depths,
            smem_cycles,
            color=palette["smem"],
            linewidth=2.8,
            marker="s",
            markersize=5.0,
            label="SMEM",
        )
        ax.fill_between(depths, dcache_cycles, color=palette["dcache"], alpha=0.08)
        ax.fill_between(depths, smem_cycles, color=palette["smem"], alpha=0.08)
        ax.set_title(title, fontsize=13, weight="bold", color=palette["text"], pad=12)
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.4,
            color="#5c5348",
        )
        ax.grid(True, axis="y", alpha=0.55, color=palette["grid"], linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.18, color=palette["grid"], linewidth=0.6)
        ax.set_xlim(1, 32)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#8e8579")
        ax.spines["bottom"].set_color("#8e8579")
        ax.tick_params(colors=palette["text"])
        ax.legend(loc="upper left", frameon=False)

    axes[0].set_ylabel("Cycles to Resolve Stream", fontsize=11, color=palette["text"])
    for ax in axes:
        ax.set_xlabel("Conflict Depth (1-32 active lanes on one bank)", fontsize=10.5, color=palette["text"])

    fig.suptitle(
        "SAXPY-Like Same-Bank Conflict Sweep",
        fontsize=17,
        weight="bold",
        color=palette["text"],
        y=0.97,
    )
    fig.text(
        0.5,
        0.028,
        "Each point runs a bank-conflicted SAXPY-like stream. "
        f"DCache uses the cache-bank model; SMEM uses the shared-memory functional model with a {SMEM_READ_CROSSBAR_PIPELINE}-cycle read crossbar return path.",
        ha="center",
        fontsize=9.4,
        color="#5c5348",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_table(rows: Iterable[ConflictRow]) -> str:
    lines = [
        f"{'Dep':>3} | {'D$ Non':>7} | {'SMEM Non':>8} | {'D$ Multi':>8} | {'SMEM Multi':>10}"
    ]
    lines.append("-" * 52)
    for row in rows:
        lines.append(
            f"{row.conflict_depth:>3} | "
            f"{row.dcache_non_multicast_cycles:>7} | "
            f"{row.smem_non_multicast_cycles:>8} | "
            f"{row.dcache_multicast_cycles:>8} | "
            f"{row.smem_multicast_cycles:>10}"
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
        f"- DCache miss latency: {MISS_LATENCY_CYCLES} cycles",
        f"- SMEM read crossbar latency: {SMEM_READ_CROSSBAR_PIPELINE} cycles",
        f"- DCache banks: {DCACHE_NUM_BANKS}",
        f"- SMEM banks: {SMEM_NUM_BANKS}",
        "",
        "Case Definitions",
        "----------------",
        "- Non-multicast: each active thread loads a distinct x[i] and updates a distinct y[i].",
        "- Multicast: all active threads share the same read-only x address while still updating distinct y[i].",
        "- Stream shape: load x, load y, store y = a*x + y, read back y.",
        "",
        "32-way Summary",
        "--------------",
        f"- Non-multicast: DCache = {last.dcache_non_multicast_cycles} cycles, SMEM = {last.smem_non_multicast_cycles} cycles",
        f"- Multicast: DCache = {last.dcache_multicast_cycles} cycles, SMEM = {last.smem_multicast_cycles} cycles",
        "",
        _build_table(rows),
        "",
        f"CSV: {CSV_PATH.name}",
        f"Report: {REPORT_PATH.name}",
        f"Plot: {PLOT_PATH.name}",
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def _print_summary(rows: List[ConflictRow]) -> None:
    print("SAXPY-Like Same-Bank Conflict Sweep")
    print("----------------------------------")
    print(_build_table(rows))
    print()
    print(f"CSV saved to:   {CSV_PATH}")
    print(f"Report saved to:{REPORT_PATH}")
    print(f"Plot saved to:  {PLOT_PATH}")


def main() -> None:
    rows = run_conflict_benchmark(max_conflict_depth=32)
    _write_csv(rows, CSV_PATH)
    _write_report(rows, REPORT_PATH)
    _plot_rows(rows, PLOT_PATH)
    _print_summary(rows)


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        main()

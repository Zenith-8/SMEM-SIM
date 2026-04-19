#!/usr/bin/env python3
"""
SAXPY benchmark: global-memory transactions per global-memory request.

This benchmark compares DCache vs SMEM using the same full SAXPY memory stream.
For each SAXPY element we issue:

- Read x
- Read y
- Write y = a*x + y

On the SMEM model, the stream is modeled explicitly as:

- global.ld x -> shmem
- global.ld y -> shmem
- sh.ld x
- sh.ld y
- sh.st y (result)
- global.st y -> dram

Sweep variable:
- x_reuse_words: how many unique x words are reused across the vector.

Metric:
- transactions_per_request = dram_transactions / global_requests
  where "global_requests" counts only global-memory requests.

Outputs:
- global_transactions_per_request.csv
- global_transactions_per_request_report.txt
- global_transactions_per_request_plot.png

Run:
    ./.venv/bin/python test_global_transactions_per_request_benchmark.py
"""

from __future__ import annotations

from collections import deque
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
CSV_PATH = ROOT / "global_transactions_per_request.csv"
REPORT_PATH = ROOT / "global_transactions_per_request_report.txt"
PLOT_PATH = ROOT / "global_transactions_per_request_plot.png"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/smem-sim-mpl-cache")

DCACHE = _load_dcache_symbols()
from simulator.mem_types import (
    BLOCK_SIZE_WORDS as DCACHE_BLOCK_WORDS,
    WORD_SIZE_BYTES as DCACHE_WORD_BYTES,
)


_SMEM_CFG = load_smem_config()

WORD_BYTES = int(DCACHE_WORD_BYTES)
WORD_MASK = 0xFFFF_FFFF
DCACHE_BLOCK_WORDS = int(DCACHE_BLOCK_WORDS)
DCACHE_BLOCK_BYTES = int(DCACHE_BLOCK_WORDS * WORD_BYTES)
NUM_LANES = 32

SAXPY_A = 3
TOTAL_ELEMENTS = 512
NUM_PASSES = 2
X_REUSE_WORDS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

DCACHE_X_BASE = 0x0000_0000
DCACHE_Y_BASE = 0x0400_0000
SMEM_DRAM_X_BASE = 0x2000_0000
SMEM_DRAM_Y_BASE = 0x2400_0000
SMEM_X_SLOT_BASE = 0x0000
SMEM_Y_SLOT_BASE = 0x0200


@dataclass
class _MemResp:
    warp_id: int
    packet: Any = None
    status: Any = None


class _PacketWords:
    def __init__(self, words: Sequence[int]):
        self._bytes = b"".join(
            int(word & WORD_MASK).to_bytes(WORD_BYTES, byteorder="little", signed=False)
            for word in words
        )

    def tobytes(self) -> bytes:
        return self._bytes


@dataclass
class SaxpyStream:
    dcache_requests: List[Any]
    dcache_mem_init: Dict[int, int]
    smem_transactions: List[Transaction]
    smem_dram_init: Dict[int, int]
    dcache_global_requests: int
    smem_global_requests: int


@dataclass
class RatioRow:
    x_reuse_words: int
    dcache_global_requests: int
    dcache_dram_transactions: int
    dcache_transactions_per_request: float
    smem_global_requests: int
    smem_dram_transactions: int
    smem_transactions_per_request: float
    dcache_cycles: int
    smem_cycles: int


class InstrumentedShmemFunctionalSimulator(ShmemFunctionalSimulator):
    """SMEM simulator with explicit DRAM issue counters."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.axi_reads_issued = 0
        self.axi_writes_issued = 0

    def _run_smem_write_controller(self, banks_used_this_cycle: set[int]) -> None:
        before = len(self.pending_dram_reads)
        super()._run_smem_write_controller(banks_used_this_cycle)
        after = len(self.pending_dram_reads)
        if after > before:
            self.axi_reads_issued += int(after - before)

    def _run_axi_bus_write_phase(self) -> None:
        before = len(self.pending_dram_writes)
        super()._run_axi_bus_write_phase()
        after = len(self.pending_dram_writes)
        if after > before:
            self.axi_writes_issued += int(after - before)


def _build_saxpy_stream(
    *,
    total_elements: int,
    x_reuse_words: int,
    num_passes: int,
) -> SaxpyStream:
    x_unique = max(1, min(int(total_elements), int(x_reuse_words)))

    x_values = [int((0x1000 + (i * 17)) & WORD_MASK) for i in range(x_unique)]
    y_values = [int((0x2000 + (i * 29)) & WORD_MASK) for i in range(total_elements)]

    dcache_mem_init: Dict[int, int] = {}
    smem_dram_init: Dict[int, int] = {}

    for i in range(x_unique):
        dcache_mem_init[DCACHE_X_BASE + (i * DCACHE_BLOCK_BYTES)] = x_values[i]
        smem_dram_init[SMEM_DRAM_X_BASE + (i * DCACHE_BLOCK_BYTES)] = x_values[i]
    for i in range(total_elements):
        dcache_mem_init[DCACHE_Y_BASE + (i * DCACHE_BLOCK_BYTES)] = y_values[i]
        smem_dram_init[SMEM_DRAM_Y_BASE + (i * DCACHE_BLOCK_BYTES)] = y_values[i]

    dcache_requests: List[Any] = []
    smem_transactions: List[Transaction] = []
    dcache_global_requests = 0
    smem_global_requests = 0

    for _pass_idx in range(int(num_passes)):
        for element_idx in range(int(total_elements)):
            lane = int(element_idx % NUM_LANES)
            x_idx = int(element_idx % x_unique)
            y_idx = int(element_idx)

            x_val = int(x_values[x_idx])
            y_val = int(y_values[y_idx])
            result = int((SAXPY_A * x_val + y_val) & WORD_MASK)
            y_values[y_idx] = result

            dcache_x_addr = int(DCACHE_X_BASE + (x_idx * DCACHE_BLOCK_BYTES))
            dcache_y_addr = int(DCACHE_Y_BASE + (y_idx * DCACHE_BLOCK_BYTES))

            smem_dram_x_addr = int(SMEM_DRAM_X_BASE + (x_idx * DCACHE_BLOCK_BYTES))
            smem_dram_y_addr = int(SMEM_DRAM_Y_BASE + (y_idx * DCACHE_BLOCK_BYTES))

            shmem_x_addr = int(SMEM_X_SLOT_BASE + (lane * WORD_BYTES))
            shmem_y_addr = int(SMEM_Y_SLOT_BASE + (lane * WORD_BYTES))

            dcache_requests.append(
                DCACHE["dCacheRequest"](addr_val=dcache_x_addr, rw_mode="read", size="word")
            )
            dcache_requests.append(
                DCACHE["dCacheRequest"](addr_val=dcache_y_addr, rw_mode="read", size="word")
            )
            dcache_requests.append(
                DCACHE["dCacheRequest"](
                    addr_val=dcache_y_addr,
                    rw_mode="write",
                    size="word",
                    store_value=result,
                )
            )
            dcache_global_requests += 3

            smem_transactions.extend(
                [
                    Transaction(
                        txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                        dram_addr=smem_dram_x_addr,
                        shmem_addr=shmem_x_addr,
                        thread_id=lane,
                    ),
                    Transaction(
                        txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                        dram_addr=smem_dram_y_addr,
                        shmem_addr=shmem_y_addr,
                        thread_id=lane,
                    ),
                    Transaction(
                        txn_type=TxnType.SH_LD,
                        shmem_addr=shmem_x_addr,
                        thread_id=lane,
                    ),
                    Transaction(
                        txn_type=TxnType.SH_LD,
                        shmem_addr=shmem_y_addr,
                        thread_id=lane,
                    ),
                    Transaction(
                        txn_type=TxnType.SH_ST,
                        shmem_addr=shmem_y_addr,
                        write_data=result,
                        thread_id=lane,
                    ),
                    Transaction(
                        txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                        dram_addr=smem_dram_y_addr,
                        shmem_addr=shmem_y_addr,
                        thread_id=lane,
                    ),
                ]
            )
            smem_global_requests += 3

    return SaxpyStream(
        dcache_requests=dcache_requests,
        dcache_mem_init=dcache_mem_init,
        smem_transactions=smem_transactions,
        smem_dram_init=smem_dram_init,
        dcache_global_requests=dcache_global_requests,
        smem_global_requests=smem_global_requests,
    )


def _dcache_stage_quiescent(
    stage: Any,
    *,
    behind: Any,
    mem_req_if: Any,
    mem_resp_if: Any,
    pending_responses: List[Tuple[int, _MemResp]],
) -> bool:
    if pending_responses:
        return False
    if getattr(behind, "valid", False):
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


def _run_dcache_saxpy_stream(
    *,
    requests: Sequence[Any],
    mem_init: Dict[int, int],
    mem_latency_cycles: int,
) -> Tuple[int, int]:
    behind = DCACHE["LatchIF"](name="saxpy_txn_per_req_lsu_to_dcache")
    mem_req_if = DCACHE["LatchIF"](name="saxpy_txn_per_req_dcache_to_mem")
    mem_resp_if = DCACHE["LatchIF"](name="saxpy_txn_per_req_mem_to_dcache")
    response_if = DCACHE["ForwardingIF"](name="saxpy_txn_per_req_dcache_to_lsu")

    stage = DCACHE["LockupFreeCacheStage"](
        name="SaxpyTxnPerRequestDCache",
        behind_latch=behind,
        forward_ifs_write={"DCache_LSU_Resp": response_if},
        mem_req_if=mem_req_if,
        mem_resp_if=mem_resp_if,
    )

    mem_image = {int(addr): int(word) & WORD_MASK for addr, word in mem_init.items()}
    pending = deque(requests)
    pending_responses: List[Tuple[int, _MemResp]] = []

    accounted_requests = 0
    dram_transactions = 0
    target_requests = len(requests)

    max_cycles = max(10000, target_requests * (int(mem_latency_cycles) + 20))
    steps = 0

    while pending or accounted_requests < target_requests or not _dcache_stage_quiescent(
        stage,
        behind=behind,
        mem_req_if=mem_req_if,
        mem_resp_if=mem_resp_if,
        pending_responses=pending_responses,
    ):
        steps += 1
        if steps > max_cycles:
            raise TimeoutError("DCache SAXPY stream did not complete in cycle budget.")

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
            dram_transactions += 1

            issue_cycle = int(stage.get_cycle_count())
            warp_id = int(req_payload.get("warp", req_payload.get("warp_id", 0)))
            rw_mode = str(req_payload.get("rw_mode", "read")).lower()
            base_addr = int(req_payload.get("addr", 0))
            ready_cycle = issue_cycle + int(mem_latency_cycles)

            if rw_mode == "read":
                words = [
                    int(mem_image.get(base_addr + (i * WORD_BYTES), 0)) & WORD_MASK
                    for i in range(DCACHE_BLOCK_WORDS)
                ]
                pending_responses.append(
                    (ready_cycle, _MemResp(warp_id=warp_id, packet=_PacketWords(words)))
                )
            else:
                data_words = req_payload.get("data", [])
                if isinstance(data_words, list):
                    for i, word in enumerate(data_words):
                        mem_image[base_addr + (i * WORD_BYTES)] = int(word) & WORD_MASK
                pending_responses.append(
                    (ready_cycle, _MemResp(warp_id=warp_id, status="WRITE_DONE"))
                )

        payload = response_if.pop()
        if payload is None:
            continue

        payload_type = str(getattr(payload, "type", ""))
        if payload_type in {"MISS_ACCEPTED", "HIT_COMPLETE", "FLUSH_COMPLETE"}:
            accounted_requests += 1

    return int(stage.get_cycle_count()), int(dram_transactions)


def _run_smem_saxpy_stream(
    *,
    transactions: Sequence[Transaction],
    dram_init: Dict[int, int],
) -> Tuple[int, int]:
    sim = InstrumentedShmemFunctionalSimulator(
        dram_init=dram_init,
        num_banks=int(_SMEM_CFG.num_banks),
        word_bytes=int(_SMEM_CFG.word_bytes),
        dram_latency_cycles=int(_SMEM_CFG.dram_latency_cycles),
        arbiter_issue_width=int(_SMEM_CFG.arbiter_issue_width),
        num_threads=max(NUM_LANES, int(_SMEM_CFG.num_threads)),
        thread_block_size_bytes=_SMEM_CFG.thread_block_size_bytes,
        read_crossbar_pipeline_cycles=int(_SMEM_CFG.read_crossbar_pipeline_cycles),
    )

    for txn in transactions:
        sim.issue(txn)

    while sim._has_pending_work():
        sim.step()

    dram_transactions = int(sim.axi_reads_issued + sim.axi_writes_issued)
    return int(sim.get_cycle_count()), dram_transactions


def run_benchmark() -> List[RatioRow]:
    rows: List[RatioRow] = []

    for x_reuse_words in X_REUSE_WORDS_SWEEP:
        stream = _build_saxpy_stream(
            total_elements=TOTAL_ELEMENTS,
            x_reuse_words=x_reuse_words,
            num_passes=NUM_PASSES,
        )

        dcache_cycles, dcache_dram_txn = _run_dcache_saxpy_stream(
            requests=stream.dcache_requests,
            mem_init=stream.dcache_mem_init,
            mem_latency_cycles=int(_SMEM_CFG.dram_latency_cycles),
        )
        smem_cycles, smem_dram_txn = _run_smem_saxpy_stream(
            transactions=stream.smem_transactions,
            dram_init=stream.smem_dram_init,
        )

        rows.append(
            RatioRow(
                x_reuse_words=int(x_reuse_words),
                dcache_global_requests=int(stream.dcache_global_requests),
                dcache_dram_transactions=int(dcache_dram_txn),
                dcache_transactions_per_request=float(dcache_dram_txn)
                / float(max(stream.dcache_global_requests, 1)),
                smem_global_requests=int(stream.smem_global_requests),
                smem_dram_transactions=int(smem_dram_txn),
                smem_transactions_per_request=float(smem_dram_txn)
                / float(max(stream.smem_global_requests, 1)),
                dcache_cycles=int(dcache_cycles),
                smem_cycles=int(smem_cycles),
            )
        )

    return rows


def _write_csv(rows: Iterable[RatioRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_reuse_words",
                "dcache_global_requests",
                "dcache_dram_transactions",
                "dcache_transactions_per_request",
                "smem_global_requests",
                "smem_dram_transactions",
                "smem_transactions_per_request",
                "dcache_cycles",
                "smem_cycles",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.x_reuse_words,
                    row.dcache_global_requests,
                    row.dcache_dram_transactions,
                    f"{row.dcache_transactions_per_request:.8f}",
                    row.smem_global_requests,
                    row.smem_dram_transactions,
                    f"{row.smem_transactions_per_request:.8f}",
                    row.dcache_cycles,
                    row.smem_cycles,
                ]
            )


def _build_table(rows: Iterable[RatioRow]) -> str:
    lines = [
        f"{'XReuse':>6} | {'D$ Txn/Req':>10} | {'SMEM Txn/Req':>12} | {'D$ DRAM':>7} | {'SMEM DRAM':>9} | {'D$ Cyc':>6} | {'SMEM Cyc':>8}"
    ]
    lines.append("-" * 80)
    for row in rows:
        lines.append(
            f"{row.x_reuse_words:>6} | "
            f"{row.dcache_transactions_per_request:>10.5f} | "
            f"{row.smem_transactions_per_request:>12.5f} | "
            f"{row.dcache_dram_transactions:>7} | "
            f"{row.smem_dram_transactions:>9} | "
            f"{row.dcache_cycles:>6} | "
            f"{row.smem_cycles:>8}"
        )
    return "\n".join(lines)


def _write_report(rows: List[RatioRow], path: Path) -> None:
    first = rows[0]
    last = rows[-1]
    sections = [
        "SAXPY: Global Transactions per Global Request (DCache vs SMEM)",
        "==============================================================",
        "",
        "Workload",
        "--------",
        f"- Full SAXPY stream over {TOTAL_ELEMENTS} elements and {NUM_PASSES} passes",
        "- DCache side: read x, read y, write y",
        "- SMEM side: global.ld x/y, sh.ld x/y, sh.st y, global.st y",
        f"- Address stride per vector element: {DCACHE_BLOCK_BYTES} bytes",
        "",
        "Metric",
        "------",
        "- transactions_per_request = dram_transactions / global_requests",
        "- global_requests counts only global-memory requests.",
        "",
        "Edge Summary",
        "------------",
        f"- Lowest x reuse ({first.x_reuse_words}): D$ txn/req={first.dcache_transactions_per_request:.5f}, SMEM txn/req={first.smem_transactions_per_request:.5f}",
        f"- Highest x reuse ({last.x_reuse_words}): D$ txn/req={last.dcache_transactions_per_request:.5f}, SMEM txn/req={last.smem_transactions_per_request:.5f}",
        "",
        _build_table(rows),
        "",
        f"CSV: {CSV_PATH.name}",
        f"Report: {REPORT_PATH.name}",
        f"Plot: {PLOT_PATH.name}",
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def _plot_rows(rows: List[RatioRow], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the benchmark plot."
        ) from exc

    xs = [row.x_reuse_words for row in rows]
    dcache_txn_req = [row.dcache_transactions_per_request for row in rows]
    smem_txn_req = [row.smem_transactions_per_request for row in rows]

    fig, ax = plt.subplots(figsize=(11.8, 6.3))
    fig.patch.set_facecolor("#f5f2eb")
    ax.set_facecolor("#fffdf8")

    ax.plot(
        xs,
        dcache_txn_req,
        color="#c05621",
        linewidth=2.8,
        marker="o",
        markersize=5.0,
        label="DCache txn/request",
    )
    ax.plot(
        xs,
        smem_txn_req,
        color="#1f6f8b",
        linewidth=2.8,
        marker="s",
        markersize=4.8,
        label="SMEM txn/request",
    )
    ax.fill_between(xs, dcache_txn_req, color="#c05621", alpha=0.08)
    ax.fill_between(xs, smem_txn_req, color="#1f6f8b", alpha=0.08)

    ax.grid(True, axis="y", alpha=0.5, color="#d8d2c8", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.2, color="#d8d2c8", linewidth=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8d8579")
    ax.spines["bottom"].set_color("#8d8579")
    ax.tick_params(colors="#2f2a24")

    ax.set_xlabel("Unique x Words in SAXPY Reuse Window", color="#2f2a24")
    ax.set_ylabel("DRAM Transactions per Global Request", color="#2f2a24")
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "SAXPY Global Transaction Intensity: DCache vs SMEM",
        fontsize=16,
        weight="bold",
        color="#2f2a24",
        y=0.96,
    )
    fig.text(
        0.5,
        0.03,
        "Same SAXPY stream, same sweep. Lower txn/request means better global-memory reuse/coalescing.",
        ha="center",
        fontsize=9.3,
        color="#5c5348",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _print_summary(rows: List[RatioRow]) -> None:
    print("SAXPY Global Transactions per Global Request (DCache vs SMEM)")
    print("-------------------------------------------------------------")
    print(_build_table(rows))
    print()
    print(f"CSV saved to:   {CSV_PATH}")
    print(f"Report saved to:{REPORT_PATH}")
    print(f"Plot saved to:  {PLOT_PATH}")


def main() -> None:
    rows = run_benchmark()
    _write_csv(rows, CSV_PATH)
    _write_report(rows, REPORT_PATH)
    _plot_rows(rows, PLOT_PATH)
    _print_summary(rows)


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        main()

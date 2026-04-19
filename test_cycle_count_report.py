#!/usr/bin/env python3
"""
Cycle-count comparison test between:
- simulator/mem/dcache.py
- main.py (SHMEM functional model)

This script prints a formatted per-case report with individual cycle counts.

Run:
    python3 test_cycle_count_report.py | tee output.txt
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from main import ShmemFunctionalSimulator, Transaction, TxnType, load_smem_config
from test_dcache_and_smem import _load_dcache_symbols


DCACHE = _load_dcache_symbols()
DEFAULT_TEST_TB_SIZE_BYTES = 0x100


def _tb_slots(*tbids: Optional[int]) -> Tuple[Optional[int], ...]:
    slots = list(tbids[:4])
    while len(slots) < 4:
        slots.append(None)
    return tuple(slots)


def _tb_kwargs(
    *,
    thread_block_id: int = 0,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
    done: bool = False,
) -> Dict[str, Any]:
    return {
        "thread_block_id": int(thread_block_id),
        "resident_thread_block_ids": tuple(
            resident_thread_block_ids
            if resident_thread_block_ids is not None
            else _tb_slots(int(thread_block_id))
        ),
        "thread_block_done_bits": [1] if done else [0],
    }


def _tb_txn(
    txn_type: TxnType,
    *,
    thread_block_id: int = 0,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
    done: bool = False,
    **kwargs: Any,
) -> Transaction:
    return Transaction(
        txn_type=txn_type,
        **kwargs,
        **_tb_kwargs(
            thread_block_id=thread_block_id,
            resident_thread_block_ids=resident_thread_block_ids,
            done=done,
        ),
    )


@dataclass
class DCacheRunResult:
    completion_cycle: int
    completion_type: str
    accept_cycle: Optional[int]
    data: Optional[int]


@dataclass
class SmemRunResult:
    completion_cycle: int
    txn_type: str
    read_data: Optional[int]
    cycle_count: int


@dataclass
class CycleCase:
    name: str
    note: str
    run_dcache: Callable[[], DCacheRunResult]
    run_smem: Callable[[], SmemRunResult]
    expect_dcache_completion: str
    expect_smem_txn_type: str


@dataclass
class _MemResp:
    warp_id: int
    packet: Any = None
    status: Any = None


class _PacketWords:
    def __init__(self, words: List[int]):
        self._bytes = b"".join(
            int(word & 0xFFFF_FFFF).to_bytes(4, byteorder="little", signed=False)
            for word in words
        )

    def tobytes(self) -> bytes:
        return self._bytes


def _format_optional_hex(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return f"0x{int(value) & 0xFFFF_FFFF:08X}"


def _preload_dcache_hit(stage: Any, req: Any, value: int) -> None:
    bank = stage.banks[req.addr.bank_id]
    frame = DCACHE["dCacheFrame"](
        valid=True,
        dirty=False,
        tag=req.addr.tag,
        block=[0] * DCACHE["BLOCK_SIZE_WORDS"],
    )
    frame.block[req.addr.block_offset] = int(value) & 0xFFFF_FFFF
    bank.sets[req.addr.set_index][0] = frame


def _preload_smem_word(
    sim: ShmemFunctionalSimulator,
    *,
    shmem_addr: int,
    value: int,
    thread_id: int = 0,
    thread_block_id: int = 0,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
) -> None:
    probe = _tb_txn(
        TxnType.SH_LD,
        shmem_addr=int(shmem_addr),
        thread_id=int(thread_id),
        thread_block_id=thread_block_id,
        resident_thread_block_ids=resident_thread_block_ids,
    )
    absolute = sim._absolute_smem_addr(probe)
    bank, slot = sim._address_crossbar(
        absolute, sim._effective_thread_block_offset(probe)
    )
    masked = int(value) & sim.word_mask
    sim.banks[bank][slot] = masked
    sim.sram_linear[absolute] = masked


def _run_smem_to_completion(
    txn: Transaction,
    *,
    dram_init: Optional[Dict[int, int]] = None,
    dram_latency_cycles: Optional[int] = None,
    preload: Optional[Callable[[ShmemFunctionalSimulator], None]] = None,
    max_cycles: int = 2000,
) -> SmemRunResult:
    cfg = load_smem_config()
    kwargs = cfg.to_sim_kwargs()
    kwargs["dram_init"] = dram_init or {}
    kwargs.pop("thread_block_offsets", None)
    kwargs["thread_block_size_bytes"] = int(
        kwargs.get("thread_block_size_bytes") or DEFAULT_TEST_TB_SIZE_BYTES
    )
    if dram_latency_cycles is not None:
        kwargs["dram_latency_cycles"] = int(dram_latency_cycles)
    sim = ShmemFunctionalSimulator(**kwargs)
    if preload is not None:
        preload(sim)

    done_before = len(sim.completions)
    sim.issue(txn)

    steps = 0
    while len(sim.completions) == done_before:
        if steps >= max_cycles:
            raise TimeoutError(
                f"SMEM transaction did not complete within {max_cycles} cycles: {txn}"
            )
        sim.step()
        steps += 1

    completion = sim.completions[-1]
    return SmemRunResult(
        completion_cycle=int(completion.cycle_completed),
        txn_type=str(completion.txn_type),
        read_data=completion.read_data,
        cycle_count=sim.get_cycle_count(),
    )


def _run_dcache_to_completion(
    req: Any,
    *,
    preload_hit_value: Optional[int] = None,
    mem_latency_cycles: int = 1,
    memory_words: Optional[Dict[int, int]] = None,
    max_cycles: int = 5000,
) -> DCacheRunResult:
    behind = DCACHE["LatchIF"](name="lsu_to_dcache_cycle")
    mem_req_if = DCACHE["LatchIF"](name="dcache_to_mem_cycle")
    mem_resp_if = DCACHE["LatchIF"](name="mem_to_dcache_cycle")
    dcache_fwd = DCACHE["ForwardingIF"](name="dcache_to_lsu_cycle")

    stage = DCACHE["LockupFreeCacheStage"](
        name="DCacheCycle",
        behind_latch=behind,
        forward_ifs_write={"DCache_LSU_Resp": dcache_fwd},
        mem_req_if=mem_req_if,
        mem_resp_if=mem_resp_if,
    )

    if preload_hit_value is not None:
        _preload_dcache_hit(stage, req, preload_hit_value)

    mem_image = {int(addr): int(word) & 0xFFFF_FFFF for addr, word in (memory_words or {}).items()}
    pending_responses: List[Tuple[int, _MemResp]] = []
    accept_cycle: Optional[int] = None

    behind.push(req)

    for _ in range(max_cycles):
        # Push one ready memory response before this compute call, if any.
        if pending_responses and mem_resp_if.ready_for_push():
            ready_cycle, resp = pending_responses[0]
            # dcache.compute increments cycle first, then checks mem_resp_if.
            if stage.get_cycle_count() >= (ready_cycle - 1):
                mem_resp_if.push(resp)
                pending_responses.pop(0)

        with redirect_stdout(io.StringIO()):
            stage.compute()

        # Harvest requests generated by cache banks and schedule synthetic memory responses.
        if mem_req_if.valid:
            req_payload = mem_req_if.pop()
            issue_cycle = stage.get_cycle_count()
            warp_id = int(req_payload.get("warp", req_payload.get("warp_id", 0)))
            rw_mode = str(req_payload.get("rw_mode", "read")).lower()
            base_addr = int(req_payload.get("addr", 0))
            ready_cycle = int(issue_cycle) + int(mem_latency_cycles)

            if rw_mode == "read":
                words = [
                    int(mem_image.get(base_addr + (i * 4), 0)) & 0xFFFF_FFFF
                    for i in range(DCACHE["BLOCK_SIZE_WORDS"])
                ]
                pending_responses.append(
                    (ready_cycle, _MemResp(warp_id=warp_id, packet=_PacketWords(words)))
                )
            else:
                data_words = req_payload.get("data", [])
                if isinstance(data_words, list):
                    for i, word in enumerate(data_words):
                        mem_image[base_addr + (i * 4)] = int(word) & 0xFFFF_FFFF
                pending_responses.append(
                    (ready_cycle, _MemResp(warp_id=warp_id, status="WRITE_DONE"))
                )

        payload = dcache_fwd.pop()
        if payload is None:
            continue

        payload_type = str(getattr(payload, "type", ""))
        if payload_type == "MISS_ACCEPTED" and accept_cycle is None:
            accept_cycle = stage.get_cycle_count()
            continue

        if payload_type in ("HIT_COMPLETE", "MISS_COMPLETE", "FLUSH_COMPLETE"):
            return DCacheRunResult(
                completion_cycle=stage.get_cycle_count(),
                completion_type=payload_type,
                accept_cycle=accept_cycle,
                data=getattr(payload, "data", None),
            )

    raise TimeoutError(
        f"DCache request did not complete within {max_cycles} cycles: {req}"
    )


def _build_cases() -> List[CycleCase]:
    sh_ld_addr = 0x20
    sh_ld_data = 0x11223344

    sh_st_addr = 0x24
    sh_st_old = 0xDEADBEEF
    sh_st_new = 0xA1B2C3D4

    global_ld_addr = 0x3000
    global_ld_data = 0xCAFED00D

    global_st_shmem_addr = 0x2C
    global_st_dram_addr = 0x3100
    global_st_data = 0x55AA66CC

    mem_latency = 4

    return [
        CycleCase(
            name="sh.ld (hit)",
            note="Direct read hit in both models.",
            run_dcache=lambda: _run_dcache_to_completion(
                DCACHE["dCacheRequest"](
                    addr_val=sh_ld_addr,
                    rw_mode="read",
                    size="word",
                ),
                preload_hit_value=sh_ld_data,
            ),
            run_smem=lambda: _run_smem_to_completion(
                _tb_txn(TxnType.SH_LD, shmem_addr=sh_ld_addr),
                preload=lambda sim: _preload_smem_word(
                    sim,
                    shmem_addr=sh_ld_addr,
                    value=sh_ld_data,
                ),
            ),
            expect_dcache_completion="HIT_COMPLETE",
            expect_smem_txn_type=TxnType.SH_LD.value,
        ),
        CycleCase(
            name="sh.st (hit)",
            note="Write hit; both report prior word in response payload.",
            run_dcache=lambda: _run_dcache_to_completion(
                DCACHE["dCacheRequest"](
                    addr_val=sh_st_addr,
                    rw_mode="write",
                    size="word",
                    store_value=sh_st_new,
                ),
                preload_hit_value=sh_st_old,
            ),
            run_smem=lambda: _run_smem_to_completion(
                _tb_txn(
                    TxnType.SH_ST,
                    shmem_addr=sh_st_addr,
                    write_data=sh_st_new,
                ),
                preload=lambda sim: _preload_smem_word(
                    sim,
                    shmem_addr=sh_st_addr,
                    value=sh_st_old,
                ),
            ),
            expect_dcache_completion="HIT_COMPLETE",
            expect_smem_txn_type=TxnType.SH_ST.value,
        ),
        CycleCase(
            name="global.ld.dram2sram",
            note="Compared to dcache read miss completion path.",
            run_dcache=lambda: _run_dcache_to_completion(
                DCACHE["dCacheRequest"](
                    addr_val=global_ld_addr,
                    rw_mode="read",
                    size="word",
                ),
                mem_latency_cycles=mem_latency,
                memory_words={global_ld_addr: global_ld_data},
            ),
            run_smem=lambda: _run_smem_to_completion(
                _tb_txn(
                    TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=global_ld_addr,
                    shmem_addr=0x28,
                ),
                dram_init={global_ld_addr: global_ld_data},
                dram_latency_cycles=mem_latency,
            ),
            expect_dcache_completion="MISS_COMPLETE",
            expect_smem_txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM.value,
        ),
        CycleCase(
            name="global.st.smem2dram",
            note="Compared to dcache write-miss allocate/complete path.",
            run_dcache=lambda: _run_dcache_to_completion(
                DCACHE["dCacheRequest"](
                    addr_val=global_st_dram_addr,
                    rw_mode="write",
                    size="word",
                    store_value=global_st_data,
                ),
                mem_latency_cycles=mem_latency,
            ),
            run_smem=lambda: _run_smem_to_completion(
                _tb_txn(
                    TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                    dram_addr=global_st_dram_addr,
                    shmem_addr=global_st_shmem_addr,
                ),
                dram_latency_cycles=mem_latency,
                preload=lambda sim: _preload_smem_word(
                    sim,
                    shmem_addr=global_st_shmem_addr,
                    value=global_st_data,
                ),
            ),
            expect_dcache_completion="MISS_COMPLETE",
            expect_smem_txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM.value,
        ),
    ]


def generate_cycle_report_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in _build_cases():
        dcache_result = case.run_dcache()
        smem_result = case.run_smem()

        if dcache_result.completion_type != case.expect_dcache_completion:
            raise AssertionError(
                f"{case.name}: dcache completion type mismatch: "
                f"{dcache_result.completion_type} != {case.expect_dcache_completion}"
            )
        if smem_result.txn_type != case.expect_smem_txn_type:
            raise AssertionError(
                f"{case.name}: smem completion type mismatch: "
                f"{smem_result.txn_type} != {case.expect_smem_txn_type}"
            )

        dcache_timeline = (
            f"accept@{dcache_result.accept_cycle}, done@{dcache_result.completion_cycle}"
            if dcache_result.accept_cycle is not None
            else f"done@{dcache_result.completion_cycle}"
        )

        rows.append(
            {
                "case": case.name,
                "dcache_cycles": int(dcache_result.completion_cycle),
                "smem_cycles": int(smem_result.cycle_count),
                "delta_smem_minus_dcache": int(smem_result.cycle_count)
                - int(dcache_result.completion_cycle),
                "dcache_result": (
                    f"{dcache_result.completion_type} "
                    f"(data={_format_optional_hex(dcache_result.data)})"
                ),
                "smem_result": (
                    f"{smem_result.txn_type} "
                    f"(data={_format_optional_hex(smem_result.read_data)})"
                ),
                "dcache_timeline": dcache_timeline,
                "note": case.note,
            }
        )
    return rows


def print_cycle_report(rows: List[Dict[str, Any]]) -> None:
    title = "Cycle Count Comparison: DCache vs SHMEM (main.py)"
    print(title)
    print("=" * len(title))
    print(
        f"{'Case':<22} {'DCache':>8} {'SHMEM':>8} {'Delta':>8} "
        f"{'DCache Timeline':<22} {'DCache Result':<32} {'SHMEM Result':<30}"
    )
    print("-" * 142)
    for row in rows:
        print(
            f"{row['case']:<22} "
            f"{row['dcache_cycles']:>8} "
            f"{row['smem_cycles']:>8} "
            f"{row['delta_smem_minus_dcache']:>8} "
            f"{row['dcache_timeline']:<22} "
            f"{row['dcache_result']:<32} "
            f"{row['smem_result']:<30}"
        )
        print(f"  note: {row['note']}")
    print("-" * 142)
    avg_delta = sum(row["delta_smem_minus_dcache"] for row in rows) / max(len(rows), 1)
    print(f"Average delta (SHMEM - DCache): {avg_delta:.2f} cycles")


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        report_rows = generate_cycle_report_rows()
        print_cycle_report(report_rows)

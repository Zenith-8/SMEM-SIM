#!/usr/bin/env python3
"""
Back-to-back sh.ld / sh.st throughput characterization for the SMEM simulator.

Motivation
----------
The SMEM model has *separate* read and write controllers that both run every
cycle. That means a stream of pure ``sh.ld`` (or pure ``sh.st``) is
bottlenecked at 1 transaction / cycle per controller, but a stream that mixes
the two can reach 2 transactions / cycle because the read controller and the
write controller retire one request each per cycle independently.

Our previous combined-warp workload serialized each warp-wide instruction
(it drained every instruction before issuing the next), which prevented the
read and write controllers from ever being busy in the same cycle. This file
measures the *true* back-to-back throughput of sh.ld / sh.st by issuing long
instruction chains into the simulator without any barriers between them.

Scenarios
---------
* Scenario A -- verbose cycle walkthrough:
    Two warps (32 threads each) issue ``sh.ld`` then ``sh.st`` back-to-back,
    with no drain between them. The verbose per-cycle dump shows the read
    controller and the write controller retiring in parallel.

* Scenario B -- pure sh.ld chain: K warp-wide sh.ld instructions back-to-back.
* Scenario C -- pure sh.st chain: K warp-wide sh.st instructions back-to-back.
* Scenario D -- strictly alternating sh.ld / sh.st chain: the simulator
  should retire two transactions per cycle at steady state.
* Scenario E -- realistic mix (LD,LD,ST,LD,ST,ST,LD,ST pattern) to show
  partial overlap throughput.

Run:
    python3 test_smem_back_to_back_throughput.py

Output is appended to ``output_extended.txt`` via ``capture_to_extended_log``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from main import (
    ShmemFunctionalSimulator,
    SmemArbiter,
    Transaction,
    TxnType,
    _expand_thread_offsets_to_num_threads,
    load_smem_config,
)


NUM_THREADS_PER_WARP: int = 32
WORD_BYTES: int = 4
DEFAULT_CHAIN_LENGTH: int = 8
INTERLEAVED_CHAIN_PAIRS: int = 8

COMPUTE_XOR_MASK: int = 0x0000_ABCD
SH_ST_SEED: int = 0xD000_0000


@dataclass
class WarpInstruction:
    """One warp-wide instruction: ``NUM_THREADS_PER_WARP`` per-thread transactions."""

    name: str
    txn_type: TxnType
    transactions: List[Transaction]


@dataclass
class ChainResult:
    """Throughput result for a single back-to-back instruction chain."""

    label: str
    instruction_names: List[str]
    total_transactions: int
    total_cycles: int

    @property
    def throughput_txns_per_cycle(self) -> float:
        return float(self.total_transactions) / float(max(self.total_cycles, 1))


def _build_sim(
    *,
    num_threads: int,
    verbose: bool = False,
    preload_words: Optional[Dict[int, int]] = None,
) -> ShmemFunctionalSimulator:
    """Build a simulator using ``.config`` defaults plus an optional SMEM preload.

    Preloaded words land in both the bank array and the linear SRAM so sh.ld
    reads return the intended value instead of zero.
    """
    cfg = load_smem_config()
    kwargs = cfg.to_sim_kwargs()
    kwargs["num_threads"] = int(num_threads)
    kwargs["thread_block_offsets"] = _expand_thread_offsets_to_num_threads(
        kwargs.get("thread_block_offsets"), int(num_threads)
    )
    kwargs["verbose"] = bool(verbose)
    sim = ShmemFunctionalSimulator(**kwargs)

    if preload_words:
        for shmem_addr, value in preload_words.items():
            probe = Transaction(txn_type=TxnType.SH_LD, shmem_addr=int(shmem_addr))
            absolute = sim._absolute_smem_addr(probe)
            bank, slot = sim._address_crossbar(
                absolute, sim._effective_thread_block_offset(probe)
            )
            masked = int(value) & sim.word_mask
            sim.banks[bank][slot] = masked
            sim.sram_linear[absolute] = masked

    return sim


def _build_sh_ld_instruction(sequence_index: int) -> WarpInstruction:
    """One warp-wide ``sh.ld`` with no intra-warp bank conflicts (lane = bank)."""
    return WarpInstruction(
        name=f"sh.ld#{sequence_index}",
        txn_type=TxnType.SH_LD,
        transactions=[
            Transaction(
                txn_type=TxnType.SH_LD,
                thread_id=lane,
                shmem_addr=lane * WORD_BYTES,
            )
            for lane in range(NUM_THREADS_PER_WARP)
        ],
    )


def _build_sh_st_instruction(sequence_index: int) -> WarpInstruction:
    """One warp-wide ``sh.st`` with no intra-warp bank conflicts (lane = bank)."""
    return WarpInstruction(
        name=f"sh.st#{sequence_index}",
        txn_type=TxnType.SH_ST,
        transactions=[
            Transaction(
                txn_type=TxnType.SH_ST,
                thread_id=lane,
                shmem_addr=lane * WORD_BYTES,
                write_data=(SH_ST_SEED + sequence_index * 0x100 + lane)
                ^ COMPUTE_XOR_MASK,
            )
            for lane in range(NUM_THREADS_PER_WARP)
        ],
    )


def _build_preload_for_sh_ld() -> Dict[int, int]:
    """Seed the SMEM banks so pure-sh.ld chains observe meaningful data."""
    return {
        lane * WORD_BYTES: 0xA000_0000 + lane
        for lane in range(NUM_THREADS_PER_WARP)
    }


def _run_chain_raw(
    instructions: Sequence[WarpInstruction],
    *,
    preload: bool,
    verbose: bool = False,
) -> Tuple[int, int, ShmemFunctionalSimulator]:
    """
    Issue every per-thread transaction of every warp-wide instruction into the
    simulator up-front, then drain. This is the "no software barrier" mode:
    the simulator's own arbiter + controllers pipeline the chain end-to-end.

    Returns the cycle count, the total transaction count, and the simulator
    (for optional completion/trace inspection).
    """
    sim = _build_sim(
        num_threads=NUM_THREADS_PER_WARP,
        verbose=verbose,
        preload_words=_build_preload_for_sh_ld() if preload else None,
    )

    total_txns = 0
    for instr in instructions:
        for txn in instr.transactions:
            sim.issue(txn)
            total_txns += 1

    while sim._has_pending_work():
        sim.step()

    return int(sim.cycle), total_txns, sim


def _run_chain_with_arbiter_verbose(
    instructions: Sequence[WarpInstruction],
    *,
    preload: bool,
) -> Tuple[int, int, ShmemFunctionalSimulator]:
    """
    Issue every warp-wide instruction through ``SmemArbiter.process_batch`` so
    the per-sub-batch ``[DEBUG]`` trace is emitted alongside the verbose
    cycle dump. Back-to-back warp instructions are issued without any
    intervening drain, so the read and write controllers can overlap.
    """
    sim = _build_sim(
        num_threads=NUM_THREADS_PER_WARP,
        verbose=True,
        preload_words=_build_preload_for_sh_ld() if preload else None,
    )
    arbiter = SmemArbiter(sim)

    total_txns = 0
    for instr in instructions:
        arbiter.process_batch(list(instr.transactions))
        total_txns += len(instr.transactions)

    while sim._has_pending_work():
        sim.step()

    return int(sim.cycle), total_txns, sim


def _dump_completions(sim: ShmemFunctionalSimulator) -> None:
    """Mirror the ``Completions:`` + ``Traceback:`` dump used by main.py."""
    print("Completions:")
    for completion in sim.snapshot()["completions"]:
        print(completion)
        print("  Traceback:")
        for trace_line in completion["trace"]:
            print(f"    {trace_line}")


def _print_chain_summary(result: ChainResult) -> None:
    print()
    print("-" * 80)
    print(f"  {result.label}")
    print("-" * 80)
    print(f"  Instructions issued back-to-back: {len(result.instruction_names)}")
    print(f"  Per-thread transactions:          {result.total_transactions}")
    print(f"  Total cycles:                     {result.total_cycles}")
    print(
        f"  Steady-state throughput:          "
        f"{result.throughput_txns_per_cycle:.3f} txn/cycle"
    )


def _print_scenario_table(results: Sequence[ChainResult]) -> None:
    print()
    print("=" * 90)
    print("Back-to-back sh.ld / sh.st throughput summary")
    print("=" * 90)
    header = (
        f"{'Scenario':<50} | {'Instr':>5} | {'Txns':>5} | {'Cycles':>6} | {'Txn/Cyc':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.label:<50} | "
            f"{len(result.instruction_names):>5} | "
            f"{result.total_transactions:>5} | "
            f"{result.total_cycles:>6} | "
            f"{result.throughput_txns_per_cycle:>8.3f}"
        )
    print()
    print(
        "Expected peak:  pure sh.ld or sh.st = 1.0 txn/cyc "
        "(single controller bottleneck),\n"
        "                alternating sh.ld/sh.st = 2.0 txn/cyc "
        "(read + write controllers retire one each per cycle)."
    )


def demo_verbose_back_to_back() -> None:
    """Scenario A: verbose trace of sh.ld -> sh.st back-to-back (no drain).

    Uses a tiny warp count (1 warp x 2 instructions = 64 per-thread txns) so
    the per-cycle queue dump stays readable while still showing the read and
    write controllers retiring in the same cycle.
    """
    print()
    print("=" * 80)
    print("=== SCENARIO A: verbose back-to-back sh.ld -> sh.st (1 warp, no drain) ===")
    print("=" * 80)
    print(
        "One 32-thread warp issues a full sh.ld instruction immediately followed "
        "by a full sh.st instruction. The verbose cycle dump below should show "
        "the read controller and the write controller retiring a transaction "
        "each in the same cycle, yielding ~2 txn/cycle steady state."
    )

    instructions: List[WarpInstruction] = [
        _build_sh_ld_instruction(sequence_index=0),
        _build_sh_st_instruction(sequence_index=0),
    ]

    cycles, total_txns, sim = _run_chain_with_arbiter_verbose(
        instructions, preload=True
    )
    _dump_completions(sim)

    result = ChainResult(
        label="A. Verbose sh.ld -> sh.st (1 warp, 2 instructions)",
        instruction_names=[instr.name for instr in instructions],
        total_transactions=total_txns,
        total_cycles=cycles,
    )
    _print_chain_summary(result)
    print(
        f"NOTE: Scenario A overlaps only the transition window between the two "
        f"instructions; the bulk of each instruction still runs single-controller, "
        f"so end-to-end throughput on this short chain is dominated by that phase."
    )


def demo_pure_sh_ld_chain(chain_length: int = DEFAULT_CHAIN_LENGTH) -> ChainResult:
    """Scenario B: pure sh.ld chain — read controller at peak."""
    print()
    print("=" * 80)
    print(
        f"=== SCENARIO B: pure sh.ld x {chain_length} warp-wide instructions "
        f"({chain_length * NUM_THREADS_PER_WARP} txns total) ==="
    )
    print("=" * 80)
    instructions = [
        _build_sh_ld_instruction(sequence_index=i) for i in range(chain_length)
    ]
    cycles, total_txns, _sim = _run_chain_raw(instructions, preload=True)

    result = ChainResult(
        label=f"B. Pure sh.ld x{chain_length} ({total_txns} txns)",
        instruction_names=[instr.name for instr in instructions],
        total_transactions=total_txns,
        total_cycles=cycles,
    )
    _print_chain_summary(result)
    print(
        "Pure sh.ld chains can only use the SMEM read controller, which retires "
        "one request per cycle => ~1.0 txn/cycle steady state."
    )
    return result


def demo_pure_sh_st_chain(chain_length: int = DEFAULT_CHAIN_LENGTH) -> ChainResult:
    """Scenario C: pure sh.st chain — write controller at peak."""
    print()
    print("=" * 80)
    print(
        f"=== SCENARIO C: pure sh.st x {chain_length} warp-wide instructions "
        f"({chain_length * NUM_THREADS_PER_WARP} txns total) ==="
    )
    print("=" * 80)
    instructions = [
        _build_sh_st_instruction(sequence_index=i) for i in range(chain_length)
    ]
    cycles, total_txns, _sim = _run_chain_raw(instructions, preload=False)

    result = ChainResult(
        label=f"C. Pure sh.st x{chain_length} ({total_txns} txns)",
        instruction_names=[instr.name for instr in instructions],
        total_transactions=total_txns,
        total_cycles=cycles,
    )
    _print_chain_summary(result)
    print(
        "Pure sh.st chains can only use the SMEM write controller, which retires "
        "one request per cycle => ~1.0 txn/cycle steady state."
    )
    return result


def demo_alternating_ld_st_chain(
    pair_count: int = INTERLEAVED_CHAIN_PAIRS,
) -> ChainResult:
    """Scenario D: strictly alternating sh.ld / sh.st — both controllers busy."""
    total_instructions = pair_count * 2
    print()
    print("=" * 80)
    print(
        f"=== SCENARIO D: alternating sh.ld / sh.st x {pair_count} pairs "
        f"({total_instructions} instructions, "
        f"{total_instructions * NUM_THREADS_PER_WARP} txns total) ==="
    )
    print("=" * 80)
    instructions: List[WarpInstruction] = []
    for i in range(pair_count):
        instructions.append(_build_sh_ld_instruction(sequence_index=i))
        instructions.append(_build_sh_st_instruction(sequence_index=i))

    cycles, total_txns, _sim = _run_chain_raw(instructions, preload=True)

    result = ChainResult(
        label=f"D. Alternating sh.ld/sh.st x{pair_count} pairs ({total_txns} txns)",
        instruction_names=[instr.name for instr in instructions],
        total_transactions=total_txns,
        total_cycles=cycles,
    )
    _print_chain_summary(result)
    print(
        "Alternating chains let the read and write controllers each retire one "
        "request per cycle in parallel => ~2.0 txn/cycle steady state."
    )
    return result


def demo_realistic_mix() -> ChainResult:
    """Scenario E: realistic 2:3 read:write mix (LD,LD,ST,LD,ST,ST,LD,ST)."""
    print()
    print("=" * 80)
    print("=== SCENARIO E: realistic mix LD,LD,ST,LD,ST,ST,LD,ST (8 warp-wide) ===")
    print("=" * 80)

    pattern = ["ld", "ld", "st", "ld", "st", "st", "ld", "st"]
    instructions: List[WarpInstruction] = []
    ld_idx = 0
    st_idx = 0
    for kind in pattern:
        if kind == "ld":
            instructions.append(_build_sh_ld_instruction(sequence_index=ld_idx))
            ld_idx += 1
        else:
            instructions.append(_build_sh_st_instruction(sequence_index=st_idx))
            st_idx += 1

    cycles, total_txns, _sim = _run_chain_raw(instructions, preload=True)

    result = ChainResult(
        label=f"E. Realistic LD,LD,ST,LD,ST,ST,LD,ST ({total_txns} txns)",
        instruction_names=[instr.name for instr in instructions],
        total_transactions=total_txns,
        total_cycles=cycles,
    )
    _print_chain_summary(result)
    print(
        "Mixed patterns fall between the 1.0 and 2.0 txn/cycle bounds depending "
        "on how many cycles have both a pending read and a pending write."
    )
    return result


def run_all_scenarios() -> None:
    """Run every back-to-back throughput scenario and print the summary table."""
    demo_verbose_back_to_back()
    results: List[ChainResult] = []
    results.append(demo_pure_sh_ld_chain())
    results.append(demo_pure_sh_st_chain())
    results.append(demo_alternating_ld_st_chain())
    results.append(demo_realistic_mix())
    _print_scenario_table(results)


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        run_all_scenarios()

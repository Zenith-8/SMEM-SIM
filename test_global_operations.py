#!/usr/bin/env python3
"""
Comprehensive regression tests for the global transaction path of the SMEM
functional simulator in ``main.py``.

These tests target the two global transaction types:
    - ``global.ld.dram2sram`` (DRAM -> SMEM global loads, a.k.a. ``ldglobal``)
    - ``global.st.smem2dram`` (SMEM -> DRAM global stores, a.k.a. ``stglobal``)

They exercise:
    - Alias parsing on ``TxnType.from_user_value`` and ``Transaction.from_dict``
    - Validation errors for missing ``dram_addr`` / ``shmem_addr``
    - Round-trip data correctness (DRAM <-> SMEM) including zero-initialized reads
    - DRAM-latency-driven completion timing for both global load and store
    - Value-snapshot semantics of ``global.st`` (captured at read-controller time)
    - Pipelining of multiple in-flight global loads / stores
    - Per-thread ``thread_block_offset`` behavior for global ops
    - Global + sync interleavings end-to-end
    - Completion record field population (``dram_addr``, ``read_data``, trace)
    - ``run_single_smem_transaction`` end-to-end coverage for global ops
    - ``ShmemCompatibleCacheStage`` emits ``MISS_COMPLETE`` for global ops

Run:
    python -m unittest test_global_operations -v
or:
    python test_global_operations.py
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from main import (
    ShmemCompatibleCacheStage,
    ShmemFunctionalSimulator,
    SmemArbiter,
    Transaction,
    TxnType,
    _sim_from_config,
    run_single_smem_transaction,
    run_smem_functional_sim,
)


DEFAULT_NUM_BANKS = 32
DEFAULT_WORD_BYTES = 4
DEFAULT_ARBITER_ISSUE_WIDTH = 4


def _completions_by_id(result: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Index the list of completion dicts by their ``txn_id`` for easy lookup."""
    return {int(c["txn_id"]): c for c in result["completions"]}


def _sim(**overrides: Any) -> ShmemFunctionalSimulator:
    """Construct a deterministic simulator instance for unit tests.

    Uses defaults that ignore ``.config`` so the tests remain stable regardless
    of repo-level tuning.
    """
    kwargs: Dict[str, Any] = {
        "num_banks": DEFAULT_NUM_BANKS,
        "word_bytes": DEFAULT_WORD_BYTES,
        "dram_latency_cycles": 1,
        "arbiter_issue_width": DEFAULT_ARBITER_ISSUE_WIDTH,
        "num_threads": 1,
        "thread_block_offsets": [0],
    }
    kwargs.update(overrides)
    return ShmemFunctionalSimulator(**kwargs)


class TestGlobalTxnTypeParsing(unittest.TestCase):
    """Verify user-facing aliases resolve to the correct global ``TxnType``."""

    def test_parses_global_ld_canonical(self) -> None:
        self.assertEqual(
            TxnType.from_user_value("global.ld.dram2sram"),
            TxnType.GLOBAL_LD_DRAM_TO_SRAM,
        )

    def test_parses_ldglobal_alias(self) -> None:
        self.assertEqual(
            TxnType.from_user_value("ldglobal"),
            TxnType.GLOBAL_LD_DRAM_TO_SRAM,
        )

    def test_parses_stglobal_alias(self) -> None:
        self.assertEqual(
            TxnType.from_user_value("stglobal"),
            TxnType.GLOBAL_ST_SMEM_TO_DRAM,
        )

    def test_parses_global_load_dram_to_sram_verbose_alias(self) -> None:
        self.assertEqual(
            TxnType.from_user_value("global_load_dram_to_sram"),
            TxnType.GLOBAL_LD_DRAM_TO_SRAM,
        )

    def test_parses_global_st_canonical(self) -> None:
        self.assertEqual(
            TxnType.from_user_value("global.st.smem2dram"),
            TxnType.GLOBAL_ST_SMEM_TO_DRAM,
        )

    def test_parses_global_store_shmem_to_dram_verbose_alias(self) -> None:
        self.assertEqual(
            TxnType.from_user_value("global_store_shmem_to_dram"),
            TxnType.GLOBAL_ST_SMEM_TO_DRAM,
        )

    def test_rejects_unknown_global_variant(self) -> None:
        with self.assertRaises(ValueError):
            TxnType.from_user_value("global.prefetch")

    def test_transaction_from_dict_accepts_ldglobal_alias(self) -> None:
        txn = Transaction.from_dict(
            {"type": "ldglobal", "dram_addr": 0x1000, "shmem_addr": 0x20}
        )
        self.assertEqual(txn.txn_type, TxnType.GLOBAL_LD_DRAM_TO_SRAM)
        self.assertEqual(txn.dram_addr, 0x1000)
        self.assertEqual(txn.shmem_addr, 0x20)


class TestGlobalTransactionValidation(unittest.TestCase):
    """Exercise ``_validate_transaction`` for global-specific required fields."""

    def test_global_load_without_dram_addr_raises(self) -> None:
        sim = _sim()
        with self.assertRaisesRegex(ValueError, "dram_addr"):
            sim.issue(
                Transaction(
                    txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    shmem_addr=0x20,
                )
            )

    def test_global_load_without_shmem_addr_raises(self) -> None:
        sim = _sim()
        with self.assertRaisesRegex(ValueError, "shmem_addr"):
            sim.issue(
                Transaction(
                    txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=0x1000,
                )
            )

    def test_global_store_without_dram_addr_raises(self) -> None:
        sim = _sim()
        with self.assertRaisesRegex(ValueError, "dram_addr"):
            sim.issue(
                Transaction(
                    txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                    shmem_addr=0x20,
                )
            )

    def test_global_store_without_shmem_addr_raises(self) -> None:
        sim = _sim()
        with self.assertRaisesRegex(ValueError, "shmem_addr"):
            sim.issue(
                Transaction(
                    txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                    dram_addr=0x1000,
                )
            )


class TestGlobalStoreSmemToDram(unittest.TestCase):
    """Behavioral tests for the ``global.st.smem2dram`` path."""

    def test_global_store_commits_smem_value_to_dram(self) -> None:
        result = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xCAFEBABE},
                {
                    "type": "global.st.smem2dram",
                    "shmem_addr": 0x20,
                    "dram_addr": 0x1000,
                },
            ]
        )
        self.assertEqual(result["dram"][0x1000], 0xCAFEBABE)

    def test_global_store_from_uninitialized_smem_writes_zero(self) -> None:
        result = run_smem_functional_sim(
            [
                {
                    "type": "global.st.smem2dram",
                    "shmem_addr": 0x80,
                    "dram_addr": 0x2000,
                }
            ]
        )
        self.assertEqual(result["dram"][0x2000], 0x0)

    def test_global_store_completion_metadata_is_populated(self) -> None:
        result = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xABCD},
                {
                    "type": "global.st.smem2dram",
                    "shmem_addr": 0x20,
                    "dram_addr": 0x1100,
                },
            ]
        )
        completions = _completions_by_id(result)
        global_completion = completions[2]

        self.assertEqual(global_completion["txn_type"], "global.st.smem2dram")
        self.assertEqual(global_completion["status"], "ok")
        self.assertEqual(global_completion["dram_addr"], 0x1100)
        self.assertEqual(global_completion["shmem_addr"], 0x20)
        self.assertIsNone(
            global_completion["read_data"],
            "global.st completions must not carry user-visible read_data.",
        )
        self.assertTrue(
            any("AXI write" in line for line in global_completion["trace"]),
            "Global store trace should reference the AXI write path.",
        )

    def test_global_store_respects_dram_latency(self) -> None:
        for latency in (1, 2, 5):
            with self.subTest(latency=latency):
                result = run_smem_functional_sim(
                    [
                        {
                            "type": "global.st.smem2dram",
                            "shmem_addr": 0x20,
                            "dram_addr": 0x1000,
                        }
                    ],
                    dram_latency_cycles=latency,
                )
                completion = _completions_by_id(result)[1]
                delta = completion["cycle_completed"] - completion["cycle_issued"]
                self.assertGreaterEqual(
                    delta,
                    latency,
                    f"Global store should take at least dram_latency_cycles="
                    f"{latency} to commit, observed delta={delta}.",
                )

    def test_global_store_value_is_snapshotted_at_issue_time(self) -> None:
        """A subsequent ``sh.st`` to the same SMEM address must not mutate the
        value already captured by an in-flight ``global.st``."""
        result = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0x111},
                {
                    "type": "global.st.smem2dram",
                    "shmem_addr": 0x20,
                    "dram_addr": 0x2000,
                },
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0x222},
            ],
            dram_latency_cycles=5,
        )
        self.assertEqual(result["dram"][0x2000], 0x111)
        completions = _completions_by_id(result)
        final_sh_st = completions[3]
        self.assertEqual(
            final_sh_st["txn_type"],
            "sh.st",
            "Sanity: the third transaction should be a sh.st completion.",
        )

    def test_multiple_global_stores_pipeline_and_preserve_values(self) -> None:
        """Multiple global stores pipeline through the AXI write port and
        each commits the correct per-transaction value to DRAM."""
        prep: List[Dict[str, Any]] = [
            {"type": "sh.st", "shmem_addr": 0x40 + i * 4, "write_data": 0xA0 + i}
            for i in range(4)
        ]
        global_stores: List[Dict[str, Any]] = [
            {
                "type": "global.st.smem2dram",
                "shmem_addr": 0x40 + i * 4,
                "dram_addr": 0x100 + i * 4,
            }
            for i in range(4)
        ]
        result = run_smem_functional_sim(
            prep + global_stores,
            dram_latency_cycles=3,
        )
        for i in range(4):
            self.assertEqual(
                result["dram"][0x100 + i * 4],
                0xA0 + i,
                f"DRAM at 0x{0x100 + i * 4:x} should hold 0x{0xA0 + i:x}",
            )

    def test_global_store_per_thread_offset_reads_correct_slot(self) -> None:
        """Two threads with distinct block offsets must store the value that
        lives at each thread's own absolute SMEM address."""
        result = run_smem_functional_sim(
            [
                {
                    "type": "sh.st",
                    "thread_id": 0,
                    "shmem_addr": 0x10,
                    "write_data": 0xAAAA,
                },
                {
                    "type": "sh.st",
                    "thread_id": 1,
                    "shmem_addr": 0x10,
                    "write_data": 0xBBBB,
                },
                {
                    "type": "global.st.smem2dram",
                    "thread_id": 0,
                    "shmem_addr": 0x10,
                    "dram_addr": 0x3000,
                },
                {
                    "type": "global.st.smem2dram",
                    "thread_id": 1,
                    "shmem_addr": 0x10,
                    "dram_addr": 0x3100,
                },
            ],
            num_threads=2,
            thread_block_offsets={0: 0x000, 1: 0x200},
        )
        self.assertEqual(result["dram"][0x3000], 0xAAAA)
        self.assertEqual(result["dram"][0x3100], 0xBBBB)


class TestGlobalLoadDramToSram(unittest.TestCase):
    """Behavioral tests for the ``global.ld.dram2sram`` path."""

    def test_global_load_populates_sram_from_dram(self) -> None:
        result = run_smem_functional_sim(
            [
                {
                    "type": "global.ld.dram2sram",
                    "dram_addr": 0x1000,
                    "shmem_addr": 0x24,
                },
                {"type": "sh.ld", "shmem_addr": 0x24},
            ],
            dram_init={0x1000: 0x1234ABCD},
        )
        completions = _completions_by_id(result)
        self.assertEqual(completions[2]["read_data"], 0x1234ABCD)

    def test_global_load_missing_dram_key_loads_zero(self) -> None:
        result = run_smem_functional_sim(
            [
                {
                    "type": "global.ld.dram2sram",
                    "dram_addr": 0x9999,
                    "shmem_addr": 0x40,
                },
                {"type": "sh.ld", "shmem_addr": 0x40},
            ],
        )
        completions = _completions_by_id(result)
        self.assertEqual(completions[2]["read_data"], 0)

    def test_global_load_overwrites_existing_sram_value(self) -> None:
        result = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xDEAD},
                {
                    "type": "global.ld.dram2sram",
                    "dram_addr": 0x1000,
                    "shmem_addr": 0x20,
                },
                {"type": "sh.ld", "shmem_addr": 0x20},
            ],
            dram_init={0x1000: 0xBEEF},
        )
        completions = _completions_by_id(result)
        self.assertEqual(completions[3]["read_data"], 0xBEEF)

    def test_global_load_completion_metadata_is_populated(self) -> None:
        result = run_smem_functional_sim(
            [
                {
                    "type": "global.ld.dram2sram",
                    "dram_addr": 0x1000,
                    "shmem_addr": 0x24,
                },
            ],
            dram_init={0x1000: 0xAAAA5555},
        )
        completion = _completions_by_id(result)[1]

        self.assertEqual(completion["txn_type"], "global.ld.dram2sram")
        self.assertEqual(completion["status"], "ok")
        self.assertEqual(completion["dram_addr"], 0x1000)
        self.assertEqual(completion["shmem_addr"], 0x24)
        self.assertIsNone(
            completion["read_data"],
            "global.ld completions write into SMEM and must not expose read_data.",
        )
        self.assertTrue(
            any("AXI read" in line for line in completion["trace"]),
            "Global load trace should reference the AXI read path.",
        )

    def test_global_load_respects_dram_latency(self) -> None:
        for latency in (1, 3, 7):
            with self.subTest(latency=latency):
                result = run_smem_functional_sim(
                    [
                        {
                            "type": "global.ld.dram2sram",
                            "dram_addr": 0x1000,
                            "shmem_addr": 0x20,
                        }
                    ],
                    dram_init={0x1000: 0xFEED},
                    dram_latency_cycles=latency,
                )
                completion = _completions_by_id(result)[1]
                delta = completion["cycle_completed"] - completion["cycle_issued"]
                self.assertGreaterEqual(
                    delta,
                    latency,
                    f"Global load should take at least dram_latency_cycles="
                    f"{latency} cycles, observed delta={delta}.",
                )

    def test_multiple_global_loads_pipeline_outstanding_requests(self) -> None:
        """Four global loads issued together should finish within roughly
        ``latency + N`` cycles, proving they pipeline through DRAM rather than
        serializing end-to-end."""
        latency = 5
        txns: List[Dict[str, Any]] = [
            {
                "type": "global.ld.dram2sram",
                "thread_id": i,
                "dram_addr": 0x100 + i * 4,
                "shmem_addr": 0x40 + i * 4,
            }
            for i in range(4)
        ]
        dram_init = {0x100: 1, 0x104: 2, 0x108: 3, 0x10C: 4}
        result = run_smem_functional_sim(
            txns,
            dram_init=dram_init,
            dram_latency_cycles=latency,
            num_threads=4,
            thread_block_offsets=[0, 0, 0, 0],
        )

        completions = _completions_by_id(result)
        for txn_id in range(1, 5):
            self.assertEqual(completions[txn_id]["status"], "ok")

        serial_bound = latency * len(txns)
        self.assertLess(
            result["cycle_count"],
            serial_bound,
            f"Expected pipelined global loads to finish well before the serial "
            f"bound of {serial_bound} cycles; took {result['cycle_count']}.",
        )

    def test_global_load_per_thread_offset_targets_correct_slot(self) -> None:
        """An global load that targets a non-trivial per-thread offset must be
        observable by a follow-up ``sh.ld`` from that same thread."""
        result = run_smem_functional_sim(
            [
                {
                    "type": "global.ld.dram2sram",
                    "thread_id": 1,
                    "dram_addr": 0x7000,
                    "shmem_addr": 0x10,
                },
                {"type": "sh.ld", "thread_id": 1, "shmem_addr": 0x10},
                {"type": "sh.ld", "thread_id": 0, "shmem_addr": 0x10},
            ],
            dram_init={0x7000: 0xC0DE},
            num_threads=2,
            thread_block_offsets={0: 0x000, 1: 0x200},
        )
        completions = _completions_by_id(result)
        self.assertEqual(completions[2]["read_data"], 0xC0DE)
        self.assertEqual(
            completions[3]["read_data"],
            0,
            "Thread 0 observes its own offset window; it must see the "
            "uninitialized value at its (thread_id=0) view of shmem 0x10.",
        )


class TestGlobalMixedWithSync(unittest.TestCase):
    """End-to-end scenarios that mix global and sync SMEM traffic.

    Global operations intentionally do NOT act as a barrier for subsequent
    sync transactions. Therefore, to observe an global op's side effect with a
    follow-up sync op, callers must drive the simulator in phases (analogous
    to a software barrier). These tests drive a single simulator instance
    across multiple ``run`` invocations so each global op fully completes
    before the dependent sync op is issued.
    """

    @staticmethod
    def _drain(sim: ShmemFunctionalSimulator) -> None:
        while sim._has_pending_work():
            sim.step()

    def test_global_store_then_global_load_round_trip(self) -> None:
        sim = _sim(dram_latency_cycles=2)

        sim.run(
            [
                Transaction(
                    txn_type=TxnType.SH_ST, shmem_addr=0x20, write_data=0x7777
                ),
                Transaction(
                    txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                    shmem_addr=0x20,
                    dram_addr=0x4000,
                ),
            ]
        )
        self._drain(sim)
        self.assertEqual(sim.dram[0x4000], 0x7777)

        sim.run(
            [
                Transaction(
                    txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=0x4000,
                    shmem_addr=0x24,
                ),
            ]
        )
        self._drain(sim)

        read_completion = sim.run_one(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=0x24)
        )
        self.assertEqual(read_completion["read_data"], 0x7777)

    def test_global_does_not_act_as_barrier_in_single_batch(self) -> None:
        """Regression guard: when issued together in one batch, a trailing
        ``sh.ld`` fires before the preceding ``global.ld`` has populated SMEM.
        This documents the current (correct) no-implicit-barrier semantic."""
        result = run_smem_functional_sim(
            [
                {
                    "type": "global.ld.dram2sram",
                    "dram_addr": 0x9100,
                    "shmem_addr": 0x24,
                },
                {"type": "sh.ld", "shmem_addr": 0x24},
            ],
            dram_init={0x9100: 0x5A5A},
            dram_latency_cycles=5,
        )
        completions = _completions_by_id(result)
        self.assertEqual(
            completions[2]["read_data"],
            0,
            "Without explicit phase-draining, sh.ld races ahead of global.ld.",
        )

    def test_global_ops_do_not_corrupt_interleaved_sh_ld_sh_st(self) -> None:
        """An interleaved sequence maintains data integrity when each global
        stage is allowed to fully complete before subsequent dependent reads.
        """
        sim = _sim(dram_latency_cycles=3)

        sim.run(
            [
                Transaction(
                    txn_type=TxnType.SH_ST, shmem_addr=0x20, write_data=0x100
                ),
                Transaction(
                    txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                    shmem_addr=0x20,
                    dram_addr=0x5000,
                ),
                Transaction(
                    txn_type=TxnType.SH_ST, shmem_addr=0x24, write_data=0x200
                ),
            ]
        )
        self._drain(sim)

        sh_ld_intermediate = sim.run_one(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=0x24)
        )
        self.assertEqual(sh_ld_intermediate["read_data"], 0x200)
        self.assertEqual(sim.dram[0x5000], 0x100)

        sim.run(
            [
                Transaction(
                    txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=0x5000,
                    shmem_addr=0x28,
                ),
            ]
        )
        self._drain(sim)

        sh_ld_final = sim.run_one(
            Transaction(txn_type=TxnType.SH_LD, shmem_addr=0x28)
        )
        self.assertEqual(sh_ld_final["read_data"], 0x100)


class TestRunSingleSmemTransactionGlobal(unittest.TestCase):
    """Cover the single-transaction convenience wrapper for global ops."""

    def test_run_single_global_store(self) -> None:
        result = run_single_smem_transaction(
            "global.st.smem2dram",
            shmem_addr=0x20,
            dram_addr=0x6000,
            num_threads=1,
        )
        completion = result["completion"]
        snapshot = result["snapshot"]

        self.assertEqual(completion["txn_type"], "global.st.smem2dram")
        self.assertEqual(completion["dram_addr"], 0x6000)
        self.assertEqual(completion["status"], "ok")
        self.assertEqual(snapshot["dram"][0x6000], 0x0)

    def test_run_single_global_load(self) -> None:
        result = run_single_smem_transaction(
            "global.ld.dram2sram",
            shmem_addr=0x24,
            dram_addr=0x6100,
            dram_init={0x6100: 0xFACE},
            num_threads=1,
        )
        completion = result["completion"]
        snapshot = result["snapshot"]

        self.assertEqual(completion["txn_type"], "global.ld.dram2sram")
        self.assertEqual(completion["dram_addr"], 0x6100)
        self.assertEqual(completion["shmem_addr"], 0x24)
        absolute = completion["absolute_shmem_addr"]
        self.assertEqual(snapshot["sram_linear"].get(absolute), 0xFACE)


class TestGlobalCompatibilityStage(unittest.TestCase):
    """Smoke test that ``ShmemCompatibleCacheStage`` handles global requests and
    emits ``MISS_COMPLETE`` responses (as required by the surrounding pipeline
    when swapping SMEM in place of the DCache)."""

    def test_global_ops_emit_miss_complete_in_compat_stage(self) -> None:
        from test_dcache_and_smem import _load_dcache_symbols

        dcache_symbols = _load_dcache_symbols()
        behind = dcache_symbols["LatchIF"](name="lsu_to_smem_global")
        fwd = dcache_symbols["ForwardingIF"](name="smem_to_lsu_global")
        stage = ShmemCompatibleCacheStage(
            name="SMEMCompatGlobal",
            behind_latch=behind,
            forward_ifs_write={"DCache_LSU_Resp": fwd},
            mem_req_if=dcache_symbols["LatchIF"](name="unused_req"),
            mem_resp_if=dcache_symbols["LatchIF"](name="unused_resp"),
        )

        global_requests: List[Dict[str, Any]] = [
            {
                "type": "global.st.smem2dram",
                "shmem_addr": 0x20,
                "dram_addr": 0x1100,
            },
            {
                "type": "global.ld.dram2sram",
                "dram_addr": 0x1100,
                "shmem_addr": 0x24,
            },
        ]

        observed_types: List[str] = []
        for req in global_requests:
            behind.push(req)
            for _ in range(32):
                stage.compute()
                payload = fwd.payload
                if payload is not None:
                    observed_types.append(
                        getattr(payload, "type", None)
                        if not isinstance(payload, dict)
                        else payload["type"]
                    )
                    fwd.pop()
                    break
            else:
                self.fail(
                    f"Compat stage never produced a response for global request {req}"
                )

        self.assertEqual(observed_types, ["MISS_COMPLETE", "MISS_COMPLETE"])


# ---------------------------------------------------------------------------
# Enhanced transaction tracking demos
#
# These mirror the style of the demos and tests in ``main.py`` (see
# ``test_32_threads_different_addresses``, ``test_integration_smem_arbiter``,
# etc.). Each demo:
#
#   1. Prints a ``=== TEST: <name> ===`` banner.
#   2. Builds a verbose simulator via ``_sim_from_config`` so every cycle
#      produces a ``CYCLE N [BEGIN]`` / ``[END]`` queue-state dump.
#   3. Issues transactions through ``SmemArbiter`` so the arbiter emits its
#      per-thread ``[DEBUG] Sub-batch K | Cycle C | Thread T | ...`` log.
#   4. Drains the simulator and prints the resulting completion records with
#      the same ``Completions:`` + ``Traceback:`` dump format used by the
#      ``main.py`` demo.
#   5. Prints a terminal ``Completed in X cycles. Sub-batches: ...`` summary.
#
# The demos focus on the global transaction paths (``global.ld.dram2sram`` and
# ``global.st.smem2dram``), which the pre-existing demos in ``main.py`` cover
# only lightly.
# ---------------------------------------------------------------------------


def _dump_completions(sim: ShmemFunctionalSimulator) -> None:
    """Mirror the ``Completions:`` + ``Traceback:`` dump used by ``main.py``.

    Prints every completion record produced so far by ``sim`` in the same
    dict/trace layout that already appears at the top of
    ``output_extended.txt``.
    """
    print("Completions:")
    for completion in sim.snapshot()["completions"]:
        print(completion)
        print("  Traceback:")
        for trace_line in completion["trace"]:
            print(f"    {trace_line}")


def _drain(sim: ShmemFunctionalSimulator) -> None:
    """Step the simulator until every queue and pending DRAM event drains."""
    while sim._has_pending_work():
        sim.step()


def _run_arbiter_phase(
    sim: ShmemFunctionalSimulator,
    transactions: List[Transaction],
) -> Dict[str, Any]:
    """Issue ``transactions`` through a ``SmemArbiter`` and fully drain.

    Returns the partitioning metadata dict from ``SmemArbiter.process_batch``
    so callers can print the familiar ``Sub-batches: M (sizes: [...])``
    summary line at the end.
    """
    arbiter = SmemArbiter(sim)
    result = arbiter.process_batch(transactions)
    _drain(sim)
    return result


def demo_global_single_thread_round_trip() -> None:
    """Trace a full global store -> global load round-trip on a single thread.

    Exercises every global code path (SMEM read controller -> AXI write ->
    DRAM commit -> AXI read -> memory read queue -> SMEM write controller)
    and shows the resulting completion trace for each of the four
    transactions.
    """
    print("\n=== TEST: Global Round-Trip (sh.st -> global.st -> global.ld -> sh.ld) ===")
    sim = _sim_from_config(num_threads=1)

    prep_txns = [
        Transaction(txn_type=TxnType.SH_ST, shmem_addr=0x20, write_data=0xCAFEBABE),
        Transaction(
            txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
            shmem_addr=0x20,
            dram_addr=0x4000,
        ),
    ]
    prep_result = _run_arbiter_phase(sim, prep_txns)

    load_txns = [
        Transaction(
            txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            dram_addr=0x4000,
            shmem_addr=0x24,
        ),
    ]
    load_result = _run_arbiter_phase(sim, load_txns)

    readback_txns = [Transaction(txn_type=TxnType.SH_LD, shmem_addr=0x24)]
    readback_result = _run_arbiter_phase(sim, readback_txns)

    _dump_completions(sim)
    total_sub_batches = (
        prep_result["num_sub_batches"]
        + load_result["num_sub_batches"]
        + readback_result["num_sub_batches"]
    )
    print(
        f"Completed in {sim.cycle} cycles across 3 phases. "
        f"Total sub-batches: {total_sub_batches}"
    )


def demo_global_loads_pipelined_no_conflict() -> None:
    """4 threads issue ``global.ld`` to distinct banks and pipeline through DRAM.

    With no bank conflicts and an arbiter issue width of 4, all four loads
    are accepted in a single sub-batch. DRAM responses stream back
    sequentially through the SMEM write controller.
    """
    print("\n=== TEST: Pipelined Global Loads (4 threads, no bank conflict) ===")
    sim = _sim_from_config(num_threads=4)
    sim.dram.update({0x100: 0xA1, 0x104: 0xB2, 0x108: 0xC3, 0x10C: 0xD4})

    txns: List[Transaction] = [
        Transaction(
            txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            dram_addr=0x100 + i * sim.word_bytes,
            shmem_addr=0x40 + i * sim.word_bytes,
            thread_id=i,
        )
        for i in range(4)
    ]
    result = _run_arbiter_phase(sim, txns)

    _dump_completions(sim)
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Sub-batches: {result['num_sub_batches']} (sizes: {result['sub_batch_sizes']})"
    )


def demo_global_stores_pipelined_no_conflict() -> None:
    """4 threads issue ``global.st`` from distinct banks and pipeline out to DRAM.

    Shows the SMEM read controller handing values off to the memory write
    queue, which the AXI write port drains one-per-cycle into DRAM.
    """
    print("\n=== TEST: Pipelined Global Stores (4 threads, no bank conflict) ===")
    sim = _sim_from_config(num_threads=4)

    prep: List[Transaction] = [
        Transaction(
            txn_type=TxnType.SH_ST,
            shmem_addr=0x40 + i * sim.word_bytes,
            write_data=0xA000 + i,
            thread_id=i,
        )
        for i in range(4)
    ]
    _run_arbiter_phase(sim, prep)

    store_txns: List[Transaction] = [
        Transaction(
            txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
            shmem_addr=0x40 + i * sim.word_bytes,
            dram_addr=0x200 + i * sim.word_bytes,
            thread_id=i,
        )
        for i in range(4)
    ]
    store_result = _run_arbiter_phase(sim, store_txns)

    _dump_completions(sim)
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Global-store sub-batches: {store_result['num_sub_batches']} "
        f"(sizes: {store_result['sub_batch_sizes']})"
    )


def demo_global_store_bank_conflict_divergence() -> None:
    """Global stores from multiple threads to the same bank must be serialized.

    All four threads target SMEM addresses on the same bank, so the arbiter
    must split them into one sub-batch per conflicting request.
    """
    print("\n=== TEST: Global Store Bank-Conflict Divergence (4-way same bank) ===")
    sim = _sim_from_config(num_threads=4)

    prep: List[Transaction] = [
        Transaction(
            txn_type=TxnType.SH_ST,
            shmem_addr=0x00,
            write_data=0x1000 + i,
            thread_id=i,
        )
        for i in range(4)
    ]
    _run_arbiter_phase(sim, prep)

    store_txns: List[Transaction] = [
        Transaction(
            txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
            shmem_addr=0x00,
            dram_addr=0x3000 + i * sim.word_bytes,
            thread_id=i,
        )
        for i in range(4)
    ]
    store_result = _run_arbiter_phase(sim, store_txns)

    _dump_completions(sim)
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Global-store sub-batches: {store_result['num_sub_batches']} "
        f"(sizes: {store_result['sub_batch_sizes']})"
    )


def demo_global_mixed_load_and_store() -> None:
    """Concurrent global loads and stores share the arbiter and DRAM ports.

    Two threads issue ``global.ld`` while two others issue ``global.st`` in
    the same batch. The verbose trace shows the read controller and write
    controller making independent forward progress per cycle.
    """
    print("\n=== TEST: Concurrent Global Loads + Stores (mixed batch) ===")
    sim = _sim_from_config(num_threads=4)
    sim.dram.update({0x700: 0xF00D, 0x704: 0xBEEF})

    prep: List[Transaction] = [
        Transaction(
            txn_type=TxnType.SH_ST,
            shmem_addr=0x40,
            write_data=0xAAAA,
            thread_id=0,
        ),
        Transaction(
            txn_type=TxnType.SH_ST,
            shmem_addr=0x44,
            write_data=0xBBBB,
            thread_id=1,
        ),
    ]
    _run_arbiter_phase(sim, prep)

    mixed_txns: List[Transaction] = [
        Transaction(
            txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
            shmem_addr=0x40,
            dram_addr=0x800,
            thread_id=0,
        ),
        Transaction(
            txn_type=TxnType.GLOBAL_ST_SMEM_TO_DRAM,
            shmem_addr=0x44,
            dram_addr=0x804,
            thread_id=1,
        ),
        Transaction(
            txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            dram_addr=0x700,
            shmem_addr=0x48,
            thread_id=2,
        ),
        Transaction(
            txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            dram_addr=0x704,
            shmem_addr=0x4C,
            thread_id=3,
        ),
    ]
    mixed_result = _run_arbiter_phase(sim, mixed_txns)

    _dump_completions(sim)
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Mixed-global sub-batches: {mixed_result['num_sub_batches']} "
        f"(sizes: {mixed_result['sub_batch_sizes']})"
    )


def demo_global_with_high_dram_latency() -> None:
    """Show how a higher ``dram_latency_cycles`` stretches the global pipeline.

    Builds the simulator manually (bypassing ``.config``) so we can force a
    5-cycle DRAM latency and make the latency-hiding behavior obvious in
    the cycle-by-cycle trace.
    """
    print("\n=== TEST: Global Load with dram_latency_cycles=5 ===")
    sim = ShmemFunctionalSimulator(
        num_banks=DEFAULT_NUM_BANKS,
        word_bytes=DEFAULT_WORD_BYTES,
        dram_latency_cycles=5,
        arbiter_issue_width=DEFAULT_ARBITER_ISSUE_WIDTH,
        num_threads=2,
        thread_block_offsets=[0, 0],
        verbose=True,
    )
    sim.dram.update({0xA00: 0xFEEDFACE, 0xA04: 0xDEADC0DE})

    txns: List[Transaction] = [
        Transaction(
            txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            dram_addr=0xA00,
            shmem_addr=0x50,
            thread_id=0,
        ),
        Transaction(
            txn_type=TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            dram_addr=0xA04,
            shmem_addr=0x54,
            thread_id=1,
        ),
    ]
    result = _run_arbiter_phase(sim, txns)

    _dump_completions(sim)
    print(
        f"Completed in {sim.cycle} cycles with dram_latency_cycles=5. "
        f"Sub-batches: {result['num_sub_batches']} (sizes: {result['sub_batch_sizes']})"
    )


def run_global_tracking_demos() -> None:
    """Run every enhanced-tracking global demo in sequence.

    Intended to be invoked from ``__main__`` with stdout redirected to
    ``output_extended.txt`` (via ``test_output.capture_to_extended_log``),
    so the demos append a consolidated global-focused tracking report to
    the shared log file.
    """
    demo_global_single_thread_round_trip()
    demo_global_loads_pipelined_no_conflict()
    demo_global_stores_pipelined_no_conflict()
    demo_global_store_bank_conflict_divergence()
    demo_global_mixed_load_and_store()
    demo_global_with_high_dram_latency()


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        run_global_tracking_demos()

        print("\n" + "=" * 78)
        print("== Global unit-test assertions")
        print("=" * 78)
        unittest.main(verbosity=2, exit=False)

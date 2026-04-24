#!/usr/bin/env python3
"""
Focused tests for the TBID-based resident thread-block offset system.

These tests exercise the new slot-derived TBO behavior directly:

- resident TBID vector parsing and validation
- done-bit encoding edge cases
- slot-derived offset/address calculation
- replacement rules for resident SMEM slots
- user-facing API reporting of effective TBO
- human-readable formatting that prints the simulator-derived TBO
- a verbose walkthrough trace so the TBID/TBO behavior can be inspected
  functionally in the generated output log
"""

from __future__ import annotations

from dataclasses import asdict
import unittest
from typing import Any, Dict, Optional, Sequence, Tuple

from main import (
    ShmemFunctionalSimulator,
    SmemArbiter,
    Transaction,
    TxnType,
    run_single_smem_transaction,
    run_smem_functional_sim,
)


DEFAULT_NUM_BANKS = 32
DEFAULT_WORD_BYTES = 4
DEFAULT_ARBITER_ISSUE_WIDTH = 32
DEFAULT_TEST_TB_SIZE_BYTES = 0x80
DEFAULT_NUM_THREADS = 4


def _tb_slots(*tbids: Optional[int]) -> Tuple[Optional[int], ...]:
    slots = list(tbids[:4])
    while len(slots) < 4:
        slots.append(None)
    return tuple(slots)


def _tb_kwargs(
    *,
    thread_block_id: int = 0,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
    done_bits: Any = None,
) -> Dict[str, Any]:
    return {
        "thread_block_id": int(thread_block_id),
        "resident_thread_block_ids": tuple(
            resident_thread_block_ids
            if resident_thread_block_ids is not None
            else _tb_slots(int(thread_block_id))
        ),
        "thread_block_done_bits": done_bits,
    }


def _tb_txn(
    txn_type: TxnType,
    *,
    thread_block_id: int = 0,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
    done_bits: Any = None,
    **kwargs: Any,
) -> Transaction:
    return Transaction(
        txn_type=txn_type,
        **kwargs,
        **_tb_kwargs(
            thread_block_id=thread_block_id,
            resident_thread_block_ids=resident_thread_block_ids,
            done_bits=done_bits,
        ),
    )


def _sim(**overrides: Any) -> ShmemFunctionalSimulator:
    kwargs: Dict[str, Any] = {
        "num_banks": DEFAULT_NUM_BANKS,
        "word_bytes": DEFAULT_WORD_BYTES,
        "dram_latency_cycles": 0,
        "arbiter_issue_width": DEFAULT_ARBITER_ISSUE_WIDTH,
        "num_threads": DEFAULT_NUM_THREADS,
        "thread_block_size_bytes": DEFAULT_TEST_TB_SIZE_BYTES,
    }
    kwargs.update(overrides)
    return ShmemFunctionalSimulator(**kwargs)


def _describe_calculated_tbo(
    *,
    resident_ids: Sequence[Optional[int]],
    thread_block_id: int,
    shmem_addr: int,
    thread_id: int = 0,
) -> str:
    probe_sim = _sim()
    probe = _tb_txn(
        TxnType.SH_LD,
        thread_id=thread_id,
        shmem_addr=shmem_addr,
        thread_block_id=thread_block_id,
        resident_thread_block_ids=resident_ids,
        done_bits=[0, 0, 0, 0],
    )
    tbo = int(probe_sim._effective_thread_block_offset(probe))
    absolute = int(probe_sim._absolute_smem_addr(probe))
    bank, slot = probe_sim._address_crossbar(absolute, tbo)
    smem_block_id = probe_sim._smem_block_id_for_transaction(probe)
    return (
        f"tbid={thread_block_id} -> smem_block={smem_block_id}, "
        f"calculated_tbo=0x{tbo:04x}, abs=0x{absolute:04x}, "
        f"bank={bank}, slot={slot}"
    )


def _dump_completions(sim: ShmemFunctionalSimulator) -> None:
    print("Completions:")
    for comp in sim.completions:
        comp_dict = asdict(comp)
        print(comp_dict)
        print("  Traceback:")
        for line in comp.trace:
            print(f"    {line}")


def demo_thread_block_offset_trace() -> None:
    """
    Emit a verbose, step-by-step residency walkthrough similar in spirit to the
    historical ``output_extended.txt`` trace.

    The scenario proves:
    - TBID 33 is placed into resident slot 2, yielding TBO 0x0100.
    - A replacement TBID 55 is blocked until TBID 33 reports all-one done bits.
    - Once TBID 33 is done, TBID 55 reuses the same SMEM slot and therefore the
      same calculated TBO.
    """
    resident_before = (11, 22, 33, 44)
    resident_after = (11, 22, 55, 44)

    print("=== TRACE: Thread-Block Offset Residency Walkthrough ===")
    print(
        "Initial resident vector: "
        f"slot0={resident_before[0]}, slot1={resident_before[1]}, "
        f"slot2={resident_before[2]}, slot3={resident_before[3]}"
    )
    print("Derived offsets before replacement:")
    for tbid in resident_before:
        print(
            "  "
            + _describe_calculated_tbo(
                resident_ids=resident_before,
                thread_block_id=int(tbid),
                shmem_addr=0x20,
            )
        )
    print("Derived offsets after replacement:")
    for tbid in (11, 22, 55, 44):
        print(
            "  "
            + _describe_calculated_tbo(
                resident_ids=resident_after,
                thread_block_id=int(tbid),
                shmem_addr=0x20,
            )
        )

    sim = _sim(num_threads=4, dram_latency_cycles=1, verbose=True)
    arbiter = SmemArbiter(sim)

    scenario = [
        (
            "Step 1: TBID 33 writes a value into its resident slot.",
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xAAAA,
                thread_block_id=33,
                resident_thread_block_ids=resident_before,
                done_bits=[0, 0, 0, 0],
            ),
        ),
        (
            "Step 2: TBID 33 reads back the value while keeping the slot reserved.",
            _tb_txn(
                TxnType.SH_LD,
                thread_id=0,
                shmem_addr=0x20,
                thread_block_id=33,
                resident_thread_block_ids=resident_before,
                done_bits=[0, 0, 0, 0],
            ),
        ),
        (
            "Step 3: TBID 33 performs a global store and marks done bits all-one.",
            _tb_txn(
                TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                thread_id=0,
                shmem_addr=0x20,
                dram_addr=0x9000,
                thread_block_id=33,
                resident_thread_block_ids=resident_before,
                done_bits="1111",
            ),
        ),
        (
            "Step 4: TBID 55 replaces TBID 33 in slot 2 and writes a new value.",
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xBBBB,
                thread_block_id=55,
                resident_thread_block_ids=resident_after,
                done_bits=[0, 0, 0, 0],
            ),
        ),
        (
            "Step 5: TBID 55 reads back from the same calculated TBO/absolute address.",
            _tb_txn(
                TxnType.SH_LD,
                thread_id=0,
                shmem_addr=0x20,
                thread_block_id=55,
                resident_thread_block_ids=resident_after,
                done_bits=[0, 0, 0, 0],
            ),
        ),
        (
            "Step 6: TBID 55 streams the replacement value back out to DRAM.",
            _tb_txn(
                TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                thread_id=0,
                shmem_addr=0x20,
                dram_addr=0x9004,
                thread_block_id=55,
                resident_thread_block_ids=resident_after,
                done_bits=[0, 0, 0, 0],
            ),
        ),
    ]

    print()
    print("Replacement attempt before done bits go all-one:")
    blocked_sim = _sim(num_threads=4)
    blocked_txn = _tb_txn(
        TxnType.SH_ST,
        thread_id=0,
        shmem_addr=0x20,
        write_data=0xDEAD,
        thread_block_id=55,
        resident_thread_block_ids=resident_after,
        done_bits=[0, 0, 0, 0],
    )
    try:
        blocked_sim.issue(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xAAAA,
                thread_block_id=33,
                resident_thread_block_ids=resident_before,
                done_bits=[0, 0, 0, 0],
            )
        )
        blocked_sim.issue(blocked_txn)
    except ValueError as exc:
        print(f"  Expected block: {exc}")

    for label, txn in scenario:
        print()
        print(label)
        print(
            "  "
            + _describe_calculated_tbo(
                resident_ids=(
                    txn.resident_thread_block_ids
                    if txn.resident_thread_block_ids is not None
                    else _tb_slots(txn.thread_block_id)
                ),
                thread_block_id=int(txn.thread_block_id),
                shmem_addr=int(txn.shmem_addr),
                thread_id=int(txn.thread_id),
            )
        )
        completion_cursor = len(sim.completions)
        arbiter.process_batch([txn])
        steps = 0
        while len(sim.completions) == completion_cursor:
            steps += 1
            if steps > 256:
                raise TimeoutError(
                    f"Trace demo transaction did not complete: {txn}"
                )
            sim.step()
        comp = sim.completions[-1]
        print(
            "  Completion: "
            f"tbid={comp.thread_block_id}, smem_block={comp.smem_block_id}, "
            f"calculated_tbo=0x{int(comp.thread_block_offset_effective):04x}, "
            f"abs=0x{int(comp.absolute_shmem_addr):04x}, "
            f"read_data={comp.read_data}, dram_addr={comp.dram_addr}"
        )

    print()
    print("Final resident state:")
    snapshot = sim.snapshot()
    print(f"  resident_thread_block_ids={snapshot['resident_thread_block_ids']}")
    print(f"  dram[0x9000]=0x{int(sim.dram.get(0x9000, 0)) & sim.word_mask:08x}")
    print(f"  dram[0x9004]=0x{int(sim.dram.get(0x9004, 0)) & sim.word_mask:08x}")
    print()
    _dump_completions(sim)


class _BitsProxy:
    def __init__(self, bits: str):
        self.bin = bits


class TestResidentVectorValidation(unittest.TestCase):
    def test_transaction_from_dict_parses_tbid_aliases(self) -> None:
        txn = Transaction.from_dict(
            {
                "type": "sh.st",
                "shmem_addr": 0x10,
                "write_data": 0x55AA,
                "tbid": 22,
                "tbids": [11, 22, None, 44],
                "done_bits": "1111",
            }
        )

        self.assertEqual(txn.thread_block_id, 22)
        self.assertEqual(txn.resident_thread_block_ids, (11, 22, None, 44))
        self.assertEqual(txn.thread_block_done_bits, "1111")

    def test_resident_thread_block_ids_must_have_exactly_four_entries(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 4 entries"):
            ShmemFunctionalSimulator._normalize_resident_thread_block_ids((1, 2, 3))

        with self.assertRaisesRegex(ValueError, "exactly 4 entries"):
            ShmemFunctionalSimulator._normalize_resident_thread_block_ids(
                (1, 2, 3, 4, 5)
            )

    def test_legacy_thread_block_offset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Legacy thread_block_offset"):
            run_single_smem_transaction(
                "sh.st",
                shmem_addr=0x10,
                write_data=1,
                thread_block_offset=0x80,
            )


class TestDoneBitsInterpretation(unittest.TestCase):
    def test_done_bits_all_one_accepts_multiple_encodings(self) -> None:
        fn = ShmemFunctionalSimulator._done_bits_all_one
        self.assertTrue(fn("1111"))
        self.assertTrue(fn("0b1111"))
        self.assertTrue(fn(0b1111))
        self.assertTrue(fn([1, 1, True, 1]))
        self.assertTrue(fn(_BitsProxy("1111")))

    def test_done_bits_all_one_rejects_partial_or_empty_patterns(self) -> None:
        fn = ShmemFunctionalSimulator._done_bits_all_one
        self.assertFalse(fn(None))
        self.assertFalse(fn(""))
        self.assertFalse(fn("1011"))
        self.assertFalse(fn(0b1011))
        self.assertFalse(fn([1, 0, 1, 1]))
        self.assertFalse(fn(_BitsProxy("1101")))

    def test_done_bits_negative_int_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ShmemFunctionalSimulator._done_bits_all_one(-1)


class TestSlotDerivedOffsets(unittest.TestCase):
    def test_offset_is_derived_from_resident_slot_not_thread_block_id(self) -> None:
        sim = _sim()
        out = sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0x1234,
                thread_block_id=99,
                resident_thread_block_ids=_tb_slots(11, 22, None, 99),
                done_bits=[0, 0, 0, 0],
            )
        )

        self.assertEqual(out["smem_block_id"], 3)
        self.assertEqual(out["thread_block_offset_effective"], 0x180)
        self.assertEqual(out["absolute_shmem_addr"], 0x1A0)

    def test_all_four_slots_produce_distinct_absolute_addresses(self) -> None:
        sim = _sim()
        resident_ids = (7, 17, 27, 37)

        for slot_idx, tbid in enumerate(resident_ids):
            out = sim.run_one(
                _tb_txn(
                    TxnType.SH_ST,
                    thread_id=slot_idx,
                    shmem_addr=0x24,
                    write_data=0x1000 + slot_idx,
                    thread_block_id=int(tbid),
                    resident_thread_block_ids=resident_ids,
                    done_bits=[0, 0, 0, 0],
                )
            )
            self.assertEqual(out["smem_block_id"], slot_idx)
            self.assertEqual(
                out["thread_block_offset_effective"],
                slot_idx * DEFAULT_TEST_TB_SIZE_BYTES,
            )
            self.assertEqual(
                out["absolute_shmem_addr"],
                0x24 + (slot_idx * DEFAULT_TEST_TB_SIZE_BYTES),
            )

    def test_existing_resident_mapping_can_be_reused_without_repeating_vector(self) -> None:
        sim = _sim()
        resident_ids = (11, 22, 33, 44)

        sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x10,
                write_data=0xCAFE_BABE,
                thread_block_id=33,
                resident_thread_block_ids=resident_ids,
                done_bits=[0, 0, 0, 0],
            )
        )

        out = sim.run_one(
            Transaction(
                txn_type=TxnType.SH_LD,
                thread_id=0,
                shmem_addr=0x10,
                thread_block_id=33,
            )
        )

        self.assertEqual(out["smem_block_id"], 2)
        self.assertEqual(out["thread_block_offset_effective"], 0x100)
        self.assertEqual(out["read_data"], 0xCAFE_BABE)


class TestResidentReplacementRules(unittest.TestCase):
    def test_slot_replacement_is_blocked_until_done_bits_are_all_one(self) -> None:
        sim = _sim()

        sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xAAAA,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                done_bits=[0, 0, 0, 0],
            )
        )

        with self.assertRaisesRegex(ValueError, "still occupied"):
            sim.issue(
                _tb_txn(
                    TxnType.SH_ST,
                    thread_id=0,
                    shmem_addr=0x20,
                    write_data=0xBBBB,
                    thread_block_id=55,
                    resident_thread_block_ids=(11, 22, 55, 44),
                    done_bits=[0, 0, 0, 0],
                )
            )

    def test_string_done_bits_allow_replacement(self) -> None:
        sim = _sim()

        sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xAAAA,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                done_bits=[0, 0, 0, 0],
            )
        )
        sim.run_one(
            _tb_txn(
                TxnType.SH_LD,
                thread_id=0,
                shmem_addr=0x20,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                done_bits="1111",
            )
        )

        replacement = sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xBBBB,
                thread_block_id=55,
                resident_thread_block_ids=(11, 22, 55, 44),
                done_bits=[0, 0, 0, 0],
            )
        )

        self.assertEqual(replacement["smem_block_id"], 2)
        self.assertEqual(replacement["thread_block_offset_effective"], 0x100)
        self.assertEqual(sim.snapshot()["resident_thread_block_ids"][2], 55)

    def test_integer_done_bits_allow_replacement(self) -> None:
        sim = _sim()

        sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xAAAA,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                done_bits=[0, 0, 0, 0],
            )
        )
        sim.run_one(
            _tb_txn(
                TxnType.SH_LD,
                thread_id=0,
                shmem_addr=0x20,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                done_bits=0b1111,
            )
        )

        replacement = sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xBBBB,
                thread_block_id=55,
                resident_thread_block_ids=(11, 22, 55, 44),
                done_bits=[0, 0, 0, 0],
            )
        )

        self.assertEqual(replacement["smem_block_id"], 2)
        self.assertEqual(sim.snapshot()["resident_thread_block_ids"][2], 55)

    def test_replacement_resets_done_state_for_the_new_resident_thread_block(self) -> None:
        sim = _sim()

        sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xAAAA,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                done_bits="1111",
            )
        )
        sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x20,
                write_data=0xBBBB,
                thread_block_id=55,
                resident_thread_block_ids=(11, 22, 55, 44),
                done_bits=[0, 0, 0, 0],
            )
        )

        with self.assertRaisesRegex(ValueError, "still occupied"):
            sim.issue(
                _tb_txn(
                    TxnType.SH_ST,
                    thread_id=0,
                    shmem_addr=0x20,
                    write_data=0xCCCC,
                    thread_block_id=66,
                    resident_thread_block_ids=(11, 22, 66, 44),
                    done_bits=[0, 0, 0, 0],
                )
            )


class TestUserFacingApis(unittest.TestCase):
    def test_run_single_smem_transaction_reports_slot_derived_tbo(self) -> None:
        out = run_single_smem_transaction(
            "sh.st",
            shmem_addr=0x10,
            write_data=0xAA55AA55,
            num_threads=2,
            thread_id=1,
            thread_block_size_bytes=DEFAULT_TEST_TB_SIZE_BYTES,
            thread_block_id=22,
            resident_thread_block_ids=(11, 22, None, None),
            thread_block_done_bits=[0, 0, 0, 0],
        )

        completion = out["completion"]
        self.assertEqual(completion["smem_block_id"], 1)
        self.assertEqual(completion["thread_block_offset_effective"], 0x80)
        self.assertEqual(completion["absolute_shmem_addr"], 0x90)

    def test_batch_api_reports_slot_derived_tbo_for_each_completion(self) -> None:
        result = run_smem_functional_sim(
            [
                {
                    "type": "sh.st",
                    "shmem_addr": 0x10,
                    "write_data": 0x1111,
                    "thread_id": 0,
                    "thread_block_id": 11,
                    "resident_thread_block_ids": (11, 22, None, None),
                    "thread_block_done_bits": [0, 0, 0, 0],
                },
                {
                    "type": "sh.st",
                    "shmem_addr": 0x10,
                    "write_data": 0x2222,
                    "thread_id": 1,
                    "thread_block_id": 22,
                    "resident_thread_block_ids": (11, 22, None, None),
                    "thread_block_done_bits": [0, 0, 0, 0],
                },
            ],
            num_threads=2,
            thread_block_size_bytes=DEFAULT_TEST_TB_SIZE_BYTES,
        )

        completions = {
            int(comp["thread_block_id"]): comp for comp in result["completions"]
        }
        self.assertEqual(completions[11]["thread_block_offset_effective"], 0x0000)
        self.assertEqual(completions[22]["thread_block_offset_effective"], 0x0080)
        self.assertEqual(completions[22]["absolute_shmem_addr"], 0x0090)


class TestDisplayFormatting(unittest.TestCase):
    def test_fmt_tagged_uses_slot_derived_tbo(self) -> None:
        sim = _sim()
        txn = _tb_txn(
            TxnType.SH_ST,
            thread_id=0,
            shmem_addr=0x08,
            write_data=0x1234,
            thread_block_id=22,
            resident_thread_block_ids=(11, 22, None, None),
            done_bits=[0, 0, 0, 0],
        )
        sim.issue(txn)

        line = sim._fmt_tagged(sim.input_queue[0])
        self.assertIn("tbo=0x0080", line)
        self.assertIn("smem_block=1", line)
        self.assertIn("tbid=22", line)


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        demo_thread_block_offset_trace()
        print()
        print("=== UNIT TESTS ===")
        unittest.main(verbosity=2, exit=False)

#!/usr/bin/env python3
"""
Comprehensive SMEM instruction tests + cycle-count comparison vs DCache.

Run:
    python3 test_smem_comprehensive_cycle_compare.py
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from main import (
    ShmemFunctionalSimulator,
    ShmemCompatibleCacheStage,
    Transaction,
    TxnType,
    load_smem_config,
    run_single_smem_transaction,
    run_smem_functional_sim,
)
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
) -> dict[str, Any]:
    return {
        "thread_block_id": int(thread_block_id),
        "resident_thread_block_ids": tuple(
            resident_thread_block_ids
            if resident_thread_block_ids is not None
            else _tb_slots(int(thread_block_id))
        ),
        "thread_block_done_bits": [1] if done else [0],
    }


def _tb_dict(
    payload: dict[str, Any],
    *,
    thread_block_id: int = 0,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
    done: bool = False,
) -> dict[str, Any]:
    out = dict(payload)
    out.update(
        _tb_kwargs(
            thread_block_id=thread_block_id,
            resident_thread_block_ids=resident_thread_block_ids,
            done=done,
        )
    )
    return out


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


def _apply_tb_attrs(
    req: Any,
    *,
    thread_block_id: int = 0,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
    done: bool = False,
) -> Any:
    for key, value in _tb_kwargs(
        thread_block_id=thread_block_id,
        resident_thread_block_ids=resident_thread_block_ids,
        done=done,
    ).items():
        setattr(req, key, value)
    return req


def _preload_dcache_hit(stage, req, value: int) -> None:
    bank = stage.banks[req.addr.bank_id]
    frame = DCACHE["dCacheFrame"](
        valid=True,
        dirty=False,
        tag=req.addr.tag,
        block=[0] * DCACHE["BLOCK_SIZE_WORDS"],
    )
    frame.block[req.addr.block_offset] = value
    bank.sets[req.addr.set_index][0] = frame


def _preload_smem_hit(stage: ShmemCompatibleCacheStage, addr: int, value: int) -> None:
    probe = _tb_txn(TxnType.SH_LD, shmem_addr=addr)
    absolute = stage.sim._absolute_smem_addr(probe)
    bank, slot = stage.sim._address_crossbar(absolute, stage.sim._effective_thread_block_offset(probe))
    stage.sim.banks[bank][slot] = value & 0xFFFFFFFF
    stage.sim.sram_linear[absolute] = value & 0xFFFFFFFF


def _run_stage_until_response(stage, response_if, max_cycles: int = 20):
    for _ in range(max_cycles):
        stage.compute()
        payload = response_if.payload
        if payload is not None:
            return stage.get_cycle_count(), payload
    raise TimeoutError("No response produced within max_cycles.")


def _make_smem_compat_stage(
    name: str,
    behind,
    fwd,
    *,
    mem_req_name: str,
    mem_resp_name: str,
):
    return ShmemCompatibleCacheStage(
        name=name,
        behind_latch=behind,
        forward_ifs_write={"DCache_LSU_Resp": fwd},
        mem_req_if=DCACHE["LatchIF"](name=mem_req_name),
        mem_resp_if=DCACHE["LatchIF"](name=mem_resp_name),
        smem_simulator_kwargs={
            "thread_block_size_bytes": DEFAULT_TEST_TB_SIZE_BYTES,
        },
    )


def _resp_field(resp, field: str):
    if hasattr(resp, field):
        return getattr(resp, field)
    if isinstance(resp, dict):
        return resp[field]
    raise TypeError(f"Unsupported response object type: {type(resp)}")


class TestSmemConfig(unittest.TestCase):
    def test_load_config_and_apply_to_run(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / ".config"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    [smem]
                    num_banks = 16
                    word_bytes = 4
                    dram_latency_cycles = 3
                    arbiter_issue_width = 2
                    num_threads = 2
                    thread_block_size_bytes = 256
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            cfg = load_smem_config(cfg_path)
            self.assertEqual(cfg.num_banks, 16)
            self.assertEqual(cfg.dram_latency_cycles, 3)
            self.assertEqual(cfg.num_threads, 2)

            out = run_smem_functional_sim(
                [
                    _tb_dict(
                        {"type": "sh.st", "thread_id": 1, "shmem_addr": 0x20, "write_data": 7},
                        thread_block_id=1,
                        resident_thread_block_ids=_tb_slots(0, 1),
                    )
                ],
                config_path=cfg_path,
            )
            self.assertIn("cycle_count", out)
            self.assertEqual(out["completions"][0]["absolute_shmem_addr"], 0x120)


class TestSmemInstructionCoverage(unittest.TestCase):
    def test_sh_ld_from_uninitialized_returns_zero(self):
        out = run_smem_functional_sim(
            [_tb_dict({"type": "sh.ld", "shmem_addr": 0x80})],
            thread_block_size_bytes=DEFAULT_TEST_TB_SIZE_BYTES,
        )
        self.assertEqual(out["completions"][0]["read_data"], 0)

    def test_sh_st_then_sh_ld_round_trip(self):
        out = run_smem_functional_sim(
            [
                _tb_dict({"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xABCD1234}),
                _tb_dict({"type": "sh.ld", "shmem_addr": 0x20}),
            ],
            thread_block_size_bytes=DEFAULT_TEST_TB_SIZE_BYTES,
        )
        self.assertEqual(out["completions"][1]["txn_type"], "sh.ld")
        self.assertEqual(out["completions"][1]["read_data"], 0xABCD1234)

    def test_global_ld_dram_to_sram_then_load(self):
        sim = ShmemFunctionalSimulator(
            dram_init={0x2000: 0x0BADC0DE},
            **{
                **load_smem_config().to_sim_kwargs(),
                "thread_block_size_bytes": DEFAULT_TEST_TB_SIZE_BYTES,
            },
        )
        sim.run(
            [
                _tb_txn(
                    TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                    dram_addr=0x2000,
                    shmem_addr=0x24,
                ),
            ]
        )
        while sim._has_pending_work():
            sim.step()
        done = sim.run_one(_tb_txn(TxnType.SH_LD, shmem_addr=0x24))
        self.assertEqual(done["read_data"], 0x0BADC0DE)

    def test_global_st_smem_to_dram(self):
        out = run_smem_functional_sim(
            [
                _tb_dict({"type": "sh.st", "shmem_addr": 0x30, "write_data": 0xFEEDFACE}),
                _tb_dict({"type": "global.st.smem2dram", "shmem_addr": 0x30, "dram_addr": 0x5000}),
            ],
            thread_block_size_bytes=DEFAULT_TEST_TB_SIZE_BYTES,
        )
        self.assertEqual(out["dram"][0x5000], 0xFEEDFACE)

    def test_all_instruction_types_in_one_sequence(self):
        sim = ShmemFunctionalSimulator(
            num_threads=2,
            thread_block_size_bytes=0x100,
            **{
                key: value
                for key, value in load_smem_config().to_sim_kwargs().items()
                if key not in ("num_threads", "thread_block_size_bytes")
            },
        )
        resident_ids = _tb_slots(0, 1)
        sim.run_one(
            _tb_txn(
                TxnType.SH_ST,
                thread_id=0,
                shmem_addr=0x10,
                write_data=0x11111111,
                thread_block_id=0,
                resident_thread_block_ids=resident_ids,
            )
        )
        done_local = sim.run_one(
            _tb_txn(
                TxnType.SH_LD,
                thread_id=0,
                shmem_addr=0x10,
                thread_block_id=0,
                resident_thread_block_ids=resident_ids,
            )
        )
        done_store = sim.run_one(
            _tb_txn(
                TxnType.GLOBAL_ST_SMEM_TO_DRAM,
                thread_id=0,
                shmem_addr=0x10,
                dram_addr=0x8000,
                thread_block_id=0,
                resident_thread_block_ids=resident_ids,
            )
        )
        done_global = sim.run_one(
            _tb_txn(
                TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                thread_id=1,
                dram_addr=0x8000,
                shmem_addr=0x10,
                thread_block_id=1,
                resident_thread_block_ids=resident_ids,
            )
        )
        done_remote = sim.run_one(
            _tb_txn(
                TxnType.SH_LD,
                thread_id=1,
                shmem_addr=0x10,
                thread_block_id=1,
                resident_thread_block_ids=resident_ids,
            )
        )

        self.assertEqual(done_local["read_data"], 0x11111111)
        self.assertEqual(done_store["txn_type"], "global.st.smem2dram")
        self.assertEqual(done_global["txn_type"], "global.ld.dram2sram")
        self.assertEqual(done_remote["read_data"], 0x11111111)

    def test_single_transaction_api_uses_config(self):
        res = run_single_smem_transaction(
            "sh.st",
            shmem_addr=0x40,
            write_data=0xAA55AA55,
            num_threads=2,
            thread_id=1,
            thread_block_size_bytes=0x200,
            thread_block_id=1,
            resident_thread_block_ids=_tb_slots(0, 1),
        )
        self.assertEqual(
            res["completion"]["absolute_shmem_addr"],
            0x240,
        )

    def test_single_transaction_thread_id_expands_offsets_if_needed(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / ".config"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    [smem]
                    num_threads = 1
                    thread_block_size_bytes = 256
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            out = run_single_smem_transaction(
                "sh.st",
                shmem_addr=0x10,
                write_data=1,
                thread_id=2,
                config_path=cfg_path,
                thread_block_id=0,
                resident_thread_block_ids=_tb_slots(0),
            )
            self.assertEqual(out["completion"]["thread_id"], 2)


class TestCycleComparisonVsDCache(unittest.TestCase):
    def test_read_hit_cycle_compare(self):
        addr = 0x20
        value = 0x12345678
        req = _apply_tb_attrs(
            DCACHE["dCacheRequest"](addr_val=addr, rw_mode="read", size="word")
        )

        dcache_behind = DCACHE["LatchIF"](name="lsu_to_dcache")
        dcache_fwd = DCACHE["ForwardingIF"](name="dcache_to_lsu")
        dcache_stage = DCACHE["LockupFreeCacheStage"](
            name="dCache",
            behind_latch=dcache_behind,
            forward_ifs_write={"DCache_LSU_Resp": dcache_fwd},
            mem_req_if=DCACHE["LatchIF"](name="dcache_to_mem"),
            mem_resp_if=DCACHE["LatchIF"](name="mem_to_dcache"),
        )
        _preload_dcache_hit(dcache_stage, req, value)

        smem_behind = DCACHE["LatchIF"](name="lsu_to_smem_compat")
        smem_fwd = DCACHE["ForwardingIF"](name="smem_to_lsu")
        smem_stage = _make_smem_compat_stage(
            "SMEMCompat",
            smem_behind,
            smem_fwd,
            mem_req_name="smem_to_mem_unused",
            mem_resp_name="mem_to_smem_unused",
        )
        _preload_smem_hit(smem_stage, addr=addr, value=value)

        dcache_behind.push(req)
        smem_behind.push(req)

        dcache_cycles, dcache_resp = _run_stage_until_response(dcache_stage, dcache_fwd)
        smem_cycles, smem_resp = _run_stage_until_response(smem_stage, smem_fwd)

        self.assertEqual(_resp_field(dcache_resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(smem_resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(dcache_resp, "data"), value)
        self.assertEqual(_resp_field(smem_resp, "data"), value)
        self.assertGreaterEqual(smem_cycles, dcache_cycles)

    def test_write_hit_cycle_compare(self):
        addr = 0x24
        req = _apply_tb_attrs(
            DCACHE["dCacheRequest"](
                addr_val=addr,
                rw_mode="write",
                size="word",
                store_value=0xA1B2C3D4,
            )
        )

        dcache_behind = DCACHE["LatchIF"](name="lsu_to_dcache_w")
        dcache_fwd = DCACHE["ForwardingIF"](name="dcache_to_lsu_w")
        dcache_stage = DCACHE["LockupFreeCacheStage"](
            name="dCache",
            behind_latch=dcache_behind,
            forward_ifs_write={"DCache_LSU_Resp": dcache_fwd},
            mem_req_if=DCACHE["LatchIF"](name="dcache_to_mem_w"),
            mem_resp_if=DCACHE["LatchIF"](name="mem_to_dcache_w"),
        )
        _preload_dcache_hit(dcache_stage, req, 0x0)

        smem_behind = DCACHE["LatchIF"](name="lsu_to_smem_compat_w")
        smem_fwd = DCACHE["ForwardingIF"](name="smem_to_lsu_w")
        smem_stage = _make_smem_compat_stage(
            "SMEMCompat",
            smem_behind,
            smem_fwd,
            mem_req_name="smem_to_mem_unused_w",
            mem_resp_name="mem_to_smem_unused_w",
        )
        _preload_smem_hit(smem_stage, addr=addr, value=0x0)

        dcache_behind.push(req)
        smem_behind.push(req)

        dcache_cycles, dcache_resp = _run_stage_until_response(dcache_stage, dcache_fwd)
        smem_cycles, smem_resp = _run_stage_until_response(smem_stage, smem_fwd)

        self.assertEqual(_resp_field(dcache_resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(smem_resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(dcache_resp, "data"), 0x0)
        self.assertEqual(_resp_field(smem_resp, "data"), 0x0)
        self.assertLessEqual(smem_cycles, dcache_cycles)

    def test_compat_stage_handles_all_smem_transaction_types(self):
        behind = DCACHE["LatchIF"](name="lsu_to_smem_compat_all")
        fwd = DCACHE["ForwardingIF"](name="smem_to_lsu_all")
        stage = _make_smem_compat_stage(
            "SMEMCompat",
            behind,
            fwd,
            mem_req_name="unused_mem_req",
            mem_resp_name="unused_mem_resp",
        )

        requests = [
            _tb_dict({"type": "sh.st", "shmem_addr": 0x20, "write_data": 0x7777}),
            _tb_dict({"type": "sh.ld", "shmem_addr": 0x20}),
            _tb_dict({"type": "global.st.smem2dram", "shmem_addr": 0x20, "dram_addr": 0x1110}),
            _tb_dict({"type": "global.ld.dram2sram", "dram_addr": 0x1110, "shmem_addr": 0x24}),
        ]

        response_types = []
        for req in requests:
            behind.push(req)
            _, resp = _run_stage_until_response(stage, fwd)
            response_types.append(_resp_field(resp, "type"))
            fwd.pop()

        self.assertEqual(response_types[0], "HIT_COMPLETE")
        self.assertEqual(response_types[1], "HIT_COMPLETE")
        self.assertEqual(response_types[2], "MISS_COMPLETE")
        self.assertEqual(response_types[3], "MISS_COMPLETE")

    def test_compat_response_has_attribute_access(self):
        behind = DCACHE["LatchIF"](name="lsu_to_smem_resp_attr")
        fwd = DCACHE["ForwardingIF"](name="smem_resp_attr")
        stage = _make_smem_compat_stage(
            "SMEMCompat",
            behind,
            fwd,
            mem_req_name="unused_mem_req_attr",
            mem_resp_name="unused_mem_resp_attr",
        )
        behind.push(_tb_dict({"type": "sh.ld", "shmem_addr": 0x10}))
        _, resp = _run_stage_until_response(stage, fwd)
        self.assertTrue(hasattr(resp, "type"))
        self.assertEqual(resp.type, "HIT_COMPLETE")

    def test_dict_rw_mode_write_maps_to_store(self):
        behind = DCACHE["LatchIF"](name="lsu_to_smem_dict_write")
        fwd = DCACHE["ForwardingIF"](name="smem_dict_write_resp")
        stage = _make_smem_compat_stage(
            "SMEMCompat",
            behind,
            fwd,
            mem_req_name="unused_mem_req_dict_write",
            mem_resp_name="unused_mem_resp_dict_write",
        )

        behind.push(
            _tb_dict({"addr_val": 0x44, "rw_mode": "write", "size": "word", "store_value": 0xABCD})
        )
        _run_stage_until_response(stage, fwd)
        fwd.pop()

        behind.push(_tb_dict({"addr_val": 0x44, "rw_mode": "read", "size": "word"}))
        _, resp = _run_stage_until_response(stage, fwd)
        self.assertEqual(_resp_field(resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(resp, "data"), 0xABCD)

    def test_dict_generic_type_with_rw_mode_falls_back_cleanly(self):
        behind = DCACHE["LatchIF"](name="lsu_to_smem_dict_generic_type")
        fwd = DCACHE["ForwardingIF"](name="smem_dict_generic_type_resp")
        stage = _make_smem_compat_stage(
            "SMEMCompat",
            behind,
            fwd,
            mem_req_name="unused_mem_req_dict_generic",
            mem_resp_name="unused_mem_resp_dict_generic",
        )

        behind.push(
            _tb_dict({
                "type": "write",
                "addr_val": 0x48,
                "rw_mode": "write",
                "size": "word",
                "store_value": 0x55AA,
            })
        )
        _run_stage_until_response(stage, fwd)
        fwd.pop()

        behind.push(_tb_dict({"type": "read", "addr_val": 0x48, "rw_mode": "read", "size": "word"}))
        _, resp = _run_stage_until_response(stage, fwd)
        self.assertEqual(_resp_field(resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(resp, "address"), 0x48)
        self.assertEqual(_resp_field(resp, "data"), 0x55AA)


class TestQueueCycleAccounting(unittest.TestCase):
    def test_sh_ld_queue_transition_costs_one_cycle(self):
        sim = ShmemFunctionalSimulator(
            num_banks=32,
            word_bytes=4,
            dram_latency_cycles=0,
            arbiter_issue_width=32,
            num_threads=1,
            thread_block_size_bytes=DEFAULT_TEST_TB_SIZE_BYTES,
        )
        sim.issue(_tb_txn(TxnType.SH_LD, shmem_addr=0x20))

        sim.step()
        self.assertEqual(len(sim.completions), 0)
        self.assertEqual(len(sim.smem_read_queue), 1)
        self.assertEqual(sim.smem_read_queue[0]["ready_cycle"], 1)

        sim.step()
        self.assertEqual(len(sim.completions), 0)
        self.assertEqual(len(sim.pending_read_crossbar_deliveries), 1)

        sim.step()
        self.assertEqual(len(sim.completions), 0)

        sim.step()
        self.assertEqual(len(sim.completions), 0)

        sim.step()
        self.assertEqual(len(sim.completions), 1)
        self.assertEqual(sim.completions[0].cycle_completed, 4)

    def test_global_ld_axi_response_queue_costs_one_cycle(self):
        sim = ShmemFunctionalSimulator(
            dram_init={0x1000: 0xA5A5},
            num_banks=32,
            word_bytes=4,
            dram_latency_cycles=0,
            arbiter_issue_width=32,
            num_threads=1,
            thread_block_size_bytes=DEFAULT_TEST_TB_SIZE_BYTES,
        )
        sim.issue(
            _tb_txn(
                TxnType.GLOBAL_LD_DRAM_TO_SRAM,
                dram_addr=0x1000,
                shmem_addr=0x20,
            )
        )

        sim.step()
        self.assertEqual(len(sim.completions), 0)
        self.assertEqual(len(sim.smem_write_queue), 1)

        sim.step()
        self.assertEqual(len(sim.completions), 0)
        self.assertEqual(len(sim.pending_dram_reads), 1)

        sim.step()
        self.assertEqual(len(sim.completions), 0)
        self.assertEqual(len(sim.axi_bus_queue), 1)
        self.assertEqual(sim.axi_bus_queue[0]["kind"], "read_resp")

        sim.step()
        self.assertEqual(len(sim.completions), 1)
        self.assertEqual(sim.completions[0].cycle_completed, 3)


class TestResidentThreadBlockMapping(unittest.TestCase):
    def test_thread_block_offset_is_derived_from_resident_smem_slot(self):
        sim = ShmemFunctionalSimulator(
            num_banks=32,
            word_bytes=4,
            dram_latency_cycles=0,
            arbiter_issue_width=32,
            num_threads=1,
            thread_block_size_bytes=0x80,
        )
        out = sim.run_one(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=0x20,
                write_data=0x1234,
                thread_id=0,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                thread_block_done_bits=[0, 0, 0, 0],
            )
        )
        self.assertEqual(out["smem_block_id"], 2)
        self.assertEqual(out["thread_block_offset_effective"], 0x100)
        self.assertEqual(out["absolute_shmem_addr"], 0x120)

    def test_thread_block_slot_stays_reserved_until_done_bits_are_all_one(self):
        sim = ShmemFunctionalSimulator(
            num_banks=32,
            word_bytes=4,
            dram_latency_cycles=0,
            arbiter_issue_width=32,
            num_threads=1,
            thread_block_size_bytes=0x80,
        )

        first = sim.run_one(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=0x20,
                write_data=0xAAAA,
                thread_id=0,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                thread_block_done_bits=[0, 0, 0, 0],
            )
        )
        self.assertEqual(first["absolute_shmem_addr"], 0x120)

        with self.assertRaisesRegex(ValueError, "still occupied"):
            sim.issue(
                Transaction(
                    txn_type=TxnType.SH_ST,
                    shmem_addr=0x20,
                    write_data=0xBBBB,
                    thread_id=0,
                    thread_block_id=55,
                    resident_thread_block_ids=(11, 22, 55, 44),
                    thread_block_done_bits=[0, 0, 0, 0],
                )
            )

        read_back = sim.run_one(
            Transaction(
                txn_type=TxnType.SH_LD,
                shmem_addr=0x20,
                thread_id=0,
                thread_block_id=33,
                resident_thread_block_ids=(11, 22, 33, 44),
                thread_block_done_bits=[1, 1, 1, 1],
            )
        )
        self.assertEqual(read_back["read_data"], 0xAAAA)

        replacement = sim.run_one(
            Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=0x20,
                write_data=0xBBBB,
                thread_id=0,
                thread_block_id=55,
                resident_thread_block_ids=(11, 22, 55, 44),
                thread_block_done_bits=[0, 0, 0, 0],
            )
        )
        self.assertEqual(replacement["smem_block_id"], 2)
        self.assertEqual(replacement["absolute_shmem_addr"], 0x120)
        self.assertEqual(sim.snapshot()["resident_thread_block_ids"][2], 55)


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        unittest.main(verbosity=2, exit=False)

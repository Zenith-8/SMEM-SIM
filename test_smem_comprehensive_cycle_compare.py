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

from main import (
    ShmemCompatibleCacheStage,
    Transaction,
    TxnType,
    load_smem_config,
    run_single_smem_transaction,
    run_smem_functional_sim,
)
from test_dcache_and_smem import _load_dcache_symbols


DCACHE = _load_dcache_symbols()


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
    probe = Transaction(txn_type=TxnType.SH_LD, shmem_addr=addr)
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
                    thread_block_offsets = [0, 256]
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
                [{"type": "sh.st", "thread_id": 1, "shmem_addr": 0x20, "write_data": 7}],
                config_path=cfg_path,
            )
            self.assertIn("cycle_count", out)
            self.assertEqual(out["completions"][0]["absolute_shmem_addr"], 0x120)


class TestSmemInstructionCoverage(unittest.TestCase):
    def test_sh_ld_from_uninitialized_returns_zero(self):
        out = run_smem_functional_sim([{"type": "sh.ld", "shmem_addr": 0x80}])
        self.assertEqual(out["completions"][0]["read_data"], 0)

    def test_sh_st_then_sh_ld_round_trip(self):
        out = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xABCD1234},
                {"type": "sh.ld", "shmem_addr": 0x20},
            ]
        )
        self.assertEqual(out["completions"][1]["txn_type"], "sh.ld")
        self.assertEqual(out["completions"][1]["read_data"], 0xABCD1234)

    def test_global_ld_dram_to_sram_then_load(self):
        out = run_smem_functional_sim(
            [
                {"type": "global.ld.dram2sram", "dram_addr": 0x2000, "shmem_addr": 0x24},
                {"type": "sh.ld", "shmem_addr": 0x24},
            ],
            dram_init={0x2000: 0x0BADC0DE},
        )
        self.assertEqual(out["completions"][1]["read_data"], 0x0BADC0DE)

    def test_global_st_smem_to_dram(self):
        out = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x30, "write_data": 0xFEEDFACE},
                {"type": "global.st.smem2dram", "shmem_addr": 0x30, "dram_addr": 0x5000},
            ]
        )
        self.assertEqual(out["dram"][0x5000], 0xFEEDFACE)

    def test_all_instruction_types_in_one_sequence(self):
        txns = [
            {"type": "sh.st", "thread_id": 0, "shmem_addr": 0x10, "write_data": 0x11111111},
            {"type": "sh.ld", "thread_id": 0, "shmem_addr": 0x10},
            {"type": "global.st.smem2dram", "thread_id": 0, "shmem_addr": 0x10, "dram_addr": 0x8000},
            {"type": "global.ld.dram2sram", "thread_id": 1, "dram_addr": 0x8000, "shmem_addr": 0x10},
            {"type": "sh.ld", "thread_id": 1, "shmem_addr": 0x10},
        ]
        out = run_smem_functional_sim(
            txns,
            num_threads=2,
            thread_block_offsets={0: 0x000, 1: 0x100},
        )
        self.assertEqual(out["completions"][1]["read_data"], 0x11111111)
        self.assertEqual(out["completions"][4]["read_data"], 0x11111111)

    def test_single_transaction_api_uses_config(self):
        res = run_single_smem_transaction(
            "sh.st",
            shmem_addr=0x40,
            write_data=0xAA55AA55,
            num_threads=2,
            thread_id=1,
            thread_block_offsets={0: 0x0, 1: 0x200},
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
                    thread_block_offsets = [0]
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
            )
            self.assertEqual(out["completion"]["thread_id"], 2)


class TestCycleComparisonVsDCache(unittest.TestCase):
    def test_read_hit_cycle_compare(self):
        addr = 0x20
        value = 0x12345678
        req = DCACHE["dCacheRequest"](addr_val=addr, rw_mode="read", size="word")

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
        smem_stage = ShmemCompatibleCacheStage(
            name="SMEMCompat",
            behind_latch=smem_behind,
            forward_ifs_write={"DCache_LSU_Resp": smem_fwd},
            mem_req_if=DCACHE["LatchIF"](name="smem_to_mem_unused"),
            mem_resp_if=DCACHE["LatchIF"](name="mem_to_smem_unused"),
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
        self.assertLessEqual(smem_cycles, dcache_cycles)

    def test_write_hit_cycle_compare(self):
        addr = 0x24
        req = DCACHE["dCacheRequest"](
            addr_val=addr,
            rw_mode="write",
            size="word",
            store_value=0xA1B2C3D4,
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
        smem_stage = ShmemCompatibleCacheStage(
            name="SMEMCompat",
            behind_latch=smem_behind,
            forward_ifs_write={"DCache_LSU_Resp": smem_fwd},
            mem_req_if=DCACHE["LatchIF"](name="smem_to_mem_unused_w"),
            mem_resp_if=DCACHE["LatchIF"](name="mem_to_smem_unused_w"),
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
        stage = ShmemCompatibleCacheStage(
            name="SMEMCompat",
            behind_latch=behind,
            forward_ifs_write={"DCache_LSU_Resp": fwd},
            mem_req_if=DCACHE["LatchIF"](name="unused_mem_req"),
            mem_resp_if=DCACHE["LatchIF"](name="unused_mem_resp"),
        )

        requests = [
            {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0x7777},
            {"type": "sh.ld", "shmem_addr": 0x20},
            {"type": "global.st.smem2dram", "shmem_addr": 0x20, "dram_addr": 0x1110},
            {"type": "global.ld.dram2sram", "dram_addr": 0x1110, "shmem_addr": 0x24},
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
        stage = ShmemCompatibleCacheStage(
            name="SMEMCompat",
            behind_latch=behind,
            forward_ifs_write={"DCache_LSU_Resp": fwd},
            mem_req_if=DCACHE["LatchIF"](name="unused_mem_req_attr"),
            mem_resp_if=DCACHE["LatchIF"](name="unused_mem_resp_attr"),
        )
        behind.push({"type": "sh.ld", "shmem_addr": 0x10})
        _, resp = _run_stage_until_response(stage, fwd)
        self.assertTrue(hasattr(resp, "type"))
        self.assertEqual(resp.type, "HIT_COMPLETE")

    def test_dict_rw_mode_write_maps_to_store(self):
        behind = DCACHE["LatchIF"](name="lsu_to_smem_dict_write")
        fwd = DCACHE["ForwardingIF"](name="smem_dict_write_resp")
        stage = ShmemCompatibleCacheStage(
            name="SMEMCompat",
            behind_latch=behind,
            forward_ifs_write={"DCache_LSU_Resp": fwd},
            mem_req_if=DCACHE["LatchIF"](name="unused_mem_req_dict_write"),
            mem_resp_if=DCACHE["LatchIF"](name="unused_mem_resp_dict_write"),
        )

        behind.push({"addr_val": 0x44, "rw_mode": "write", "size": "word", "store_value": 0xABCD})
        _run_stage_until_response(stage, fwd)
        fwd.pop()

        behind.push({"addr_val": 0x44, "rw_mode": "read", "size": "word"})
        _, resp = _run_stage_until_response(stage, fwd)
        self.assertEqual(_resp_field(resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(resp, "data"), 0xABCD)

    def test_dict_generic_type_with_rw_mode_falls_back_cleanly(self):
        behind = DCACHE["LatchIF"](name="lsu_to_smem_dict_generic_type")
        fwd = DCACHE["ForwardingIF"](name="smem_dict_generic_type_resp")
        stage = ShmemCompatibleCacheStage(
            name="SMEMCompat",
            behind_latch=behind,
            forward_ifs_write={"DCache_LSU_Resp": fwd},
            mem_req_if=DCACHE["LatchIF"](name="unused_mem_req_dict_generic"),
            mem_resp_if=DCACHE["LatchIF"](name="unused_mem_resp_dict_generic"),
        )

        behind.push(
            {
                "type": "write",
                "addr_val": 0x48,
                "rw_mode": "write",
                "size": "word",
                "store_value": 0x55AA,
            }
        )
        _run_stage_until_response(stage, fwd)
        fwd.pop()

        behind.push({"type": "read", "addr_val": 0x48, "rw_mode": "read", "size": "word"})
        _, resp = _run_stage_until_response(stage, fwd)
        self.assertEqual(_resp_field(resp, "type"), "HIT_COMPLETE")
        self.assertEqual(_resp_field(resp, "address"), 0x48)
        self.assertEqual(_resp_field(resp, "data"), 0x55AA)


if __name__ == "__main__":
    from test_output import capture_to_test_log

    with capture_to_test_log(__file__):
        unittest.main(verbosity=2, exit=False)

#!/usr/bin/env python3
"""
Combined regression tests for:
- simulator/mem/dcache.py
- main.py (SMEM functional simulator)

Run:
    python3 test_dcache_and_smem.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from main import run_smem_functional_sim


ROOT = Path(__file__).resolve().parent


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_bitstring_stub() -> None:
    if "bitstring" in sys.modules:
        return

    bitstring_module = types.ModuleType("bitstring")

    class Bits:
        def __init__(self, *, bytes: bytes | None = None, uint: int | None = None, length: int | None = None):
            if bytes is not None:
                self._bytes = bytes
            elif uint is not None and length is not None:
                nbytes = max(1, (int(length) + 7) // 8)
                self._bytes = int(uint).to_bytes(nbytes, byteorder="little", signed=False)
            else:
                self._bytes = b""

        def tobytes(self) -> bytes:
            return self._bytes

        @property
        def uint(self) -> int:
            if not self._bytes:
                return 0
            return int.from_bytes(self._bytes, byteorder="little", signed=False)

    bitstring_module.Bits = Bits
    sys.modules["bitstring"] = bitstring_module


def _ensure_common_stub() -> None:
    if "common.custom_enums_multi" in sys.modules:
        return

    common_pkg = types.ModuleType("common")
    common_pkg.__path__ = []
    sys.modules["common"] = common_pkg

    enum_mod = types.ModuleType("common.custom_enums_multi")

    class Op(Enum):
        NOP = 0

    enum_mod.Op = Op
    sys.modules["common.custom_enums_multi"] = enum_mod


def _load_dcache_symbols() -> Dict[str, Any]:
    """
    Try normal imports first. If optional deps are missing in this local env,
    load dcache modules directly with lightweight stubs.
    """
    try:
        from simulator.mem.dcache import CacheBank, LockupFreeCacheStage, MSHRBuffer
        from simulator.mem_types import (
            BLOCK_SIZE_WORDS,
            MSHR_BUFFER_LEN,
            NUM_SETS_PER_BANK,
            NUM_WAYS,
            dCacheFrame,
            dCacheRequest,
        )
        from simulator.interfaces import ForwardingIF, LatchIF

        return {
            "CacheBank": CacheBank,
            "LockupFreeCacheStage": LockupFreeCacheStage,
            "MSHRBuffer": MSHRBuffer,
            "BLOCK_SIZE_WORDS": BLOCK_SIZE_WORDS,
            "MSHR_BUFFER_LEN": MSHR_BUFFER_LEN,
            "NUM_SETS_PER_BANK": NUM_SETS_PER_BANK,
            "NUM_WAYS": NUM_WAYS,
            "dCacheFrame": dCacheFrame,
            "dCacheRequest": dCacheRequest,
            "ForwardingIF": ForwardingIF,
            "LatchIF": LatchIF,
        }
    except ModuleNotFoundError:
        pass

    _ensure_bitstring_stub()
    _ensure_common_stub()

    for name in list(sys.modules):
        if name == "simulator" or name.startswith("simulator."):
            del sys.modules[name]

    simulator_pkg = types.ModuleType("simulator")
    simulator_pkg.__path__ = [str(ROOT / "simulator")]
    sys.modules["simulator"] = simulator_pkg

    simulator_mem_pkg = types.ModuleType("simulator.mem")
    simulator_mem_pkg.__path__ = [str(ROOT / "simulator" / "mem")]
    sys.modules["simulator.mem"] = simulator_mem_pkg

    _load_module("simulator.interfaces", ROOT / "simulator" / "interfaces.py")
    _load_module("simulator.stage", ROOT / "simulator" / "stage.py")

    instruction_mod = types.ModuleType("simulator.instruction")

    class Instruction:
        pass

    instruction_mod.Instruction = Instruction
    sys.modules["simulator.instruction"] = instruction_mod

    _load_module("simulator.mem_types", ROOT / "simulator" / "mem_types.py")
    _load_module("simulator.mem.dcache", ROOT / "simulator" / "mem" / "dcache.py")

    from simulator.mem.dcache import CacheBank, LockupFreeCacheStage, MSHRBuffer
    from simulator.mem_types import (
        BLOCK_SIZE_WORDS,
        MSHR_BUFFER_LEN,
        NUM_SETS_PER_BANK,
        NUM_WAYS,
        dCacheFrame,
        dCacheRequest,
    )
    from simulator.interfaces import ForwardingIF, LatchIF

    return {
        "CacheBank": CacheBank,
        "LockupFreeCacheStage": LockupFreeCacheStage,
        "MSHRBuffer": MSHRBuffer,
        "BLOCK_SIZE_WORDS": BLOCK_SIZE_WORDS,
        "MSHR_BUFFER_LEN": MSHR_BUFFER_LEN,
        "NUM_SETS_PER_BANK": NUM_SETS_PER_BANK,
        "NUM_WAYS": NUM_WAYS,
        "dCacheFrame": dCacheFrame,
        "dCacheRequest": dCacheRequest,
        "ForwardingIF": ForwardingIF,
        "LatchIF": LatchIF,
    }


DCACHE = _load_dcache_symbols()


class TestDCacheMSHR(unittest.TestCase):
    def test_primary_miss_becomes_ready_after_countdown(self):
        mshr = DCACHE["MSHRBuffer"](buffer_len=4, bank_id=0)
        req = DCACHE["dCacheRequest"](addr_val=0x20, rw_mode="read", size="word")

        uuid, is_new = mshr.add_miss(req)
        self.assertTrue(is_new)
        self.assertEqual(len(mshr.buffer), 1)
        self.assertIsNone(mshr.get_head())

        for _ in range(DCACHE["MSHR_BUFFER_LEN"]):
            mshr.cycle()

        head = mshr.get_head()
        self.assertIsNotNone(head)
        self.assertEqual(head.uuid, uuid)

    def test_secondary_write_miss_merges_into_existing_entry(self):
        mshr = DCACHE["MSHRBuffer"](buffer_len=4, bank_id=0)
        primary = DCACHE["dCacheRequest"](addr_val=0x20, rw_mode="read", size="word")
        secondary = DCACHE["dCacheRequest"](
            addr_val=0x24,
            rw_mode="write",
            size="word",
            store_value=0xA5A5A5A5,
        )

        uuid_primary, is_new_primary = mshr.add_miss(primary)
        uuid_secondary, is_new_secondary = mshr.add_miss(secondary)

        self.assertTrue(is_new_primary)
        self.assertFalse(is_new_secondary)
        self.assertEqual(uuid_primary, uuid_secondary)
        self.assertEqual(len(mshr.buffer), 1)

        entry = mshr.buffer[0]
        offset = secondary.addr.block_offset
        self.assertTrue(entry.write_status[offset])
        self.assertEqual(entry.write_block[offset], 0xA5A5A5A5)

    def test_stall_when_mshr_full_and_bank_busy(self):
        mshr = DCACHE["MSHRBuffer"](buffer_len=1, bank_id=0)
        req = DCACHE["dCacheRequest"](addr_val=0x20, rw_mode="read", size="word")
        mshr.add_miss(req)

        self.assertTrue(mshr.check_stall(bank_empty=False))
        self.assertFalse(mshr.check_stall(bank_empty=True))


class TestDCacheBankAndStage(unittest.TestCase):
    def test_word_write_hit_updates_cacheline_and_sets_dirty(self):
        bank = DCACHE["CacheBank"](
            bank_id=0,
            num_sets=DCACHE["NUM_SETS_PER_BANK"],
            num_ways=DCACHE["NUM_WAYS"],
            mem_req_if=DCACHE["LatchIF"](),
        )
        req = DCACHE["dCacheRequest"](
            addr_val=0x20,
            rw_mode="write",
            size="word",
            store_value=0xDEADBEEF,
        )

        frame = DCACHE["dCacheFrame"](
            valid=True,
            dirty=False,
            tag=req.addr.tag,
            block=[0] * DCACHE["BLOCK_SIZE_WORDS"],
        )
        frame.block[req.addr.block_offset] = 0x11223344
        bank.sets[req.addr.set_index][0] = frame

        hit, old_word = bank.check_hit(
            req.addr,
            req.rw_mode,
            req.store_value,
            size=req.size,
            raw_addr=req.addr_val,
        )
        self.assertTrue(hit)
        self.assertEqual(old_word, 0x11223344)
        self.assertEqual(
            bank.sets[req.addr.set_index][0].block[req.addr.block_offset], 0xDEADBEEF
        )
        self.assertTrue(bank.sets[req.addr.set_index][0].dirty)

    def test_hit_pipeline_emits_hit_complete_event(self):
        behind = DCACHE["LatchIF"](name="lsu_to_dcache")
        mem_req_if = DCACHE["LatchIF"](name="dcache_to_mem")
        mem_resp_if = DCACHE["LatchIF"](name="mem_to_dcache")
        dcache_resp_if = DCACHE["ForwardingIF"](name="dcache_to_lsu")

        stage = DCACHE["LockupFreeCacheStage"](
            name="DCache",
            behind_latch=behind,
            forward_ifs_write={"DCache_LSU_Resp": dcache_resp_if},
            mem_req_if=mem_req_if,
            mem_resp_if=mem_resp_if,
        )

        req = DCACHE["dCacheRequest"](addr_val=0x20, rw_mode="read", size="word")
        bank = stage.banks[req.addr.bank_id]

        frame = DCACHE["dCacheFrame"](
            valid=True,
            dirty=False,
            tag=req.addr.tag,
            block=[0] * DCACHE["BLOCK_SIZE_WORDS"],
        )
        frame.block[req.addr.block_offset] = 0x11223344
        bank.sets[req.addr.set_index][0] = frame

        behind.push(req)

        stage.compute()  # accept request and place into hit pipeline
        stage.compute()  # pipeline advance
        stage.compute()  # HIT_COMPLETE should appear here

        self.assertEqual(stage.get_cycle_count(), 3)

        resp = dcache_resp_if.payload
        self.assertIsNotNone(resp)
        self.assertEqual(resp.type, "HIT_COMPLETE")
        self.assertTrue(resp.hit)
        self.assertEqual(resp.address, req.addr_val)
        self.assertEqual(resp.data, 0x11223344)


class TestSMEMFunctionalSimulator(unittest.TestCase):
    def test_sh_store_then_load_round_trip(self):
        result = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xDEADBEEF},
                {"type": "sh.ld", "shmem_addr": 0x20},
            ]
        )
        self.assertEqual(result["cycle_count"], result["cycle"])
        completions = {c["txn_id"]: c for c in result["completions"]}
        self.assertEqual(completions[2]["read_data"], 0xDEADBEEF)

    def test_async_store_writes_to_dram(self):
        result = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xCAFEBABE},
                {"type": "async.st.smem2dram", "shmem_addr": 0x20, "dram_addr": 0x1000},
            ]
        )
        self.assertEqual(result["dram"][0x1000], 0xCAFEBABE)

    def test_async_load_then_sh_load_reads_back_dram_value(self):
        result = run_smem_functional_sim(
            [
                {"type": "async.ld.dram2sram", "dram_addr": 0x1000, "shmem_addr": 0x24},
                {"type": "sh.ld", "shmem_addr": 0x24},
            ],
            dram_init={0x1000: 0x1234ABCD},
        )
        completions = {c["txn_id"]: c for c in result["completions"]}
        self.assertEqual(completions[2]["read_data"], 0x1234ABCD)

    def test_bank_conflicts_are_split_but_order_preserved(self):
        result = run_smem_functional_sim(
            [
                {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0x1},
                {"type": "sh.st", "shmem_addr": 0xA0, "write_data": 0x2},
                {"type": "sh.st", "shmem_addr": 0x24, "write_data": 0x3},
            ],
            arbiter_issue_width=4,
        )
        completions = {c["txn_id"]: c for c in result["completions"]}

        self.assertGreater(
            completions[2]["cycle_completed"], completions[1]["cycle_completed"]
        )
        self.assertGreaterEqual(
            completions[3]["cycle_completed"], completions[2]["cycle_completed"]
        )
        self.assertTrue(
            any(
                "arbiter stall on bank conflict" in step
                for step in completions[2]["trace"]
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

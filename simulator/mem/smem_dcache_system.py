from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from simulator.interfaces import ForwardingIF, LatchIF
from simulator.stage import Stage
from simulator.mem.dcache import LockupFreeCacheStage

from main import ShmemFunctionalSimulator, Transaction

try:
    from simulator.execute.stage import MemorySystemInterfaces, create_memory_system_interfaces
except Exception:
    @dataclass
    class MemorySystemInterfaces:
        lsu_dcache_latch: LatchIF
        dcache_lsu_forward: ForwardingIF
        dcache_mem_latch: LatchIF
        mem_dcache_latch: LatchIF
        lsu_smem_latch: LatchIF
        smem_lsu_forward: ForwardingIF

    def create_memory_system_interfaces() -> MemorySystemInterfaces:
        lsu_dcache_latch = LatchIF(name="LSU-DCache Latch")
        dcache_lsu_forward = ForwardingIF(name="dcache_lsu_forward")
        lsu_dcache_latch.forward_if = dcache_lsu_forward

        dcache_mem_latch = LatchIF(name="DCache-Mem Latch")
        mem_dcache_latch = LatchIF(name="Mem-DCache Latch")

        lsu_smem_latch = LatchIF(name="LSU-SMEM Latch")
        smem_lsu_forward = ForwardingIF(name="smem_lsu_forward")
        lsu_smem_latch.forward_if = smem_lsu_forward

        return MemorySystemInterfaces(
            lsu_dcache_latch=lsu_dcache_latch,
            dcache_lsu_forward=dcache_lsu_forward,
            dcache_mem_latch=dcache_mem_latch,
            mem_dcache_latch=mem_dcache_latch,
            lsu_smem_latch=lsu_smem_latch,
            smem_lsu_forward=smem_lsu_forward,
        )


class SharedMemoryStage(Stage):
    """
    Stage wrapper around ShmemFunctionalSimulator so it can run cycle-by-cycle
    using the same latch/forward-if pattern as other pipeline stages.
    """

    SMEM_LSU_IF_NAME = "SMEM_LSU_Resp"

    def __init__(
        self,
        name: str,
        behind_latch: Optional[LatchIF],
        forward_ifs_write: Optional[Dict[str, ForwardingIF]] = None,
        *,
        simulator: Optional[ShmemFunctionalSimulator] = None,
        simulator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            behind_latch=behind_latch,
            forward_ifs_write=forward_ifs_write or {},
        )
        self.simulator = simulator or ShmemFunctionalSimulator(
            **(simulator_kwargs or {})
        )
        self.cycle_count = 0
        self.output_buffer = deque()
        self._completion_cursor = 0

        if self.behind_latch and (self.SMEM_LSU_IF_NAME in self.forward_ifs_write):
            self.behind_latch.forward_if = self.forward_ifs_write[self.SMEM_LSU_IF_NAME]

    def get_cycle_count(self) -> int:
        return int(self.cycle_count)

    def compute(self) -> None:
        self.cycle_count += 1

        if self.behind_latch and self.behind_latch.valid:
            in_payload = self.behind_latch.pop()
            if in_payload is not None:
                txn = (
                    in_payload
                    if isinstance(in_payload, Transaction)
                    else Transaction.from_dict(in_payload)
                )
                self.simulator.issue(txn)

        self.simulator.step()

        if len(self.simulator.completions) > self._completion_cursor:
            new_done = self.simulator.completions[self._completion_cursor :]
            for done in new_done:
                event = asdict(done)
                event["source"] = "smem"
                event["smem_cycle_count"] = self.simulator.get_cycle_count()
                self.output_buffer.append(event)
            self._completion_cursor = len(self.simulator.completions)

        if self.SMEM_LSU_IF_NAME in self.forward_ifs_write:
            interface = self.forward_ifs_write[self.SMEM_LSU_IF_NAME]
            if not interface.wait:
                if self.output_buffer:
                    interface.push(self.output_buffer.popleft())
                else:
                    interface.push(None)


class MemoryCompareSystem:
    """
    Small integration harness that wires DCache + SMEM through latch interfaces
    and provides cycle counter comparison.
    """

    def __init__(
        self,
        *,
        interfaces: Optional[MemorySystemInterfaces] = None,
        smem_simulator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.interfaces = interfaces or create_memory_system_interfaces()

        self.dcache_stage = LockupFreeCacheStage(
            name="dCache",
            behind_latch=self.interfaces.lsu_dcache_latch,
            forward_ifs_write={"DCache_LSU_Resp": self.interfaces.dcache_lsu_forward},
            mem_req_if=self.interfaces.dcache_mem_latch,
            mem_resp_if=self.interfaces.mem_dcache_latch,
        )

        self.smem_stage = SharedMemoryStage(
            name="SMEM",
            behind_latch=self.interfaces.lsu_smem_latch,
            forward_ifs_write={"SMEM_LSU_Resp": self.interfaces.smem_lsu_forward},
            simulator_kwargs=smem_simulator_kwargs or {},
        )

        self.cycle_count = 0

    def tick(self) -> None:
        self.dcache_stage.compute()
        self.smem_stage.compute()
        self.cycle_count += 1

    def submit_dcache_request(self, req: Any) -> bool:
        return self.interfaces.lsu_dcache_latch.push(req)

    def submit_smem_transaction(self, txn: Transaction | Dict[str, Any]) -> bool:
        return self.interfaces.lsu_smem_latch.push(txn)

    def pop_dcache_response(self) -> Any:
        return self.interfaces.dcache_lsu_forward.pop()

    def pop_smem_response(self) -> Any:
        return self.interfaces.smem_lsu_forward.pop()

    def get_cycle_counters(self) -> Dict[str, int]:
        return {
            "system_cycle_count": int(self.cycle_count),
            "dcache_cycle_count": int(self.dcache_stage.get_cycle_count()),
            "smem_stage_cycle_count": int(self.smem_stage.get_cycle_count()),
            "smem_model_cycle_count": int(self.smem_stage.simulator.get_cycle_count()),
        }

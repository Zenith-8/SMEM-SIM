from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

from clos_network_sim import (
    ClosNetwork,
    ERR_GOOD as CLOS_ERR_GOOD,
    Flit as ClosFlit,
    NUM_THREADS as CLOS_NUM_THREADS,
)


class TxnType(str, Enum):
    """
    Enumeration of supported shared memory transaction types.
    """
    SH_LD = "sh.ld"
    SH_ST = "sh.st"
    GLOBAL_LD_DRAM_TO_SRAM = "global.ld.dram2sram"
    GLOBAL_ST_SMEM_TO_DRAM = "global.st.smem2dram"

    @classmethod
    def from_user_value(cls, raw_value: str) -> "TxnType":
        """
        Parse a user-provided string into a TxnType enum.
        
        Normalizes the input by removing non-alphanumeric characters and converting to lowercase.
        Maps common aliases to their corresponding TxnType.
        
        Args:
            raw_value: The raw string input representing the transaction type.
            
        Returns:
            The corresponding TxnType enum value.
            
        Raises:
            ValueError: If the raw_value does not match any known aliases.
        """
        key = re.sub(r"[^a-z0-9]+", "", raw_value.lower())
        aliases = {
            "shld": cls.SH_LD,
            "shst": cls.SH_ST,
            "globallddram2sram": cls.GLOBAL_LD_DRAM_TO_SRAM,
            "globalloaddramtosram": cls.GLOBAL_LD_DRAM_TO_SRAM,
            "ldglobal": cls.GLOBAL_LD_DRAM_TO_SRAM,
            "globalstsmem2dram": cls.GLOBAL_ST_SMEM_TO_DRAM,
            "globalstoreshmemtodram": cls.GLOBAL_ST_SMEM_TO_DRAM,
            "stglobal": cls.GLOBAL_ST_SMEM_TO_DRAM,
        }
        if key not in aliases:
            supported = ", ".join(t.value for t in cls)
            raise ValueError(
                f"Unsupported transaction type '{raw_value}'. Supported: {supported}"
            )
        return aliases[key]


@dataclass
class Transaction:
    """
    Represents a single memory transaction request.
    """
    txn_type: TxnType
    dram_addr: Optional[int] = None
    shmem_addr: Optional[int] = None
    write_data: Optional[int] = None
    thread_id: int = 0
    thread_block_offset: Optional[int] = None
    thread_block_id: Optional[int] = None
    resident_thread_block_ids: Optional[Tuple[Optional[int], ...]] = None
    thread_block_done_bits: Any = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Transaction":
        """
        Construct a Transaction instance from a dictionary payload.
        
        Extracts and normalizes transaction fields, handling fallback keys (e.g., type vs txn_type).
        
        Args:
            payload: Dictionary containing transaction parameters.
            
        Returns:
            A new Transaction instance.
            
        Raises:
            ValueError: If the transaction type is missing.
        """
        raw_txn_type = payload.get("type", payload.get("txn_type"))
        if raw_txn_type is None:
            raise ValueError("Transaction is missing 'type'/'txn_type'.")

        if isinstance(raw_txn_type, TxnType):
            txn_type = raw_txn_type
        else:
            txn_type = TxnType.from_user_value(str(raw_txn_type))

        resident_thread_block_ids = payload.get(
            "resident_thread_block_ids",
            payload.get("tbids", payload.get("smem_tbids")),
        )
        if resident_thread_block_ids is not None:
            resident_thread_block_ids = tuple(
                int(tbid) if tbid is not None else None
                for tbid in resident_thread_block_ids
            )

        return cls(
            txn_type=txn_type,
            dram_addr=payload.get("dram_addr"),
            shmem_addr=payload.get("shmem_addr"),
            write_data=payload.get("write_data"),
            thread_id=int(payload.get("thread_id", 0)),
            thread_block_id=(
                int(payload["thread_block_id"])
                if payload.get("thread_block_id") is not None
                else (
                    int(payload["tbid"])
                    if payload.get("tbid") is not None
                    else None
                )
            ),
            resident_thread_block_ids=resident_thread_block_ids,
            thread_block_done_bits=payload.get(
                "thread_block_done_bits",
                payload.get("done_bits"),
            ),
        )


@dataclass
class Completion:
    """
    Represents the completion state of a memory transaction.
    """
    txn_id: int
    txn_type: str
    status: str
    cycle_issued: int
    cycle_completed: int
    thread_id: int
    thread_block_id: Optional[int]
    smem_block_id: Optional[int]
    thread_block_offset_effective: int
    dram_addr: Optional[int]
    shmem_addr: Optional[int]
    absolute_shmem_addr: Optional[int]
    read_data: Optional[int] = None
    note: str = ""
    trace: List[str] = field(default_factory=list)


DEFAULT_SMEM_CONFIG_PATH = Path(".config")
DEFAULT_RESIDENT_THREAD_BLOCK_SLOTS = 4
READ_CROSSBAR_PIPELINE_CYCLES = 3


def _resolve_smem_capacity_fields(
    *,
    resident_thread_block_slots: int = DEFAULT_RESIDENT_THREAD_BLOCK_SLOTS,
    thread_block_size_bytes: Optional[int] = None,
    total_smem_size_bytes: Optional[int] = None,
) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Resolve and validate the resident-slot / block-size / total-size trio.

    The simulator models SMEM capacity as:

    ``total_smem_size_bytes = resident_thread_block_slots * thread_block_size_bytes``

    Any two of the three values are enough to derive the third. When all three
    are provided, the block-size/slot-count product is treated as authoritative
    and ``total_smem_size_bytes`` is normalized to that product.
    """
    slots = int(resident_thread_block_slots)
    if slots <= 0:
        raise ValueError("resident_thread_block_slots must be > 0.")

    block_size = (
        int(thread_block_size_bytes)
        if thread_block_size_bytes is not None
        else None
    )
    total_size = (
        int(total_smem_size_bytes)
        if total_smem_size_bytes is not None
        else None
    )

    if block_size is not None and block_size <= 0:
        raise ValueError("thread_block_size_bytes must be > 0 when provided.")
    if total_size is not None and total_size <= 0:
        raise ValueError("total_smem_size_bytes must be > 0 when provided.")

    if total_size is None and block_size is not None:
        total_size = slots * block_size
    elif total_size is not None and block_size is None:
        if total_size % slots != 0:
            raise ValueError(
                "total_smem_size_bytes must be divisible by "
                "resident_thread_block_slots."
        )
        block_size = total_size // slots
    elif total_size is not None and block_size is not None:
        total_size = slots * block_size

    return slots, block_size, total_size


@dataclass
class SmemSimulatorConfig:
    """
    Configuration parameters for the shared memory simulator.
    """
    num_banks: int = 32
    word_bytes: int = 4
    dram_latency_cycles: int = 1
    arbiter_issue_width: int = 4
    num_threads: int = 1
    resident_thread_block_slots: int = DEFAULT_RESIDENT_THREAD_BLOCK_SLOTS
    thread_block_size_bytes: Optional[int] = None
    total_smem_size_bytes: Optional[int] = None
    read_crossbar_pipeline_cycles: int = 3

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SmemSimulatorConfig":
        """
        Construct a SmemSimulatorConfig instance from a dictionary payload.

        Args:
            payload: Dictionary containing configuration parameters.
            
        Returns:
            A new SmemSimulatorConfig instance.
            
        Raises:
            TypeError: If the payload is not a dict.
        """
        if not isinstance(payload, dict):
            raise TypeError("SMEM config payload must be a dict.")

        resident_thread_block_slots = int(
            payload.get(
                "resident_thread_block_slots",
                payload.get("num_blocks", DEFAULT_RESIDENT_THREAD_BLOCK_SLOTS),
            )
        )
        _, thread_block_size_bytes, total_smem_size_bytes = _resolve_smem_capacity_fields(
            resident_thread_block_slots=resident_thread_block_slots,
            thread_block_size_bytes=payload.get(
                "thread_block_size_bytes",
                payload.get("block_size_bytes"),
            ),
            total_smem_size_bytes=payload.get(
                "total_smem_size_bytes",
                payload.get("total_smem_size"),
            ),
        )

        return cls(
            num_banks=int(payload.get("num_banks", 32)),
            word_bytes=int(payload.get("word_bytes", 4)),
            dram_latency_cycles=int(payload.get("dram_latency_cycles", 1)),
            arbiter_issue_width=int(payload.get("arbiter_issue_width", 4)),
            num_threads=int(payload.get("num_threads", 1)),
            resident_thread_block_slots=resident_thread_block_slots,
            thread_block_size_bytes=thread_block_size_bytes,
            total_smem_size_bytes=total_smem_size_bytes,
            read_crossbar_pipeline_cycles=int(
                payload.get(
                    "read_crossbar_pipeline_cycles",
                    READ_CROSSBAR_PIPELINE_CYCLES,
                )
            ),
        )

    @classmethod
    def from_file(
        cls, config_path: Union[str, Path] = DEFAULT_SMEM_CONFIG_PATH
    ) -> "SmemSimulatorConfig":
        """
        Load a SmemSimulatorConfig from a TOML configuration file.
        
        Args:
            config_path: Path to the TOML configuration file.
            
        Returns:
            A SmemSimulatorConfig instance populated with values from the file, or defaults if the file does not exist.
        """
        path = Path(config_path)
        if not path.exists():
            return cls()
        with path.open("rb") as f:
            raw = tomllib.load(f)
        smem_section = raw.get("smem", {})
        return cls.from_dict(smem_section)

    def to_sim_kwargs(self) -> Dict[str, Any]:
        """
        Convert the configuration into a dictionary of keyword arguments suitable for the simulator.
        
        Returns:
            Dictionary of simulator initialization arguments.
        """
        return {
            "num_banks": int(self.num_banks),
            "word_bytes": int(self.word_bytes),
            "dram_latency_cycles": int(self.dram_latency_cycles),
            "arbiter_issue_width": int(self.arbiter_issue_width),
            "num_threads": int(self.num_threads),
            "resident_thread_block_slots": int(self.resident_thread_block_slots),
            "thread_block_size_bytes": self.thread_block_size_bytes,
            "total_smem_size_bytes": self.total_smem_size_bytes,
            "read_crossbar_pipeline_cycles": int(self.read_crossbar_pipeline_cycles),
        }


def load_smem_config(
    config_path: Union[str, Path] = DEFAULT_SMEM_CONFIG_PATH,
) -> SmemSimulatorConfig:
    """
    Helper function to load the shared memory simulator configuration from a file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        The loaded SmemSimulatorConfig.
    """
    return SmemSimulatorConfig.from_file(config_path)


try:
    from simulator.mem_types import dMemResponse as _SimDMemResponse
except Exception:
    _SimDMemResponse = None


@dataclass
class _CompatDMemResponse:
    """
    A compatibility response object matching the expected interface of the simulator.
    """
    type: str
    req: Any = None
    address: Optional[int] = None
    replay: bool = False
    is_secondary: bool = False
    data: Optional[Any] = None
    miss: bool = False
    hit: bool = False
    stall: bool = False
    uuid: Optional[int] = None
    flushed: bool = False


_DMEM_RESPONSE_CLS = _SimDMemResponse or _CompatDMemResponse
class ShmemFunctionalSimulator:
    """
    Functional-level SMEM model based on the provided flow:
    - Shared memory arbiter
    - SMEM read/write queues
    - SMEM read/write controllers
    - Write crossbar (direct XOR-mapped bank/slot routing)
    - Read crossbar (3-cycle pipelined Clos network)
    - AXI memory read/write queues
    - XOR map
    - Address crossbar
    - Banks

    Bank conflicts are handled by splitting conflicting requests across cycles
    while preserving original transaction issue order.
    """

    def __init__(
        self,
        dram_init: Optional[Dict[int, int]] = None,
        *,
        num_banks: int = 32,
        word_bytes: int = 4,
        dram_latency_cycles: int = 1,
        arbiter_issue_width: int = 4,
        num_threads: int = 1,
        thread_block_offsets: Optional[Dict[int, int] | List[int] | Tuple[int, ...]] = None,
        resident_thread_block_slots: int = DEFAULT_RESIDENT_THREAD_BLOCK_SLOTS,
        thread_block_size_bytes: Optional[int] = None,
        total_smem_size_bytes: Optional[int] = None,
        read_crossbar_pipeline_cycles: int = READ_CROSSBAR_PIPELINE_CYCLES,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the shared memory functional simulator.
        
        Args:
            dram_init: Initial state of the DRAM.
            num_banks: Number of memory banks.
            word_bytes: Number of bytes per word.
            dram_latency_cycles: Latency of DRAM accesses in cycles.
            arbiter_issue_width: Number of requests the arbiter can issue per cycle.
            num_threads: Total number of threads.
            resident_thread_block_slots: Number of resident SMEM thread-block slots.
            thread_block_size_bytes: Size, in bytes, of one resident SMEM thread-block slot.
            total_smem_size_bytes: Total modeled SMEM capacity, in bytes.
            read_crossbar_pipeline_cycles: Latency, in cycles, of the pipelined Clos read crossbar.
            verbose: When True, print per-cycle state summaries during step().
        """
        if num_banks <= 0:
            raise ValueError("num_banks must be > 0.")
        if word_bytes <= 0:
            raise ValueError("word_bytes must be > 0.")
        if dram_latency_cycles < 0:
            raise ValueError("dram_latency_cycles must be >= 0.")
        if arbiter_issue_width <= 0:
            raise ValueError("arbiter_issue_width must be > 0.")
        if num_threads <= 0:
            raise ValueError("num_threads must be > 0.")
        if int(read_crossbar_pipeline_cycles) <= 0:
            raise ValueError("read_crossbar_pipeline_cycles must be > 0.")

        (
            resident_thread_block_slots,
            thread_block_size_bytes,
            total_smem_size_bytes,
        ) = _resolve_smem_capacity_fields(
            resident_thread_block_slots=resident_thread_block_slots,
            thread_block_size_bytes=thread_block_size_bytes,
            total_smem_size_bytes=total_smem_size_bytes,
        )

        self.verbose = verbose
        self.num_banks = num_banks
        self.word_bytes = word_bytes
        self.word_mask = (1 << (8 * word_bytes)) - 1
        self.dram_latency_cycles = dram_latency_cycles
        self.arbiter_issue_width = arbiter_issue_width
        self.num_threads = num_threads
        if thread_block_offsets is not None:
            raise ValueError(
                "Legacy thread_block_offsets are no longer supported. "
                "Use thread_block_size_bytes plus thread_block_id/"
                "resident_thread_block_ids."
            )
        self.resident_thread_block_slots = int(resident_thread_block_slots)
        self.thread_block_size_bytes = (
            int(thread_block_size_bytes)
            if thread_block_size_bytes is not None
            else None
        )
        self.total_smem_size_bytes = (
            int(total_smem_size_bytes)
            if total_smem_size_bytes is not None
            else None
        )
        self.read_crossbar_pipeline_cycles = int(read_crossbar_pipeline_cycles)
        self.resident_thread_block_ids: List[Optional[int]] = [
            None
            for _ in range(self.resident_thread_block_slots)
        ]
        self.resident_thread_block_done_bits: List[Any] = [
            None
            for _ in range(self.resident_thread_block_slots)
        ]
        self.resident_thread_block_done: List[bool] = [
            False
            for _ in range(self.resident_thread_block_slots)
        ]
        self.read_crossbar = ClosNetwork()

        self.cycle = 0
        self.cycle_count = 0
        self._next_txn_id = 1

        self.dram: Dict[int, int] = dict(dram_init or {})
        self.sram_linear: Dict[int, int] = {}
        self.banks: List[Dict[int, int]] = [dict() for _ in range(num_banks)]

        self.input_queue: Deque[Dict[str, Any]] = deque()
        self.smem_read_queue: Deque[Dict[str, Any]] = deque()
        self.smem_write_queue: Deque[Dict[str, Any]] = deque()
        # Unified AXI bus queue.
        #
        # A real AXI bus is a single shared resource between the DRAM read-response
        # path and the DRAM write-issue path; at any given time only one "blocking
        # set" (all reads or all writes) can occupy it, so the previously-separate
        # ``memory_read_queue`` and ``memory_write_queue`` have been merged into
        # one FIFO with kind-tagged entries:
        #
        #   ("read_resp", tagged, value)
        #       An AXI read response that completed DRAM latency and now needs
        #       to deposit ``value`` into the destination SMEM bank/slot.
        #
        #   ("write_req", tagged, dram_addr, value)
        #       A value that was read out of SMEM and is waiting to be issued
        #       to DRAM via the AXI write port.
        #
        # The ``_run_axi_bus`` step services at most ONE head entry per cycle,
        # modeling the single-port nature of the AXI bus and serializing the
        # two directions against each other.
        self.axi_bus_queue: Deque[Dict[str, Any]] = deque()

        self.pending_dram_reads: List[Tuple[int, Dict[str, Any], int]] = []
        self.pending_dram_writes: List[Tuple[int, Dict[str, Any], int, int]] = []
        self.pending_read_crossbar_deliveries: List[Dict[str, Any]] = []

        self.completions: List[Completion] = []

    def issue(self, transaction: Transaction) -> int:
        """
        Issue a new transaction to the simulator.
        
        Validates the transaction and adds it to the input queue with a unique transaction ID.
        
        Args:
            transaction: The transaction to issue.
            
        Returns:
            The unique transaction ID assigned to this request.
        """
        self._validate_transaction(transaction)
        thread_id = int(transaction.thread_id)
        smem_block_id = self._smem_block_id_for_transaction(transaction)
        if smem_block_id is not None:
            if self.thread_block_size_bytes is None:
                raise ValueError(
                    "thread_block_size_bytes must be configured when using "
                    "thread_block_id-based SMEM residency."
                )
            effective_offset = int(smem_block_id) * int(self.thread_block_size_bytes)
        else:
            effective_offset = 0
        accept_parts = [
            f"thread={thread_id}",
            f"tbo=0x{effective_offset:x}",
        ]
        if transaction.thread_block_id is not None:
            accept_parts.append(f"tbid={int(transaction.thread_block_id)}")
        if smem_block_id is not None:
            accept_parts.append(f"smem_block={int(smem_block_id)}")
        tagged = {
            "txn_id": self._next_txn_id,
            "txn": transaction,
            "cycle_issued": self.cycle,
            "ready_cycle": self.cycle,
            "trace": [
                f"cycle {self.cycle}: accepted by simulator input "
                f"({', '.join(accept_parts)})"
            ],
        }
        self._next_txn_id += 1
        self.input_queue.append(tagged)
        return tagged["txn_id"]

    def _enqueue_for_next_cycle(self, queue: Deque[Dict[str, Any]], tagged: Dict[str, Any]) -> None:
        """
        Enqueue a transaction into a downstream pipeline queue.

        Internal queue hops consume one cycle: an entry produced during cycle
        ``N`` cannot be consumed by the downstream stage until cycle ``N + 1``.
        """
        tagged["ready_cycle"] = self.cycle + 1
        queue.append(tagged)

    @staticmethod
    def _queue_head_ready(queue: Deque[Dict[str, Any]], cycle: int) -> bool:
        """
        Check whether the queue head is eligible to be consumed this cycle.
        """
        if not queue:
            return False
        return int(queue[0].get("ready_cycle", cycle)) <= int(cycle)

    @staticmethod
    def _normalize_resident_thread_block_ids(
        resident_thread_block_ids: Sequence[Optional[int]],
        *,
        expected_slots: int = DEFAULT_RESIDENT_THREAD_BLOCK_SLOTS,
    ) -> Tuple[Optional[int], ...]:
        """
        Normalize the resident SMEM thread-block ID vector to the fixed slot count.
        """
        normalized = tuple(
            int(tbid) if tbid is not None else None
            for tbid in resident_thread_block_ids
        )
        if len(normalized) != int(expected_slots):
            raise ValueError(
                f"resident_thread_block_ids must contain exactly "
                f"{int(expected_slots)} entries."
            )
        return normalized

    @staticmethod
    def _done_bits_all_one(done_bits: Any) -> bool:
        """
        Determine whether a thread block's done bits indicate completion.
        """
        if done_bits is None:
            return False

        if isinstance(done_bits, str):
            bits = done_bits.strip().lower().replace("_", "")
            if bits.startswith("0b"):
                bits = bits[2:]
            return bool(bits) and all(ch == "1" for ch in bits)

        if isinstance(done_bits, int):
            if done_bits < 0:
                raise ValueError("done bits must be non-negative.")
            bits = bin(done_bits)[2:]
            return bool(bits) and all(ch == "1" for ch in bits)

        if hasattr(done_bits, "bin"):
            bits = str(done_bits.bin)
            return bool(bits) and all(ch == "1" for ch in bits)

        if isinstance(done_bits, Iterable):
            bits_list = list(done_bits)
            return bool(bits_list) and all(bool(bit) for bit in bits_list)

        return bool(done_bits)

    def _find_resident_thread_block_slot(self, thread_block_id: int) -> Optional[int]:
        """
        Locate the resident SMEM slot currently assigned to ``thread_block_id``.
        """
        for slot_idx, resident_tbid in enumerate(self.resident_thread_block_ids):
            if resident_tbid == int(thread_block_id):
                return slot_idx
        return None

    def _update_resident_thread_block_state(self, txn: Transaction) -> None:
        """
        Synchronize the 4-slot resident thread-block table from the transaction metadata.

        Each slot retains its SMEM allocation until the *currently resident*
        thread block reports all-one done bits. Only then may a newly supplied
        thread block ID replace that slot.
        """
        resident_ids = txn.resident_thread_block_ids
        if resident_ids is not None:
            incoming_ids = self._normalize_resident_thread_block_ids(
                resident_ids,
                expected_slots=self.resident_thread_block_slots,
            )
            for slot_idx, incoming_tbid in enumerate(incoming_ids):
                resident_tbid = self.resident_thread_block_ids[slot_idx]
                if resident_tbid == incoming_tbid:
                    continue

                if resident_tbid is None or self.resident_thread_block_done[slot_idx]:
                    self.resident_thread_block_ids[slot_idx] = incoming_tbid
                    self.resident_thread_block_done_bits[slot_idx] = None
                    self.resident_thread_block_done[slot_idx] = False

        if txn.thread_block_id is None or txn.thread_block_done_bits is None:
            return

        slot_idx = self._find_resident_thread_block_slot(int(txn.thread_block_id))
        if slot_idx is None:
            return

        self.resident_thread_block_done_bits[slot_idx] = txn.thread_block_done_bits
        self.resident_thread_block_done[slot_idx] = self._done_bits_all_one(
            txn.thread_block_done_bits
        )

    def _smem_block_id_for_transaction(self, txn: Transaction) -> Optional[int]:
        """
        Resolve the resident SMEM slot assigned to a transaction's thread block.
        """
        if txn.thread_block_id is None:
            return None

        self._update_resident_thread_block_state(txn)
        slot_idx = self._find_resident_thread_block_slot(int(txn.thread_block_id))
        if slot_idx is not None:
            return slot_idx

        if txn.resident_thread_block_ids is not None:
            requested_ids = self._normalize_resident_thread_block_ids(
                txn.resident_thread_block_ids,
                expected_slots=self.resident_thread_block_slots,
            )
            if int(txn.thread_block_id) in requested_ids:
                requested_slot = requested_ids.index(int(txn.thread_block_id))
                resident_tbid = self.resident_thread_block_ids[requested_slot]
                resident_done = self.resident_thread_block_done[requested_slot]
                raise ValueError(
                    f"thread_block_id {int(txn.thread_block_id)} cannot take SMEM "
                    f"slot {requested_slot}: slot is still occupied by "
                    f"thread_block_id {resident_tbid} "
                    f"(done={int(resident_done)})."
                )

        return None

    @staticmethod
    def _clos_thread_lane(thread_id: int) -> int:
        """
        Map the simulator thread id onto one of the 32 Clos network thread lanes.
        """
        return int(thread_id) % int(CLOS_NUM_THREADS)

    def _write_crossbar_target(self, txn: Transaction) -> Tuple[int, int, int]:
        """
        Route a write-side transaction through the simple write crossbar.

        The write crossbar is a direct mapping: the XOR-based address crossbar
        selects the bank/slot, and the write is applied at that location.
        """
        absolute = self._absolute_smem_addr(txn)
        bank, bank_slot = self._address_crossbar(
            absolute, self._effective_thread_block_offset(txn)
        )
        return absolute, bank, bank_slot

    def _multicast_read_key(self, txn: Transaction) -> Optional[Tuple[int, int]]:
        """
        Return a coalescing key for read-side multicast candidates.

        Only ``sh.ld`` transactions can exploit the Clos read crossbar's
        multicast capability: multiple lanes reading the same final shared
        memory word can be serviced by one bank read plus one multicast flit.
        """
        if txn.txn_type is not TxnType.SH_LD:
            return None
        absolute = self._absolute_smem_addr(txn)
        bank = self._bank_for_transaction(txn)
        return (int(bank), int(absolute))

    def _multicast_group_members(
        self, tagged: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Return the logical read cohort represented by a queue entry.
        """
        return [tagged, *list(tagged.get("multicast_peers", []))]

    def _can_multicast_merge(
        self, leader: Dict[str, Any], candidate: Dict[str, Any]
    ) -> bool:
        """
        Decide whether ``candidate`` can merge into ``leader``'s multicast read.

        The Clos fabric fans one flit out by destination-lane mask, so every
        merged logical recipient must map onto a distinct Clos thread lane.
        """
        candidate_lane = self._clos_thread_lane(int(candidate["txn"].thread_id))
        for member in self._multicast_group_members(leader):
            lane = self._clos_thread_lane(int(member["txn"].thread_id))
            if int(lane) == int(candidate_lane):
                return False
        return True

    def _launch_read_crossbar(
        self,
        tagged_group: Sequence[Dict[str, Any]],
        bank: int,
        value: int,
    ) -> None:
        """
        Launch a load response into the 3-cycle pipelined Clos read crossbar.
        """
        recipients: List[Dict[str, Any]] = []
        dest_mask = 0
        ready_cycle = self.cycle + int(self.read_crossbar_pipeline_cycles)
        is_multicast = len(tagged_group) > 1

        for tagged in tagged_group:
            txn: Transaction = tagged["txn"]
            lane = self._clos_thread_lane(int(txn.thread_id))
            if dest_mask & (1 << int(lane)):
                raise RuntimeError(
                    f"Clos read crossbar multicast collision on lane {lane} "
                    f"in cycle {self.cycle}."
                )
            dest_mask |= 1 << int(lane)
            recipients.append(
                {
                    "tagged": tagged,
                    "lane": int(lane),
                    "thread_id": int(txn.thread_id),
                }
            )
            launch_kind = "multicast launch" if is_multicast else "launch"
            tagged["trace"].append(
                f"cycle {self.cycle}: Clos read crossbar {launch_kind} "
                f"(bank {int(bank)} -> lane {int(lane)}), ready cycle {ready_cycle}"
            )

        self.pending_read_crossbar_deliveries.append(
            {
                "ready_cycle": ready_cycle,
                "bank": int(bank),
                "dest_mask": int(dest_mask),
                "value": int(value) & self.word_mask,
                "recipients": recipients,
            }
        )

    def _service_read_crossbar(self) -> None:
        """
        Retire any Clos-read-crossbar responses whose 3-cycle pipeline has completed.
        """
        if not self.pending_read_crossbar_deliveries:
            return

        ready_events: List[Dict[str, Any]] = []
        next_pending: List[Dict[str, Any]] = []
        for event in self.pending_read_crossbar_deliveries:
            if int(event["ready_cycle"]) <= self.cycle:
                ready_events.append(event)
            else:
                next_pending.append(event)
        if not ready_events:
            return

        flits_from_banks: Dict[int, ClosFlit] = {}
        for event in ready_events:
            bank = int(event["bank"])
            if bank in flits_from_banks:
                raise RuntimeError(
                    f"Clos read crossbar received more than one ready flit for bank {bank} "
                    f"in cycle {self.cycle}."
                )
            flits_from_banks[bank] = ClosFlit.make(
                dest_mask=int(event["dest_mask"]),
                data=int(event["value"]),
                error=CLOS_ERR_GOOD,
            )

        deliveries = self.read_crossbar.send(flits_from_banks)
        for event in ready_events:
            recipients = list(event.get("recipients", []))
            is_multicast = len(recipients) > 1
            for recipient in recipients:
                lane = int(recipient["lane"])
                thread_id = int(recipient["thread_id"])
                tagged = recipient["tagged"]
                rxs = deliveries.get(lane, [])
                if not rxs:
                    raise RuntimeError(
                        f"Clos read crossbar dropped delivery for thread {thread_id} "
                        f"(lane {lane}) in cycle {self.cycle}."
                    )

                data, error = rxs.pop(0)
                if int(error) != int(CLOS_ERR_GOOD):
                    raise RuntimeError(
                        f"Clos read crossbar returned error {error} for thread {thread_id} "
                        f"in cycle {self.cycle}."
                    )

                route_note = (
                    "Clos multicast read crossbar"
                    if is_multicast
                    else "Clos read crossbar"
                )
                tagged["trace"].append(
                    f"cycle {self.cycle}: {route_note} -> thread {thread_id} "
                    f"(lane {lane})"
                )
                self._complete(
                    tagged,
                    read_data=int(data) & self.word_mask,
                    note=(
                        f"Read returned through {self.read_crossbar_pipeline_cycles}-cycle "
                        f"{route_note}."
                    ),
                )

        self.pending_read_crossbar_deliveries = next_pending

    def run(
        self, transactions: Iterable[Transaction | Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run the simulator for a batch of transactions until completion.
        
        Args:
            transactions: An iterable of transactions (or dict payloads) to process.
            
        Returns:
            A snapshot of the simulator state after all transactions have completed.
        """
        for raw in transactions:
            txn = raw if isinstance(raw, Transaction) else Transaction.from_dict(raw)
            self.issue(txn)

        while self._has_pending_work():
            self.step()

        return self.snapshot()

    def run_one(self, transaction: Transaction | Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single transaction through the simulator until it completes.
        
        Args:
            transaction: The transaction (or dict payload) to process.
            
        Returns:
            The completion record for the transaction as a dictionary.
        """
        start_idx = len(self.completions)
        txn = (
            transaction
            if isinstance(transaction, Transaction)
            else Transaction.from_dict(transaction)
        )
        self.issue(txn)
        while len(self.completions) == start_idx:
            self.step()
        return asdict(self.completions[-1])

    def step(self) -> None:
        """
        Advance the simulator by a single clock cycle.
        
        Executes all sub-components: DRAM events, SMEM arbiter, read/write controllers, and AXI ports.
        When ``verbose`` is enabled, prints the full pipeline state before and
        after the cycle, plus a list of completions.
        """
        if self.verbose:
            self._log_cycle_summary(phase="BEGIN")

        prev_completions = len(self.completions)

        arbiter_banks_issued_this_cycle: Set[int] = set()
        banks_used_by_controllers_this_cycle: Set[int] = set()

        self._service_read_crossbar()
        self._service_dram_events()
        self._run_shared_memory_arbiter(arbiter_banks_issued_this_cycle)
        self._run_smem_write_controller(banks_used_by_controllers_this_cycle)
        # The AXI bus is driven in two phases per cycle, both acting on the
        # same unified ``axi_bus_queue``:
        #
        #   Phase 1 (read-response): runs BEFORE the SMEM read controller so
        #     a DRAM read response at the queue head is deposited into SMEM
        #     in the same cycle the read controller observes the bank.
        #
        #   Phase 2 (write-request): runs AFTER the SMEM read controller so
        #     a ``global.st`` just enqueued by the read controller is issued
        #     to DRAM in the same cycle (matching the old timing where the
        #     dedicated AXI write port ran last).
        #
        # Each phase consumes at most ONE matching head entry; the queue
        # stays strict-FIFO. This preserves the old parallel-channel AXI
        # throughput while unifying the two directions into a single queue.
        self._run_axi_bus_read_phase(banks_used_by_controllers_this_cycle)
        self._run_smem_read_controller(banks_used_by_controllers_this_cycle)
        self._run_axi_bus_write_phase()

        if self.verbose:
            self._log_cycle_completions(prev_completions)
            self._log_cycle_summary(phase="END")

        self.cycle += 1
        self.cycle_count = self.cycle

    def snapshot(self) -> Dict[str, Any]:
        """
        Capture the current state of the simulator.
        
        Returns:
            A dictionary containing cycle counts, memory states, and completion records.
        """
        return {
            "cycle": self.cycle,
            "cycle_count": self.cycle_count,
            "dram": dict(self.dram),
            "sram_linear": dict(self.sram_linear),
            "banks": [dict(bank) for bank in self.banks],
            "resident_thread_block_ids": list(self.resident_thread_block_ids),
            "resident_thread_block_done": list(self.resident_thread_block_done),
            "resident_thread_block_slots": int(self.resident_thread_block_slots),
            "thread_block_size_bytes": self.thread_block_size_bytes,
            "total_smem_size_bytes": self.total_smem_size_bytes,
            "pending_read_crossbar_deliveries": [
                dict(event) for event in self.pending_read_crossbar_deliveries
            ],
            "completions": [asdict(c) for c in self.completions],
        }

    def get_cycle_count(self) -> int:
        """
        Get the current cycle count of the simulator.
        
        Returns:
            The total number of cycles executed.
        """
        return int(self.cycle_count)

    def _run_shared_memory_arbiter(self, banks_issued_this_cycle: Set[int]) -> None:
        """
        Execute the shared memory arbiter logic for the current cycle.

        Warp-synchronous bank-parallel issue model:

        * The arbiter scans the entire ``input_queue`` from head to tail.
        * Each cycle it issues at most one request per distinct bank (so up
          to ``num_banks`` requests in parallel when ``arbiter_issue_width``
          is >= num_banks), modeling the per-bank output ports.
        * When a request at position *k* targets a bank that has already
          been issued to this cycle, the request is *deferred* (left in the
          queue in its original order) instead of breaking the scan. This
          matches how a real hardware arbiter serializes bank conflicts
          across cycles without blocking unrelated lanes.
        * ``arbiter_issue_width`` still caps the total number of parallel
          issues per cycle (useful for modeling narrower arbiter hardware).

        Relative order within each bank is preserved, which is what
        downstream order-sensitive logic (e.g. read-after-write to the same
        SMEM slot) relies on.

        Args:
            banks_issued_this_cycle: Set of bank indices already issued to in this cycle.
        """
        if not self.input_queue:
            return

        issued_count = 0
        deferred: Deque[Dict[str, Any]] = deque()
        issued_multicast_leaders: Dict[Tuple[int, int], Dict[str, Any]] = {}

        while self.input_queue:
            tagged = self.input_queue.popleft()
            txn: Transaction = tagged["txn"]
            bank = self._bank_for_transaction(txn)
            multicast_key = self._multicast_read_key(txn)

            if issued_count >= self.arbiter_issue_width:
                # Arbiter saturated this cycle; keep the rest in FIFO order.
                deferred.append(tagged)
                continue

            if bank in banks_issued_this_cycle:
                leader = (
                    issued_multicast_leaders.get(multicast_key)
                    if multicast_key is not None
                    else None
                )
                if leader is not None and self._can_multicast_merge(leader, tagged):
                    leader.setdefault("multicast_peers", []).append(tagged)
                    leader_txn: Transaction = leader["txn"]
                    tagged["trace"].append(
                        f"cycle {self.cycle}: arbiter multicast-merged with "
                        f"thread {int(leader_txn.thread_id)} on bank {bank}"
                    )
                    leader["trace"].append(
                        f"cycle {self.cycle}: arbiter multicast-merged thread "
                        f"{int(txn.thread_id)}"
                    )
                    continue
                # Bank already claimed by an earlier-lane request this cycle.
                # Defer this lane to a later cycle but keep scanning the tail
                # for requests to still-free banks.
                tagged["trace"].append(
                    f"cycle {self.cycle}: arbiter stall on bank conflict (bank {bank})"
                )
                deferred.append(tagged)
                continue

            banks_issued_this_cycle.add(bank)
            issued_count += 1

            if txn.txn_type in (TxnType.SH_LD, TxnType.GLOBAL_ST_SMEM_TO_DRAM):
                if multicast_key is not None:
                    tagged.setdefault("multicast_peers", [])
                    issued_multicast_leaders[multicast_key] = tagged
                self._enqueue_for_next_cycle(self.smem_read_queue, tagged)
                tagged["trace"].append(f"cycle {self.cycle}: arbiter -> SMEM Read Queue")
            else:
                self._enqueue_for_next_cycle(self.smem_write_queue, tagged)
                tagged["trace"].append(
                    f"cycle {self.cycle}: arbiter -> SMEM Write Queue"
                )

        # Restore deferred requests to the head of the input queue for next
        # cycle, preserving their original relative order.
        self.input_queue = deferred

    def _run_smem_read_controller(self, banks_used_this_cycle: Set[int]) -> None:
        """
        Execute the SMEM read controller logic for the current cycle.
        
        Processes read requests and global store requests from the read queue.
        
        Args:
            banks_used_this_cycle: Set of bank indices already accessed in this cycle.
        """
        if not self.smem_read_queue:
            return
        if not self._queue_head_ready(self.smem_read_queue, self.cycle):
            return

        tagged = self.smem_read_queue[0]
        txn: Transaction = tagged["txn"]
        _, bank, bank_slot = self._write_crossbar_target(txn)

        if bank in banks_used_this_cycle:
            tagged["trace"].append(
                f"cycle {self.cycle}: SMEM Read Controller stall (bank {bank} busy)"
            )
            return

        self.smem_read_queue.popleft()
        banks_used_this_cycle.add(bank)

        if txn.txn_type == TxnType.SH_LD:
            # Read the bank-selected word, then return it through the pipelined Clos read crossbar.
            value = self.banks[bank].get(bank_slot, 0)
            multicast_group = self._multicast_group_members(tagged)
            tagged["trace"].append(
                f"cycle {self.cycle}: SMEM Read Controller read bank {bank}, slot {bank_slot}"
            )
            if len(multicast_group) > 1:
                leader_tid = int(txn.thread_id)
                for peer in tagged.get("multicast_peers", []):
                    peer_txn: Transaction = peer["txn"]
                    peer["trace"].append(
                        f"cycle {self.cycle}: SMEM Read Controller multicast-read "
                        f"bank {bank}, slot {bank_slot} with thread {leader_tid}"
                    )
            self._launch_read_crossbar(multicast_group, bank, value)
            return

        if txn.txn_type == TxnType.GLOBAL_ST_SMEM_TO_DRAM:
            # Read the value from SMEM and enqueue an AXI write on the unified AXI bus.
            value = self.banks[bank].get(bank_slot, 0)
            self._enqueue_for_next_cycle(
                self.axi_bus_queue,
                {
                    "kind": "write_req",
                    "tagged": tagged,
                    "dram_addr": txn.dram_addr,
                    "value": value,
                    "ready_cycle": self.cycle,
                },
            )
            tagged["trace"].append(
                f"cycle {self.cycle}: SMEM Read Controller -> AXI Bus Queue (write_req)"
            )
            return

        raise RuntimeError(f"Unexpected read-controller transaction: {txn.txn_type}")

    def _run_smem_write_controller(self, banks_used_this_cycle: Set[int]) -> None:
        """
        Execute the SMEM write controller logic for the current cycle.

        Processes standard store requests (``sh.st``) and global-load issue
        (``global.ld.dram2sram``) from the SMEM write queue. AXI read responses
        that deposit data back into SMEM are now handled by the unified AXI
        bus (see ``_run_axi_bus``).

        Args:
            banks_used_this_cycle: Set of bank indices already accessed in this cycle.
        """
        if not self.smem_write_queue:
            return
        if not self._queue_head_ready(self.smem_write_queue, self.cycle):
            return

        tagged = self.smem_write_queue[0]
        txn: Transaction = tagged["txn"]
        absolute, bank, bank_slot = self._write_crossbar_target(txn)

        if txn.txn_type == TxnType.SH_ST:
            # Mask the write data to ensure it fits within the configured word size
            value = int(txn.write_data) & self.word_mask

            if bank in banks_used_this_cycle:
                tagged["trace"].append(
                    f"cycle {self.cycle}: SMEM Write Controller stall (bank {bank} busy)"
                )
                return

            self.smem_write_queue.popleft()
            banks_used_this_cycle.add(bank)
            prior_value = self.banks[bank].get(bank_slot, 0)
            self.banks[bank][bank_slot] = value
            self.sram_linear[absolute] = value
            tagged["trace"].append(
                f"cycle {self.cycle}: Write Crossbar -> bank {bank}, slot {bank_slot}"
            )
            self._complete(tagged, read_data=prior_value)
            return

        if txn.txn_type == TxnType.GLOBAL_LD_DRAM_TO_SRAM:
            self.smem_write_queue.popleft()
            # Simulate an AXI read by fetching data from DRAM and scheduling its arrival based on latency
            dram_data = int(self.dram.get(txn.dram_addr, 0)) & self.word_mask
            ready = self.cycle + self.dram_latency_cycles
            self.pending_dram_reads.append((ready, tagged, dram_data))
            tagged["trace"].append(
                f"cycle {self.cycle}: issued AXI read @0x{txn.dram_addr:x}, "
                f"response due cycle {ready}"
            )
            return

        raise RuntimeError(f"Unexpected write-controller transaction: {txn.txn_type}")

    def _run_axi_bus_read_phase(self, banks_used_this_cycle: Set[int]) -> None:
        """
        AXI bus -- read-response phase.

        If the head of ``axi_bus_queue`` is a ``read_resp`` entry, deposit
        its payload into the destination SMEM bank/slot (subject to
        bank-busy arbitration) and complete the transaction. If the head
        is a ``write_req`` (or the queue is empty), this phase is a no-op
        and the write_req will be handled by ``_run_axi_bus_write_phase``
        at the end of the cycle.

        Strict FIFO semantics are preserved: at most ONE entry is consumed
        per cycle by this phase, and only if that entry is at the head.

        Args:
            banks_used_this_cycle: Set of bank indices already accessed in this cycle.
        """
        if not self.axi_bus_queue:
            return

        head = self.axi_bus_queue[0]
        if not self._queue_head_ready(self.axi_bus_queue, self.cycle):
            return
        if head["kind"] != "read_resp":
            return

        tagged = head["tagged"]
        value = head["value"]
        txn: Transaction = tagged["txn"]
        absolute, bank, bank_slot = self._write_crossbar_target(txn)

        if bank in banks_used_this_cycle:
            tagged["trace"].append(
                f"cycle {self.cycle}: AXI Bus (read phase) stall (bank {bank} busy)"
            )
            return

        self.axi_bus_queue.popleft()
        banks_used_this_cycle.add(bank)
        self.banks[bank][bank_slot] = value
        self.sram_linear[absolute] = value
        tagged["trace"].append(
            f"cycle {self.cycle}: AXI Bus Queue -> Write Crossbar -> SMEM bank "
            f"{bank}, slot {bank_slot}"
        )
        self._complete(tagged)

    def _run_axi_bus_write_phase(self) -> None:
        """
        AXI bus -- write-request phase.

        If the head of ``axi_bus_queue`` is a ``write_req`` entry, issue the
        AXI write toward DRAM by scheduling it in ``pending_dram_writes``
        with the configured DRAM latency. If the head is a ``read_resp`` that
        hasn't been serviced this cycle (e.g. because of a bank conflict),
        that read_resp blocks any trailing write_req -- this preserves the
        "single AXI bus, strict FIFO" contract.

        At most ONE entry is consumed per cycle by this phase.
        """
        if not self.axi_bus_queue:
            return

        head = self.axi_bus_queue[0]
        if not self._queue_head_ready(self.axi_bus_queue, self.cycle):
            return
        if head["kind"] != "write_req":
            return

        tagged = head["tagged"]
        dram_addr = head["dram_addr"]
        value = head["value"]
        self.axi_bus_queue.popleft()
        ready = self.cycle + self.dram_latency_cycles
        self.pending_dram_writes.append((ready, tagged, dram_addr, value))
        tagged["trace"].append(
            f"cycle {self.cycle}: AXI write issued @0x{int(dram_addr):x}, "
            f"commit due cycle {ready}"
        )

    def _service_dram_events(self) -> None:
        """
        Service pending DRAM read and write events that have completed their latency period.

        Completed reads are pushed onto the unified AXI bus queue as
        ``read_resp`` entries (to be deposited into SMEM by ``_run_axi_bus``),
        and completed writes commit their payload to the DRAM model and
        mark the originating transaction complete.
        """
        next_pending_reads: List[Tuple[int, Dict[str, Any], int]] = []
        for ready_cycle, tagged, value in self.pending_dram_reads:
            if ready_cycle <= self.cycle:
                self._enqueue_for_next_cycle(
                    self.axi_bus_queue,
                    {
                        "kind": "read_resp",
                        "tagged": tagged,
                        "value": value,
                        "ready_cycle": self.cycle,
                    },
                )
                tagged["trace"].append(
                    f"cycle {self.cycle}: AXI read response -> AXI Bus Queue"
                )
            else:
                next_pending_reads.append((ready_cycle, tagged, value))
        self.pending_dram_reads = next_pending_reads

        next_pending_writes: List[Tuple[int, Dict[str, Any], int, int]] = []
        for ready_cycle, tagged, dram_addr, value in self.pending_dram_writes:
            if ready_cycle <= self.cycle:
                self.dram[dram_addr] = int(value) & self.word_mask
                tagged["trace"].append(
                    f"cycle {self.cycle}: AXI write committed @0x{dram_addr:x}"
                )
                self._complete(tagged)
            else:
                next_pending_writes.append((ready_cycle, tagged, dram_addr, value))
        self.pending_dram_writes = next_pending_writes

    def _xor_map(self, absolute_smem_addr: int, thread_block_offset: int) -> int:
        """
        Compute the XOR-mapped word address to reduce bank conflicts.
        
        Args:
            absolute_smem_addr: The absolute shared memory address.
            thread_block_offset: The offset for the current thread block.
            
        Returns:
            The XOR-mapped word address.
        """
        # Convert byte addresses to word addresses for banking logic
        word_addr = absolute_smem_addr // self.word_bytes
        offset_words = thread_block_offset // self.word_bytes
        # XOR the word address with the offset to distribute accesses across banks and minimize conflicts
        return word_addr ^ offset_words

    def _address_crossbar(
        self, absolute_smem_addr: int, thread_block_offset: int
    ) -> Tuple[int, int]:
        """
        Route an address through the crossbar to determine its bank and slot.
        
        Args:
            absolute_smem_addr: The absolute shared memory address.
            thread_block_offset: The offset for the current thread block.
            
        Returns:
            A tuple of (bank_index, bank_slot).
        """
        # Calculate the absolute word index in the memory space
        absolute_word = absolute_smem_addr // self.word_bytes
        # Apply XOR mapping to determine the effective word for bank selection
        mapped_word = self._xor_map(absolute_smem_addr, thread_block_offset)
        # The bank is determined by taking the modulo of the mapped word
        bank = mapped_word % self.num_banks
        # Preserve uniqueness in storage location across thread offsets:
        # XOR remaps bank selection, while slot still tracks absolute address space.
        bank_slot = absolute_word // self.num_banks
        return bank, bank_slot

    def _bank_for_transaction(self, txn: Transaction) -> int:
        """
        Determine the target bank for a given transaction.
        
        Args:
            txn: The transaction to evaluate.
            
        Returns:
            The index of the target bank.
        """
        absolute = self._absolute_smem_addr(txn)
        bank, _ = self._address_crossbar(absolute, self._effective_thread_block_offset(txn))
        return bank

    def _complete(
        self, tagged: Dict[str, Any], *, read_data: Optional[int] = None, note: str = ""
    ) -> None:
        """
        Mark a transaction as complete and record its completion state.
        
        Args:
            tagged: The tagged transaction dictionary.
            read_data: Optional data read during the transaction.
            note: Optional note to attach to the completion record.
        """
        txn: Transaction = tagged["txn"]
        smem_block_id = self._smem_block_id_for_transaction(txn)
        effective_offset = self._effective_thread_block_offset(txn)
        completion = Completion(
            txn_id=tagged["txn_id"],
            txn_type=txn.txn_type.value,
            status="ok",
            cycle_issued=tagged["cycle_issued"],
            cycle_completed=self.cycle,
            thread_id=int(txn.thread_id),
            thread_block_id=(
                int(txn.thread_block_id)
                if txn.thread_block_id is not None
                else None
            ),
            smem_block_id=smem_block_id,
            thread_block_offset_effective=effective_offset,
            dram_addr=txn.dram_addr,
            shmem_addr=txn.shmem_addr,
            absolute_shmem_addr=self._absolute_smem_addr(txn),
            read_data=read_data,
            note=note,
            trace=list(tagged["trace"]),
        )
        self.completions.append(completion)

    def _absolute_smem_addr(self, txn: Transaction) -> int:
        """
        Calculate the absolute shared memory address for a transaction.
        
        Args:
            txn: The transaction.
            
        Returns:
            The absolute memory address including thread block offsets.
        """
        return int(txn.shmem_addr) + self._effective_thread_block_offset(txn)

    def _effective_thread_block_offset(self, txn: Transaction) -> int:
        """
        Determine the effective thread block offset for a transaction.
        
        Args:
            txn: The transaction.
            
        Returns:
            The effective offset value.
            
        Raises:
            ValueError: If the thread ID is out of range.
        """
        smem_block_id = self._smem_block_id_for_transaction(txn)
        if smem_block_id is not None:
            if self.thread_block_size_bytes is None:
                raise ValueError(
                    "thread_block_size_bytes must be configured when using "
                    "thread_block_id-based SMEM residency."
                )
            return int(smem_block_id) * int(self.thread_block_size_bytes)
        return 0

    def _has_pending_work(self) -> bool:
        """
        Check if there is any pending work remaining in the simulator queues.
        
        Returns:
            True if there is pending work, False otherwise.
        """
        return any(
            (
                self.input_queue,
                self.smem_read_queue,
                self.smem_write_queue,
                self.axi_bus_queue,
                self.pending_dram_reads,
                self.pending_dram_writes,
                self.pending_read_crossbar_deliveries,
            )
        )

    def _fmt_tagged(self, tagged: Dict[str, Any]) -> str:
        """
        Format a tagged transaction dict into a compact multi-field description.

        Includes thread ID, txn ID, type, shmem address, absolute address,
        TBO, bank/slot mapping, write data, dram address, and current bank
        content (for reads).

        Args:
            tagged: The tagged transaction dictionary.

        Returns:
            A human-readable string with full per-thread request details.
        """
        txn: Transaction = tagged["txn"]
        txn_id: int = tagged["txn_id"]
        tid = int(txn.thread_id)
        addr = int(txn.shmem_addr) if txn.shmem_addr is not None else 0
        tbo = self._effective_thread_block_offset(txn)
        absolute = self._absolute_smem_addr(txn)
        bank, slot = self._address_crossbar(absolute, tbo)
        smem_block_id = self._smem_block_id_for_transaction(txn)
        ready_cycle = int(tagged.get("ready_cycle", self.cycle))

        line = (
            f"T{tid:02d} #{txn_id} {txn.txn_type.value:<20s}"
            f"  shmem=0x{addr:04x}  abs=0x{absolute:04x}  tbo=0x{tbo:04x}"
            f"  -> bank {bank:2d}, slot {slot}"
        )
        if txn.thread_block_id is not None:
            line += f"  | tbid={int(txn.thread_block_id)}"
        if smem_block_id is not None:
            line += f"  smem_block={smem_block_id}"
        if ready_cycle > self.cycle:
            line += f"  ready@{ready_cycle}"

        if txn.txn_type == TxnType.SH_ST and txn.write_data is not None:
            line += f"  | write_data=0x{int(txn.write_data) & self.word_mask:08x}"

        if txn.txn_type == TxnType.SH_LD:
            current = self.banks[bank].get(slot, 0)
            line += f"  | bank_content=0x{current & self.word_mask:08x}"

        if txn.dram_addr is not None:
            line += f"  | dram_addr=0x{txn.dram_addr:04x}"
            if txn.txn_type == TxnType.GLOBAL_LD_DRAM_TO_SRAM:
                dram_val = int(self.dram.get(txn.dram_addr, 0)) & self.word_mask
                line += f"  dram_content=0x{dram_val:08x}"

        return line

    def _log_cycle_summary(self, phase: str = "BEGIN") -> None:
        """
        Print a formatted summary of every queue in the simulator, showing
        the thread and request state for each entry.

        Args:
            phase: Label printed in the header (e.g. ``"BEGIN"`` or ``"END"``).
        """
        separator = "-" * 78
        print(f"\n{separator}")
        print(f"  CYCLE {self.cycle}  [{phase}]")
        print(separator)
        resident_summary = ", ".join(
            (
                f"slot{slot_idx}=tbid:{resident_tbid}"
                f"/done:{int(self.resident_thread_block_done[slot_idx])}"
            )
            for slot_idx, resident_tbid in enumerate(self.resident_thread_block_ids)
        )
        print(f"  Resident Thread Blocks: {resident_summary}")

        def _print_queue(
            label: str, items: Iterable[Any], extractor: Any
        ) -> None:
            entries = list(items)
            if not entries:
                print(f"  {label}: (empty)")
                return
            print(f"  {label} ({len(entries)}):")
            for item in entries:
                tagged = extractor(item)
                print(f"    {self._fmt_tagged(tagged)}")

        _print_queue(
            "Input Queue",
            self.input_queue,
            lambda t: t,
        )
        _print_queue(
            "SMEM Read Queue",
            self.smem_read_queue,
            lambda t: t,
        )
        _print_queue(
            "SMEM Write Queue",
            self.smem_write_queue,
            lambda t: t,
        )
        if self.axi_bus_queue:
            print(f"  AXI Bus Queue ({len(self.axi_bus_queue)}):")
            for entry in self.axi_bus_queue:
                kind = entry["kind"]
                if kind == "read_resp":
                    tagged = entry["tagged"]
                    value = entry["value"]
                    print(
                        f"    [read_resp]  {self._fmt_tagged(tagged)}"
                        f"  | fetched_data=0x{int(value) & self.word_mask:08x}"
                    )
                elif kind == "write_req":
                    tagged = entry["tagged"]
                    dram_addr = entry["dram_addr"]
                    value = entry["value"]
                    print(
                        f"    [write_req]  {self._fmt_tagged(tagged)}"
                        f"  | dest_dram=0x{int(dram_addr):04x}"
                        f"  payload=0x{int(value) & self.word_mask:08x}"
                    )
                else:
                    print(f"    [unknown kind={kind!r}]  {entry!r}")
        else:
            print("  AXI Bus Queue: (empty)")

        if self.pending_dram_reads:
            print(f"  Pending DRAM Reads ({len(self.pending_dram_reads)}):")
            for ready_cycle, tagged, value in self.pending_dram_reads:
                print(
                    f"    {self._fmt_tagged(tagged)}"
                    f"  | fetched_data=0x{int(value) & self.word_mask:08x}"
                    f"  ready @cycle {ready_cycle}"
                )
        else:
            print("  Pending DRAM Reads: (empty)")

        if self.pending_dram_writes:
            print(f"  Pending DRAM Writes ({len(self.pending_dram_writes)}):")
            for ready_cycle, tagged, dram_addr, value in self.pending_dram_writes:
                print(
                    f"    {self._fmt_tagged(tagged)}"
                    f"  | dest_dram=0x{dram_addr:04x}"
                    f"  payload=0x{int(value) & self.word_mask:08x}"
                    f"  ready @cycle {ready_cycle}"
                )
        else:
            print("  Pending DRAM Writes: (empty)")

        if self.pending_read_crossbar_deliveries:
            print(
                f"  Pending Read Crossbar Deliveries "
                f"({len(self.pending_read_crossbar_deliveries)}):"
            )
            for event in self.pending_read_crossbar_deliveries:
                recipients = list(event.get("recipients", []))
                if recipients:
                    first_tagged = recipients[0]["tagged"]
                    thread_desc = ", ".join(
                        f"T{int(recipient['thread_id']):02d}->L{int(recipient['lane']):02d}"
                        for recipient in recipients
                    )
                else:
                    first_tagged = event["tagged"]
                    thread_desc = (
                        f"T{int(event['thread_id']):02d}->L{int(event['lane']):02d}"
                    )
                print(
                    f"    {self._fmt_tagged(first_tagged)}"
                    f"  | bank={int(event['bank'])}"
                    f"  routes=[{thread_desc}]"
                    f"  value=0x{int(event['value']) & self.word_mask:08x}"
                    f"  ready @cycle {int(event['ready_cycle'])}"
                )
        else:
            print("  Pending Read Crossbar Deliveries: (empty)")

        print(separator)

    def _log_cycle_completions(self, prev_count: int) -> None:
        """
        Print transactions that completed during the current cycle, including
        full address and data details for each thread.

        Args:
            prev_count: Length of ``self.completions`` before the cycle ran.
        """
        new_completions = self.completions[prev_count:]
        if not new_completions:
            print("  >> No completions this cycle.")
            return
        print(f"  >> Completed this cycle ({len(new_completions)}):")
        for comp in new_completions:
            shmem = comp.shmem_addr
            abs_addr = comp.absolute_shmem_addr
            tbo = comp.thread_block_offset_effective

            line = (
                f"    T{comp.thread_id:02d} #{comp.txn_id} {comp.txn_type:<20s}"
                f"  shmem=0x{shmem:04x}  abs=0x{abs_addr:04x}  tbo=0x{tbo:04x}"
                f"  (issued @{comp.cycle_issued}, completed @{comp.cycle_completed})"
            )
            if comp.thread_block_id is not None:
                line += f"  | tbid={comp.thread_block_id}"
            if comp.smem_block_id is not None:
                line += f"  smem_block={comp.smem_block_id}"
            if comp.read_data is not None:
                line += f"  | read_data=0x{comp.read_data:08x}"
            if comp.dram_addr is not None:
                line += f"  | dram_addr=0x{comp.dram_addr:04x}"
            print(line)

    def _validate_transaction(self, txn: Transaction) -> None:
        """
        Validate the parameters of a transaction before issuing it.
        
        Args:
            txn: The transaction to validate.
            
        Raises:
            ValueError: If any transaction parameters are invalid or missing.
        """
        thread_id = int(txn.thread_id)
        if thread_id < 0 or thread_id >= self.num_threads:
            raise ValueError(
                f"thread_id {thread_id} out of range for num_threads={self.num_threads}"
            )

        if txn.thread_block_offset is not None:
            raise ValueError(
                "Legacy thread_block_offset is no longer supported. "
                "Use thread_block_id plus resident_thread_block_ids."
            )
        if txn.thread_block_id is not None:
            int(txn.thread_block_id)
        if txn.resident_thread_block_ids is not None:
            self._normalize_resident_thread_block_ids(
                txn.resident_thread_block_ids,
                expected_slots=self.resident_thread_block_slots,
            )

        # Ensure that all transaction types that interact with shared memory have a valid shmem_addr
        if txn.txn_type in (
            TxnType.SH_LD,
            TxnType.SH_ST,
            TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            TxnType.GLOBAL_ST_SMEM_TO_DRAM,
        ) and txn.shmem_addr is None:
            raise ValueError(f"{txn.txn_type.value} requires shmem_addr.")

        # Store operations must provide the data to be written
        if txn.txn_type == TxnType.SH_ST and txn.write_data is None:
            raise ValueError("sh.st requires write_data.")

        # Ensure that all global-memory transactions that interact with DRAM have a valid dram_addr
        if txn.txn_type in (
            TxnType.GLOBAL_LD_DRAM_TO_SRAM,
            TxnType.GLOBAL_ST_SMEM_TO_DRAM,
        ) and txn.dram_addr is None:
            raise ValueError(f"{txn.txn_type.value} requires dram_addr.")

def _resolve_simulator_kwargs(
    *,
    config_path: Union[str, Path] = DEFAULT_SMEM_CONFIG_PATH,
    num_banks: Optional[int] = None,
    word_bytes: Optional[int] = None,
    dram_latency_cycles: Optional[int] = None,
    arbiter_issue_width: Optional[int] = None,
    num_threads: Optional[int] = None,
    thread_block_offsets: Optional[Dict[int, int] | List[int] | Tuple[int, ...]] = None,
    resident_thread_block_slots: Optional[int] = None,
    thread_block_size_bytes: Optional[int] = None,
    total_smem_size_bytes: Optional[int] = None,
    read_crossbar_pipeline_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve simulator initialization arguments by combining config file defaults with explicit overrides.
    
    Args:
        config_path: Path to the configuration file.
        num_banks: Number of memory banks.
        word_bytes: Number of bytes per word.
        dram_latency_cycles: Latency of DRAM accesses in cycles.
        arbiter_issue_width: Number of requests the arbiter can issue per cycle.
        num_threads: Total number of threads.
        resident_thread_block_slots: Number of resident SMEM thread-block slots.
        thread_block_size_bytes: Size, in bytes, of one resident SMEM thread-block slot.
        total_smem_size_bytes: Total modeled SMEM capacity, in bytes.
        read_crossbar_pipeline_cycles: Latency of the pipelined Clos read crossbar.
        
    Returns:
        A dictionary of resolved keyword arguments.
    """
    if thread_block_offsets is not None:
        raise ValueError(
            "Legacy thread_block_offsets are no longer supported. "
            "Use thread_block_size_bytes plus thread_block_id/"
            "resident_thread_block_ids."
        )
    cfg = load_smem_config(config_path)
    resolved = cfg.to_sim_kwargs()

    explicit_slots = resident_thread_block_slots is not None
    explicit_block_size = thread_block_size_bytes is not None
    explicit_total_size = total_smem_size_bytes is not None

    merged_slots = (
        resident_thread_block_slots
        if explicit_slots
        else resolved.get("resident_thread_block_slots", DEFAULT_RESIDENT_THREAD_BLOCK_SLOTS)
    )
    merged_block_size = (
        thread_block_size_bytes
        if explicit_block_size
        else resolved.get("thread_block_size_bytes")
    )
    merged_total_size = (
        total_smem_size_bytes
        if explicit_total_size
        else resolved.get("total_smem_size_bytes")
    )

    if explicit_total_size and not explicit_block_size:
        merged_block_size = None
    if (explicit_slots or explicit_block_size) and not explicit_total_size:
        merged_total_size = None

    (
        merged_slots,
        merged_block_size,
        merged_total_size,
    ) = _resolve_smem_capacity_fields(
        resident_thread_block_slots=merged_slots,
        thread_block_size_bytes=merged_block_size,
        total_smem_size_bytes=merged_total_size,
    )
    resolved["resident_thread_block_slots"] = merged_slots
    resolved["thread_block_size_bytes"] = merged_block_size
    resolved["total_smem_size_bytes"] = merged_total_size

    overrides = {
        "num_banks": num_banks,
        "word_bytes": word_bytes,
        "dram_latency_cycles": dram_latency_cycles,
        "arbiter_issue_width": arbiter_issue_width,
        "num_threads": num_threads,
        "read_crossbar_pipeline_cycles": read_crossbar_pipeline_cycles,
    }
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def run_smem_functional_sim(
    transactions: Iterable[Transaction | Dict[str, Any]],
    dram_init: Optional[Dict[int, int]] = None,
    *,
    config_path: Union[str, Path] = DEFAULT_SMEM_CONFIG_PATH,
    num_banks: Optional[int] = None,
    word_bytes: Optional[int] = None,
    dram_latency_cycles: Optional[int] = None,
    arbiter_issue_width: Optional[int] = None,
    num_threads: Optional[int] = None,
    thread_block_offsets: Optional[Dict[int, int] | List[int] | Tuple[int, ...]] = None,
    resident_thread_block_slots: Optional[int] = None,
    thread_block_size_bytes: Optional[int] = None,
    total_smem_size_bytes: Optional[int] = None,
    read_crossbar_pipeline_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main function to run the SMEM functional simulator.

    Each transaction may be a Transaction object or a dict with:
    - type / txn_type: sh.ld | sh.st | global.ld.dram2sram | global.st.smem2dram
    - dram_addr: int (required for global-memory transactions)
    - shmem_addr: int (required for all transactions in this model)
    - write_data: int (required for sh.st)
    - thread_id: int (optional, default 0)
    - thread_block_id / tbid: int (optional resident thread-block ID)
    - resident_thread_block_ids / tbids: resident-slot vector mapping SMEM slots to TBIDs
    - thread_block_done_bits / done_bits: completion bits for the transaction's TBID

    Global thread parameters:
    - num_threads: total thread count in this simulation instance
    - resident_thread_block_slots: number of resident SMEM thread-block slots
    - thread_block_size_bytes: when TBID residency is used, offset =
      smem_block_id * thread_block_size_bytes.
    - total_smem_size_bytes: total modeled SMEM capacity in bytes.
    - read_crossbar_pipeline_cycles: latency of the pipelined Clos read crossbar.

    Config:
    - config_path: TOML file path (default `.config`) with `[smem]` defaults.
      Explicit function arguments override config values.
    """
    sim_kwargs = _resolve_simulator_kwargs(
        config_path=config_path,
        num_banks=num_banks,
        word_bytes=word_bytes,
        dram_latency_cycles=dram_latency_cycles,
        arbiter_issue_width=arbiter_issue_width,
        num_threads=num_threads,
        thread_block_offsets=thread_block_offsets,
        resident_thread_block_slots=resident_thread_block_slots,
        thread_block_size_bytes=thread_block_size_bytes,
        total_smem_size_bytes=total_smem_size_bytes,
        read_crossbar_pipeline_cycles=read_crossbar_pipeline_cycles,
    )
    sim = ShmemFunctionalSimulator(dram_init=dram_init, **sim_kwargs)
    return sim.run(transactions)


def run_single_smem_transaction(
    txn_type: str | TxnType,
    *,
    dram_addr: Optional[int] = None,
    shmem_addr: Optional[int] = None,
    write_data: Optional[int] = None,
    thread_id: int = 0,
    thread_block_offset: Optional[int] = None,
    dram_init: Optional[Dict[int, int]] = None,
    config_path: Union[str, Path] = DEFAULT_SMEM_CONFIG_PATH,
    num_banks: Optional[int] = None,
    word_bytes: Optional[int] = None,
    dram_latency_cycles: Optional[int] = None,
    arbiter_issue_width: Optional[int] = None,
    num_threads: Optional[int] = None,
    thread_block_offsets: Optional[Dict[int, int] | List[int] | Tuple[int, ...]] = None,
    resident_thread_block_slots: Optional[int] = None,
    thread_block_size_bytes: Optional[int] = None,
    total_smem_size_bytes: Optional[int] = None,
    read_crossbar_pipeline_cycles: Optional[int] = None,
    thread_block_id: Optional[int] = None,
    resident_thread_block_ids: Optional[Sequence[Optional[int]]] = None,
    thread_block_done_bits: Any = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for one transaction using the exact input fields:
    txn_type, dram_addr, shmem_addr, write_data, thread_id, thread_block_id.
    Uses `.config` defaults unless overridden by explicit arguments.
    """
    if thread_block_offset is not None:
        raise ValueError(
            "Legacy thread_block_offset is no longer supported. "
            "Use thread_block_id plus resident_thread_block_ids."
        )
    txn_enum = (
        txn_type
        if isinstance(txn_type, TxnType)
        else TxnType.from_user_value(txn_type)
    )
    sim_kwargs = _resolve_simulator_kwargs(
        config_path=config_path,
        num_banks=num_banks,
        word_bytes=word_bytes,
        dram_latency_cycles=dram_latency_cycles,
        arbiter_issue_width=arbiter_issue_width,
        num_threads=num_threads,
        thread_block_offsets=thread_block_offsets,
        resident_thread_block_slots=resident_thread_block_slots,
        thread_block_size_bytes=thread_block_size_bytes,
        total_smem_size_bytes=total_smem_size_bytes,
        read_crossbar_pipeline_cycles=read_crossbar_pipeline_cycles,
    )
    sim_kwargs["num_threads"] = max(int(sim_kwargs["num_threads"]), int(thread_id) + 1)
    sim = ShmemFunctionalSimulator(dram_init=dram_init, **sim_kwargs)
    completion = sim.run_one(
        Transaction(
            txn_type=txn_enum,
            dram_addr=dram_addr,
            shmem_addr=shmem_addr,
            write_data=write_data,
            thread_id=thread_id,
            thread_block_id=(
                int(thread_block_id) if thread_block_id is not None else None
            ),
            resident_thread_block_ids=(
                tuple(
                    int(tbid) if tbid is not None else None
                    for tbid in resident_thread_block_ids
                )
                if resident_thread_block_ids is not None
                else None
            ),
            thread_block_done_bits=thread_block_done_bits,
        )
    )
    return {
        "completion": completion,
        "snapshot": sim.snapshot(),
    }


class ShmemCompatibleCacheStage:
    """
    Drop-in compatibility wrapper for swapping this SMEM model in place of
    simulator.mem.dcache.LockupFreeCacheStage in integration tests.

    It keeps the same constructor shape and `compute()` contract, and emits
    DCache-like responses on the `DCache_LSU_Resp` forwarding interface.
    """

    DCACHE_LSU_IF_NAME = "DCache_LSU_Resp"

    def __init__(
        self,
        name: str,
        behind_latch: Any,
        forward_ifs_write: Optional[Dict[str, Any]],
        mem_req_if: Any,
        mem_resp_if: Any,
        *,
        config_path: Union[str, Path] = DEFAULT_SMEM_CONFIG_PATH,
        smem_simulator: Optional[ShmemFunctionalSimulator] = None,
        smem_simulator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the compatibility cache stage wrapper.
        
        Args:
            name: The name of the cache stage.
            behind_latch: The latch providing input requests.
            forward_ifs_write: Interfaces for forwarding responses.
            mem_req_if: Interface for memory requests.
            mem_resp_if: Interface for memory responses.
            config_path: Path to the configuration file.
            smem_simulator: An optional pre-initialized simulator instance.
            smem_simulator_kwargs: Additional arguments for simulator initialization.
        """
        self.name = name
        self.behind_latch = behind_latch
        self.forward_ifs_write = forward_ifs_write or {}
        self.mem_req_if = mem_req_if
        self.mem_resp_if = mem_resp_if

        if smem_simulator is not None:
            self.sim = smem_simulator
        else:
            resolved = _resolve_simulator_kwargs(config_path=config_path)
            if smem_simulator_kwargs:
                resolved.update(smem_simulator_kwargs)
            self.sim = ShmemFunctionalSimulator(**resolved)

        self.cycle_count = 0
        self.cycle = 0
        self.output_buffer: Deque[Any] = deque()
        self._completion_cursor = 0
        self._origin_req_by_txn_id: Dict[int, Any] = {}

        if self.behind_latch and (self.DCACHE_LSU_IF_NAME in self.forward_ifs_write):
            self.behind_latch.forward_if = self.forward_ifs_write[self.DCACHE_LSU_IF_NAME]

    def get_cycle_count(self) -> int:
        """
        Get the current cycle count of the simulator.
        
        Returns:
            The total number of cycles executed.
        """
        return int(self.cycle_count)

    def compute(self) -> None:
        """
        Perform the computation for a single cycle in the cache stage.
        
        Pops requests from the latch, issues them to the simulator, steps the simulator,
        and pushes completions to the output buffer.
        """
        self.cycle_count += 1
        self.cycle = self.cycle_count

        if (
            self.behind_latch is not None
            and getattr(self.behind_latch, "forward_if", None) is not None
        ):
            self.behind_latch.forward_if.set_wait(0)

        if self.behind_latch is not None and getattr(self.behind_latch, "valid", False):
            req = self.behind_latch.pop()
            if req is not None:
                if bool(getattr(req, "halt", False)):
                    self.output_buffer.append(self._make_flush_response())
                else:
                    txn = self._request_to_transaction(req)
                    txn_id = self.sim.issue(txn)
                    self._origin_req_by_txn_id[txn_id] = req

        self.sim.step()
        self._collect_new_completions()
        self._push_output()

    def _collect_new_completions(self) -> None:
        """
        Collect newly completed transactions from the simulator and format them as responses.
        """
        if len(self.sim.completions) <= self._completion_cursor:
            return
        new_done = self.sim.completions[self._completion_cursor :]
        for done in new_done:
            original_req = self._origin_req_by_txn_id.pop(done.txn_id, None)
            self.output_buffer.append(
                self._completion_to_compat_response(done, original_req)
            )
        self._completion_cursor = len(self.sim.completions)

    def _push_output(self) -> None:
        """
        Push formatted responses from the output buffer to the forward interface.
        """
        if self.DCACHE_LSU_IF_NAME not in self.forward_ifs_write:
            return
        interface = self.forward_ifs_write[self.DCACHE_LSU_IF_NAME]
        if not getattr(interface, "wait", False):
            if self.output_buffer:
                interface.push(self.output_buffer.popleft())
            else:
                interface.push(None)

    def _request_to_transaction(self, req: Any) -> Transaction:
        """
        Convert an incoming request object or dictionary into a Transaction instance.
        
        Args:
            req: The raw request object or dictionary.
            
        Returns:
            A formatted Transaction instance.
            
        Raises:
            ValueError: If the request type is unsupported.
        """
        if isinstance(req, Transaction):
            return req

        if isinstance(req, dict):
            if req.get("thread_block_offset") is not None:
                raise ValueError(
                    "Legacy thread_block_offset is no longer supported. "
                    "Use thread_block_id plus resident_thread_block_ids."
                )
            if ("type" in req or "txn_type" in req) and ("rw_mode" not in req):
                return Transaction.from_dict(req)
            parsed_type: Optional[TxnType] = None
            raw_candidates = [
                req.get("smem_txn_type"),
                req.get("txn_type"),
                req.get("type"),
            ]
            for raw_candidate in raw_candidates:
                if raw_candidate is None:
                    continue
                try:
                    parsed_type = TxnType.from_user_value(str(raw_candidate))
                    break
                except ValueError:
                    continue

            if parsed_type is None:
                if "rw_mode" in req:
                    rw_mode = str(req.get("rw_mode", "read")).lower()
                    parsed_type = TxnType.SH_ST if rw_mode == "write" else TxnType.SH_LD
                elif any(raw_candidate is not None for raw_candidate in raw_candidates):
                    raw_type = req.get("smem_txn_type", req.get("txn_type", req.get("type")))
                    raise ValueError(
                        f"Unsupported transaction type '{raw_type}' for dict request."
                    )
                else:
                    parsed_type = TxnType.SH_LD
            return Transaction(
                txn_type=parsed_type,
                dram_addr=req.get("dram_addr"),
                shmem_addr=int(req.get("addr_val", req.get("shmem_addr", 0))),
                write_data=req.get("store_value", req.get("write_data")),
                thread_id=int(req.get("thread_id", 0)),
                thread_block_id=(
                    int(req["thread_block_id"])
                    if req.get("thread_block_id") is not None
                    else (
                        int(req["tbid"])
                        if req.get("tbid") is not None
                        else None
                    )
                ),
                resident_thread_block_ids=(
                    tuple(
                        int(tbid) if tbid is not None else None
                        for tbid in req.get(
                            "resident_thread_block_ids",
                            req.get("tbids", req.get("smem_tbids", ())),
                        )
                    )
                    if (
                        req.get("resident_thread_block_ids") is not None
                        or req.get("tbids") is not None
                        or req.get("smem_tbids") is not None
                    )
                    else None
                ),
                thread_block_done_bits=req.get(
                    "thread_block_done_bits",
                    req.get("done_bits"),
                ),
            )

        addr = int(getattr(req, "addr_val", 0))
        rw_mode = str(getattr(req, "rw_mode", "read")).lower()
        size = str(getattr(req, "size", "word")).lower()
        thread_id = int(getattr(req, "thread_id", 0))
        tbo = getattr(req, "thread_block_offset", None)
        if tbo is not None:
            raise ValueError(
                "Legacy thread_block_offset is no longer supported. "
                "Use thread_block_id plus resident_thread_block_ids."
            )
        tbid = getattr(req, "thread_block_id", getattr(req, "tbid", None))
        tbid_int = int(tbid) if tbid is not None else None
        resident_ids = getattr(
            req,
            "resident_thread_block_ids",
            getattr(req, "tbids", getattr(req, "smem_tbids", None)),
        )
        resident_ids_tuple = (
            tuple(int(slot_tbid) if slot_tbid is not None else None for slot_tbid in resident_ids)
            if resident_ids is not None
            else None
        )
        done_bits = getattr(
            req,
            "thread_block_done_bits",
            getattr(req, "done_bits", None),
        )

        if rw_mode == "write":
            raw_store = int(getattr(req, "store_value", 0))
            merged_store = self._format_store_data_for_size(
                addr=addr,
                data=raw_store,
                size=size,
                thread_id=thread_id,
                thread_block_id=tbid_int,
                resident_thread_block_ids=resident_ids_tuple,
                thread_block_done_bits=done_bits,
            )
            return Transaction(
                txn_type=TxnType.SH_ST,
                shmem_addr=addr,
                write_data=merged_store,
                thread_id=thread_id,
                thread_block_id=tbid_int,
                resident_thread_block_ids=resident_ids_tuple,
                thread_block_done_bits=done_bits,
            )

        return Transaction(
            txn_type=TxnType.SH_LD,
            shmem_addr=addr,
            thread_id=thread_id,
            thread_block_id=tbid_int,
            resident_thread_block_ids=resident_ids_tuple,
            thread_block_done_bits=done_bits,
        )

    def _format_store_data_for_size(
        self,
        *,
        addr: int,
        data: int,
        size: str,
        thread_id: int,
        thread_block_id: Optional[int],
        resident_thread_block_ids: Optional[Tuple[Optional[int], ...]],
        thread_block_done_bits: Any,
    ) -> int:
        """
        Format store data based on the requested access size (word, half, byte).
        
        Performs read-modify-write for sub-word accesses.
        
        Args:
            addr: The target memory address.
            data: The raw data to store.
            size: The access size ("word", "half", "byte").
            thread_id: The ID of the requesting thread.
            thread_block_id: The resident thread-block ID, when TBID mode is used.
            resident_thread_block_ids: The 4-slot resident SMEM TBID vector.
            thread_block_done_bits: Done bits for the current thread block.
            
        Returns:
            The formatted data word to store.
        """
        if size == "word":
            return int(data) & 0xFFFF_FFFF

        tx_probe = Transaction(
            txn_type=TxnType.SH_LD,
            shmem_addr=addr,
            thread_id=thread_id,
            thread_block_id=thread_block_id,
            resident_thread_block_ids=resident_thread_block_ids,
            thread_block_done_bits=thread_block_done_bits,
        )
        absolute = self.sim._absolute_smem_addr(tx_probe)
        old_word = int(self.sim.sram_linear.get(absolute, 0)) & 0xFFFF_FFFF
        byte_offset = int(addr) & 0x3
        shift = byte_offset * 8

        if size == "half":
            mask = 0xFFFF << shift
            return (old_word & ~mask) | ((int(data) & 0xFFFF) << shift)
        if size == "byte":
            mask = 0xFF << shift
            return (old_word & ~mask) | ((int(data) & 0xFF) << shift)
        return int(data) & 0xFFFF_FFFF

    def _completion_to_compat_response(self, done: Completion, req: Any) -> Any:
        """
        Convert a simulator Completion record into a compatible memory response object.
        
        Args:
            done: The completion record.
            req: The original request object.
            
        Returns:
            A memory response object compatible with the surrounding system.
        """
        if isinstance(req, dict):
            addr = req.get("addr_val", done.absolute_shmem_addr)
        else:
            addr = getattr(req, "addr_val", done.absolute_shmem_addr)
        req_for_resp = req if req is not None else None

        if done.txn_type in (TxnType.SH_LD.value, TxnType.SH_ST.value):
            resp_type = "HIT_COMPLETE"
            hit = True
            miss = False
            replay = False
        else:
            # Global memory operations are completion-based in this model.
            resp_type = "MISS_COMPLETE"
            hit = False
            miss = True
            replay = True

        return _DMEM_RESPONSE_CLS(
            type=resp_type,
            req=req_for_resp,
            address=addr,
            replay=replay,
            is_secondary=False,
            data=done.read_data,
            miss=miss,
            hit=hit,
            stall=False,
            uuid=done.txn_id,
            flushed=False,
        )

    def _make_flush_response(self) -> Any:
        """
        Create a flush completion response object.
        
        Returns:
            A flush response object.
        """
        return _DMEM_RESPONSE_CLS(
            type="FLUSH_COMPLETE",
            req=None,
            address=0,
            replay=False,
            is_secondary=False,
            data=None,
            miss=False,
            hit=False,
            stall=False,
            uuid=0,
            flushed=True,
        )


LockupFreeCacheStageCompat = ShmemCompatibleCacheStage
LockupFreeCacheStage = ShmemCompatibleCacheStage


class SmemArbiter:
    """
    SMEM Arbiter -- the entry point to the shared memory subsystem.

    Accepts a batch of transactions (e.g. one per thread in a warp), detects
    bank conflicts via the simulator's XOR-mapped address crossbar, and
    partitions conflicting requests into multiple conflict-free sub-batches.
    Each sub-batch is issued to the simulator and separated by a ``step()``
    call so that conflicting accesses are serialized across cycles.
    """

    def __init__(self, simulator: ShmemFunctionalSimulator) -> None:
        """
        Initialize the SMEM Arbiter.

        Args:
            simulator: The underlying shared memory functional simulator.
        """
        self.simulator = simulator

    def _get_bank(self, txn: Transaction) -> int:
        """
        Determine the target bank for a given transaction.

        Delegates to the simulator to avoid duplicating address-mapping logic.

        Args:
            txn: The transaction to evaluate.

        Returns:
            The index of the target bank.
        """
        return self.simulator._bank_for_transaction(txn)

    def _get_bank_and_slot(self, txn: Transaction) -> Tuple[int, int]:
        """
        Determine the target bank and slot for a given transaction.

        Args:
            txn: The transaction to evaluate.

        Returns:
            A tuple of (bank_index, bank_slot).
        """
        absolute = self.simulator._absolute_smem_addr(txn)
        offset = self.simulator._effective_thread_block_offset(txn)
        return self.simulator._address_crossbar(absolute, offset)

    def _log_thread_state(
        self,
        txn: Transaction,
        cycle: int,
        bank: int,
        bank_slot: int,
        absolute_addr: int,
        sub_batch_idx: int,
    ) -> None:
        """
        Log the state of a thread's memory access for debugging, including
        all address fields and data payloads.

        Args:
            txn: The transaction being processed.
            cycle: The current cycle count.
            bank: The target bank index.
            bank_slot: The target bank slot.
            absolute_addr: The absolute memory address.
            sub_batch_idx: The index of the sub-batch this transaction belongs to.
        """
        sim = self.simulator
        tbo = sim._effective_thread_block_offset(txn)
        line = (
            f"[DEBUG] Sub-batch {sub_batch_idx} | Cycle {cycle} | "
            f"Thread {txn.thread_id:2d} | {txn.txn_type.value:<20s} | "
            f"Addr 0x{txn.shmem_addr:04x} | AbsAddr 0x{absolute_addr:04x} | "
            f"TBO 0x{tbo:04x} | XOR Map -> Bank {bank:2d} | "
            f"Slot {bank_slot:4d}"
        )

        if txn.txn_type == TxnType.SH_ST and txn.write_data is not None:
            line += f" | write_data=0x{int(txn.write_data) & sim.word_mask:08x}"
        if txn.txn_type == TxnType.SH_LD:
            current = sim.banks[bank].get(bank_slot, 0)
            line += f" | bank_content=0x{current & sim.word_mask:08x}"
        if txn.dram_addr is not None:
            line += f" | dram_addr=0x{txn.dram_addr:04x}"

        print(line)

    def process_batch(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """
        Issue a warp-wide batch of thread transactions into the SMEM arbiter
        atomically, then drive the simulator until every one of them has
        been accepted by the arbiter (i.e. left ``input_queue``).

        Warp-synchronous issue model: all ``N`` per-thread transactions
        enter the arbiter in the same cycle. The simulator's internal
        bank-parallel arbiter then drains them at up to one-per-bank per
        cycle, naturally serializing any bank conflicts across subsequent
        cycles. This replaces the previous software-side sub-batch
        partitioning, which was a pre-arbitration model that did not match
        how warp instructions reach the arbiter in real hardware.

        Args:
            transactions: A list of per-thread transactions comprising one
                warp-wide instruction.

        Returns:
            A dict with arbitration metadata:

            - ``total_transactions``: total input count
            - ``num_arbiter_cycles``: number of cycles the warp occupied the
              arbiter before all lanes were dispatched to the SMEM queues
            - ``per_cycle_issue_counts``: list of how many lanes were issued
              each cycle (length == ``num_arbiter_cycles``)
            - ``num_sub_batches`` / ``sub_batch_sizes``: backward-compatible
              aliases for ``num_arbiter_cycles`` / ``per_cycle_issue_counts``
              so existing callers and assertions keep working.
        """
        total = len(transactions)

        if total == 0:
            print(
                "\n[DEBUG] --- SmemArbiter received an empty warp-wide batch; "
                "nothing to issue. ---\n"
            )
            return {
                "total_transactions": 0,
                "num_arbiter_cycles": 0,
                "per_cycle_issue_counts": [],
                "num_sub_batches": 0,
                "sub_batch_sizes": [],
            }

        sim = self.simulator

        print(
            f"\n[DEBUG] --- SmemArbiter: atomically issuing {total} warp-wide "
            f"thread transactions into input_queue @cycle {sim.cycle} ---"
        )
        input_len_before = len(sim.input_queue)
        for txn in transactions:
            absolute_addr = sim._absolute_smem_addr(txn)
            bank, bank_slot = self._get_bank_and_slot(txn)
            self._log_thread_state(
                txn,
                sim.cycle,
                bank,
                bank_slot,
                absolute_addr,
                sub_batch_idx=0,
            )
            sim.issue(txn)

        per_cycle_issue_counts: List[int] = []
        target_input_len = input_len_before
        max_cycles = max(4096, total * 64)
        drain_steps = 0

        # A single ``step()`` call invokes the internal arbiter once, so the
        # number of steps it takes for ``input_queue`` to shrink back to the
        # pre-issue length equals the number of arbiter cycles this warp
        # spent being dispatched.
        while len(sim.input_queue) > target_input_len:
            drain_steps += 1
            if drain_steps > max_cycles:
                raise RuntimeError(
                    f"SmemArbiter.process_batch did not drain {total} "
                    f"transactions within {max_cycles} cycles."
                )
            before = len(sim.input_queue)
            sim.step()
            after = len(sim.input_queue)
            issued_this_cycle = before - after
            per_cycle_issue_counts.append(issued_this_cycle)

        num_arbiter_cycles = len(per_cycle_issue_counts)
        print(
            f"[DEBUG] --- SmemArbiter: warp cleared arbiter in "
            f"{num_arbiter_cycles} cycle(s); "
            f"per-cycle issue counts = {per_cycle_issue_counts} ---\n"
        )

        return {
            "total_transactions": total,
            "num_arbiter_cycles": num_arbiter_cycles,
            "per_cycle_issue_counts": list(per_cycle_issue_counts),
            "num_sub_batches": num_arbiter_cycles,
            "sub_batch_sizes": list(per_cycle_issue_counts),
        }


def _sim_from_config(*, num_threads: int, verbose: bool = True) -> ShmemFunctionalSimulator:
    """
    Create a simulator whose hardware params (num_banks, word_bytes,
    dram_latency_cycles, arbiter_issue_width) come from ``.config``.
    ``num_threads`` is always test-specific so it is passed explicitly.
    """
    cfg = load_smem_config()
    kwargs = cfg.to_sim_kwargs()
    kwargs["num_threads"] = num_threads
    kwargs["verbose"] = verbose
    return ShmemFunctionalSimulator(**kwargs)


def test_32_threads_different_addresses() -> None:
    """
    Baseline: 32 threads accessing different addresses without bank conflicts.
    With no bank conflicts, the number of sub-batches is ceil(N / issue_width).
    """
    print("\n=== TEST: 32 Threads with Different Addresses (No Conflicts) ===")
    sim = _sim_from_config(num_threads=32)
    arbiter = SmemArbiter(sim)

    num_threads = 32
    txns = [
        Transaction(
            txn_type=TxnType.SH_LD,
            shmem_addr=i * sim.word_bytes,
            thread_id=i,
        )
        for i in range(min(num_threads, sim.num_banks))
    ]

    result = arbiter.process_batch(txns)
    while sim._has_pending_work():
        sim.step()

    expected_batches = -(-len(txns) // sim.arbiter_issue_width)
    assert result['num_sub_batches'] == expected_batches, (
        f"Expected {expected_batches} sub-batch(es) "
        f"(issue_width={sim.arbiter_issue_width}), "
        f"got {result['num_sub_batches']}"
    )
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Sub-batches: {result['num_sub_batches']} "
        f"(issue_width={sim.arbiter_issue_width})"
    )


def test_divergence() -> None:
    """
    Multicast-read scenario: all 32 threads hit the same shmem_addr (bank 0).
    The arbiter should coalesce the warp into one multicast read and let the
    Clos read crossbar fan the value back out to every lane.
    """
    print("\n=== TEST: Divergence -- All Threads Hit Same Bank ===")
    num_threads = 32
    sim = _sim_from_config(num_threads=num_threads)
    arbiter = SmemArbiter(sim)

    txns = [
        Transaction(txn_type=TxnType.SH_LD, shmem_addr=0x00, thread_id=i)
        for i in range(num_threads)
    ]

    result = arbiter.process_batch(txns)
    while sim._has_pending_work():
        sim.step()

    assert result['num_sub_batches'] == 1, (
        f"Expected 1 multicast arbiter cycle, "
        f"got {result['num_sub_batches']}"
    )
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Sub-batches: {result['num_sub_batches']} (sizes: {result['sub_batch_sizes']})"
    )


def test_integration_smem_arbiter() -> None:
    """
    Integration test with deliberate bank conflicts among stores.
    Threads 0 and 2 both target shmem_addr 0x10 (same bank), so they must be
    split across two sub-batches.
    """
    print("\n=== TEST: Integration with SMEM Arbiter (Bank Conflicts in Stores) ===")
    sim = _sim_from_config(num_threads=4)
    arbiter = SmemArbiter(sim)

    txns = [
        Transaction(txn_type=TxnType.SH_ST, shmem_addr=0x10, write_data=0xAA, thread_id=0),
        Transaction(txn_type=TxnType.SH_ST, shmem_addr=0x14, write_data=0xBB, thread_id=1),
        Transaction(txn_type=TxnType.SH_ST, shmem_addr=0x10, write_data=0xCC, thread_id=2),
        Transaction(txn_type=TxnType.SH_ST, shmem_addr=0x1C, write_data=0xDD, thread_id=3),
    ]

    result = arbiter.process_batch(txns)
    while sim._has_pending_work():
        sim.step()

    assert result['num_sub_batches'] == 2, (
        f"Expected 2 sub-batches due to bank conflict, got {result['num_sub_batches']}"
    )
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Sub-batches: {result['num_sub_batches']} (sizes: {result['sub_batch_sizes']})"
    )


def test_multicast_divergence() -> None:
    """
    Multi-way bank conflict: half the threads hit bank 0, half hit bank 1.
    Each bank's identical-address read should coalesce into one multicast
    request, so the arbiter only needs to schedule one logical read per bank.
    """
    print("\n=== TEST: Multicast Divergence (2-bank, 16-way conflict) ===")
    sim = _sim_from_config(num_threads=32)
    arbiter = SmemArbiter(sim)

    per_bank = 16
    txns: List[Transaction] = []
    for i in range(per_bank):
        txns.append(Transaction(txn_type=TxnType.SH_LD, shmem_addr=0x00, thread_id=i))
    for i in range(per_bank, 2 * per_bank):
        txns.append(Transaction(
            txn_type=TxnType.SH_LD,
            shmem_addr=sim.word_bytes,
            thread_id=i,
        ))

    result = arbiter.process_batch(txns)
    while sim._has_pending_work():
        sim.step()

    logical_reads = 2
    per_batch = min(logical_reads, sim.arbiter_issue_width)
    expected_batches = -(-logical_reads // per_batch)
    assert result['num_sub_batches'] == expected_batches, (
        f"Expected {expected_batches} sub-batches "
        f"(per_batch={per_batch}, issue_width={sim.arbiter_issue_width}), "
        f"got {result['num_sub_batches']}"
    )
    print(
        f"Completed in {sim.cycle} cycles. "
        f"Sub-batches: {result['num_sub_batches']} (sizes: {result['sub_batch_sizes']})"
    )

if __name__ == "__main__":
    # Redirect extended traceback and debug output to output_extended.txt
    original_stdout = sys.stdout
    with open("output_extended.txt", "w") as f:
        sys.stdout = f

        demo_transactions = [
            {"type": "sh.st", "shmem_addr": 0x20, "write_data": 0xDEADBEEF},
            {"type": "sh.ld", "shmem_addr": 0x20},
            {"type": "global.st.smem2dram", "shmem_addr": 0x20, "dram_addr": 0x1000},
            {"type": "global.ld.dram2sram", "dram_addr": 0x1000, "shmem_addr": 0x24},
            {"type": "sh.ld", "shmem_addr": 0x24},
        ]

        result = run_smem_functional_sim(demo_transactions, dram_init={0x2000: 0x1234ABCD})

        print("Completions:")
        for completion in result["completions"]:
            print(completion)
            print("  Traceback:")
            for trace_line in completion["trace"] if isinstance(completion, dict) else completion.trace:
                print(f"    {trace_line}")

        # Run new tests
        test_32_threads_different_addresses()
        test_divergence()
        test_integration_smem_arbiter()
        test_multicast_divergence()

    sys.stdout = original_stdout
    print("Extended traceback and test output have been written to output_extended.txt")

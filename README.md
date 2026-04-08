# SHMEM Functional Simulator (Minimal)

## What This Is
- `main.py` contains a functional shared-memory simulator (data crossbar intentionally not modeled).
- `simulator/mem/dcache.py` is the reference dcache implementation used for cycle comparison.

## Quick Run
```bash
python3 main.py
```

## Config
- Edit `.config` (`[smem]` section) to change simulator parameters:
  - `num_banks`
  - `word_bytes`
  - `dram_latency_cycles`
  - `arbiter_issue_width`
  - `num_threads`
  - `thread_block_offsets`

## Tests
```bash
python3 test_dcache_and_smem.py
python3 test_smem_comprehensive_cycle_compare.py
```

## Cycle Report (DCache vs SHMEM)
```bash
python3 test_cycle_count_report.py | tee output.txt
```
- Report is saved to `output.txt`.

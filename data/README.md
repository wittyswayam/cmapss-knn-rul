# Data

## Source

NASA **Turbofan Engine Degradation Simulation Data Set** (C-MAPSS), also known as the PHM08 challenge
data, published by the NASA Ames Prognostics Center of Excellence. The data is synthetic: it was
generated with the C-MAPSS engine simulator, not recorded from real aircraft.

Citation requested by the NASA repository:

> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
> NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Method paper (also included in the archive as `Damage Propagation Modeling.pdf`):

> A. Saxena, K. Goebel, D. Simon and N. Eklund (2008). "Damage Propagation Modeling for Aircraft
> Engine Run-to-Failure Simulation", Proc. 1st International Conference on Prognostics and Health
> Management (PHM08), Denver, CO.

## Licence

**No licence is stated** in the archive or in its readme, and the NASA catalogue entry I looked at
listed none. The data is publicly distributed by NASA, but that is not the same as an explicit
licence. For that reason this repository does **not** redistribute the data: you download it
yourself. Before publishing anything derived from it, check the terms on the NASA Prognostics Data
Repository page (NASA Intelligent Systems Division) and cite the sources above.

## Getting the data

1. Download the original `CMAPSSData.zip` from the NASA Prognostics Data Repository.
2. Put it, unextracted, in `data/raw/`. (Alternatively extract it so that `train_FD001.txt`,
   `test_FD001.txt`, `RUL_FD001.txt`, ... sit directly in `data/raw/`.)
3. Run `python scripts/verify_data.py`.

If you downloaded a bundle that contains `CMAPSSData.zip` inside another zip, extract the outer
one first and use the inner `CMAPSSData.zip`.

The project was developed against a `CMAPSSData.zip` of 12,425,978 bytes with SHA-256
`74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`. A different digest only
produces a warning; the structural validation still decides whether the data is usable.

## Files in the archive

| File | Content |
|---|---|
| `train_FDxxx.txt` | Run-to-failure trajectories: every engine is observed until it fails. |
| `test_FDxxx.txt` | Trajectories truncated some cycles before failure. |
| `RUL_FDxxx.txt` | True remaining useful life after the last observed test cycle, one value per test engine, in engine order. |
| `readme.txt` | Original dataset description. |

## Layout of each row

One row is one operating cycle (one flight) of one engine, 26 whitespace-separated numbers:

| Column(s) | Name here | Meaning |
|---|---|---|
| 1 | `unit` | Engine id (1..N) |
| 2 | `cycle` | Cycle index, starting at 1 |
| 3-5 | `op1`, `op2`, `op3` | Operational settings |
| 6-26 | `s1` ... `s21` | 21 sensor channels |

The original readme calls all 26 columns "sensor measurements"; in fact only 21 are sensors.

## Subsets

| | Operating conditions | Fault modes | Train engines | Test engines |
|---|---|---|---|---|
| FD001 | 1 | HPC degradation | 100 | 100 |
| FD002 | 6 | HPC degradation | 260 | 259 |
| FD003 | 1 | HPC + fan degradation | 100 | 100 |
| FD004 | 6 | HPC + fan degradation | **249** | **248** |

**Readme discrepancy:** the original readme lists FD004 as 248 training and 249 test engines.
The files contain 249 training and 248 test engines (the readme has them swapped).
`scripts/verify_data.py` checks the counts against the files, not against the readme.

## Target conventions

* Training: RUL = last cycle - current cycle (0 on the final row).
* Test: the RUL file gives the cycles remaining *after* the last observed cycle.
* Primary training target: RUL clipped at 125 cycles (`configs/default.yaml`).

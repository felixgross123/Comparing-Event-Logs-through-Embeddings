"""
Timing benchmark on the log pairs of the validity and reliability experiments -- the
BPIC11 age split, the ten BPIC15 municipality pairs and the six BPIC18 department pairs:

1. CWINDOW embeddings -- the compass on both logs combined, then one frozen-output
   model per log -- timed once on the CPU and once on the GPU.
2. Only with ``RUN_EMSC``: EMSC (Earth Mover's Stochastic Conformance) between the
   two whole logs.

The two stages are run exactly as the validity and reliability experiments run them,
through ``log_comparison_cwindow`` and ``emsc_runner``, and every pair with the training
schedule it has there, so the numbers describe the pipeline those experiments use and
not a re-implementation of it.

Reading the logs happens once, up front, outside every timed region. Each stage's
encoding, however, sits INSIDE its timed region, because the library encodes as part of
training. That work is identical on both devices, so it flattens the CPU/GPU gap rather
than exaggerating it.

Results are written to timing_results.txt line by line, so an interrupted run keeps what
it already has.

    python timing_benchmark.py
"""

import platform
import time
from pathlib import Path

import emsc_runner
import log_comparison_cwindow as lcw
import pm4py
import torch

HERE = Path(__file__).resolve().parent  # so the job can be submitted from anywhere
LOGS_DIR = HERE / "logs"
OUT = HERE / "timing_results.txt"

# --- the global settings of the validity and reliability experiments ---
WINDOW = 4  # context window c
DIM = 32  # embedding dimension d
UNIT_NORM = True  # keep the activity embeddings on the unit sphere
BALANCE_COMPASS = True  # both logs weigh equally in the compass loss
SEED = 42

RUN_EMSC = False  # also time EMSC on the whole logs

# Seconds allowed per EMSC before the pair is abandoned; raise it if a pair that matters
# is being cut off, and raise --time in run_timing.sh with it.
EMSC_TIMEOUT = 3600

CASE_COL, ACTIVITY_COL, TIME_COL = "case:concept:name", "concept:name", "time:timestamp"

# --- One comparison entry ----------------------------------------------------
# Every pair states its own full set of per-pair hyperparameters, as in the validity and
# reliability experiments. Nothing is inherited: a missing or misspelled hyperparameter
# raises instead of silently falling back to a default.
#
#   compass_learning_rate / compass_epochs    compass stage (the shared space)
#   log_learning_rate / log_epochs            per-log frozen-output retraining
HYPERPARAM_KEYS = (
    "compass_learning_rate",
    "compass_epochs",
    "log_learning_rate",
    "log_epochs",
)


def pair(l1, l2, **hyperparams):
    """One comparison: the two log names plus every hyperparameter it runs with."""
    missing = [k for k in HYPERPARAM_KEYS if k not in hyperparams]
    unknown = [k for k in hyperparams if k not in HYPERPARAM_KEYS]
    if missing or unknown:
        raise ValueError(
            f"{l1} vs {l2}: "
            + (f"missing hyperparameters {missing}. " if missing else "")
            + (f"unknown hyperparameters {unknown}." if unknown else "")
        )
    return {"l1": l1, "l2": l2, **hyperparams}


def pair_id(p):
    return f"{p['l1']} vs {p['l2']}"


def family(p):
    """The benchmark log a pair is split from: BPIC11, BPIC15 or BPIC18."""
    return p["l1"].split("_")[0]


PAIRS = [
    # --- BPIC11 age ---
    pair(
        "BPIC11_age_low",
        "BPIC11_age_high",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    # --- BPIC15 municipalities ---
    pair(
        "BPIC15_M1",
        "BPIC15_M2",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M1",
        "BPIC15_M3",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M1",
        "BPIC15_M4",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M1",
        "BPIC15_M5",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M2",
        "BPIC15_M3",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M2",
        "BPIC15_M4",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M2",
        "BPIC15_M5",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M3",
        "BPIC15_M4",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M3",
        "BPIC15_M5",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC15_M4",
        "BPIC15_M5",
        compass_learning_rate=1e-2,
        compass_epochs=1500,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    # --- BPIC18 departments ---
    pair(
        "BPIC18_D4e",
        "BPIC18_D6b",
        compass_learning_rate=1e-2,
        compass_epochs=1000,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC18_D4e",
        "BPIC18_Dd4",
        compass_learning_rate=1e-2,
        compass_epochs=1000,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC18_D4e",
        "BPIC18_De7",
        compass_learning_rate=1e-2,
        compass_epochs=1000,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC18_D6b",
        "BPIC18_Dd4",
        compass_learning_rate=1e-2,
        compass_epochs=1000,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC18_D6b",
        "BPIC18_De7",
        compass_learning_rate=1e-2,
        compass_epochs=1000,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
    pair(
        "BPIC18_Dd4",
        "BPIC18_De7",
        compass_learning_rate=1e-2,
        compass_epochs=1000,
        log_learning_rate=1e-2,
        log_epochs=500,
    ),
]

# every log the pairs name, each read once, in the order they are first named
LOG_NAMES = list(dict.fromkeys(name for p in PAIRS for name in (p["l1"], p["l2"])))
FAMILIES = list(dict.fromkeys(family(p) for p in PAIRS))
PAIR_W = max(len(pair_id(p)) for p in PAIRS) + 2  # the pair column fits every label


def gpu_device():
    """CUDA if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for the accelerator — CUDA and MPS queue their work asynchronously."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def read_log(name):
    """One log, reduced to the three columns the benchmark needs.

    A log preprocessed by ``00_preprocessing`` already carries the readable label in
    ``concept:name``. A raw BPIC15 export instead carries the internal activity code
    there and the label in ``activityNameEN``; that case is relabelled, and which of the
    two happened is reported, because comparing on codes silently splits one activity
    into several numbered variants and changes every number below. The BPIC11 and
    BPIC18 splits exist only as ``00_preprocessing`` output, so they always take the
    first path.
    """
    df = pm4py.read_xes(str(LOGS_DIR / f"{name}.xes"))
    if ACTIVITY_COL in df.columns and "activityNameEN" not in df.columns:
        source = "concept:name (preprocessed)"
    else:
        df[ACTIVITY_COL] = df["activityNameEN"]
        source = "activityNameEN (raw export)"
    df = df[[CASE_COL, ACTIVITY_COL, TIME_COL]]
    print(
        f"  {name}: {df[CASE_COL].nunique()} cases, {len(df)} events, "
        f"{df[ACTIVITY_COL].nunique()} activities, from {source}",
        flush=True,
    )
    return df


def time_cwindow(traces_i, traces_j, p, device):
    """Seconds for the compass, for each log's frozen-output retraining, and in total.

    The stages are the ones :func:`log_comparison_cwindow.compare_cwindow_event_logs`
    runs, in the same order, with the same balancing and with the pair's own schedule,
    split apart only so each can be timed on its own.
    """
    lcw.set_seed(SEED)

    # the compass sees both logs, rescaled so neither dominates, and merged
    balanced, _ = lcw._balance_case_counts(
        {p["l1"]: traces_i, p["l2"]: traces_j}, BALANCE_COMPASS
    )
    combined = lcw._combine_logs(balanced)

    t0 = time.perf_counter()
    compass, _ = lcw._train_cwindow(
        combined,
        WINDOW,
        DIM,
        p["compass_learning_rate"],
        p["compass_epochs"],
        unit_norm=UNIT_NORM,
        device=device,
        verbose=False,
    )
    sync(device)
    compass_s = time.perf_counter() - t0

    retrain_s = []
    for traces in (traces_i, traces_j):
        t0 = time.perf_counter()
        lcw._train_cwindow(
            traces,
            WINDOW,
            DIM,
            p["log_learning_rate"],
            p["log_epochs"],
            word_to_ix=compass.word_to_ix,
            compass=compass,
            unit_norm=UNIT_NORM,
            device=device,
            verbose=False,
        )
        sync(device)
        retrain_s.append(time.perf_counter() - t0)

    return compass_s, retrain_s[0], retrain_s[1], compass_s + sum(retrain_s)


def time_emsc(df_i, df_j):
    """Seconds and value of one EMSC comparison, through :mod:`emsc_runner`.

    The timing covers what a caller actually pays: the trace extraction, the child
    process the Ebi call is isolated in, and the call itself.
    """
    t0 = time.perf_counter()
    value = emsc_runner.emsc_with_timeout(df_i, df_j, timeout=EMSC_TIMEOUT)
    return time.perf_counter() - t0, value


def main():
    # fail before any work if a log is missing, naming all of them at once
    missing = [n for n in LOG_NAMES if not (LOGS_DIR / f"{n}.xes").exists()]
    if missing:
        raise SystemExit(
            f"missing in {LOGS_DIR}: {', '.join(f'{n}.xes' for n in missing)}"
        )

    out = open(OUT, "w", encoding="utf-8")

    def write(text=""):
        out.write(text + "\n")
        out.flush()
        print(text, flush=True)

    def per_pair(n):
        return f"{n} pair" if n == 1 else f"{n} pairs"

    gpu = gpu_device()
    title = "CWINDOW embeddings (CPU vs GPU)" + (" and EMSC" if RUN_EMSC else "")
    write(f"{title}, per log pair")
    write(f"machine:  {platform.node()} | {platform.platform()}")
    write(f"torch:    {torch.__version__} | cpu device 'cpu' | gpu device '{gpu}'")
    write(
        f"settings: c = {WINDOW}, d = {DIM}, unit_norm = {UNIT_NORM}, "
        f"balance_compass = {BALANCE_COMPASS}, seed = {SEED}"
    )
    if RUN_EMSC:
        write(f"          EMSC timeout = {EMSC_TIMEOUT}s per pair")
    if gpu.type == "cuda":
        write(f"GPU:      {torch.cuda.get_device_name(0)}")
    elif gpu.type == "cpu":
        write("GPU:      none found — the GPU run repeats the CPU run")
    write()
    write("schedule per pair (lr x epochs)")
    write(f"{'pair':<{PAIR_W}}{'compass':>16}{'per-log':>16}")
    for p in PAIRS:
        compass = f"{p['compass_learning_rate']:g} x {p['compass_epochs']}"
        per_log = f"{p['log_learning_rate']:g} x {p['log_epochs']}"
        write(f"{pair_id(p):<{PAIR_W}}{compass:>16}{per_log:>16}")
    write()

    print("reading the logs ...", flush=True)
    logs = {name: read_log(name) for name in LOG_NAMES}
    traces = {name: lcw._convert_to_simpleEventLog(df) for name, df in logs.items()}

    write("CWINDOW (seconds)")
    write(
        f"{'pair':<{PAIR_W}}{'device':<8}{'compass':>10}"
        f"{'retrain A':>11}{'retrain B':>11}{'total':>10}"
    )
    totals = {}  # {(device, family): [seconds per pair]}
    for label, device in (("cpu", torch.device("cpu")), ("gpu", gpu)):
        for p in PAIRS:
            compass_s, retrain_i, retrain_j, total = time_cwindow(
                traces[p["l1"]], traces[p["l2"]], p, device
            )
            totals.setdefault((label, family(p)), []).append(total)
            write(
                f"{pair_id(p):<{PAIR_W}}{label:<8}{compass_s:>10.2f}"
                f"{retrain_i:>11.2f}{retrain_j:>11.2f}{total:>10.2f}"
            )
    # per family: the logs differ too much in size for one mean to describe them all
    for label in ("cpu", "gpu"):
        for fam in FAMILIES:
            values = totals[(label, fam)]
            write(
                f"  {label} {fam}: {sum(values):.2f}s for {per_pair(len(values))} "
                f"({sum(values) / len(values):.2f}s per pair)"
            )
    write()

    if RUN_EMSC:
        write("EMSC (seconds, whole logs)")
        write(f"{'pair':<{PAIR_W}}{'seconds':>10}{'EMSC':>12}")
        emsc_s = {fam: [] for fam in FAMILIES}
        for p in PAIRS:
            try:
                seconds, value = time_emsc(logs[p["l1"]], logs[p["l2"]])
                emsc_s[family(p)].append(seconds)
                write(f"{pair_id(p):<{PAIR_W}}{seconds:>10.2f}{value:>12.4f}")
            except Exception as exc:
                write(f"{pair_id(p):<{PAIR_W}}{'failed':>10}   {exc!r}")
        for fam in FAMILIES:
            values = emsc_s[fam]
            n_pairs = sum(family(p) == fam for p in PAIRS)
            if values:
                write(
                    f"  {fam}: {sum(values):.2f}s for {len(values)} of "
                    f"{per_pair(n_pairs)} ({sum(values) / len(values):.2f}s per pair)"
                )
            else:
                write(f"  {fam}: none of {per_pair(n_pairs)} finished")

    out.close()
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

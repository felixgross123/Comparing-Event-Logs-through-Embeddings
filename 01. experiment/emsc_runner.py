"""Earth Mover's Stochastic Conformance between two projected sub-logs, with a timeout.

``ebi.conformance_earth_movers`` runs inside a compiled Rust extension, so a long call
cannot be interrupted from Python: a ``signal.alarm`` only fires once the call has already
returned (measured: a 2 s alarm on a 39 s call raised after 39 s). The only way to bound it
is to run it in a separate process and kill that process, which is what
:func:`emsc_with_timeout` does -- one short-lived subprocess per call, no pool.

A subprocess rather than ``multiprocessing`` on purpose: ``spawn`` re-executes the parent's
``__main__``, which misbehaves in a notebook kernel, whereas ``subprocess.run(timeout=...)``
kills the child on overrun anywhere.

Kept out of ``log_comparison_simple`` so the child imports only pandas / pm4py / ebi and
not torch.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT = 120


def _to_event_log(df):
    """DataFrame (case:concept:name, concept:name, time:timestamp) -> PM4Py EventLog."""
    from pm4py.objects.log.obj import Event, EventLog, Trace

    log = EventLog()
    for case_id, case_df in df.groupby("case:concept:name", sort=False):
        trace = Trace()
        trace.attributes["concept:name"] = str(case_id)
        for _, row in case_df.sort_values("time:timestamp").iterrows():
            trace.append(
                Event(
                    {
                        "concept:name": str(row["concept:name"]),
                        "time:timestamp": row["time:timestamp"],
                    }
                )
            )
        log.append(trace)
    return log


def _to_float(v):
    """Ebi returns a float, a list, or a string such as 'Approximately 0.42'."""
    if isinstance(v, (list, tuple)):
        v = v[0]
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and "Approximately" in v:
        return float(v.split("Approximately")[-1].strip())
    return float(str(v).strip())


def emsc(df_1, df_2):
    """EMSC between two projected sub-logs (Ebi: conformance = 1 - distance)."""
    import ebi

    return _to_float(
        ebi.conformance_earth_movers(_to_event_log(df_1), _to_event_log(df_2))
    )


def emsc_with_timeout(df_1, df_2, timeout=DEFAULT_TIMEOUT):
    """
    :func:`emsc`, but abandoned after ``timeout`` seconds.

    Runs the computation in a subprocess and kills it on overrun, so a stuck Ebi call costs
    at most ``timeout`` seconds and only that one activity is lost. Raises ``TimeoutError``
    in that case; a crash in the child becomes ``RuntimeError``. ``timeout=None`` waits
    indefinitely.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.pkl", Path(tmp) / "out.pkl"
        with in_path.open("wb") as fh:
            pickle.dump((df_1, df_2), fh, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            proc = subprocess.run(
                [sys.executable, __file__, str(in_path), str(out_path)],
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"EMSC exceeded {timeout}s") from None

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            raise RuntimeError(
                tail[-1] if tail else f"child exited with {proc.returncode}"
            )

        with out_path.open("rb") as fh:
            return pickle.load(fh)


if __name__ == "__main__":
    _in, _out = Path(sys.argv[1]), Path(sys.argv[2])
    with _in.open("rb") as _fh:
        _df_1, _df_2 = pickle.load(_fh)
    with _out.open("wb") as _fh:
        pickle.dump(emsc(_df_1, _df_2), _fh, protocol=pickle.HIGHEST_PROTOCOL)

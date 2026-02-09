#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_root", required=True, help="z.B. records/throws/fs_2026-02-08_ALL_THROWS_FIXED")
    ap.add_argument("--runs_root", default="records/throws", help="Ordner mit den Einzel-runs fs_*_job_wurf_*")
    ap.add_argument("--glob", default="fs_*_job_wurf_*", help="Welche Run-Ordner einlesen")
    args = ap.parse_args()

    combined_root = Path(args.combined_root).resolve()
    runs_root = Path(args.runs_root).resolve()

    data_path = combined_root/"data/chunk-000/file-000.parquet"
    epi_path  = combined_root/"meta/episodes/chunk-000/file-000.parquet"
    tasks_path= combined_root/"meta/tasks.parquet"

    assert data_path.exists(), data_path
    assert epi_path.exists(), epi_path

    # 1) Alle Run-Targets in der Reihenfolge einsammeln, wie sie im Combine eingegangen sind:
    #    Dein Combine hatte run_dirs = sorted(glob). Wir machen exakt das gleiche Sorting.
    run_dirs = sorted([p for p in runs_root.glob(args.glob) if p.is_dir()])
    labels = []
    for rd in run_dirs:
        lp = rd/"meta/throw_label.json"
        if not lp.exists():
            continue
        j = json.loads(lp.read_text(encoding="utf-8"))
        labels.append((rd.name, tuple(map(float, j["target_xy_mm"])), bool(j.get("success", True))))

    if not labels:
        raise SystemExit("Keine throw_label.json gefunden.")

    # Hinweis: du hattest beim Combine --only_success, also nehmen wir auch hier nur success==True
    labels = [x for x in labels if x[2]]
    print("labels (success) :", len(labels))

    # 2) Unique Targets -> task_index
    uniq = sorted(set(t for _, t, _ in labels))
    target_to_tid = {t:i for i,t in enumerate(uniq)}
    print("unique targets:", len(uniq))

    # 3) Episode-Tabelle laden und pro Episode task_index setzen
    epi = pq.read_table(epi_path).to_pandas()
    n_ep = epi.shape[0]
    if n_ep != len(labels):
        raise SystemExit(f"Episodenanzahl ({n_ep}) passt nicht zu labels ({len(labels)}). "
                         f"Das bedeutet: Combine-Reihenfolge/Filter weicht ab. Sag Bescheid, dann matchen wir über run_name.")

    ep_task_index = np.array([target_to_tid[labels[i][1]] for i in range(n_ep)], dtype=np.int64)
    epi["task_index"] = ep_task_index

    # Optional: falls eine Spalte 'tasks' existiert, kann man sie sprechend setzen
    # (meist ist das ein String oder eine Liste; wir setzen hier einen stabilen Namen)
    if "tasks" in epi.columns:
        epi["tasks"] = [f"target_{t[0]:.1f}_{t[1]:.1f}" for _, t, _ in labels]

    # 4) Data-Parquet laden und task_index pro Frame via episode_index setzen
    dt = pq.read_table(data_path)
    ep_idx = np.asarray(dt["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    new_ti = ep_task_index[ep_idx]  # broadcast über frames

    # task_index ersetzen
    if "task_index" in dt.column_names:
        col_i = dt.column_names.index("task_index")
        dt = dt.set_column(col_i, "task_index", pa.array(new_ti, type=pa.int64()))
    else:
        dt = dt.append_column("task_index", pa.array(new_ti, type=pa.int64()))

    # 5) tasks.parquet schreiben (41 Zeilen)
    idx = [f"target_{t[0]:.1f}_{t[1]:.1f}" for t in uniq]
    tasks_df = pd.DataFrame({"task_index": list(range(len(uniq)))}, index=idx)

    # Backups
    (combined_root/"meta").mkdir(parents=True, exist_ok=True)
    epi_bak = epi_path.with_suffix(".parquet.bak")
    data_bak = data_path.with_suffix(".parquet.bak")
    tasks_bak = tasks_path.with_suffix(".parquet.bak")
    if not epi_bak.exists():
        epi_path.rename(epi_bak)
    if not data_bak.exists():
        data_path.rename(data_bak)
    if tasks_path.exists() and not tasks_bak.exists():
        tasks_path.rename(tasks_bak)

    # Neu schreiben
    pq.write_table(pa.Table.from_pandas(epi, preserve_index=False), epi_path)
    pq.write_table(dt, data_path)
    tasks_df.to_parquet(tasks_path)

    print("[OK] gepatcht:")
    print(" -", epi_path)
    print(" -", data_path)
    print(" -", tasks_path)
    print("Backups:")
    print(" -", epi_bak)
    print(" -", data_bak)
    if tasks_bak.exists():
        print(" -", tasks_bak)


if __name__ == "__main__":
    main()

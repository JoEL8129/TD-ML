import json
import ast
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

# ——— Globals ———
pipeline = None
classes = None
n_feat = None

# Kept for backward-compatible loading (default filename next to the .toe unless Savefolder is set)


# ——— Minimal I/O helpers ———

def _maybe_write_metadata(meta: dict):
    """Best-effort: mirror metadata into a TouchDesigner DAT named 'metadata'."""
    try:
        td = op("metadata")
        td.text = json.dumps(meta, indent=2, sort_keys=False, default=str)
    except Exception:
        pass


def _meta_with_name_and_path_first(meta: dict, path: Path) -> dict:
    base = {"model_name": path.name, "model_path": str(path)}
    for k, v in meta.items():
        if k not in base:
            base[k] = v
    return base


# --- Header-aware helpers ---

def _header_flags():
    """Return (first_row_header, first_col_header) as booleans from the parent COMP."""
    try:
        frh = bool(parent().par.Firstrowheader.eval())
    except Exception:
        frh = False
    try:
        fch = bool(parent().par.Firstcolheader.eval())
    except Exception:
        fch = False
    return frh, fch


def _table_to_numpy(table, first_row_header=False, first_col_header=False):
    """
    Convert a DAT table to a 2D float numpy array, optionally skipping the first row and/or column.
    """
    r0 = 1 if first_row_header else 0
    c0 = 1 if first_col_header else 0
    if table.numRows - r0 <= 0 or table.numCols - c0 <= 0:
        raise ValueError(
            f"After header slicing (row={r0}, col={c0}), table has no data "
            f"(rows={table.numRows}, cols={table.numCols})."
        )
    out = []
    for r in range(r0, table.numRows):
        row_vals = []
        for c in range(c0, table.numCols):
            try:
                row_vals.append(float(table[r, c].val))
            except Exception as e:
                raise ValueError(
                    f"Non-numeric value at r={r}, c={c} ('{table[r, c].val}') after header slicing."
                ) from e
        out.append(row_vals)
    return np.asarray(out, dtype=float)


def _labels_from_table(table, first_row_header=False, first_col_header=False):
    """Extract 1D labels from a DAT table. Labels live in the first data column."""
    r0 = 1 if first_row_header else 0
    c = 1 if first_col_header else 0
    if table.numCols <= c:
        raise ValueError(
            f"Label column index {c} is out of range for table with {table.numCols} columns."
        )
    return np.array([str(table[r, c].val) for r in range(r0, table.numRows)], dtype=object)


def _weights_from_table(table, first_row_header=False, first_col_header=False):
    """Extract 1D float sample weights from a DAT table. Same convention as labels."""
    r0 = 1 if first_row_header else 0
    c = 1 if first_col_header else 0
    if table.numCols <= c:
        raise ValueError(
            f"Weight column index {c} is out of range for table with {table.numCols} columns."
        )
    vals = []
    for r in range(r0, table.numRows):
        try:
            vals.append(float(table[r, c].val))
        except Exception as e:
            raise ValueError(
                f"Non-numeric weight at r={r}, c={c} ('{table[r, c].val}')."
            ) from e
    return np.asarray(vals, dtype=float)


# ——— Load (kept, default path inferred) ———

def _default_model_path() -> Path:
    """Where we *would* expect the model by default (for load_models)."""
    try:
        raw = (parent().par.Savefolder.eval() or "").strip()
    except Exception:
        raw = ""
    base = Path(raw) if raw else Path(project.folder)
    base.mkdir(parents=True, exist_ok=True)
    return base / 'model_default.joblib'


def load_model(path_str: str = parent().par.Loadfile.eval()) -> bool:
    """Load a model from an explicit path, or fall back to the default bundle path if not provided."""
    global pipeline, classes, n_feat

    path = Path(path_str) if path_str else _default_model_path()
    if not path.exists():
        pipeline = None
        classes = None
        n_feat = None
        msg = f"[ml] No model bundle at '{path}'. Train first or save a model."
        print(msg)
        try:
            op("labelOut").text = msg
        except Exception:
            pass
        _maybe_write_metadata({})
        return False

    loaded = joblib.load(str(path))
    if isinstance(loaded, dict) and "pipeline" in loaded:
        pipeline = loaded["pipeline"]
        meta = dict(loaded.get("meta", {}))
    else:  # legacy plain pipeline
        pipeline = loaded
        meta = {}

    clf = pipeline.named_steps["mlpclassifier"]
    scaler = pipeline.named_steps["standardscaler"]
    classes = clf.classes_
    n_feat = scaler.scale_.shape[0]

    meta = _meta_with_name_and_path_first(meta, path)
    _maybe_write_metadata(meta)

    msg = f"[ml] Loaded model from '{path}'."
    print(msg)
    try:
        op("labelOut").text = msg
    except Exception:
        pass
    return True


def save_model(path_str: str = parent().par.Savefile.eval(), note: str = "manual save") -> str:
    global pipeline
    if pipeline is None:
        print("[ml] No pipeline in memory to save. Train first.")
        return ""

    out_path = Path(path_str)
    print(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(out_path)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": note,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "op_path": parent().path,
    }

    joblib.dump({"pipeline": pipeline, "meta": _meta_with_name_and_path_first(meta, out_path)}, str(out_path), compress=3)
    msg = f"[ml] Saved model to '{out_path}'."
    print(msg)
    _maybe_write_metadata(_meta_with_name_and_path_first(meta, out_path))
    print(f"[ml] Saved bundle to '{out_path}'.")
    return str(out_path)


# ——— Train / Retrain (no implicit saving) ———

def train(X_table, y_table, param_table, use_weights: bool = False, weights_table=None):
    """Formerly train_and_save. Trains in-memory; call save_model(path) explicitly to persist."""
    global pipeline, classes, n_feat
    frh, fch = _header_flags()

    X = _table_to_numpy(X_table, first_row_header=frh, first_col_header=fch)
    y = _labels_from_table(y_table, first_row_header=frh, first_col_header=fch)

    if use_weights and weights_table is not None:
        w = _weights_from_table(weights_table, first_row_header=frh, first_col_header=fch)
    else:
        w = None

    # Parse params (skip header row)
    params = {}
    for row in param_table.rows()[1:]:
        key, raw = row[0].val, row[1].val
        try:
            params[key] = ast.literal_eval(raw)
        except Exception:
            params[key] = raw

    clf = make_pipeline(StandardScaler(), MLPClassifier(**params))
    try:
        clf.fit(X, y, mlpclassifier__sample_weight=w) if w is not None else clf.fit(X, y)
    except TypeError:
        clf.fit(X, y)

    pipeline = clf
    classes = clf.named_steps["mlpclassifier"].classes_
    n_feat = clf.named_steps["standardscaler"].scale_.shape[0]

    meta_core = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "train",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": [str(c) for c in classes],
        "params": params,
        "use_weights": bool(use_weights),
        "headers": {"first_row": bool(frh), "first_col": bool(fch)},
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "op_path": parent().path,
    }
    _maybe_write_metadata(meta_core)
    save_model()
    return pipeline, meta_core


def retrain(X_table, y_table, use_weights: bool = False, weights_table=None):
    """Formerly retrain_and_save. Updates the in-memory model; call save_model(path) to persist."""
    global pipeline, classes, n_feat
    frh, fch = _header_flags()

    if pipeline is None:
        # Backstop: if no model exists, behave like train() using default params DAT
        return train(X_table, y_table, op("training_params"), use_weights, weights_table)

    X_new = _table_to_numpy(X_table, first_row_header=frh, first_col_header=fch)
    y_new = _labels_from_table(y_table, first_row_header=frh, first_col_header=fch)

    if use_weights and weights_table is not None:
        w = _weights_from_table(weights_table, first_row_header=frh, first_col_header=fch)
    else:
        w = None

    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["mlpclassifier"]

    try:
        scaler.partial_fit(X_new)
        X_scaled = scaler.transform(X_new)
    except Exception:
        new_scaler = StandardScaler().fit(X_new)
        pipeline.named_steps["standardscaler"] = new_scaler
        X_scaled = new_scaler.transform(X_new)

    if hasattr(clf, "partial_fit"):
        try:
            clf.partial_fit(X_scaled, y_new, classes=clf.classes_, sample_weight=w if w is not None else None)
        except TypeError:
            clf.partial_fit(X_scaled, y_new, classes=clf.classes_)
    else:
        clf.fit(X_scaled, y_new, sample_weight=w if w is not None else None)

    classes = clf.classes_
    n_feat = getattr(pipeline.named_steps["standardscaler"], "scale_", np.zeros(X_new.shape[1])).shape[0]

    meta_core = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "retrain",
        "n_samples_added": int(X_new.shape[0]),
        "n_features": int(X_new.shape[1]),
        "classes": [str(c) for c in classes],
        "use_weights": bool(use_weights),
        "headers": {"first_row": bool(frh), "first_col": bool(fch)},
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "op_path": parent().path,
    }
    _maybe_write_metadata(meta_core)
    save_model()
    return pipeline, meta_core


# ——— TouchDesigner callbacks ———

def onSetupParameters(scriptOp):
    return


def cook(scriptOp):
    global pipeline, classes, n_feat
    scriptOp.isTimeSlice = False

    # No model loaded yet -> friendly message + zero output
    if pipeline is None or classes is None or n_feat is None:
        msg = "No model found. Click TRAIN to create one."
        try:
            td = op("labelOut")
            if td.text != msg:
                td.text = msg
        except Exception:
            pass

        scriptOp.clear()
        scriptOp.numSamples = 1
        scriptOp.rate = parent().par.Samplerate
        scriptOp.appendChan("out1")[0] = 0
        return

    method = parent().par.Predictmethod.eval()

    arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 1:
        vals = arr[:, 0]  # shape (channels,)
    elif arr.ndim == 2:
        vals = arr[:, 0]  # (channels, samples>1): first sample per channel
    elif arr.ndim == 1:
        vals = arr
    else:
        raise ValueError("Unexpected input shape: %s" % (arr.shape,))

    # 2) Pad/truncate to n_feat
    L = vals.shape[0]
    if L < n_feat:
        padded = np.zeros((n_feat,), dtype=np.float32)
        padded[:L] = vals
        vals = padded
    elif L > n_feat:
        vals = vals[:n_feat]

    # 3) Scale and predict
    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["mlpclassifier"]
    Xs = scaler.transform(vals.reshape(1, -1))

    if method == "predict":
        pred_label = clf.predict(Xs)[0]
        out_text = str(pred_label)
    elif method == "predict_proba":
        proba = clf.predict_proba(Xs)[0]
        best_i = int(np.argmax(proba))
        pred_label = classes[best_i]
        out_text = "\n".join(f"{cls}: {p*100:.1f}%" for cls, p in zip(classes, proba))

    elif method == "predict_log_proba":
        logp = clf.predict_log_proba(Xs)[0]
        best_i = int(np.argmax(logp))
        pred_label = classes[best_i]
        out_text = "\n".join(f"{cls}: {v:.3f}" for cls, v in zip(classes, logp))
    else:
        raise ValueError(f"Unknown Predictmethod: {method!r}")

    # 4) Write text & CHOP out
    try:
        td = op("labelOut")
        if td.text != out_text:
            td.text = out_text
    except Exception:
        pass

    scriptOp.clear()
    scriptOp.numSamples = 1
    scriptOp.rate = parent().par.Samplerate
    chan = scriptOp.appendChan('empty')
    chan[0] = 1 if pred_label == "silence" else 0

# ——— Optional: try loading on drop ———

import json
import ast
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

# Optional smoothing filters
from scipy.signal import savgol_filter, medfilt, wiener, butter, filtfilt, firwin, convolve

# ——— Globals ———
pipeline = None          # sklearn Pipeline(StandardScaler -> TransformedTargetRegressor(MLPRegressor))
n_feat = None            # number of input features expected
n_outputs = None         # number of outputs (targets)
last_params = None       # remember last known training params so save_model can persist them
input_channel_names = [] # remembered input channel names from headers (if any)
output_channel_names = []# remembered output channel names from headers (if any)

BUNDLE_FILENAME = "bundle.joblib"  # used when Filepath is a folder or has no suffix


# ——— Minimal I/O helpers (same style as classifier) ———

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


# --- Header-aware helpers (match the classifier’s behavior) ---

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


def _is_floatish(s) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _has_first_row_header_auto(table) -> bool:
    """Auto-detect 'new' format with header row (row 0) and index col (col 0)."""
    return (
        table is not None
        and table.numRows > 0
        and table.numCols > 1
        and not _is_floatish(table[0, 0].val)
    )


def _channel_names_from_table(table):
    """
    Return list of column names if we believe a header row exists; else [].
    Priority:
      1) Respect Firstrowheader/Firstcolheader if available.
      2) Fallback to auto-detect: header if [0,0] is non-numeric.
    """
    if table is None or table.numRows == 0 or table.numCols == 0:
        return []

    frh, fch = _header_flags()
    if frh:
        start_col = 1 if fch else 0
        return [table[0, c].val for c in range(start_col, table.numCols)]

    # Fallback auto-detection (old behavior)
    if _has_first_row_header_auto(table):
        return [table[0, c].val for c in range(1, table.numCols)]

    return []


def _table_to_numpy(table, first_row_header=False, first_col_header=False):
    """
    Convert a DAT table to a 2D float numpy array, optionally skipping the first row and/or column.
    """
    r0 = 1 if first_row_header else 0
    c0 = 1 if first_col_header else 0
    if table is None or table.numRows - r0 <= 0 or table.numCols - c0 <= 0:
        raise ValueError(
            f"After header slicing (row={r0}, col={c0}), table has no data "
            f"(rows={getattr(table,'numRows',0)}, cols={getattr(table,'numCols',0)})."
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


def _weights_from_table(table, first_row_header=False, first_col_header=False):
    """Extract 1D float sample weights from a DAT table. Same convention as labels/weights in classifier."""
    r0 = 1 if first_row_header else 0
    c = 1 if first_col_header else 0
    if table is None or table.numCols <= c:
        raise ValueError(
            f"Weight column index {c} is out of range for table with {getattr(table,'numCols',0)} columns."
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


# ——— Path helpers ———

def _default_model_path() -> Path:
    """Where we *would* expect the model by default (same idea as classifier)."""
    try:
        raw = (parent().par.Savefolder.eval() or "").strip()
    except Exception:
        raw = ""
    base = Path(raw) if raw else Path(project.folder)
    base.mkdir(parents=True, exist_ok=True)
    return base / 'model_default.joblib'


def _ensure_joblib_ext(p: Path) -> Path:
    """Ensure .joblib extension is present."""
    return p if p.suffix.lower() == ".joblib" else Path(str(p) + ".joblib")


def _resolve_model_path() -> Path:
    """
    Prefer explicitly configured paths in this order:
      1) parent().par.Filepath (if set). If it's a folder or has no suffix, append BUNDLE_FILENAME.
      2) parent().par.Loadfile (if set, ensure .joblib).
      3) parent().par.Savefile (if set, ensure .joblib).
      4) default bundle path.
    """
    def _get_par(name):
        try:
            return (getattr(parent().par, name).eval() or "").strip()
        except Exception:
            return ""

    # 1) Filepath (compat with old script)
    raw = _get_par("Filepath")
    if raw:
        p = Path(raw)
        if not p.suffix:
            p = p / BUNDLE_FILENAME
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) Loadfile / 3) Savefile
    for raw in (_get_par("Loadfile"), _get_par("Savefile")):
        if raw:
            p = _ensure_joblib_ext(Path(raw))
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

    # 4) Default
    return _default_model_path()


# ——— Bundle meta I/O (merge-preserving) ———

def _load_existing_meta(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        loaded = joblib.load(str(path))
        if isinstance(loaded, dict):
            return dict(loaded.get("meta", {}))
        return {}
    except Exception:
        return {}


def _merge_and_dump_bundle(pipeline_obj, new_meta: dict, path: Path) -> Path:
    """
    Merge existing meta (if any) with new_meta (new values win), then save.
    Ensures model_name/model_path fields are first.
    """
    existing_meta = _load_existing_meta(path)
    merged = dict(existing_meta)
    merged.update(dict(new_meta))
    merged = _meta_with_name_and_path_first(merged, path)
    payload = {"pipeline": pipeline_obj, "meta": merged}
    joblib.dump(payload, str(path), compress=3)
    _maybe_write_metadata(merged)
    print(f"[ml] Saved bundle to '{path}'.")
    return path


# ——— Load ———

def load_model(path_str: str = None) -> bool:
    """Load a model from an explicit path or the resolved bundle path."""
    global pipeline, n_feat, n_outputs, last_params, input_channel_names, output_channel_names

    path = _ensure_joblib_ext(Path(path_str)) if path_str else _resolve_model_path()
    if not path.exists():
        pipeline = None
        n_feat = None
        n_outputs = None
        last_params = None
        input_channel_names = []
        output_channel_names = []
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
        last_params = dict(meta.get("params", {})) if isinstance(meta.get("params", {}), dict) else None
        input_channel_names = list(meta.get("input_channel_names", []) or [])
        output_channel_names = list(meta.get("output_channel_names", []) or [])
    else:  # legacy plain estimator
        pipeline = loaded
        meta = {}
        last_params = None
        input_channel_names = []
        output_channel_names = []

    # Infer shapes
    scaler_X = pipeline.named_steps["standardscaler"]
    n_feat = int(scaler_X.scale_.shape[0])

    try:
        ttr = pipeline.named_steps["transformedtargetregressor"]
        if hasattr(ttr, "regressor_") and hasattr(ttr.regressor_, "n_outputs_"):
            n_outputs = int(getattr(ttr.regressor_, "n_outputs_", 1))
        else:
            n_outputs = int(meta.get("n_outputs", 1))
    except Exception:
        n_outputs = int(meta.get("n_outputs", 1))

    meta = _meta_with_name_and_path_first(meta, path)
    _maybe_write_metadata(meta)

    # Echo saved output channel names to a helper DAT (optional quality-of-life)
    try:
        write_channel_names("output", "chan_from_meta", style="csv")
    except Exception:
        pass

    msg = f"[ml] Loaded model from '{path}'."
    print(msg)
    try:
        op("labelOut").text = msg
    except Exception:
        pass
    return True


# ——— Save ———

def _extract_regressor_params(clf) -> dict:
    """Best-effort: pull params from the inner regressor."""
    try:
        ttr = clf.named_steps["transformedtargetregressor"]
        reg = getattr(ttr, "regressor_", None) or getattr(ttr, "regressor", None)
        if hasattr(reg, "get_params"):
            return dict(reg.get_params(deep=False))
    except Exception:
        pass
    return {}


def save_model(path_str: str = None, note: str = "manual save") -> str:
    """
    Persist the current pipeline + merged metadata to disk (.joblib).
    Preserves prior fields (incl. channel names) and updates params, counts, etc.
    """
    global pipeline, n_feat, n_outputs, last_params, input_channel_names, output_channel_names
    if pipeline is None:
        print("[ml] No pipeline in memory to save. Train first.")
        return ""

    path = _ensure_joblib_ext(Path(path_str)) if path_str else _resolve_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep params if we know them; otherwise pull from the fitted regressor
    params_to_save = dict(last_params) if isinstance(last_params, dict) else _extract_regressor_params(pipeline)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": note,
        "n_features": int(n_feat) if n_feat is not None else None,
        "n_outputs": int(n_outputs) if n_outputs is not None else None,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "op_path": parent().path,
        "params": params_to_save,
    }

    # Add/refresh channel names if we have them
    if input_channel_names:
        meta["input_channel_names"] = list(input_channel_names)
    if output_channel_names:
        meta["output_channel_names"] = list(output_channel_names)

    _merge_and_dump_bundle(pipeline, meta, path)
    return str(path)


# ——— Train / Retrain ———

def train(X_table, y_table, param_table, use_weights: bool = False, weights_table=None):
    """Train an MLP *Regressor* in memory; call save_model(...) to persist."""
    global pipeline, n_feat, n_outputs, last_params, input_channel_names, output_channel_names
    frh, fch = _header_flags()

    X = _table_to_numpy(X_table, first_row_header=frh, first_col_header=fch)
    y = _table_to_numpy(y_table, first_row_header=frh, first_col_header=fch)

    if use_weights and weights_table is not None:
        w = _weights_from_table(weights_table, first_row_header=frh, first_col_header=fch)
    else:
        w = None

    # Grab channel names if header rows are present (or auto-detected)
    input_channel_names = _channel_names_from_table(X_table)
    output_channel_names = _channel_names_from_table(y_table)

    # Parse params (skip header row)
    params = {}
    for row in param_table.rows()[1:]:
        key, raw = row[0].val, row[1].val
        try:
            params[key] = ast.literal_eval(raw)
        except Exception:
            params[key] = raw

    # Remember params so save_model can persist them
    last_params = dict(params)

    # Build pipeline: scale X, scale y internally via TTR, then regress
    reg = MLPRegressor(**params)
    ttr = TransformedTargetRegressor(
        regressor=reg,
        transformer=StandardScaler(),
        check_inverse=False,  # suppress warning for non-perfect inverse
    )
    clf = make_pipeline(StandardScaler(), ttr)

    try:
        clf.fit(X, y, transformedtargetregressor__sample_weight=w) if w is not None else clf.fit(X, y)
    except TypeError:
        clf.fit(X, y)

    pipeline = clf
    n_feat = clf.named_steps["standardscaler"].scale_.shape[0]
    n_outputs = int(y.shape[1]) if y.ndim == 2 else 1

    meta_core = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "train",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_outputs": int(n_outputs),
        "params": last_params,
        "use_weights": bool(use_weights),
        "headers": {"first_row": bool(frh), "first_col": bool(fch)},
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "op_path": parent().path,
    }
    if input_channel_names:
        meta_core["input_channel_names"] = list(input_channel_names)
    if output_channel_names:
        meta_core["output_channel_names"] = list(output_channel_names)

    # Persist immediately (merge with any existing meta)
    save_model(note="train")
    # Also write output names to a helper DAT for convenience
    try:
        write_channel_names("output", "chan_from_meta", style="csv")
    except Exception:
        pass

    return pipeline, meta_core


def retrain(X_table, y_table, use_weights: bool = False, weights_table=None):
    """Update the in-memory regressor; call save_model(...) to persist (merge-preserving)."""
    global pipeline, n_feat, n_outputs, last_params, input_channel_names, output_channel_names
    frh, fch = _header_flags()

    if pipeline is None:
        # Backstop: behave like train() using default params DAT (mirrors classifier)
        return train(X_table, y_table, op("training_params"), use_weights, weights_table)

    X_new = _table_to_numpy(X_table, first_row_header=frh, first_col_header=fch)
    y_new = _table_to_numpy(y_table, first_row_header=frh, first_col_header=fch)
    if use_weights and weights_table is not None:
        w = _weights_from_table(weights_table, first_row_header=frh, first_col_header=fch)
    else:
        w = None

    # Update remembered channel names if new headers are present
    new_in_names = _channel_names_from_table(X_table)
    new_out_names = _channel_names_from_table(y_table)
    if new_in_names:
        input_channel_names = new_in_names
    if new_out_names:
        output_channel_names = new_out_names

    scaler_X = pipeline.named_steps["standardscaler"]
    ttr = pipeline.named_steps["transformedtargetregressor"]
    reg = ttr.regressor if hasattr(ttr, "regressor") else getattr(ttr, "regressor_", None)

    # 1) Update X scaler
    try:
        scaler_X.partial_fit(X_new)
        Xs = scaler_X.transform(X_new)
    except Exception:
        new_scaler = StandardScaler().fit(X_new)
        pipeline.named_steps["standardscaler"] = new_scaler
        Xs = new_scaler.transform(X_new)

    # 2) Update y transformer (lives inside TTR)
    try:
        yt = getattr(ttr, "transformer_", ttr.transformer)
        if hasattr(yt, "partial_fit"):
            yt.partial_fit(y_new)
            ys = yt.transform(y_new)
        else:
            yt = StandardScaler().fit(y_new)
            ttr.transformer_ = yt
            ys = yt.transform(y_new)
    except Exception:
        yt = StandardScaler().fit(y_new)
        ttr.transformer_ = yt
        ys = yt.transform(y_new)

    # 3) Try incremental update on the regressor; else refit on the chunk
    did_partial = False
    if hasattr(reg, "partial_fit"):
        try:
            if w is not None:
                reg.partial_fit(Xs, ys, sample_weight=w)
            else:
                reg.partial_fit(Xs, ys)
            did_partial = True
        except TypeError:
            pass

    if not did_partial:
        try:
            if w is not None:
                reg.fit(Xs, ys, sample_weight=w)
            else:
                reg.fit(Xs, ys)
        except TypeError:
            reg.fit(Xs, ys)

    # Keep the most current params so saves include them
    try:
        last_params = dict(reg.get_params(deep=False)) if hasattr(reg, "get_params") else last_params
    except Exception:
        pass

    n_feat = pipeline.named_steps["standardscaler"].scale_.shape[0]
    n_outputs = int(y_new.shape[1]) if y_new.ndim == 2 else 1

    meta_core = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "retrain",
        "n_samples_added": int(X_new.shape[0]),
        "n_features": int(X_new.shape[1]),
        "n_outputs": int(n_outputs),
        "use_weights": bool(use_weights),
        "headers": {"first_row": bool(frh), "first_col": bool(fch)},
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "op_path": parent().path,
    }
    if input_channel_names:
        meta_core["input_channel_names"] = list(input_channel_names)
    if output_channel_names:
        meta_core["output_channel_names"] = list(output_channel_names)

    # Persist (merge with existing meta)
    save_model(note="retrain")
    try:
        write_channel_names("output", "chan_from_meta", style="csv")
    except Exception:
        pass

    return pipeline, meta_core


# ——— Public helpers for reading meta & channel names ———

def _read_bundle_meta():
    """Robustly read the saved bundle's meta, using configured paths."""
    try:
        p = _resolve_model_path()
        if not p.exists():
            return {}
        loaded = joblib.load(str(p))
        return dict(loaded.get("meta", {})) if isinstance(loaded, dict) else {}
    except Exception as e:
        print(f"[ml] _read_bundle_meta error: {e}")
        return {}


def get_channel_names(which="input"):
    """
    which: 'input' or 'output'
    Returns the saved header names list (or []).
    """
    meta = _read_bundle_meta()
    key = "input_channel_names" if which == "input" else "output_channel_names"
    names = meta.get(key, [])
    if not isinstance(names, (list, tuple)):
        names = []
    return [str(n) for n in names]


def write_channel_names(which="input", dest_dat="chan_from_meta", style="lines"):
    """
    Write saved channel names to a Text DAT.
      which: 'input' or 'output'
      dest_dat: name of the Text DAT (e.g., 'chan_from_meta')
      style: 'lines' -> one name per line, 'csv' -> comma-separated
    """
    names = get_channel_names(which)
    try:
        td = op(dest_dat)
    except Exception:
        print(f"[ml] write_channel_names: missing DAT '{dest_dat}'")
        return False

    if not names:
        td.text = ""  # clear if nothing saved
        print(f"[ml] No {which} channel names found in bundle metadata.")
        return False

    td.text = ("\n".join(names)) if style == "lines" else (",".join(names))
    return True


# ——— TouchDesigner callbacks ———

def onSetupParameters(scriptOp):
    return


def onCook(scriptOp):
    global pipeline, n_feat, n_outputs
    scriptOp.isTimeSlice = False

    # No model loaded yet -> friendly message + zero output
    if pipeline is None or n_feat is None:
        msg = "No model found. Click TRAIN to create one."
        try:
            td = op("labelOut")
            if td.text != msg:
                td.text = msg
        except Exception:
            pass

        scriptOp.clear()
        scriptOp.numSamples = 1
        try:
            scriptOp.rate = parent().par.Samplerate
        except Exception:
            scriptOp.rate = 60
        scriptOp.appendChan("predictions")[0] = 0
        return

    # 1) Gather input
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

    # 3) Predict through the pipeline (handles X scaling + y inverse transform)
    preds = pipeline.predict(vals.reshape(1, -1))[0].astype(np.float32)

    # 4) Optional smoothing over the output vector
    try:
        if int(parent().par.Smooth) == 1:
            stype = str(parent().par.Smoothtype)
            if stype == 'thresholded_median':
                preds = _smooth_thresholded_median(
                    preds,
                    window=int(parent().par.Window),
                    thresh=float(parent().par.Threshold)
                ).astype(np.float32)
            elif stype == 'savgol_filter':
                preds = savgol_filter(
                    preds,
                    window_length=int(parent().par.Window),
                    polyorder=int(parent().par.Polyorder)
                ).astype(np.float32)
            elif stype == 'medfilt':
                win = int(parent().par.Window)
                if win % 2 == 0:
                    win += 1
                preds = medfilt(preds, kernel_size=win).astype(np.float32)
            elif stype == 'wiener':
                preds = wiener(preds.astype(np.float32), mysize=7, noise=1e-8).astype(np.float32)
            elif stype == 'butter':
                b, a = butter(
                    N=int(parent().par.Nbutter),
                    Wn=float(parent().par.Wnbutter),
                    btype='low',
                    analog=False
                )
                preds = filtfilt(b, a, preds.astype(np.float32)).astype(np.float32)
            elif stype == 'firwin':
                kernel = firwin(numtaps=int(parent().par.Numtaps), cutoff=float(parent().par.Cutoff))
                preds = convolve(preds.astype(np.float32), kernel, mode='same').astype(np.float32)
    except Exception as e:
        try:
            debug("[ml] Smoothing error: {}".format(e))
        except Exception:
            print("[ml] Smoothing error: {}".format(e))

    # 5) Output CHOP
    scriptOp.clear()
    scriptOp.numSamples = preds.size
    try:
        scriptOp.rate = parent().par.Samplerate
    except Exception:
        scriptOp.rate = 60
    chan = scriptOp.appendChan('predictions')
    chan.copyNumpyArray(preds)


# ——— Small internal helper for the custom smoothing type ———

def _smooth_thresholded_median(x, window=5, thresh=0.005):
    half = int(window) // 2
    xp = np.pad(x, half, mode='edge')
    out = x.copy()
    for i in range(len(x)):
        w = xp[i : i + int(window)]
        m = float(np.median(w))
        if abs(float(x[i]) - m) < float(thresh):
            out[i] = m
    return out


# ——— Convenience buttons (optional) ———

def train_from_ops():
    X = op('X')
    y = op('y')
    params = op('training_params')
    use_w = bool(parent().par.Useweights.eval()) if hasattr(parent().par, "Useweights") else False
    wtab = op('weight') if use_w else None
    return train(X, y, params, use_w, wtab)


def retrain_from_ops():
    X = op('X')
    y = op('y')
    use_w = bool(parent().par.Useweights.eval()) if hasattr(parent().par, "Useweights") else False
    wtab = op('weight') if use_w else None
    return retrain(X, y, use_w, wtab)

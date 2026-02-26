import ast
import re
from pathlib import Path

import numpy as np

# Shared helpers for TD ML: table/CHOP conversion, paths, custom pages and parameters.


def _table_to_numpy(table):
    """DAT table -> 2D float ndarray. Uses full table (r0=0, c0=0); raises if empty or non-numeric cell."""
    r0 = 0
    c0 = 0
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


def _weights_from_table(table):
    """1D float weights from first column of DAT table. Same convention as classifier/regressor weights."""
    r0 = 0
    c = 0
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


def _sk_param_table_to_dict(paramDAT):
    """Parse param DAT: col0=key, col1=raw value; literal_eval where possible, else string."""
    params = {}
    for row in paramDAT.rows()[0:]:
        key = row[0].val
        raw = row[1].val
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw
        params[key] = val
    return params	


def _folder_or_project(s: str, replacement: str = "project.folder") -> str:
    """Path up to and including last '/' (trailing slash), or replacement if no separator. Backslashes normalized to '/'."""
    if s is None:
        return replacement
    s = str(s).strip()
    if not s:
        return replacement
    s = s.replace("\\", "/")
    if "/" in s:
        head, sep, _ = s.rpartition("/")
        return head + sep
    return replacement

def has_suffix(s: str, ref: str) -> bool:
    """True iff s ends with ref and the character before the match is '.' (real file extension). Case-sensitive."""
    if not s or not ref:
        return False
    if not s.endswith(ref):
        return False
    i = len(s) - len(ref)
    if i == 0:
        return False
    return s[i - 1] == '.'


def _append_chop_to_table(chopOp, tableDAT, include_header=False):
    """Append CHOP to table: one row per channel [chanName, s0, s1, ...]. include_header=True adds row ['chan','s0',...]. Uses Channel.numpyArray()."""
    if chopOp is None or tableDAT is None:
        return
    if include_header:
        header = ['chan'] + [f's{i}' for i in range(chopOp.numSamples)]
        tableDAT.appendRow(header)
    for i in range(chopOp.numChans):
        samples = chopOp[i].numpyArray()
        if include_header:
            row = [chopOp[i].name] + samples.tolist()
        else: 
            row = samples.tolist()
        tableDAT.appendRow(row)

def next_incremented_path(path: Path, width=3, force_suffix=".joblib") -> Path:
    """Next available path by incrementing trailing _NNN (or _001). force_suffix applied; search in path's parent."""
    path = Path(path)
    if force_suffix and path.suffix.lower() != force_suffix:
        path = path.with_suffix(force_suffix)

    stem = path.stem
    m = re.match(r"^(.*?)(?:_(\d+))?$", stem)  # optional trailing _NNN
    base = m.group(1)
    n = int(m.group(2)) + 1 if m.group(2) else 1
    candidate = path.with_name(f"{base}_{n:0{width}d}{path.suffix}")
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{base}_{n:0{width}d}{path.suffix}")
    return candidate




def CopyOpsFromContainer(contentOp, targetOp):
    """Copy all children of contentOp into targetOp; position with nodeX/nodeY."""
    containers = contentOp.ops('*')
    for i, cont in enumerate(containers):
        newOp = targetOp.copy(cont)
        newOp.nodeX = 400 + i * 200
        newOp.nodeY = -400

def _labels_from_table(table, first_row_header=False, first_col_header=False):
    """1D label array from DAT table; first data column (after optional header row/col)."""
    r0 = 1 if first_row_header else 0
    c = 1 if first_col_header else 0
    if table.numCols <= c:
        raise ValueError(
            f"Label column index {c} is out of range for table with {table.numCols} columns."
        )
    return np.array([str(table[r, c].val) for r in range(r0, table.numRows)], dtype=object)


def _labels_or_multilabel_from_table(table, first_row_header=False, first_col_header=False):
    """Labels from DAT: one data column => 1D label array; multiple columns => list of lists (one list of label strings per row, empty cells skipped, for MultiLabelBinarizer)."""
    r0 = 1 if first_row_header else 0
    c0 = 1 if first_col_header else 0
    if table is None or table.numRows <= r0 or table.numCols <= c0:
        raise ValueError(
            f"Table has no data (rows={getattr(table, 'numRows', 0)}, cols={getattr(table, 'numCols', 0)}, "
            f"first_row={r0}, first_col={c0})."
        )
    n_cols = table.numCols - c0
    if n_cols == 1:
        return np.array(
            [str(table[r, c0].val) for r in range(r0, table.numRows)],
            dtype=object,
        )
    out = []
    for r in range(r0, table.numRows):
        row_labels = []
        for c in range(c0, table.numCols):
            val = table[r, c].val
            s = str(val).strip() if val is not None else ""
            if s:
                row_labels.append(s)
        out.append(row_labels)
    return out


def resolve_relative_op_paths_in_menu_source(par, base_op):
    """Rewrite par.menuSource: replace relative op('...') inside tdu.TableMenu with absolute path from base_op. Returns adjusted string or original."""
    if par is None or base_op is None:
        return getattr(par, 'menuSource', None) if par else None
    menu_source = par.menuSource
    if isinstance(menu_source, str) and menu_source.startswith('tdu.TableMenu('):
        match = re.search(r"op\(['\"](.+?)['\"]\)", menu_source)
        if match:
            rel_path = match.group(1)
            dat_op = base_op.op(rel_path)
            if dat_op:
                abs_path = dat_op.path
                new_menu_source = re.sub(
                    r"op\(['\"].+?['\"]\)",
                    "op('{}')".format(abs_path),
                    menu_source
                )
                return new_menu_source
    return menu_source


def get_or_create_custom_page(target_op, page_name):
    """Custom page by name on target_op; create if missing."""
    if target_op is None:
        raise ValueError("target_op cannot be None")
    
    if page_name not in [p.name for p in target_op.customPages]:
        custom_page = target_op.appendCustomPage(page_name)
    else:
        custom_page = target_op.customPages[page_name]
    
    return custom_page


def copy_custom_parameters(source_op, target_page, base_op=None, skip_existing=True, exclude_names=None):
    """Copy source_op custom pars to target_page (type -> append method). base_op optional to resolve Menu/StrMenu op() paths. skip_existing, exclude_names. Returns list of created pars."""
    if source_op is None or target_page is None:
        return []
    if exclude_names is None:
        exclude_names = set()
    elif not isinstance(exclude_names, set):
        exclude_names = set(exclude_names)
    type_map = {
        'Float': target_page.appendFloat,
        'Int': target_page.appendInt,
        'Str': target_page.appendStr,
        'Menu': target_page.appendMenu,
        'StrMenu': target_page.appendStrMenu,
        'Toggle': target_page.appendToggle,
        'Pulse': target_page.appendPulse,
        'RGB': target_page.appendRGB,
        'RGBA': target_page.appendRGBA,
        'Folder': target_page.appendFolder,
    }
    created_pars = []
    par_list = source_op.customPars
    for par in par_list:
        name = par.name
        if name in exclude_names:
            continue
        label = par.label
        par_type = par.style
        append_func = type_map.get(par_type, None)
        if append_func:
            if skip_existing and any(p.name == name for p in target_page.pars):
                continue
            new_par = append_func(name, label=label)
            created_pars.append(new_par)
            if par_type in ('Menu', 'StrMenu'):
                menu_source_path = resolve_relative_op_paths_in_menu_source(par, base_op) if base_op else par.menuSource
                try:
                    new_par.menuSource = menu_source_path
                except Exception:
                    pass
            try:
                new_par.default = par.default
            except Exception:
                pass
    return created_pars


def bind_parameters_to_target(source_pars, target_op_path):
    """Set bindExpr on each par to op(target_op_path).par.{par.name}."""
    if source_pars is None or target_op_path is None:
        return
    
    for par in source_pars:
        try:
            par.bindExpr = "op('{}').par.{}".format(target_op_path, par.name)
        except Exception:
            pass
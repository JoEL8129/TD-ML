import ast
import re
from pathlib import Path

import numpy as np



def _table_to_numpy(table):
    """
    Convert a DAT table to a 2D float numpy array, optionally skipping the first row and/or column.
    """
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
    """Extract 1D float sample weights from a DAT table. Same convention as labels/weights in classifier."""
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
    """
    If `s` contains a path separator, return everything up to and including
    the last '/' (i.e., the containing folder, with trailing slash).
    If `s` has no '/', return `replacement` (default: 'project.folder').

    Also treats backslashes as separators (normalizes to '/').
    """
    if s is None:
        return replacement

    s = str(s).strip()
    if not s:
        return replacement

    # Normalize Windows-style separators to forward slashes
    s = s.replace("\\", "/")

    # If there's at least one slash, keep up to and including the last slash
    if "/" in s:
        head, sep, _ = s.rpartition("/")
        return head + sep  # keeps the trailing slash

    # No slash -> replace with project folder token
    return replacement

def has_suffix(s: str, ref: str) -> bool:
    """
    True iff `s` ends with `ref` and the character immediately before that
    match is a '.' (i.e., a real file suffix like '.ref'). No case folding.
    """
    if not s or not ref:
        return False
    if not s.endswith(ref):
        return False
    i = len(s) - len(ref)
    if i == 0:  # ref would start at position 0 -> no dot before it
        return False
    return s[i - 1] == '.'


def _append_chop_to_table(chopCOMP, tableDAT, include_header=False):
    """
    Append the samples of every channel in a CHOP to a Table DAT.
    - One row per channel: [chanName, s0, s1, s2, ...]
    - Uses Channel.numpyArray() for speed and simplicity.

    Args:
        chopCOMP (CHOP): The CHOP whose channels/samples you want to append.
        tableDAT (DAT):  The Table DAT to append rows to.
        include_header (bool): If True, append a header row first:
                               ['chan', 's0', 's1', ... 'sN'].

    Example:
        append_chop_to_table(op('noise1'), op('table1'), include_header=True)
    """
    if chopCOMP is None or tableDAT is None:
        return

    # optional header showing sample indices
    if include_header:
        header = ['chan'] + [f's{i}' for i in range(chopCOMP.numSamples)]
        tableDAT.appendRow(header)

    # append one row per channel
    for i in range(chopCOMP.numChans):
        # Channel.numpyArray() -> 1D numpy array of samples
        samples = chopCOMP[i].numpyArray()
        if include_header:
            row = [chopCOMP[i].name] + samples.tolist()
        else: 
            row = samples.tolist()
        tableDAT.appendRow(row)

def next_incremented_path(path: Path, width=3, force_suffix=".joblib") -> Path:
    """
    Given a desired path (with or without .joblib), return the next available path by
    incrementing a trailing _NNN if present, otherwise starting at _001.
    Works within the path's parent directory.
    """
    path = Path(path)
    if force_suffix and path.suffix.lower() != force_suffix:
        path = path.with_suffix(force_suffix)

    stem = path.stem
    m = re.match(r"^(.*?)(?:_(\d+))?$", stem)
    base = m.group(1)
    n = int(m.group(2)) + 1 if m.group(2) else 1

    candidate = path.with_name(f"{base}_{n:0{width}d}{path.suffix}")
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{base}_{n:0{width}d}{path.suffix}")
    return candidate




def CopyOpsFromContainer(contentOp,targetOp):
    containers = contentOp.ops('*')
    for i, cont in enumerate(containers):
        newOp = targetOp.copy(cont)
        newOp.nodeX = 400 + i * 200
        newOp.nodeY = -400

def _labels_from_table(table, first_row_header=False, first_col_header=False):
    """Extract 1D labels from a DAT table. Labels live in the first data column."""
    r0 = 1 if first_row_header else 0
    c = 1 if first_col_header else 0
    if table.numCols <= c:
        raise ValueError(
            f"Label column index {c} is out of range for table with {table.numCols} columns."
        )
    return np.array([str(table[r, c].val) for r in range(r0, table.numRows)], dtype=object)


def resolve_relative_op_paths_in_menu_source(par, base_op):
    """
    Convert relative op() paths in a parameter's menuSource to absolute paths.
    
    If the menuSource contains a tdu.TableMenu() with a relative op() path,
    this function resolves it relative to base_op and returns the updated menuSource
    with an absolute path.
    
    Args:
        par: The parameter whose menuSource should be adjusted.
        base_op: The operator to use as the base for resolving relative paths.
    
    Returns:
        str: The adjusted menuSource string, or the original if no adjustment was needed.
    
    Example:
        # If par.menuSource is "tdu.TableMenu(op('./dat1'))"
        # and base_op is op('container1')
        # Returns: "tdu.TableMenu(op('/project1/container1/dat1'))"
    """
    if par is None or base_op is None:
        return getattr(par, 'menuSource', None) if par else None
    
    menu_source = par.menuSource
    if isinstance(menu_source, str) and menu_source.startswith('tdu.TableMenu('):
        # Try to extract the inner op() path and rewrite as absolute
        match = re.search(r"op\(['\"](.+?)['\"]\)", menu_source)
        if match:
            rel_path = match.group(1)
            dat_op = base_op.op(rel_path)
            if dat_op:
                abs_path = dat_op.path
                # Replace the op('./...') call with the absolute path
                new_menu_source = re.sub(
                    r"op\(['\"].+?['\"]\)",
                    "op('{}')".format(abs_path),
                    menu_source
                )
                return new_menu_source
    return menu_source  # fallback, return as-is


def get_or_create_custom_page(target_op, page_name):
    """
    Get an existing custom page by name, or create it if it doesn't exist.
    
    Args:
        target_op: The operator to get/create the custom page on.
        page_name: The name of the custom page.
    
    Returns:
        The custom page object.
    
    Example:
        page = get_or_create_custom_page(op('container1'), 'Presets')
    """
    if target_op is None:
        raise ValueError("target_op cannot be None")
    
    if page_name not in [p.name for p in target_op.customPages]:
        custom_page = target_op.appendCustomPage(page_name)
    else:
        custom_page = target_op.customPages[page_name]
    
    return custom_page


def copy_custom_parameters(source_op, target_page, base_op=None, skip_existing=True, exclude_names=None):
    """
    Copy custom parameters from a source operator to a target custom page.
    
    This function maps parameter types to their corresponding append methods
    and copies all custom parameters from source_op to target_page, preserving
    labels, defaults, and menu sources (with optional path adjustment).
    
    Args:
        source_op: The operator whose custom parameters should be copied.
        target_page: The custom page to add the parameters to.
        base_op: Optional operator to use for adjusting menu source paths.
                 If provided, relative op() paths in Menu parameters will be
                 converted to absolute paths.
        skip_existing: If True, skip parameters that already exist on the target page.
        exclude_names: Optional list/set of parameter names to exclude from copying.
    
    Returns:
        list: A list of the newly created parameter objects.
    
    Example:
        page = get_or_create_custom_page(op('container1'), 'Presets')
        new_pars = copy_custom_parameters(op('source1'), page, base_op=op('container1'))
        # Exclude specific parameters:
        new_pars = copy_custom_parameters(op('source1'), page, exclude_names=['Targetop', 'Otherpar'])
    """
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
        # Add more as needed
    }
    
    created_pars = []
    par_list = source_op.customPars
    
    for par in par_list:
        name = par.name
        
        # Skip excluded parameters
        if name in exclude_names:
            continue
        
        label = par.label
        par_type = par.style
        append_func = type_map.get(par_type, None)
        
        if append_func:
            # Check if parameter already exists
            if skip_existing and any(p.name == name for p in target_page.pars):
                continue
            
            new_par = append_func(name, label=label)
            created_pars.append(new_par)
            
            # Handle Menu and StrMenu parameters with menuSource
            if par_type in ('Menu', 'StrMenu'):
                menu_source_path = resolve_relative_op_paths_in_menu_source(par, base_op) if base_op else par.menuSource
                try:
                    new_par.menuSource = menu_source_path
                except Exception:
                    pass
            
            # Copy default value
            try:
                new_par.default = par.default
            except Exception:
                pass
    
    return created_pars


def bind_parameters_to_target(source_pars, target_op_path):
    """
    Set bindExpr on source parameters to bind them to corresponding parameters
    on a target operator.
    
    Args:
        source_pars: Iterable of parameters to bind (e.g., op('source1').customPars).
        target_op_path: The path of the target operator (e.g., op('target1').path).
    
    Example:
        bind_parameters_to_target(op('source1').customPars, op('target1').path)
    """
    if source_pars is None or target_op_path is None:
        return
    
    for par in source_pars:
        try:
            par.bindExpr = "op('{}').par.{}".format(target_op_path, par.name)
        except Exception:
            pass
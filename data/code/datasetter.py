def onPulse(par):
    """
    Route each input (X, Y, W) from either a CHOP or a DAT depending on:
      - parent().par.Input1switchtype  -> X
      - parent().par.Input2switchtype  -> Y
      - parent().par.Input3switchtype  -> W
    """

    # Map sources + destinations (adjust names if your nodes differ)
    inputs = {
        'X': {
            'chop': op('X'),
            'dat':  op('X_dat'),
            'dest': op('table_X'),
            'switch': parent().par.Input1
        },
        'Y': {
            'chop': op('y'),
            'dat':  op('y_dat'),
            'dest': op('table_y'),
            'switch': parent().par.Input2
        },
        'W': {
            'chop': op('weight'),
            'dat':  op('weight_dat'),
            'dest': op('table_weight'),
            'switch': parent().par.Input3
        },
    }

    # Always log X and Y
    _appendInput(inputs['X'], key='X')
    _appendInput(inputs['Y'], key='Y')

    # Optionally log W with CHOP-specific rules if coming from CHOP
    if parent().par.Useweights:
        _appendInput(inputs['W'], key='W', is_weight=True)

    return


# ──────────────────────────────────────────────────────────────────────────────
# Routing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _appendInput(src, key='X', is_weight=False):
    """
    Append one input to its destination table based on its switch mode.
    - src['switch'] should be a menu param with values like 'CHOP' / 'DAT'.
    - When 'CHOP': use _appendChopToTable (weights use _appendWeightToTable).
    - When 'DAT' : use _appendDatToTable (generic, header in row 0).
    """
    mode = _switch_mode(src['switch'])

    if mode == 'dat':
        if src['dat'] is None:
            print(f"⚠️ {key}: DAT operator not found.")
            return
        _appendDatToTable(src['dat'], src['dest'])
    else:
        if src['chop'] is None:
            print(f"⚠️ {key}: CHOP operator not found.")
            return
        if is_weight:
            _appendWeightToTable(src['chop'], src['dest'])
        else:
            _appendChopToTable(src['chop'], src['dest'])


def _switch_mode(par):
    """
    Robustly detect CHOP vs DAT from a menu/string parameter.
    Treat anything containing 'dat' or 'table' (case-insensitive) as DAT; else CHOP.
    """
    try:
        v = str(par.eval())
    except Exception:
        v = str(par)
    v = v.lower()
    if 'dat' in v or 'table' in v:
        return 'dat'
    return 'chop'


# ──────────────────────────────────────────────────────────────────────────────
# CHOP/DAT → table helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensureHeaderFromChop(table, chop):
    chan_names = [c.name for c in chop.chans()]
    _ensureHeaderFromNames(table, chan_names)

def _ensureHeaderFromSamples(table, chop):
    """
    Ensure header is ['', s1, s2, ..., sN] where N = chop.numSamples.
    """
    names = [f's{i+1}' for i in range(chop.numSamples)]
    _ensureHeaderFromNames(table, names)

def _ensureHeaderFromNames(table, names):
    """
    Ensure header is ['', <name1>, <name2>, ...].
    """
    desired = [''] + list(names)

    if table.numRows == 0:
        table.appendRow(desired)
        return

    # read existing header row
    existing = []
    for col in range(table.numCols):
        cell = table[0, col]
        existing.append(cell.val if cell is not None else '')

    if len(existing) != len(desired) or any(a != b for a, b in zip(existing, desired)):
        table.clear()
        table.appendRow(desired)


def _nextSampleLabel(table):
    """
    Returns 'sample n' where n = data-row index (header excluded) + 1.
    Header is row 0 → first data row is 'sample 1'.
    """
    n = 1 if table.numRows == 0 else table.numRows  # because header consumes row 0
    return f"sample {n}"


def _appendChopToTable_old(chop, table):
    """
    Columns are channel names. First column is the row label 'sample n'.
    • Multi-sample → one row per sample, all channels as columns.
    • Single-sample → one row snapshot, all channels as columns.
    """
    _ensureHeaderFromChop(table, chop)

    if chop.numSamples > 1:
        for i in range(chop.numSamples):
            row = [_nextSampleLabel(table)] + [chan.eval(i) for chan in chop.chans()]
            table.appendRow(row)
    else:
        row = [_nextSampleLabel(table)] + [c.eval() for c in chop.chans()]
        table.appendRow(row)

def _appendChopToTable(chop, table):
    """
    CHOP → table with the 4 requested cases:

    1) 1 channel, 1 sample
       - add 1 data sample
       - feature size = 1 (channel)
       - header from channel names

    2) 1 channel, many samples
       - add 1 data sample
       - feature size = #samples
       - header from sample indices (s1..sN)

    3) many channels, 1 sample
       - add 1 data sample
       - feature size = #channels
       - header from channel names

    4) many channels, many samples
       - add many data samples (one per channel)
       - feature size = #samples
       - header from sample indices (s1..sN)
    """
    nC, nS = chop.numChans, chop.numSamples

    # Case 1: 1C, 1S  → one row, features = channels (size 1)
    if nC == 1 and nS == 1:
        _ensureHeaderFromChop(table, chop)  # header ['', <chanName>]
        row = [_nextSampleLabel(table), chop[0].eval()]
        table.appendRow(row)
        return

    # Case 2: 1C, many S → one row, features = samples
    if nC == 1 and nS > 1:
        _ensureHeaderFromSamples(table, chop)  # header ['', s1..sN]
        row = [_nextSampleLabel(table)] + [chop[0].eval(i) for i in range(nS)]
        table.appendRow(row)
        return

    # Case 3: many C, 1S → one row, features = channels
    if nC > 1 and nS == 1:
        _ensureHeaderFromChop(table, chop)  # header ['', chan1..chanC]
        row = [_nextSampleLabel(table)] + [chan.eval() for chan in chop.chans()]
        table.appendRow(row)
        return

    # Case 4: many C, many S → many rows (one per channel), features = samples
    if nC > 1 and nS > 1:
        _ensureHeaderFromSamples(table, chop)  # header ['', s1..sN]
        for chan in chop.chans():
            row = [_nextSampleLabel(table)] + [chan.eval(i) for i in range(nS)]
            table.appendRow(row)
        return

def _appendDatToTable(dat, table, header_row=0):
    """
    Generic DAT → table append.
    Assumes the DAT has a header row at `header_row` (default 0) containing column names.
    Appends each subsequent row as one 'sample n'.
    """
    if dat.numRows == 0:
        print("⚠️ DAT is empty; nothing to append.")
        return

    # Build header from the DAT's header row
    if parent().par.Input2header==True:
        names = []
        for c in range(dat.numCols):
            cell = dat[header_row, c]
            names.append(cell.val if cell is not None else f'col{c+1}')

        _ensureHeaderFromNames(table, names)

    # Append all data rows after the header
        for r in range(header_row + 1, dat.numRows):
            
            values = []
            for c in range(dat.numCols):
                cell = dat[r, c]
                values.append(cell.val if cell is not None else '')
            table.appendRow([_nextSampleLabel(table)] + values)

    else:
        if table.numRows == 0:
            table.appendRow("")
            table.appendCol("")
            table[0,1].val = "Label"
        
        for r in range(0, dat.numRows):
            values = []
            for c in range(dat.numCols):
                cell = dat[r, c]
                values.append(cell.val if cell is not None else '')
            table.appendRow([_nextSampleLabel(table)] + values)

def _appendWeightToTable(chop, table):
    """
    Special append for W, but keep the same table shape:
    • ERROR if both nC>1 and nS>1 (invalid).
    • Multiple channels (single sample) → one row, columns are channel names.
    • Multiple samples (single channel) → one row per sample, single param column.
    • Single channel & sample → one row.
    Always includes blank top-left header and 'sample n' in the first column.
    """
    nC, nS = chop.numChans, chop.numSamples

    if nC > 1 and nS > 1:
        print("⚠️ Weight CHOP has both multiple channels and multiple samples — cannot append.")
        return

    _ensureHeaderFromChop(table, chop)

    if nC > 1:
        # single row, many params
        row = [_nextSampleLabel(table)] + [chan.eval() for chan in chop.chans()]
        table.appendRow(row)

    elif nS > 1:
        # many rows, single param
        for i in range(nS):
            row = [_nextSampleLabel(table), chop[0].eval(i)]
            table.appendRow(row)

    else:
        # single row, single param
        row = [_nextSampleLabel(table), chop[0].eval()]
        table.appendRow(row)

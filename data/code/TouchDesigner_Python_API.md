# TouchDesigner Python API Documentation

## Introduction

TouchDesigner is a visual programming environment that supports Python as its primary scripting language. The Python API allows for extensive control and customization of TouchDesigner projects through Python scripting. This document provides a comprehensive reference for the TouchDesigner Python API based on official documentation and common usage patterns.

**Official Documentation**: [https://docs.derivative.ca/Python](https://docs.derivative.ca/Python)

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Operator Access](#operator-access)
3. [OP Class](#op-class)
4. [CHOP Operators](#chop-operators)
5. [DAT Operators](#dat-operators)
6. [Parameters](#parameters)
7. [Extensions](#extensions)
8. [Common Patterns](#common-patterns)
9. [Examples](#examples)

---

## Core Concepts

### Virtual Environments (venv)

A virtual environment is an isolated directory containing a specific Python version and its own set of installed packages.

- **Isolation**: Packages installed in a `venv` are only available when that environment is active
- **Project-Specific**: Typically create one `venv` per project
- **Python Version**: TouchDesigner currently uses Python 3.11

### Python Snippets

You can experiment with Python code within a TouchDesigner Textport, which can be opened by typing `ALT`-`T`.

---

## Operator Access

### The `op()` Function

The `op()` function is used to get a reference to any operator in the TouchDesigner network.

```python
# Get operator by path
my_op = op('/project1/geo1')

# Get operator by relative path (from current operator)
child_op = op('child1')

# Get operator by name (searches from current operator)
sibling_op = op('sibling1')
```

### The `ownerComp` Object

In extension classes, `ownerComp` refers to the component to which the extension is attached.

```python
class MyExtension:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        # Access child operators
        self.child_op = self.ownerComp.op('child1')
```

### The `me` Object

In parameter expressions and callbacks, `me` refers to the current operator.

```python
# In a parameter expression
me.par.value.eval()

# In a callback
def onPulse(par):
    print(me.path)
```

---

## OP Class

The `OP` class is the base class for all operators in TouchDesigner.

### Properties

- `path` - The full path to the operator (e.g., `/project1/geo1`)
- `name` - The name of the operator
- `parent` - The parent operator
- `children` - List of child operators
- `customPages` - List of custom parameter pages

### Methods

#### `op(path)`
Get a child operator by path.

```python
child = my_op.op('child1')
```

#### `appendCustomPage(name)`
Create a new custom parameter page.

```python
page = my_op.appendCustomPage('MyPage')
```

#### `destroy()`
Destroy the operator.

```python
my_op.destroy()
```

---

## CHOP Operators

CHOP (Channel Operator) operators handle channel data (time-series data).

### Properties

- `numChans` - Number of channels
- `numSamples` - Number of samples per channel
- `sampleRate` - Sample rate in Hz

### Methods

#### `chans()`
Get all channels in the CHOP.

```python
channels = chop_op.chans()
for ch in channels:
    print(ch.name)
```

#### `chan(index)` or `[index]`
Get a specific channel by index.

```python
channel = chop_op[0]  # First channel
channel = chop_op.chan(0)  # Same as above
```

### Channel Object

#### Properties

- `name` - Channel name
- `vals` - List of all sample values
- `length` - Number of samples

#### Methods

#### `eval(sample_index)`
Get the value at a specific sample index.

```python
value = channel.eval(0)  # First sample
```

#### `numpyArray()`
Get channel data as a NumPy array.

```python
import numpy as np
data = channel.numpyArray()
```

### Common CHOP Operations

```python
# Get channel names
chan_names = [ch.name for ch in chop_op.chans()]

# Get all channel data as arrays
chan_arrays = [chop_op[i].numpyArray() for i in range(chop_op.numChans)]

# Access specific channel value
value = chop_op[0].eval(5)  # Channel 0, sample 5
```

---

## DAT Operators

DAT (Data Operator) operators handle text and table data.

### Properties

- `numRows` - Number of rows
- `numCols` - Number of columns

### Methods

#### `row(index)`
Get a row by index (returns a Row object).

```python
row = dat_op.row(0)  # First row
```

#### `rows()`
Get all rows.

```python
for row in dat_op.rows():
    print(row)
```

#### `col(index)`
Get a column by index.

```python
col = dat_op.col(0)  # First column
```

#### `appendRow(values)`
Add a new row to the table.

```python
dat_op.appendRow(['value1', 'value2', 'value3'])
```

#### `appendCol(values)`
Add a new column to the table.

```python
dat_op.appendCol(['header', 'data1', 'data2'])
```

#### `deleteRow(index)`
Delete a row by index.

```python
dat_op.deleteRow(0)  # Delete first row
```

#### `deleteCol(index)`
Delete a column by index.

```python
dat_op.deleteCol(0)  # Delete first column
```

#### `clear()`
Clear all data from the table.

```python
dat_op.clear()
```

### Cell Access

Access individual cells using indexing.

```python
# Access cell at row 0, column 0
cell = dat_op[0, 0]

# Get cell value
value = cell.val if cell is not None else ''

# Set cell value
dat_op[0, 0].val = 'new_value'
```

### Row Object

#### Properties

- `vals` - List of cell values in the row

#### Methods

#### `[index]`
Access a cell in the row by column index.

```python
row = dat_op.row(0)
value = row[0]  # First cell in row
```

---

## Parameters

Parameters control operator behavior and can be accessed and modified via Python.

### Accessing Parameters

```python
# Get parameter by name
par = my_op.par.value

# Evaluate parameter value
value = my_op.par.value.eval()

# Set parameter value
my_op.par.value = 0.5

# Pulse a pulse parameter
my_op.par.reset.pulse()
```

### Parameter Properties

- `name` - Parameter name
- `val` - Current value
- `default` - Default value
- `min` - Minimum value
- `max` - Maximum value
- `menuIndex` - Current menu selection index (for menu parameters)
- `menuNames` - List of menu option names
- `menuLabels` - List of menu option labels

### Parameter Methods

#### `eval()`
Evaluate the parameter's current value.

```python
value = my_op.par.float1.eval()
```

#### `pulse()`
Trigger a pulse parameter.

```python
my_op.par.reset.pulse()
```

### Parameter Types

Common parameter types include:

- `float` - Floating point number
- `int` - Integer
- `str` - String
- `toggle` - Boolean (0 or 1)
- `menu` - Menu selection
- `pulse` - Button/pulse
- `rgb` - RGB color (tuple)
- `folder` - Folder path

### Creating Custom Parameters

Custom parameters can be created using the `ParTemplate` class or directly via the API.

```python
from ParTemplate import ParTemplate

page = my_op.appendCustomPage('MyPage')
par_template = ParTemplate(
    'MyFloat',
    par_type='float',
    label='My Float',
    default=0.5,
    min=0.0,
    max=1.0,
    callback=True
)
par_template.createPar(page)
```

---

## Extensions

Extensions are Python classes that enhance TouchDesigner components with custom functionality.

### Extension Structure

```python
class MyExtension:
    def __init__(self, ownerComp):
        # The component to which this extension is attached
        self.ownerComp = ownerComp
        
        # Access child operators
        self.child_op = self.ownerComp.op('child1')
        
        # Access parameters
        self.value = self.ownerComp.par.value.eval()
```

### Promoted Attributes

Attributes with capitalized names can be accessed externally if the extension is promoted.

```python
class MyExtension:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.PromotedFunction = self.my_function  # Can be accessed externally
    
    def my_function(self):
        print("Called from outside")
```

### Accessing Extensions

```python
# From within the component
ext.MyExtension.SomeMethod()

# From outside (if promoted)
op('myComp').PromotedFunction()
```

### Storage Manager

Use `StorageManager` to create persistent properties that survive saves and re-initialization.

```python
from TDStoreTools import StorageManager

storedItems = [
    {
        'name': 'MyProperty',
        'default': None,
        'readOnly': False,
        'property': True,
        'dependable': True
    }
]

self.stored = StorageManager(self, ownerComp, storedItems)
# Access via: self.stored['MyProperty']
```

#### Clearing and Resetting Storage

There are several ways to clear or reset storage in TouchDesigner:

**1. Clear Specific Stored Items using `unstore()`:**

Remove individual items from storage by their key name:

```python
# Remove a specific stored item
self.ownerComp.unstore('MyProperty')

# Or from any operator
op('myOperator').unstore('keyName')
```

**2. Clear All Storage using `storage.clear()`:**

Remove all items from an operator's storage dictionary:

```python
# Clear all storage for the owner component
self.ownerComp.storage.clear()

# Or from any operator
op('myOperator').storage.clear()
```

**3. Clear Storage via Component Editor:**

You can also clear stored items using the Component Editor's Storage section:

1. Select the component
2. Open the Component Editor (press `Alt+E`)
3. Navigate to the "Storage" section
4. Use the interface to clear individual items or all stored data

**Best Practices:**

- **Key Naming**: Avoid using spaces in storage keys, as this can lead to issues when unstoring items. Use simple, alphanumeric key names (e.g., `'MyProperty'` instead of `'My Property'`).

- **Data Types**: Only objects that can be serialized by Python can be stored. This includes most built-in Python objects (lists, dicts, strings, numbers) but excludes many TouchDesigner objects, notably Operators. If you need to store an Operator reference, store its `path` as a string instead:

```python
# Store operator path instead of operator object
self.stored['MyOperator'] = op('child1').path

# Later, retrieve the operator
my_op = op(self.stored['MyOperator'])
```

- **Resetting to Defaults**: To reset a stored item to its default value, you can either use `unstore()` and let it reinitialize, or explicitly set it:

```python
# Reset to default by unstoring (will use default on next access)
self.ownerComp.unstore('MyProperty')

# Or explicitly set to default
self.stored['MyProperty'] = None  # or whatever the default value is
```

### TDFunctions

The `TDFunctions` module provides utility functions for extensions.

```python
import TDFunctions as TDF

# Create a property
TDF.createProperty(
    self,
    'MyProperty',
    value=[],
    dependable=True,
    readOnly=False
)
```

---

## Common Patterns

### Getting Operator References

```python
# In an extension
self.input_chop = self.ownerComp.op('input_chop')
self.output_table = self.ownerComp.op('output_table')
```

### Reading CHOP Data

```python
# Get all channel names
chan_names = [ch.name for ch in chop_op.chans()]

# Get channel data as NumPy arrays
chan_arrays = [chop_op[i].numpyArray() for i in range(chop_op.numChans)]

# Process each sample
for sample_idx in range(chop_op.numSamples):
    for chan_idx in range(chop_op.numChans):
        value = chop_op[chan_idx].eval(sample_idx)
        # Process value
```

### Writing to DAT Tables

```python
# Clear table
table_op.clear()

# Add header row
table_op.appendRow(['Column1', 'Column2', 'Column3'])

# Add data rows
for i in range(10):
    table_op.appendRow([i, i*2, i*3])

# Access and modify cells
table_op[0, 0].val = 'NewValue'
```

### Parameter Callbacks

```python
def onPulse(par):
    # Called when a pulse parameter is triggered
    print(f"Pulse triggered: {par.name}")

def onValueChange(par, val):
    # Called when a parameter value changes
    print(f"Parameter {par.name} changed to {val}")
```

### Custom Parameter Pages

```python
def GetPage(self, pageName, create_if_missing=True):
    """Get or create a custom parameter page."""
    lname = pageName.lower()
    for p in self.ownerComp.customPages:
        label = getattr(p, 'label', p.name)
        if p.name.lower() == lname or str(label).lower() == lname:
            return p
    
    if create_if_missing:
        return self.ownerComp.appendCustomPage(pageName)
    
    return None
```

### File Operations

```python
# Set file path
file_op.par.file = '/path/to/file.txt'

# Trigger file read
file_op.par.syncfile.pulse()

# Trigger file write
file_op.par.write.pulse()
```

---

## Examples

### Example 1: Extension with CHOP Processing

```python
from TDStoreTools import StorageManager
import TDFunctions as TDF

class CHOPProcessor:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.input_chop = self.ownerComp.op('input1')
        self.output_table = self.ownerComp.op('table1')
    
    def ProcessChop(self):
        # Clear output
        self.output_table.clear()
        
        # Add header row with channel names
        chan_names = [ch.name for ch in self.input_chop.chans()]
        self.output_table.appendRow(chan_names)
        
        # Add data rows (one per sample)
        for sample_idx in range(self.input_chop.numSamples):
            row_data = []
            for chan_idx in range(self.input_chop.numChans):
                value = self.input_chop[chan_idx].eval(sample_idx)
                row_data.append(value)
            self.output_table.appendRow(row_data)
```

### Example 2: Parameter Callback

```python
def onPulse(par):
    if par.name == 'Process':
        # Get the extension
        ext = op('myComp').ext.MyExtension
        
        # Call a method
        ext.ProcessChop()
        
        print("Processing complete")
```

### Example 3: Reading Table Data

```python
def ReadTable(table_op):
    """Read all data from a table DAT."""
    data = []
    
    # Read header row (if exists)
    if table_op.numRows > 0:
        header = table_op.row(0)
        headers = [cell.val for cell in header]
        data.append(headers)
    
    # Read data rows
    for row_idx in range(1, table_op.numRows):
        row = table_op.row(row_idx)
        row_data = [cell.val for cell in row]
        data.append(row_data)
    
    return data
```

### Example 4: Creating Custom Parameters

```python
from ParTemplate import ParTemplate

def SetupParameters(self):
    page = self.GetPage('Controls')
    
    pars = [
        ParTemplate(
            'Process',
            par_type='pulse',
            label='Process Data',
            callback=True
        ),
        ParTemplate(
            'Threshold',
            par_type='float',
            label='Threshold',
            default=0.5,
            min=0.0,
            max=1.0,
            callback=True
        ),
        ParTemplate(
            'Mode',
            par_type='menu',
            label='Processing Mode',
            menu_names=['mode1', 'mode2', 'mode3'],
            menu_labels=['Mode 1', 'Mode 2', 'Mode 3'],
            default='mode1',
            callback=True
        )
    ]
    
    for p in pars:
        p.createPar(page)
```

### Example 5: CHOP to Table Conversion

```python
def Chop_to_table(self, chopOp, tableOp, channels='cols', samples='rows'):
    """Convert CHOP data to table format."""
    num_chans = chopOp.numChans
    num_samples = chopOp.numSamples
    chan_names = [ch.name for ch in chopOp.chans()]
    chan_arrays = [chopOp[i].numpyArray() for i in range(num_chans)]
    
    if channels == 'cols' and samples == 'rows':
        # Columns are channels, rows are samples
        # Add header row with channel names
        tableOp.appendRow(chan_names)
        
        # Add data rows (one per sample)
        for si in range(num_samples):
            row_data = [arr[si] for arr in chan_arrays]
            tableOp.appendRow(row_data)
    elif channels == 'rows' and samples == 'cols':
        # Rows are channels, columns are samples
        # Add data rows (one per channel)
        for ci in range(num_chans):
            tableOp.appendRow(chan_arrays[ci].tolist())
```

---

## Additional Resources

- **Official Documentation**: [https://docs.derivative.ca/Python](https://docs.derivative.ca/Python)
- **Python API Reference**: [https://docs.derivative.ca/Python_API](https://docs.derivative.ca/Python_API)
- **Introduction to Python**: [https://docs.derivative.ca/Introduction_to_Python](https://docs.derivative.ca/Introduction_to_Python)
- **TouchDesigner Wiki**: [https://docs.derivative.ca/](https://docs.derivative.ca/)

---

## Notes

- TouchDesigner uses Python 3.11
- The `op()` function is available globally in TouchDesigner Python scripts
- Extensions are automatically instantiated when a component is created
- Parameters can be accessed via `operator.par.parameterName`
- Use `eval()` to get parameter values in expressions
- CHOP channels are 0-indexed
- DAT rows and columns are 0-indexed
- Always check if operators exist before accessing them (use try/except or check for None)

---

*This documentation is based on the official TouchDesigner Python API and common usage patterns. For the most up-to-date information, please refer to the official documentation.*


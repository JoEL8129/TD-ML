# TouchDesigner Python Extensions - Complete Lookup Sheet

A comprehensive reference guide for creating and using Python extensions in TouchDesigner components (COMPs).

**Official Documentation**: [COMP Extensions Page](https://docs.derivative.ca/COMP_Extensions_Page)

---

## Table of Contents

1. [Overview](#overview)
2. [Creating Extensions](#creating-extensions)
3. [Extension Structure](#extension-structure)
4. [Accessing Extensions](#accessing-extensions)
5. [Promotion System](#promotion-system)
6. [Properties and Attributes](#properties-and-attributes)
7. [StorageManager - Persistent Storage](#storagemanager---persistent-storage)
8. [TDFunctions Utilities](#tdfunctions-utilities)
9. [Lifecycle Methods](#lifecycle-methods)
10. [Callbacks and Event Handlers](#callbacks-and-event-handlers)
11. [Extension Parameters](#extension-parameters)
12. [Best Practices](#best-practices)
13. [Common Patterns](#common-patterns)
14. [Examples](#examples)
15. [Troubleshooting](#troubleshooting)

---

## Overview

Python extensions allow you to add custom functionality to any COMP (Component Operator) in TouchDesigner. Extensions are Python classes that:

- Add custom methods and properties to components
- Provide a clean interface for component interaction
- Support both internal and external access patterns
- Persist data across saves and re-initialization
- Integrate with TouchDesigner's parameter and callback system

### Key Concepts

- **Extension**: A Python class that enhances a COMP
- **ownerComp**: The COMP to which the extension is attached
- **Promotion**: Making extension members accessible directly from the COMP
- **StorageManager**: System for persistent data storage
- **TDFunctions**: Utility module for extension helpers

---

## Creating Extensions

### Method 1: Component Editor (Recommended)

1. **Right-click** on the desired COMP (e.g., Base COMP)
2. Select **"Customize Component..."** to open the Component Editor
3. Navigate to the **"Extension"** section
4. Enter a name for your extension (e.g., `SampleExt`)
5. Click **"Add"**
6. A Text DAT with the extension name is created inside the COMP
7. The COMP's Extensions parameter is automatically configured

### Method 2: Manual Setup

1. Create a **Text DAT** inside your COMP
2. Name it following the convention: `YourExtensionNameExt` (capitalized, ending with `Ext`)
3. Define your Python class in the Text DAT
4. In the COMP's parameters, go to the **"Extensions"** page
5. Add a new extension entry:
   - **Object**: Reference to your class (e.g., `SampleExt`)
   - **Name**: Optional alias for the extension
   - **Promote**: Whether to promote the extension

### Naming Conventions

- **Extension Class**: Capitalized name ending with `Ext` (e.g., `SampleExt`, `MyCustomExt`)
- **Promoted Members**: Capitalized names (e.g., `MyProperty`, `MyMethod`)
- **Non-Promoted Members**: Lowercase names (e.g., `myProperty`, `myMethod`)

---

## Extension Structure

### Basic Template

```python
"""
Extension classes enhance TouchDesigner components with Python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF

class SampleExt:
    """
    SampleExt description
    """
    def __init__(self, ownerComp):
        # The component to which this extension is attached
        self.ownerComp = ownerComp
        
        # Properties using TDF.createProperty
        TDF.createProperty(self, 'MyProperty', value=0, 
                          dependable=True, readOnly=False)
        
        # Regular attributes
        self.a = 0  # Non-promoted attribute
        self.B = 1  # Promoted attribute
        
        # Stored items (persistent across saves and re-initialization)
        storedItems = [
            {
                'name': 'StoredProperty',
                'default': None,
                'readOnly': False,
                'property': True,
                'dependable': True
            },
        ]
        # Uncomment to enable storage
        # self.stored = StorageManager(self, ownerComp, storedItems)
    
    # Non-promoted method
    def myFunction(self, v):
        debug(v)
    
    # Promoted method
    def PromotedFunction(self, v):
        debug(v)
    
    # Lifecycle methods (optional)
    def onInitTD(self):
        """
        Called after the extension is fully initialized and attached to the 
        component. Use this instead of __init__ for tasks that require other
        components' extensions to be available, or that use promoted members.
        """
        debug("onInitTD called")
    
    def onDestroyTD(self):
        """
        Called when the extension or component is being deleted. Use this
        instead of __del__ for cleanup tasks.
        """
        debug("onDestroyTD called")
```

### Required Elements

- **`__init__(self, ownerComp)`**: Constructor that receives the owner COMP
- **`self.ownerComp`**: Reference to the component (always store this)

### Optional Elements

- **Properties**: Created with `TDF.createProperty()`
- **Stored Items**: Defined in `storedItems` list and managed by `StorageManager`
- **Lifecycle Methods**: `onInitTD()`, `onDestroyTD()`
- **Callbacks**: `onCook()`, `onPreCook()`, `onParmChange()`, etc.

---

## Accessing Extensions

### From Within the Component

Access extensions via the `ext` object:

```python
# Access non-promoted members
op('SampleComp').ext.SampleExt.a
op('SampleComp').ext.SampleExt.myFunction('Hello')

# Access promoted members (if extension is promoted)
op('SampleComp').ext.SampleExt.B
op('SampleComp').ext.SampleExt.PromotedFunction('Hello')
```

### From Outside the Component

**Non-Promoted Extension:**
```python
# Must use .ext
op('SampleComp').ext.SampleExt.myFunction('Hello')
```

**Promoted Extension:**
```python
# Can access promoted members directly
op('SampleComp').B
op('SampleComp').PromotedFunction('Hello')

# Still can access via .ext
op('SampleComp').ext.SampleExt.PromotedFunction('Hello')
```

### From Within the Extension Itself

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
    
    def myMethod(self):
        # Access owner component
        child_op = self.ownerComp.op('child1')
        
        # Access parameters
        value = self.ownerComp.par.value.eval()
        
        # Access other extensions (if they exist)
        other_ext = self.ownerComp.ext.OtherExt
```

### Using `me` in Parameter Expressions

```python
# In a parameter expression or callback
me.ext.SampleExt.myFunction('Hello')

# Or if promoted
me.PromotedFunction('Hello')
```

---

## Promotion System

### What is Promotion?

Promotion determines whether extension members are accessible directly from the COMP or only through the `.ext` member.

### Promotion Rules

1. **Naming Convention**: 
   - **Capitalized names** = Promoted (e.g., `MyProperty`, `MyMethod`)
   - **Lowercase names** = Non-promoted (e.g., `myProperty`, `myMethod`)

2. **Extension-Level Promotion**:
   - Set in COMP's Extensions parameter page
   - Controls whether the entire extension is promoted
   - When promoted, capitalized members become directly accessible

### Examples

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.a = 0          # Non-promoted
        self.B = 1          # Promoted (capitalized)
    
    def myFunction(self):   # Non-promoted
        pass
    
    def PromotedFunction(self):  # Promoted (capitalized)
        pass
```

**Access Patterns:**

```python
# Non-promoted (always via .ext)
op('comp').ext.SampleExt.a
op('comp').ext.SampleExt.myFunction()

# Promoted (if extension is promoted)
op('comp').B              # Direct access
op('comp').PromotedFunction()  # Direct access
```

### When to Promote

**Promote when:**
- Members are part of the component's public API
- External operators need to access the functionality
- You want a cleaner interface

**Don't promote when:**
- Members are internal implementation details
- You want to encapsulate functionality
- Members might conflict with COMP's built-in attributes

---

## Properties and Attributes

### Regular Attributes

Simple Python attributes:

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.myList = []
        self.myDict = {}
        self.counter = 0
```

### Properties with TDF.createProperty

Create properties with special features:

```python
import TDFunctions as TDF

class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        
        # Create a property
        TDF.createProperty(
            self,              # Extension instance
            'MyProperty',      # Property name
            value=0,          # Initial value
            dependable=True,  # Triggers updates when changed
            readOnly=False    # Can be modified
        )
```

**TDF.createProperty Parameters:**

- **`self`**: The extension instance
- **`name`**: Property name (string)
- **`value`**: Initial/default value
- **`dependable`**: If `True`, changes trigger dependency updates
- **`readOnly`**: If `True`, property cannot be modified

**Accessing Properties:**

```python
# Get value
value = self.MyProperty

# Set value
self.MyProperty = 10

# From outside (if promoted)
op('comp').MyProperty = 10
value = op('comp').MyProperty
```

### Dependable Properties

Dependable properties trigger updates in the dependency graph:

```python
TDF.createProperty(self, 'Threshold', value=0.5, 
                  dependable=True, readOnly=False)

# When Threshold changes, dependent operators update
```

---

## StorageManager - Persistent Storage

`StorageManager` provides persistent storage that survives saves and re-initialization.

### Setting Up Storage

```python
from TDStoreTools import StorageManager

class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        
        # Define stored items
        storedItems = [
            {
                'name': 'MyStoredProperty',
                'default': None,
                'readOnly': False,
                'property': True,
                'dependable': True
            },
            {
                'name': 'MyStoredList',
                'default': [],
                'readOnly': False,
                'property': False,
                'dependable': False
            },
        ]
        
        # Initialize storage
        self.stored = StorageManager(self, ownerComp, storedItems)
```

### Stored Item Dictionary Keys

- **`name`**: Required. Name of the stored item
- **`default`**: Default value if not set
- **`readOnly`**: Whether the item can be modified
- **`property`**: Whether to create a property accessor
- **`dependable`**: Whether changes trigger dependency updates

### Accessing Stored Items

```python
# Get value
value = self.stored['MyStoredProperty']

# Set value
self.stored['MyStoredProperty'] = new_value

# If property=True, can also access as property
value = self.stored.MyStoredProperty
self.stored.MyStoredProperty = new_value
```

### Clearing Storage

**Clear specific item:**
```python
self.ownerComp.unstore('MyStoredProperty')
```

**Clear all storage:**
```python
self.ownerComp.storage.clear()
```

**Via Component Editor:**
1. Select the component
2. Open Component Editor (`Alt+E`)
3. Navigate to "Storage" section
4. Clear items via interface

### Storage Best Practices

- **Key Naming**: Use simple, alphanumeric names (avoid spaces)
- **Serializable Data**: Only store data that can be serialized (no Operator objects)
- **Operator References**: Store operator paths as strings, not operator objects

```python
# Store operator path
self.stored['MyOperator'] = op('child1').path

# Later, retrieve operator
my_op = op(self.stored['MyOperator'])
```

---

## TDFunctions Utilities

The `TDFunctions` module provides utility functions for extensions.

### Import

```python
import TDFunctions as TDF
```

### createProperty

Create a property with special features:

```python
TDF.createProperty(
    self,
    'PropertyName',
    value=initial_value,
    dependable=True,
    readOnly=False
)
```

### Common TDF Functions

While the full API may vary, common patterns include:

- **`createProperty()`**: Create extension properties
- Property management utilities
- Helper functions for common tasks

**Note**: Refer to TouchDesigner documentation for the complete `TDFunctions` API.

---

## Lifecycle Methods

Extensions support lifecycle methods that are called at specific times.

### onInitTD()

Called after the extension is fully initialized and all extensions are available.

```python
def onInitTD(self):
    """
    Called after the extension is fully initialized and attached to the 
    component. Use this instead of __init__ for tasks that require other
    components' extensions to be available, or that use promoted members.
    """
    # Safe to access other extensions here
    other_ext = self.ownerComp.ext.OtherExt
    
    # Safe to use promoted members
    self.PromotedFunction()
    
    # Setup tasks that need full initialization
    self.setupUI()
```

**When to use `onInitTD()`:**
- Accessing other extensions
- Using promoted members
- Setup tasks requiring full component initialization
- UI setup that depends on other components

### onDestroyTD()

Called when the extension or component is being deleted.

```python
def onDestroyTD(self):
    """
    Called when the extension or component is being deleted. Use this
    instead of __del__ for cleanup tasks.
    """
    # Cleanup tasks
    if hasattr(self, 'timer'):
        self.timer.destroy()
    
    # Close connections
    if hasattr(self, 'connection'):
        self.connection.close()
    
    # Save final state
    self.saveState()
```

**When to use `onDestroyTD()`:**
- Cleanup resources (timers, connections, etc.)
- Final state saving
- Releasing external resources
- Unsubscribing from events

### Lifecycle Order

1. `__init__(ownerComp)` - Extension constructor
2. `onInitTD()` - After all extensions initialized
3. Component is ready for use
4. `onDestroyTD()` - When component/extension is deleted

---

## Callbacks and Event Handlers

Extensions can respond to various TouchDesigner events.

### onCook()

Called every frame when the component cooks.

```python
def onCook(self):
    """
    Called every frame when the component cooks.
    """
    # Update logic that runs every frame
    self.updateAnimation()
    
    # Process data
    self.processInput()
```

**Use cases:**
- Animation updates
- Real-time data processing
- Continuous monitoring
- Frame-by-frame updates

### onPreCook()

Called before the component cooks.

```python
def onPreCook(self):
    """
    Called before the component cooks each frame.
    """
    # Setup before cooking
    self.prepareData()
    
    # Validate inputs
    if not self.validateInputs():
        return
```

**Use cases:**
- Pre-processing setup
- Input validation
- Data preparation
- Early exit conditions

### onParmChange(par)

Called when a parameter value changes.

```python
def onParmChange(self, par):
    """
    Called when a parameter value changes.
    
    Args:
        par: The parameter that changed
    """
    if par.name == 'threshold':
        self.threshold = par.eval()
        self.updateProcessing()
    
    elif par.name == 'mode':
        mode = par.eval()
        self.switchMode(mode)
```

**Use cases:**
- Responding to parameter changes
- Updating internal state
- Triggering recalculation
- UI updates

### onValueChange(par, val)

Alternative callback for parameter value changes.

```python
def onValueChange(self, par, val):
    """
    Called when a parameter value changes.
    
    Args:
        par: The parameter that changed
        val: The new value
    """
    if par.name == 'intensity':
        self.setIntensity(val)
```

### Parameter Callbacks in Extensions

To use parameter callbacks in extensions, you typically need to set them up in the component's parameter callbacks or use the extension's methods:

```python
# In component's parameter callback script
def onPulse(par):
    if par.name == 'Process':
        # Call extension method
        op(me).ext.SampleExt.ProcessData()

# Or if extension is promoted
def onPulse(par):
    if par.name == 'Process':
        op(me).ProcessData()
```

### Custom Callbacks

You can create custom callback mechanisms:

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.callbacks = []
    
    def RegisterCallback(self, callback):
        """Register a custom callback."""
        self.callbacks.append(callback)
    
    def TriggerCallbacks(self, data):
        """Trigger all registered callbacks."""
        for cb in self.callbacks:
            cb(data)
```

---

## Extension Parameters

Extensions are managed through the COMP's Extensions parameter page.

### Extension Parameter Fields

For each extension:

- **Object** (`ext0object`, `ext1object`, etc.):
  - The class instance/name to attach
  - References the Text DAT containing the class

- **Name** (`ext0name`, `ext1name`, etc.):
  - Optional name to reference the extension
  - Alternative to using the class name
  - Useful for searching/accessing extensions

- **Promote** (`ext0promote`, `ext1promote`, etc.):
  - Whether to promote the extension
  - When `True`, capitalized members are accessible directly from COMP
  - When `False`, must access via `.ext` member

### Managing Extensions

**Re-Init Extensions:**
- Recompiles all extension objects
- Use when extension code changes
- Parameter: `reinitextensions`

**Init Extensions On Start:**
- Automatically re-initialize extensions when TouchDesigner starts
- Parameter: `initextonstart`

**Multiple Extensions:**
- A COMP can have multiple extensions
- Extensions are processed in order
- Access each via its name or class name

```python
# Access multiple extensions
comp = op('myComp')
ext1 = comp.ext.FirstExt
ext2 = comp.ext.SecondExt
```

---

## Best Practices

### Naming Conventions

- **Extension Classes**: Capitalized, ending with `Ext` (e.g., `SampleExt`)
- **Promoted Members**: Capitalized names (e.g., `MyMethod`, `MyProperty`)
- **Non-Promoted Members**: Lowercase names (e.g., `myMethod`, `myProperty`)
- **Private Members**: Prefix with underscore (e.g., `_internalMethod`)

### Code Organization

```python
class SampleExt:
    def __init__(self, ownerComp):
        # 1. Store ownerComp
        self.ownerComp = ownerComp
        
        # 2. Initialize properties
        TDF.createProperty(...)
        
        # 3. Initialize attributes
        self.myAttr = value
        
        # 4. Setup storage
        self.stored = StorageManager(...)
        
        # 5. Get operator references
        self.input_op = self.ownerComp.op('input1')
    
    # 6. Lifecycle methods
    def onInitTD(self):
        pass
    
    # 7. Public methods (promoted)
    def PublicMethod(self):
        pass
    
    # 8. Private methods (non-promoted)
    def _privateMethod(self):
        pass
```

### Error Handling

```python
def myMethod(self):
    try:
        # Get operator
        op_ref = self.ownerComp.op('child1')
        if op_ref is None:
            debug("Operator not found")
            return
        
        # Use operator
        value = op_ref.par.value.eval()
        
    except Exception as e:
        debug(f"Error in myMethod: {e}")
```

### Operator Existence Checks

```python
def safeGetOperator(self, name):
    """Safely get an operator, returning None if not found."""
    try:
        return self.ownerComp.op(name)
    except:
        return None

# Usage
input_op = self.safeGetOperator('input1')
if input_op:
    value = input_op.par.value.eval()
```

### Documentation

```python
class SampleExt:
    """
    SampleExt provides custom functionality for components.
    
    This extension adds data processing capabilities and custom
    parameter management.
    """
    
    def ProcessData(self, data):
        """
        Process input data and return results.
        
        Args:
            data: Input data to process (list or dict)
        
        Returns:
            Processed data
        """
        # Implementation
        pass
```

### Performance Considerations

- **onCook()**: Keep lightweight - runs every frame
- **Caching**: Cache expensive computations
- **Lazy Initialization**: Initialize heavy resources in `onInitTD()`
- **Early Returns**: Use early returns in callbacks to avoid unnecessary work

---

## Common Patterns

### Pattern 1: Operator Reference Management

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self._input_op = None
        self._output_op = None
    
    @property
    def inputOp(self):
        """Lazy-load input operator."""
        if self._input_op is None:
            self._input_op = self.ownerComp.op('input1')
        return self._input_op
    
    @property
    def outputOp(self):
        """Lazy-load output operator."""
        if self._output_op is None:
            self._output_op = self.ownerComp.op('output1')
        return self._output_op
```

### Pattern 2: Parameter Page Management

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

### Pattern 3: State Management

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        
        storedItems = [
            {'name': 'State', 'default': 'idle', 'readOnly': False},
            {'name': 'Data', 'default': [], 'readOnly': False},
        ]
        self.stored = StorageManager(self, ownerComp, storedItems)
    
    def SetState(self, newState):
        """Change state and trigger updates."""
        oldState = self.stored['State']
        self.stored['State'] = newState
        self.onStateChange(oldState, newState)
    
    def onStateChange(self, oldState, newState):
        """Handle state changes."""
        debug(f"State changed: {oldState} -> {newState}")
```

### Pattern 4: Timer/Callback Management

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.timers = []
    
    def CreateTimer(self, callback, interval):
        """Create a timer that calls callback at interval."""
        timer = self.ownerComp.op('timer1')
        if timer:
            # Setup timer callback
            timer.par.callback = f"op('{self.ownerComp.path}').ext.SampleExt.TimerCallback"
            timer.par.interval = interval
            self.timers.append(timer)
        return timer
    
    def TimerCallback(self):
        """Called by timer."""
        debug("Timer fired")
    
    def onDestroyTD(self):
        """Cleanup timers."""
        for timer in self.timers:
            if timer:
                timer.par.active = False
```

### Pattern 5: Data Processing Pipeline

```python
class SampleExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.input_chop = self.ownerComp.op('input1')
        self.output_table = self.ownerComp.op('output1')
    
    def ProcessPipeline(self):
        """Process data through pipeline."""
        # 1. Read input
        data = self.readInput()
        
        # 2. Transform
        transformed = self.transform(data)
        
        # 3. Filter
        filtered = self.filter(transformed)
        
        # 4. Write output
        self.writeOutput(filtered)
    
    def readInput(self):
        """Read data from input CHOP."""
        if not self.input_chop:
            return None
        # Read logic
        return data
    
    def transform(self, data):
        """Transform data."""
        # Transform logic
        return transformed_data
    
    def filter(self, data):
        """Filter data."""
        # Filter logic
        return filtered_data
    
    def writeOutput(self, data):
        """Write data to output table."""
        if not self.output_table:
            return
        # Write logic
        pass
```

---

## Examples

### Example 1: Basic Extension with Properties

```python
from TDStoreTools import StorageManager
import TDFunctions as TDF

class CounterExt:
    """Extension that maintains a counter."""
    
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        
        # Create property
        TDF.createProperty(self, 'Count', value=0, 
                          dependable=True, readOnly=False)
    
    def Increment(self):
        """Increment the counter."""
        self.Count += 1
    
    def Reset(self):
        """Reset the counter."""
        self.Count = 0
```

**Usage:**
```python
# From outside (if promoted)
op('myComp').Increment()
count = op('myComp').Count
op('myComp').Reset()
```

### Example 2: CHOP Data Processor

```python
class CHOPProcessorExt:
    """Extension for processing CHOP data."""
    
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.input_chop = self.ownerComp.op('input1')
        self.output_table = self.ownerComp.op('output1')
    
    def ProcessChop(self):
        """Convert CHOP data to table."""
        if not self.input_chop or not self.output_table:
            return
        
        # Clear output
        self.output_table.clear()
        
        # Get channel names
        chan_names = [ch.name for ch in self.input_chop.chans()]
        self.output_table.appendRow(chan_names)
        
        # Add data rows
        for sample_idx in range(self.input_chop.numSamples):
            row_data = []
            for chan_idx in range(self.input_chop.numChans):
                value = self.input_chop[chan_idx].eval(sample_idx)
                row_data.append(value)
            self.output_table.appendRow(row_data)
```

### Example 3: Extension with Storage

```python
from TDStoreTools import StorageManager
import TDFunctions as TDF

class ConfigExt:
    """Extension for managing configuration."""
    
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        
        storedItems = [
            {
                'name': 'Settings',
                'default': {'threshold': 0.5, 'mode': 'auto'},
                'readOnly': False,
                'property': True,
                'dependable': True
            },
        ]
        self.stored = StorageManager(self, ownerComp, storedItems)
    
    def GetSetting(self, key):
        """Get a setting value."""
        settings = self.stored['Settings']
        return settings.get(key)
    
    def SetSetting(self, key, value):
        """Set a setting value."""
        settings = self.stored['Settings'].copy()
        settings[key] = value
        self.stored['Settings'] = settings
```

### Example 4: Extension with Lifecycle Methods

```python
class ManagedExt:
    """Extension with proper lifecycle management."""
    
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.initialized = False
        self.resources = []
    
    def onInitTD(self):
        """Initialize after all extensions are ready."""
        # Setup that requires other extensions
        other_ext = self.ownerComp.ext.OtherExt
        if other_ext:
            other_ext.RegisterListener(self)
        
        # Initialize resources
        self.setupResources()
        
        self.initialized = True
        debug("Extension initialized")
    
    def setupResources(self):
        """Setup resources."""
        # Create operators, connections, etc.
        pass
    
    def onDestroyTD(self):
        """Cleanup on destruction."""
        # Cleanup resources
        for resource in self.resources:
            if resource:
                try:
                    resource.destroy()
                except:
                    pass
        
        debug("Extension destroyed")
```

### Example 5: Extension with Callbacks

```python
class ReactiveExt:
    """Extension that reacts to parameter changes."""
    
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.threshold = 0.5
    
    def onCook(self):
        """Called every frame."""
        # Update processing
        self.updateProcessing()
    
    def onPreCook(self):
        """Called before cooking."""
        # Validate inputs
        if not self.validateInputs():
            return
    
    def onParmChange(self, par):
        """Called when parameter changes."""
        if par.name == 'threshold':
            self.threshold = par.eval()
            self.updateProcessing()
        elif par.name == 'mode':
            mode = par.eval()
            self.switchMode(mode)
    
    def updateProcessing(self):
        """Update processing based on current state."""
        # Processing logic
        pass
    
    def validateInputs(self):
        """Validate inputs before processing."""
        # Validation logic
        return True
    
    def switchMode(self, mode):
        """Switch processing mode."""
        # Mode switching logic
        pass
```

---

## Troubleshooting

### Extension Not Found

**Problem**: `AttributeError: 'COMP' object has no attribute 'ext'`

**Solutions**:
- Ensure extension is properly added in Component Editor
- Check that extension Text DAT exists and contains valid Python class
- Re-initialize extensions using "Re-Init Extensions" parameter
- Verify extension class name matches the Text DAT name

### Extension Not Initializing

**Problem**: Extension methods not working, errors in initialization

**Solutions**:
- Check Textport for Python errors
- Verify `__init__` method signature: `def __init__(self, ownerComp)`
- Ensure `self.ownerComp = ownerComp` is set
- Check for syntax errors in extension code
- Verify all imports are available

### Promoted Members Not Accessible

**Problem**: Cannot access promoted members directly from COMP

**Solutions**:
- Verify extension is promoted in Extensions parameter page
- Check that member names are capitalized
- Ensure extension is re-initialized after changes
- Try accessing via `.ext` as fallback

### Storage Not Persisting

**Problem**: Stored items not saving across sessions

**Solutions**:
- Verify `StorageManager` is initialized: `self.stored = StorageManager(...)`
- Check that stored items are properly defined in `storedItems` list
- Ensure data is serializable (no Operator objects)
- Check Component Editor Storage section for stored items

### Operator Not Found

**Problem**: `op('child1')` returns None or raises error

**Solutions**:
- Verify operator name/path is correct
- Check operator exists in component
- Use try/except for error handling
- Implement existence checks before use

### Performance Issues

**Problem**: Extension causing performance problems

**Solutions**:
- Avoid heavy computation in `onCook()` (runs every frame)
- Cache expensive operations
- Use `onPreCook()` for early exits
- Profile code to identify bottlenecks
- Consider lazy initialization

### Import Errors

**Problem**: Import errors in extension

**Solutions**:
- Verify module is available in TouchDesigner's Python environment
- Check Python path settings
- Use absolute imports
- Verify module compatibility with TouchDesigner's Python version (3.11)

---

## Additional Resources

- **Official Documentation**: [COMP Extensions Page](https://docs.derivative.ca/COMP_Extensions_Page)
- **Python API**: [TouchDesigner Python API](https://docs.derivative.ca/Python_API)
- **TouchDesigner Wiki**: [TouchDesigner Wiki](https://docs.derivative.ca/)
- **Community Forums**: [TouchDesigner Forum](https://forum.derivative.ca/)

---

## Quick Reference

### Extension Template

```python
from TDStoreTools import StorageManager
import TDFunctions as TDF

class YourExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        TDF.createProperty(self, 'Prop', value=0, 
                          dependable=True, readOnly=False)
    
    def YourMethod(self):
        pass
```

### Access Patterns

```python
# Non-promoted
comp.ext.YourExt.YourMethod()

# Promoted
comp.YourMethod()
```

### Lifecycle

```python
def __init__(self, ownerComp):  # Initialization
def onInitTD(self):             # After all extensions ready
def onDestroyTD(self):          # Cleanup
```

### Callbacks

```python
def onCook(self):               # Every frame
def onPreCook(self):            # Before cooking
def onParmChange(self, par):    # Parameter changed
```

---

*Last Updated: Based on TouchDesigner documentation and common usage patterns. For the most current information, refer to the official TouchDesigner documentation.*



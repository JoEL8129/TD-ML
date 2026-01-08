"""
Script DAT Callbacks

me - this DAT

scriptOp - the OP which is cooking
"""

# press 'Setup Parameters' in the OP to call this function to re-create the
# parameters.
def onSetupParameters(scriptOp: scriptDAT):
	"""
	Called to setup custom parameters for the Script DAT.
	"""
	page = scriptOp.appendCustomPage('Custom')
	p = page.appendFloat('Valuea', label='Value A')
	p = page.appendFloat('Valueb', label='Value B')
	return

def onPulse(par: Par):
	"""
	Called when a custom pulse parameter is pushed.
	
	Args:
		par: The parameter that was pulsed
	"""
	return

def cook(scriptOp):
	"""
	Called when the Script DAT needs to cook.
	Deletes rows where the first column matches any name in parent().par.Notparameters.
	"""
	so = scriptOp
	
	# Copy input table
	so.copy(so.inputs[0])
	
	# Get the Notparameters string from parent
	try:
		not_params_str = parent().par.Notparameters.eval()
	except Exception:
		# If parameter doesn't exist or error, return without filtering
		return
	
	# If empty, nothing to filter
	if not not_params_str or not not_params_str.strip():
		return
	
	# Split by spaces to get list of parameter names to exclude
	exclude_names = [name.strip() for name in not_params_str.split() if name.strip()]
	
	if not exclude_names:
		return
	
	# Convert to set for faster lookup
	exclude_set = set(exclude_names)
	
	# Iterate backwards through rows (skip header row 0)
	# This avoids index issues when deleting rows
	for r in range(so.numRows - 1, 0, -1):
		if so.numCols < 1:
			continue
		
		# Get the first column value (row name)
		row_name = str(so[r, 0].val).strip()
		
		# If row name matches any excluded name, delete the row
		if row_name in exclude_set:
			so.deleteRow(r)
	
	return

def onGetCookLevel(scriptOp: scriptDAT) -> CookLevel:
	"""
	Sets the scriptOp's cook level, the conditions necessary to cause a cook.

	Return one of the following:
		CookLevel.AUTOMATIC - inputs changed and output being used. TD default
		                      behavior.
		CookLevel.ON_CHANGE - inputs changed, output used or not.
		CookLevel.WHEN_USED - every frame when output is being used
		CookLevel.ALWAYS - every frame
	"""

	return CookLevel.AUTOMATIC

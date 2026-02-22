"""
Data sanitization functions for cleaning and preparing tables before machine learning training.

This module provides functions to:
- Fill empty cells
- Normalize/standardize data
- Remove rows/columns with missing values
- Detect and handle outliers
- Assign weights based on conditions

All functions work with TouchDesigner DAT table operators.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Callable, Any


# ============================================================================
# Helper Functions
# ============================================================================

def _is_empty_cell(cell) -> bool:
	"""
	Check if a cell is empty (None, '', whitespace, NaN-like).
	
	Args:
		cell: TouchDesigner DAT cell object or value
		
	Returns:
		bool: True if cell is considered empty
	"""
	if cell is None:
		return True
	
	# Handle TouchDesigner cell objects
	if hasattr(cell, 'val'):
		val = cell.val
	else:
		val = cell
	
	# Check for None
	if val is None:
		return True
	
	# Check for empty string or whitespace
	if isinstance(val, str):
		return len(val.strip()) == 0
	
	# Check for NaN (float)
	try:
		if isinstance(val, float) and np.isnan(val):
			return True
	except (TypeError, ValueError):
		pass
	
	return False


def _get_numeric_columns(table, preserve_header: bool = True) -> List[int]:
	"""
	Identify which columns contain numeric data (skip headers).
	
	Args:
		table: TouchDesigner DAT table operator
		preserve_header: If True, skip first row when checking
		
	Returns:
		List[int]: List of column indices that contain numeric data
	"""
	if table is None or table.numRows == 0 or table.numCols == 0:
		return []
	
	start_row = 1 if preserve_header and table.numRows > 1 else 0
	numeric_cols = []
	
	for c in range(table.numCols):
		is_numeric = False
		has_data = False
		
		for r in range(start_row, table.numRows):
			cell = table[r, c]
			if not _is_empty_cell(cell):
				has_data = True
				try:
					val = cell.val if hasattr(cell, 'val') else cell
					float(val)
					is_numeric = True
					break  # Found at least one numeric value
				except (ValueError, TypeError):
					pass
		
		if is_numeric or (not has_data):  # Include empty columns as potentially numeric
			numeric_cols.append(c)
	
	return numeric_cols


def _compute_column_stats(table, column_idx: int, preserve_header: bool = True) -> Dict[str, float]:
	"""
	Compute mean, std, min, max, median for a numeric column.
	
	Args:
		table: TouchDesigner DAT table operator
		column_idx: Column index to compute stats for
		preserve_header: If True, skip first row
		
	Returns:
		Dict with keys: 'mean', 'std', 'min', 'max', 'median', 'count'
	"""
	if table is None or table.numRows == 0 or column_idx >= table.numCols:
		return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'count': 0}
	
	start_row = 1 if preserve_header and table.numRows > 1 else 0
	values = []
	
	for r in range(start_row, table.numRows):
		cell = table[r, column_idx]
		if not _is_empty_cell(cell):
			try:
				val = cell.val if hasattr(cell, 'val') else cell
				values.append(float(val))
			except (ValueError, TypeError):
				pass
	
	if len(values) == 0:
		return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'count': 0}
	
	values = np.array(values)
	return {
		'mean': float(np.mean(values)),
		'std': float(np.std(values)),
		'min': float(np.min(values)),
		'max': float(np.max(values)),
		'median': float(np.median(values)),
		'count': len(values)
	}


def _has_header_row(table) -> bool:
	"""
	Heuristic to detect if first row is a header (non-numeric values).
	
	Args:
		table: TouchDesigner DAT table operator
		
	Returns:
		bool: True if first row appears to be a header
	"""
	if table is None or table.numRows == 0:
		return False
	
	# Check if first row has mostly non-numeric values
	non_numeric_count = 0
	for c in range(min(5, table.numCols)):  # Check first 5 columns
		cell = table[0, c]
		if not _is_empty_cell(cell):
			try:
				val = cell.val if hasattr(cell, 'val') else cell
				float(val)
			except (ValueError, TypeError):
				non_numeric_count += 1
	
	# If more than half are non-numeric, likely a header
	return non_numeric_count > (min(5, table.numCols) / 2)


# ============================================================================
# Core Sanitization Functions
# ============================================================================

def fill_empty_cells(tables: List, fill_value: Union[float, str], 
					columns: Optional[List[int]] = None, 
					preserve_header: bool = True) -> Dict[str, int]:
	"""
	Fill empty cells in specified table(s) with a given value.
	
	Args:
		tables: List of TouchDesigner DAT table operators
		fill_value: Value to use for filling (float or str)
		columns: Optional list of column indices to target (None = all columns)
		preserve_header: If True, skip first row
		
	Returns:
		Dict with 'tables_processed' and 'cells_filled' counts
	"""
	if not isinstance(tables, list):
		tables = [tables]
	
	total_filled = 0
	start_row = 1 if preserve_header else 0
	
	for table in tables:
		if table is None or table.numRows == 0 or table.numCols == 0:
			continue
		
		has_header = _has_header_row(table) if preserve_header else False
		row_start = 1 if has_header else 0
		
		cols_to_process = columns if columns is not None else list(range(table.numCols))
		
		for r in range(row_start, table.numRows):
			for c in cols_to_process:
				if c >= table.numCols:
					continue
				cell = table[r, c]
				if _is_empty_cell(cell):
					table[r, c].val = fill_value
					total_filled += 1
	
	return {'tables_processed': len(tables), 'cells_filled': total_filled}


def normalize_table(tables: List, columns: Optional[List[int]] = None, 
				   preserve_header: bool = True) -> Dict[str, Any]:
	"""
	Normalize numeric columns to [0, 1] range using min-max scaling.
	
	Args:
		tables: List of TouchDesigner DAT table operators
		columns: Optional list of column indices to process (None = all numeric columns)
		preserve_header: If True, skip first row
		
	Returns:
		Dict with normalization statistics
	"""
	if not isinstance(tables, list):
		tables = [tables]
	
	stats = {}
	
	for table_idx, table in enumerate(tables):
		if table is None or table.numRows == 0 or table.numCols == 0:
			continue
		
		has_header = _has_header_row(table) if preserve_header else False
		row_start = 1 if has_header else 0
		
		if columns is None:
			cols_to_process = _get_numeric_columns(table, preserve_header)
		else:
			cols_to_process = columns
		
		table_stats = {}
		
		for c in cols_to_process:
			if c >= table.numCols:
				continue
			
			# Compute stats for this column
			col_stats = _compute_column_stats(table, c, preserve_header)
			min_val = col_stats['min']
			max_val = col_stats['max']
			range_val = max_val - min_val
			
			if range_val == 0:
				# All values are the same, set to 0.5
				for r in range(row_start, table.numRows):
					cell = table[r, c]
					if not _is_empty_cell(cell):
						table[r, c].val = 0.5
			else:
				# Normalize: (x - min) / (max - min)
				for r in range(row_start, table.numRows):
					cell = table[r, c]
					if not _is_empty_cell(cell):
						try:
							val = float(cell.val if hasattr(cell, 'val') else cell)
							normalized = (val - min_val) / range_val
							table[r, c].val = normalized
						except (ValueError, TypeError):
							pass
			
			table_stats[c] = {'min': min_val, 'max': max_val, 'range': range_val}
		
		stats[f'table_{table_idx}'] = table_stats
	
	return {'tables_processed': len(tables), 'statistics': stats}


def standardize_table(tables: List, columns: Optional[List[int]] = None, 
					 preserve_header: bool = True) -> Dict[str, Any]:
	"""
	Standardize numeric columns to mean=0, std=1 using z-score.
	
	Args:
		tables: List of TouchDesigner DAT table operators
		columns: Optional list of column indices to process (None = all numeric columns)
		preserve_header: If True, skip first row
		
	Returns:
		Dict with standardization statistics
	"""
	if not isinstance(tables, list):
		tables = [tables]
	
	stats = {}
	
	for table_idx, table in enumerate(tables):
		if table is None or table.numRows == 0 or table.numCols == 0:
			continue
		
		has_header = _has_header_row(table) if preserve_header else False
		row_start = 1 if has_header else 0
		
		if columns is None:
			cols_to_process = _get_numeric_columns(table, preserve_header)
		else:
			cols_to_process = columns
		
		table_stats = {}
		
		for c in cols_to_process:
			if c >= table.numCols:
				continue
			
			# Compute stats for this column
			col_stats = _compute_column_stats(table, c, preserve_header)
			mean_val = col_stats['mean']
			std_val = col_stats['std']
			
			if std_val == 0:
				# All values are the same, set to 0
				for r in range(row_start, table.numRows):
					cell = table[r, c]
					if not _is_empty_cell(cell):
						table[r, c].val = 0.0
			else:
				# Standardize: (x - mean) / std
				for r in range(row_start, table.numRows):
					cell = table[r, c]
					if not _is_empty_cell(cell):
						try:
							val = float(cell.val if hasattr(cell, 'val') else cell)
							standardized = (val - mean_val) / std_val
							table[r, c].val = standardized
						except (ValueError, TypeError):
							pass
			
			table_stats[c] = {'mean': mean_val, 'std': std_val}
		
		stats[f'table_{table_idx}'] = table_stats
	
	return {'tables_processed': len(tables), 'statistics': stats}


def remove_rows_with_missing(tables: List, threshold: Union[float, int] = 0.5, 
							 mode: str = 'threshold', preserve_header: bool = True) -> Dict[str, int]:
	"""
	Remove rows where cells are empty based on configurable criteria.
	
	Args:
		tables: List of TouchDesigner DAT table operators
		threshold: Float (0.0-1.0) for percentage, or int for absolute count
		mode: 'any' (any missing), 'all' (all missing), or 'threshold' (percentage)
		preserve_header: If True, skip first row
		
	Returns:
		Dict with 'tables_processed' and 'rows_removed' counts
	"""
	if not isinstance(tables, list):
		tables = [tables]
	
	total_removed = 0
	
	for table in tables:
		if table is None or table.numRows == 0 or table.numCols == 0:
			continue
		
		has_header = _has_header_row(table) if preserve_header else False
		row_start = 1 if has_header else 0
		
		rows_to_delete = []
		
		for r in range(row_start, table.numRows):
			empty_count = 0
			total_count = table.numCols
			
			for c in range(table.numCols):
				cell = table[r, c]
				if _is_empty_cell(cell):
					empty_count += 1
			
			should_remove = False
			
			if mode == 'any':
				should_remove = empty_count > 0
			elif mode == 'all':
				should_remove = empty_count == total_count
			elif mode == 'threshold':
				if isinstance(threshold, float):
					ratio = empty_count / total_count if total_count > 0 else 0
					should_remove = ratio >= threshold
				else:  # int
					should_remove = empty_count >= threshold
			
			if should_remove:
				rows_to_delete.append(r)
		
		# Delete rows in reverse order to maintain indices
		for r in reversed(rows_to_delete):
			table.deleteRow(r)
			total_removed += 1
	
	return {'tables_processed': len(tables), 'rows_removed': total_removed}


def remove_columns_with_missing(tables: List, threshold: Union[float, int] = 0.5, 
							   mode: str = 'threshold', preserve_header: bool = True) -> Dict[str, int]:
	"""
	Remove columns where cells are empty based on configurable criteria.
	
	Args:
		tables: List of TouchDesigner DAT table operators
		threshold: Float (0.0-1.0) for percentage, or int for absolute count
		mode: 'any' (any missing), 'all' (all missing), or 'threshold' (percentage)
		preserve_header: If True, skip first row
		
	Returns:
		Dict with 'tables_processed' and 'columns_removed' counts
	"""
	if not isinstance(tables, list):
		tables = [tables]
	
	total_removed = 0
	
	for table in tables:
		if table is None or table.numRows == 0 or table.numCols == 0:
			continue
		
		has_header = _has_header_row(table) if preserve_header else False
		row_start = 1 if has_header else 0
		
		cols_to_delete = []
		
		for c in range(table.numCols):
			empty_count = 0
			total_count = table.numRows - row_start
			
			for r in range(row_start, table.numRows):
				cell = table[r, c]
				if _is_empty_cell(cell):
					empty_count += 1
			
			should_remove = False
			
			if mode == 'any':
				should_remove = empty_count > 0
			elif mode == 'all':
				should_remove = empty_count == total_count
			elif mode == 'threshold':
				if isinstance(threshold, float):
					ratio = empty_count / total_count if total_count > 0 else 0
					should_remove = ratio >= threshold
				else:  # int
					should_remove = empty_count >= threshold
			
			if should_remove:
				cols_to_delete.append(c)
		
		# Delete columns in reverse order to maintain indices
		for c in reversed(cols_to_delete):
			table.deleteCol(c)
			total_removed += 1
	
	return {'tables_processed': len(tables), 'columns_removed': total_removed}


def detect_outliers(tables: List, method: str = 'iqr', threshold: float = 1.5, 
				   columns: Optional[List[int]] = None, 
				   preserve_header: bool = True) -> Dict[str, Any]:
	"""
	Detect outliers in numeric columns using IQR or Z-score method.
	
	Args:
		tables: List of TouchDesigner DAT table operators
		method: 'iqr' or 'zscore'
		threshold: Multiplier for IQR (default 1.5) or z-score (default 3.0)
		columns: Optional list of column indices to process
		preserve_header: If True, skip first row
		
	Returns:
		Dict with outlier detection results
	"""
	if not isinstance(tables, list):
		tables = [tables]
	
	results = {}
	
	for table_idx, table in enumerate(tables):
		if table is None or table.numRows == 0 or table.numCols == 0:
			continue
		
		has_header = _has_header_row(table) if preserve_header else False
		row_start = 1 if has_header else 0
		
		if columns is None:
			cols_to_process = _get_numeric_columns(table, preserve_header)
		else:
			cols_to_process = columns
		
		table_results = {}
		
		for c in cols_to_process:
			if c >= table.numCols:
				continue
			
			# Get all values for this column
			values = []
			value_positions = []  # Track (row, col) for each value
			
			for r in range(row_start, table.numRows):
				cell = table[r, c]
				if not _is_empty_cell(cell):
					try:
						val = float(cell.val if hasattr(cell, 'val') else cell)
						values.append(val)
						value_positions.append((r, c))
					except (ValueError, TypeError):
						pass
			
			if len(values) == 0:
				continue
			
			values = np.array(values)
			outliers = []
			
			if method == 'iqr':
				q1 = np.percentile(values, 25)
				q3 = np.percentile(values, 75)
				iqr = q3 - q1
				lower_bound = q1 - threshold * iqr
				upper_bound = q3 + threshold * iqr
				
				for i, val in enumerate(values):
					if val < lower_bound or val > upper_bound:
						outliers.append({
							'position': value_positions[i],
							'value': float(val),
							'bound': 'lower' if val < lower_bound else 'upper'
						})
			
			elif method == 'zscore':
				mean_val = np.mean(values)
				std_val = np.std(values)
				
				if std_val > 0:
					z_scores = np.abs((values - mean_val) / std_val)
					for i, z_score in enumerate(z_scores):
						if z_score > threshold:
							outliers.append({
								'position': value_positions[i],
								'value': float(values[i]),
								'z_score': float(z_score)
							})
			
			table_results[c] = {
				'outlier_count': len(outliers),
				'outliers': outliers
			}
		
		results[f'table_{table_idx}'] = table_results
	
	return {'tables_processed': len(tables), 'results': results}


def handle_outliers(tables: List, method: str = 'iqr', threshold: float = 1.5, 
				   action: str = 'cap', columns: Optional[List[int]] = None, 
				   preserve_header: bool = True) -> Dict[str, Any]:
	"""
	Detect and handle outliers by capping, removing, or logging them.
	
	Args:
		tables: List of TouchDesigner DAT table operators
		method: 'iqr' or 'zscore'
		threshold: Multiplier for IQR (default 1.5) or z-score (default 3.0)
		action: 'cap' (cap to bounds), 'remove' (remove rows), 'log' (log only), or 'report' (return report)
		columns: Optional list of column indices to process
		preserve_header: If True, skip first row
		
	Returns:
		Dict with handling results
	"""
	if not isinstance(tables, list):
		tables = [tables]
	
	# First detect outliers
	detection_results = detect_outliers(tables, method, threshold, columns, preserve_header)
	
	if action == 'report' or action == 'log':
		return detection_results
	
	results = {'tables_processed': len(tables), 'actions_taken': {}}
	
	for table_idx, table in enumerate(tables):
		if table is None or table.numRows == 0 or table.numCols == 0:
			continue
		
		has_header = _has_header_row(table) if preserve_header else False
		row_start = 1 if has_header else 0
		
		table_key = f'table_{table_idx}'
		if table_key not in detection_results['results']:
			continue
		
		table_results = detection_results['results'][table_key]
		rows_to_remove = set()
		capped_count = 0
		
		for col_idx, col_data in table_results.items():
			c = int(col_idx)
			if c >= table.numCols:
				continue
			
			# Get column stats for capping
			col_stats = _compute_column_stats(table, c, preserve_header)
			
			if method == 'iqr':
				q1 = np.percentile([float(table[r, c].val) for r in range(row_start, table.numRows) 
								   if not _is_empty_cell(table[r, c])], 25)
				q3 = np.percentile([float(table[r, c].val) for r in range(row_start, table.numRows) 
								   if not _is_empty_cell(table[r, c])], 75)
				iqr = q3 - q1
				lower_bound = q1 - threshold * iqr
				upper_bound = q3 + threshold * iqr
			else:  # zscore
				mean_val = col_stats['mean']
				std_val = col_stats['std']
				lower_bound = mean_val - threshold * std_val
				upper_bound = mean_val + threshold * std_val
			
			for outlier_info in col_data['outliers']:
				r, _ = outlier_info['position']
				
				if action == 'remove':
					rows_to_remove.add(r)
				elif action == 'cap':
					val = outlier_info['value']
					if val < lower_bound:
						table[r, c].val = lower_bound
						capped_count += 1
					elif val > upper_bound:
						table[r, c].val = upper_bound
						capped_count += 1
		
		# Remove rows if needed
		if action == 'remove':
			for r in sorted(rows_to_remove, reverse=True):
				table.deleteRow(r)
		
		results['actions_taken'][table_key] = {
			'rows_removed': len(rows_to_remove) if action == 'remove' else 0,
			'values_capped': capped_count if action == 'cap' else 0
		}
	
	return results


def assign_weights(weight_table, condition: str = 'missing', 
				  weight_strategy: Optional[Dict[str, float]] = None,
				  source_tables: Optional[List] = None,
				  default_weight: float = 1.0,
				  preserve_header: bool = True) -> Dict[str, Any]:
	"""
	Assign or update weights in w_table based on conditions in source tables.
	
	Args:
		weight_table: TouchDesigner DAT table operator for weights
		condition: 'missing', 'outliers', or 'custom'
		weight_strategy: Dict mapping conditions to weights (e.g., {'missing': 0.5, 'normal': 1.0})
		source_tables: List of tables to analyze for conditions (None = use weight_table)
		default_weight: Default weight for rows not matching conditions
		preserve_header: If True, skip first row
		
	Returns:
		Dict with assignment results
	"""
	if weight_table is None:
		raise ValueError("weight_table cannot be None")
	
	if weight_strategy is None:
		weight_strategy = {'missing': 0.5, 'normal': 1.0}
	
	if source_tables is None:
		source_tables = [weight_table]
	
	if not isinstance(source_tables, list):
		source_tables = [source_tables]
	
	has_header = _has_header_row(weight_table) if preserve_header else False
	row_start = 1 if has_header else 0
	
	# Ensure weight_table has enough rows (create if needed)
	max_rows = max([t.numRows for t in source_tables if t is not None] + [weight_table.numRows])
	
	if weight_table.numRows < max_rows:
		# Add rows to match source tables
		for _ in range(max_rows - weight_table.numRows):
			weight_table.appendRow([default_weight])
	
	# Ensure weight_table has at least one column
	if weight_table.numCols == 0:
		weight_table.appendCol([default_weight] * weight_table.numRows)
	
	weights_assigned = 0
	
	# Detect outliers if needed
	outlier_info = None
	if condition == 'outliers':
		outlier_info = detect_outliers(source_tables, method='iqr', preserve_header=preserve_header)
	
	for r in range(row_start, min(weight_table.numRows, max_rows)):
		weight = default_weight
		
		if condition == 'missing':
			# Check if any source table has missing values in this row
			has_missing = False
			for table in source_tables:
				if table is None or r >= table.numRows:
					continue
				for c in range(table.numCols):
					if _is_empty_cell(table[r, c]):
						has_missing = True
						break
				if has_missing:
					break
			
			if has_missing:
				weight = weight_strategy.get('missing', default_weight)
			else:
				weight = weight_strategy.get('normal', default_weight)
		
		elif condition == 'outliers':
			# Check if this row contains outliers
			is_outlier = False
			if outlier_info:
				for table_key, table_data in outlier_info['results'].items():
					for col_idx, col_data in table_data.items():
						for outlier in col_data['outliers']:
							if outlier['position'][0] == r:
								is_outlier = True
								break
						if is_outlier:
							break
					if is_outlier:
						break
			
			if is_outlier:
				weight = weight_strategy.get('outliers', default_weight)
			else:
				weight = weight_strategy.get('normal', default_weight)
		
		# Assign weight (use first column)
		if weight_table.numCols > 0:
			weight_table[r, 0].val = weight
			weights_assigned += 1
	
	return {
		'weights_assigned': weights_assigned,
		'weight_strategy': weight_strategy,
		'condition': condition
	}



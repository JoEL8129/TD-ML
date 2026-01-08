def createParCallback(self, page: Any) -> Any:
	owner = getattr(page, 'owner', None) or getattr(page, 'comp', None)
	if owner is None:
		raise RuntimeError("Cannot resolve owner COMP for Page; cannot create callback DAT.")

	# 1) ensure 'par_callbacks' base COMP
	callbacks_comp = owner.op('par_callbacks')
	if callbacks_comp is None:
		callbacks_comp = owner.create(baseCOMP, 'par_callbacks')

	# 2) ensure Parameter Execute DAT 'par_exec_<ParName>'
	dat_name = f'par_exec_{self.name}'
	ped = callbacks_comp.op(dat_name)
	if ped is None:
		ped = callbacks_comp.create(parameterexecuteDAT, dat_name)


	# 3) set parameters on the Parameter Execute DAT
	try:
		ped.par.op = owner.path
	except Exception:
		pass
	try:
		ped.par.pars = self.name
	except Exception:
		pass

	# 4) choose callback style & event flags
	is_pulse = (self.par_type == 'pulse')
	try:
		ped.par.pulse = bool(is_pulse)
	except Exception:
		pass
	try:
		ped.par.valuechange = bool(not is_pulse)
	except Exception:
		pass

	# 5) write the DAT script (overwrite text)
	func_name = f'On{self.name}'
	if is_pulse:
		code = (
			"def onPulse(par):\n"
			"    try:\n"
			"        owner = parent(2)\n"
			f"        fn = getattr(owner, '{func_name}', None)\n"
			"        if callable(fn):\n"
			"            fn(par)\n"
			"    except Exception as e:\n"
			"        debug('par_exec pulse error:', e)\n"
			"    return\n"
		)
	else:
		code = (
			"def onValueChange(par, prev):\n"
			"    try:\n"
			"        owner = parent(2)\n"
			f"        fn = getattr(owner, '{func_name}', None)\n"
			"        if callable(fn):\n"
			"            fn(par)\n"
			"    except Exception as e:\n"
			"        debug('par_exec valuechange error:', e)\n"
			"    return\n"
		)
	try:
		ped.text = code
	except Exception:
		try:
			ped.clear(); ped.write(code)
		except Exception:
			pass

	return ped

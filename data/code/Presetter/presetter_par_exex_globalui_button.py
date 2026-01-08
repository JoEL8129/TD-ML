# par - the Par object that has changed
# prev - the previous value

def onValueChange(par, prev):
	if not parent().panel.select :
		parent().panel.state = par.eval()
	
	if int(par.eval()) == 1:
		current_child = ui.panes.current.owner.currentChild
		print(current_child.type)
		if current_child is not None:
			if current_child.isCOMP:
				page_exists = False
				for page in scriptOp.pages:
					if page.name == 'Presetter':
						page_exists = True
						break
				if not page_exists:
					source_op = op.PRESETTER_TOOL.op('Presetter_Module')
					if source_op is not None:
						copy = current_child.copy(source_op)
						run(lambda: copy.par.Copybindparstotarget.pulse(), delayFrames=20)
	return
# me - this DAT
# 
# frame - the current frame
# state - True if the timeline is paused
# 
# Make sure the corresponding toggle is enabled in the Execute DAT.

import re




def adjust_menu_source(par, parent1):
    menu_source = par.menuSource
    if isinstance(menu_source, str) and menu_source.startswith('tdu.TableMenu('):
        # Try to extract the inner op() path and rewrite as absolute
        match = re.search(r"op\(['\"](.+?)['\"]\)", menu_source)
        if match:
            rel_path = match.group(1)
            dat_op = parent1.op(rel_path)
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

def onStart():
	return

def onCreate():
    #op('enteredText').text = '_init'
    #parent().par.Presetfolder = parent(2).name
    #parent().par.Presetfolder = 'Data/Presets/'+parent(2).path.replace('/','.')
    #op('fileout1').par.write.pulse()
    parent1 = parent()  # Adjust path as needed
    parent2 = parent(2)  # Adjust path as needed

    # Exclude list: parameters that should not be copied or bound
    exclude_pars = {'Targetop', 'Copybindparstotarget'}  # Add parameter names to exclude, e.g., {'par1', 'par2'}

    page_name = "Presetter"
    if page_name not in [p.name for p in parent2.customPages]:
        custom_page = parent2.appendCustomPage(page_name)
    else:
        custom_page = parent2.customPages[page_name]

    par_list = parent1.customPars  # <-- custom parameters only!

    type_map = {
        'Float': custom_page.appendFloat,
        'Int': custom_page.appendInt,
        'Str': custom_page.appendStr,
        'Menu': custom_page.appendMenu,
        'StrMenu': custom_page.appendStrMenu,
        'Toggle': custom_page.appendToggle,
        'Pulse': custom_page.appendPulse,
        'RGB': custom_page.appendRGB,
        'RGBA': custom_page.appendRGBA,
        'Folder': custom_page.appendFolder,
        # Add more as needed
    }

    for par in par_list:
        name = par.name
        # Skip excluded parameters
        if name in exclude_pars:
            continue
        label = par.label
        par_type = par.style
        append_func = type_map.get(par_type, None)
        if append_func and not any(p.name == name for p in custom_page.pars):
            new_par = append_func(name, label=label)
            if par_type == 'Menu':
                menu_source_path = adjust_menu_source(par, parent1)
                try:
                    new_par.menuSource = menu_source_path
                except Exception:
                    pass
            try:
                new_par.default = par.default
            except Exception:
                pass
            # Copy current value from original parameter to new parameter
            try:
                new_par.val = par.val
            except Exception:
                # If direct val assignment fails, try using eval()
                try:
                    new_par.val = par.eval()
                except Exception:
                    pass
            # ----- Use bindExpr! -----
            #new_par.bindExpr = "op('{}').par.{}".format(parent1.path, name)

    
   #parent(2).par.Delscope = '*'
    #parent(2).par.Presetfolder.expr = "'Data/Presets/'+me.path.replace('/','.')"
    
    #'Presets'+me.path.replace('/','.')


    for par in parent1.customPars:
        # Skip excluded parameters
        if par.name in exclude_pars:
            continue
        par.bindExpr = "op('{}').par.{}".format(parent2.path, par.name)












def onExit():
	return

def onFrameStart(frame):
	return

def onFrameEnd(frame):
	return

def onPlayStateChange(state):
	return

def onDeviceChange():
	return

def onProjectPreSave():
	return

def onProjectPostSave():
	return

	
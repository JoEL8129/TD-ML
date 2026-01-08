# ParTemplate.py

import re
from typing import Any, Dict, Optional, Sequence, Union

ParLike = Any
ParGroupLike = Union[ParLike, Sequence[ParLike]]

class ParTemplate:
    _TYPE_TO_APPEND: Dict[str, str] = {
        'toggle':'appendToggle','pulse':'appendPulse','momentary':'appendMomentary','python':'appendPython',
        'int':'appendInt','float':'appendFloat','str':'appendStr','strmenu':'appendStrMenu','menu':'appendMenu',
        'file':'appendFile','filesave':'appendFileSave','folder':'appendFolder',
        'op':'appendOP','comp':'appendCOMP','object':'appendObject','panelcomp':'appendPanelCOMP',
        'top':'appendTOP','chop':'appendCHOP','sop':'appendSOP','dat':'appendDAT','mat':'appendMAT',
        'xy':'appendXY','xyz':'appendXYZ','xyzw':'appendXYZW','wh':'appendWH','uv':'appendUV',
        'uvw':'appendUVW','rgb':'appendRGB','rgba':'appendRGBA',
    }
    _SEQUENCE_TYPES = {'xy','xyz','xyzw','wh','uv','uvw','rgb','rgba','int','float'}

    def __init__(
        self,
        name: str,
        par_type: str,
        *,
        label: Optional[str] = None,
        size: Optional[int] = None,
        # numeric / tuples
        default: Any = None,        # for tuple-types: pass a tuple/list
        value: Any = None,          # optional explicit initial value (par.val)
        norm_min: Optional[float] = None,
        norm_max: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        clamp: bool = False,
        clamp_min: Optional[bool] = None,
        clamp_max: Optional[bool] = None,
        # behavior
        enable: bool = True,
        order: Optional[int] = None,
        replace: bool = True,
        # menus
        menu_names: Optional[Sequence[str]] = None,
        menu_labels: Optional[Sequence[str]] = None,
        # callbacks
        callback: bool = False,
        # custom-par extras
        enable_express: Optional[Union[bool, str]] = None,  # bool toggles .enable; str goes to .enableExpr
        help: Optional[str] = None,                         # each component's .help
        menu_source: Optional[Any] = None,                  # .menuSource for menu/strmenu
        # extras go straight into append* call
        **append_kwargs: Any
    ) -> None:
        self.par_type = par_type.lower().strip()
        if self.par_type not in self._TYPE_TO_APPEND:
            raise ValueError(f"Unsupported par_type '{par_type}'. Supported: {sorted(self._TYPE_TO_APPEND)}")
        self._validate_or_die(name=name, par_type=self.par_type, size=size)

        # store basics
        self.name = name
        self.label = label
        self.size = size

        self.default = default
        self.value = value
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.min = min
        self.max = max
        self.clamp_min = clamp if clamp_min is None else clamp_min
        self.clamp_max = clamp if clamp_max is None else clamp_max

        self.enable = enable
        self.order = order
        self.replace = replace

        self.menu_names = list(menu_names) if menu_names is not None else None
        self.menu_labels = list(menu_labels) if menu_labels is not None else None

        self.callback = bool(callback)
        self.append_kwargs = append_kwargs

        # custom-par extras
        self.enable_express = enable_express
        # alias: allow enableExpress= as a kwargs alternative
        if 'enableExpress' in self.append_kwargs and self.enable_express is None:
            self.enable_express = self.append_kwargs.pop('enableExpress')
        self.help = help
        self.menu_source = menu_source

        if (self.menu_labels is not None) ^ (self.menu_names is not None):
            raise ValueError("menu_names and menu_labels must be provided together (or not at all).")
        if self.menu_labels is not None and len(self.menu_labels) != len(self.menu_names):
            raise ValueError("menu_names and menu_labels must be the same length.")

    # ---------- creation (idempotent + optional callback) ----------
    def createPar(self, page: Any) -> ParGroupLike:
        # 0) Reuse existing parameter if present (no duplicate creation)
        existing = self._find_existing(page, self.name)
        created = existing
        if created is None:
            # Create fresh
            method_name = self._TYPE_TO_APPEND[self.par_type]
            append = getattr(page, method_name, None)
            if append is None:
                raise RuntimeError(f"Page has no method {method_name} for par_type '{self.par_type}'")

            call_kwargs: Dict[str, Any] = dict(self.append_kwargs)
            if self.label is not None:
                call_kwargs['label'] = self.label
            if self.order is not None:
                call_kwargs['order'] = int(self.order)
            if self.replace is not None:
                call_kwargs['replace'] = bool(self.replace)
            if self.par_type in ('int','float') and self.size:
                call_kwargs['size'] = int(self.size)

            created = append(self.name, **call_kwargs)

            # Menus first (choices before default/value), then numeric/tuple, then enable, then custom extras.
            if self.par_type in ('menu','strmenu'):
                self._apply_menu(created)
            if self.par_type in ('int','float','xy','xyz','xyzw','wh','uv','uvw','rgb','rgba'):
                self._apply_numeric(created)
            self._apply_enable(created)
            self._apply_custom(created)  # extras (help / enableExpr / menuSource)

        # 1) Always attach callback if requested (even if param existed already)
        if self.callback:
            self.createParCallback(page)
            print('callback for',self.name,'created')

        return created


    # ---------- callback creation ----------
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
            ped.nodeY = -150 * len(callbacks_comp.ops('*'))


        # 3) set parameters on the Parameter Execute DAT
        try:
            ped.par.op = '../..'
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

    # ---------- helpers ----------
    @classmethod
    def _validate_or_die(cls, name: str, par_type: str, size: Optional[int]) -> None:
        problems = []
        if not re.match(r'^[A-Z][A-Za-z0-9]*$', name or ''):
            problems.append("Parameter name must be CamelCase alphanumeric, start with uppercase, no spaces/underscores.")
        is_sequence = (par_type in cls._SEQUENCE_TYPES) and (par_type not in ('int','float') or (size or 1) > 1)
        if is_sequence and re.search(r'\d$', name or ''):
            problems.append("Sequence/tuplet names (XYZ/RGBA or Int/Float with size>1) must not end with a digit.")
        if problems:
            msg = " ".join(problems)
            print(f"[ParTemplate WARNING] Invalid parameter name '{name}': {msg}")
            raise ValueError(f"Invalid parameter name '{name}': {msg}")

    def _tuple_suffixes(self) -> Optional[Sequence[str]]:
        t = self.par_type
        if t == 'rgb':  return ['r','g','b']
        if t == 'rgba': return ['r','g','b','a']
        if t == 'xy':   return ['x','y']
        if t == 'xyz':  return ['x','y','z']
        if t == 'xyzw': return ['x','y','z','w']
        if t == 'wh':   return ['w','h']
        if t == 'uv':   return ['u','v']
        if t == 'uvw':  return ['u','v','w']
        return None  # int/float handled separately via size

    def _find_existing(self, page: Any, name: str) -> Optional[ParGroupLike]:
        owner = getattr(page, 'owner', None) or getattr(page, 'comp', None)

        # 1) direct ParGroup lookups
        if owner is not None:
            try:
                grp = owner.par[name]
                if grp is not None:
                    return grp
            except Exception:
                pass
            try:
                grp = getattr(owner.par, name)
                if grp is not None:
                    return grp
            except Exception:
                pass

        # 2) scan page.pars / owner.customPars
        def _iter_candidate_pars():
            ps = getattr(page, 'pars', None)
            if ps:
                for p in ps:
                    yield p
            if owner is not None:
                for p in getattr(owner, 'customPars', []) or []:
                    yield p

        # (a) named tuple types
        suffixes = self._tuple_suffixes()
        if suffixes:
            found = []
            for suf in suffixes:
                comp_name = f"{name}{suf}"
                comp = None
                if owner is not None:
                    try:
                        comp = owner.par[comp_name]
                    except Exception:
                        try:
                            comp = getattr(owner.par, comp_name)
                        except Exception:
                            comp = None
                if comp is None:
                    for p in _iter_candidate_pars():
                        if getattr(p, 'name', None) == comp_name:
                            comp = p
                            break
                if comp is None:
                    found = []
                    break
                found.append(comp)
            if found:
                if owner is not None:
                    try:
                        grp = getattr(owner.par, name)
                        if grp is not None:
                            return grp
                    except Exception:
                        pass
                return found

        # (b) int/float tuplets with size>1
        if self.par_type in ('int','float') and (self.size or 1) > 1:
            n = int(self.size or 1)
            comps = []
            for i in range(1, n + 1):
                comp_name = f"{name}{i}"
                comp = None
                if owner is not None:
                    try:
                        comp = owner.par[comp_name]
                    except Exception:
                        try:
                            comp = getattr(owner.par, comp_name)
                        except Exception:
                            comp = None
                if comp is None:
                    for p in _iter_candidate_pars():
                        if getattr(p, 'name', None) == comp_name:
                            comp = p
                            break
                if comp is None:
                    comps = []
                    break
                comps.append(comp)
            if comps:
                if owner is not None:
                    try:
                        grp = getattr(owner.par, name)
                        if grp is not None:
                            return grp
                    except Exception:
                        pass
                return comps

        # 3) last-resort: match single par by exact name
        for p in _iter_candidate_pars():
            if getattr(p, 'name', None) == name:
                return p

        return None

    def _iter_pars(self, created: ParGroupLike):
        try:
            return list(created)
        except Exception:
            return [created]

    def _apply_enable(self, created: ParGroupLike) -> None:
        for par in self._iter_pars(created):
            try:
                par.enable = bool(self.enable)
            except Exception:
                pass

    def _apply_numeric(self, created: ParGroupLike) -> None:
        items = self._iter_pars(created)
        is_group = len(items) > 1

        if is_group and self.default is not None:
            try:
                setattr(created, 'default', tuple(self.default))
            except Exception:
                pass
            initial = self.value if (self.value is not None) else self.default
            try:
                setattr(created, 'val', tuple(initial))
            except Exception:
                pass

        for par in items:
            if self.min is not None: self._safe_set(par, 'min', self.min)
            if self.max is not None: self._safe_set(par, 'max', self.max)
            if self.clamp_min is not None: self._safe_set(par, 'clampMin', bool(self.clamp_min))
            if self.clamp_max is not None: self._safe_set(par, 'clampMax', bool(self.clamp_max))
            if self.norm_min is not None: self._safe_set(par, 'normMin', self.norm_min)
            if self.norm_max is not None: self._safe_set(par, 'normMax', self.norm_max)

            if not is_group:
                if self.default is not None: self._safe_set(par, 'default', self.default)
                if self.value is not None: self._safe_set(par, 'val', self.value)
                elif self.default is not None: self._safe_set(par, 'val', self.default)

    def _apply_menu(self, created: ParGroupLike) -> None:
        par = next(iter(self._iter_pars(created)), None)
        if par is None:
            return

        if self.menu_names is not None: self._safe_set(par, 'menuNames', list(self.menu_names))
        if self.menu_labels is not None: self._safe_set(par, 'menuLabels', list(self.menu_labels))

        if self.par_type == 'menu':
            def _to_index(val) -> Optional[int]:
                if val is None: return None
                if isinstance(val, int): return val
                try: return list(par.menuNames).index(val)
                except Exception: return None
            idx_default = _to_index(self.default)
            if idx_default is not None:
                self._safe_set(par, 'default', idx_default)
                self._safe_set(par, 'menuIndex', idx_default)
            idx_value = _to_index(self.value)
            if idx_value is not None:
                self._safe_set(par, 'menuIndex', idx_value)

        elif self.par_type == 'strmenu':
            if self.default is not None:
                self._safe_set(par, 'default', self.default)
                self._safe_set(par, 'val', self.default)
            if self.value is not None:
                self._safe_set(par, 'val', self.value)

    # custom extras applied on creation
    def _apply_custom(self, created: ParGroupLike) -> None:
        items = self._iter_pars(created)

        # help text
        if self.help is not None:
            for par in items:
                self._safe_set(par, 'help', self.help)

        # enable_express: bool -> .enable; str -> .enableExpr
        if self.enable_express is not None:
            if isinstance(self.enable_express, str):
                for par in items:
                    # set expression string; TD will evaluate it
                    self._safe_set(par, 'enableExpr', self.enable_express)
            else:
                val = bool(self.enable_express)
                for par in items:
                    # clear any existing expr and use explicit enable toggle
                    self._safe_set(par, 'enableExpr', '')
                    self._safe_set(par, 'enable', val)

        # menu source (only for menu/strmenu)
        if self.par_type in ('menu', 'strmenu') and self.menu_source is not None:
            par = next(iter(items), None)
            if par is not None:
                self._safe_set(par, 'menuSource', self.menu_source)

    @staticmethod
    def _safe_set(obj: Any, attr: str, value: Any) -> None:
        try:
            setattr(obj, attr, value)
        except Exception:
            pass

try:
    assert INTERACTIVE
except Exception:
    from setups import *
    from heavy_setups import *

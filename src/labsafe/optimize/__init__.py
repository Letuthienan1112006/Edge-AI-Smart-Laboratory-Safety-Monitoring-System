"""Model-optimisation tooling: probe the device, then choose the technique.

Imports are lazy so that `python3 -m labsafe.optimize.arm_probe` does not
re-import the package mid-execution, and so the Pi never loads torch just
because something touched this package.
"""

__all__ = ["probe_cpu", "int8_verdict", "report"]


def __getattr__(name):
    if name in __all__:
        from . import arm_probe

        return getattr(arm_probe, name)
    raise AttributeError(name)

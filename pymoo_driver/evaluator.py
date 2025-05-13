import numpy as np
from openmdao.core.driver import Driver


class OpenMDAOEvaluator:
    """
    Wraps an OpenMDAO problem so that it can be evaluated by pymoo.
    """

    def __init__(self, driver: Driver):
        self.driver = driver

    def set_design_vars(self, x):
        """Assign values to OpenMDAO design variables from a flat numpy array."""
        i = 0
        for name, meta in self.driver._designvars.items():
            size = meta["size"]
            self.driver.set_design_var(name, x[i : i + size])
            i += size

    def evaluate(self, x):
        """
        Run the OpenMDAO model with design vector x and return objective and constraint values.
        """
        self.set_design_vars(x)

        try:
            self.driver._run_solve_nonlinear()
        except Exception:
            nF = len(self.driver._objs)
            nG = len(self.driver._cons)
            return np.full(nF, 1e20), np.full(nG, 1e20)

        F = np.array(
            [
                self.driver._get_voi_val(name, meta, self.driver._remote_objs)
                for name, meta in self.driver._objs.items()
            ]
        )
        G = np.array(
            [
                self.driver._get_voi_val(name, meta, self.driver._remote_cons)
                for name, meta in self.driver._cons.items()
            ]
        )
        return F, G

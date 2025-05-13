import numpy as np
from openmdao.core.driver import Driver
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import Termination
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .evaluator import OpenMDAOEvaluator


class PymooDriver(Driver):
    """
    OpenMDAO driver using pymoo for evolutionary multi-objective optimization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.supports["optimization"] = True
        self.supports["multiple_objectives"] = True
        self.supports["inequality_constraints"] = True
        self.supports["equality_constraints"] = True
        self.supports["integer_design_vars"] = True
        self.supports._read_only = True

        self.options.declare(
            "algorithm", default=NSGA2(pop_size=40), desc="pymoo algorithm instance"
        )
        self.options.declare(
            "termination",
            default=("n_gen", 100),
            desc="pymoo Termination condition (tuple or Termination instance)",
        )
        self.options.declare(
            "verbose",
            default=True,
            desc="Whether to run the pymoo minimization in verbose mode",
        )

    def _declare_options(self):
        pass

    def run(self):
        self.result.reset()
        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        evaluator = OpenMDAOEvaluator(self)

        # Prepare problem dimensions and bounds
        n_var = sum(meta["size"] for meta in self._designvars.values())
        xl = np.concatenate(
            [np.ravel(meta["lower"]) for meta in self._designvars.values()]
        )
        xu = np.concatenate(
            [np.ravel(meta["upper"]) for meta in self._designvars.values()]
        )
        n_obj = len(self._objs)
        n_constr = len(self._cons)

        class Problem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu
                )
                self._evaluator = evaluator

            def _evaluate(self, x, out, *args, **kwargs):
                F, G = self._evaluator.evaluate(x)
                out["F"] = F
                out["G"] = G

        prob = Problem()

        termination_opt = self.options["termination"]

        res = minimize(
            prob,
            self.options["algorithm"],
            termination=termination_opt
            if isinstance(termination_opt, Termination)
            else get_termination(*termination_opt),
            verbose=self.options["verbose"],
        )

        self._apply_solution(res.X)
        return False

    def _apply_solution(self, x):
        """Assign optimal design variable values and run the final model."""
        i = 0
        for name, meta in self._designvars.items():
            size = meta["size"]
            self.set_design_var(name, x[i : i + size])
            i += size

        self._run_solve_nonlinear()

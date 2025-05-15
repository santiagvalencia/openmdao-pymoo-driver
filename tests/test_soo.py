import os
from typing import Callable

import openmdao.api as om
import pytest
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.cmaes import BIPOPCMAES, CMAES, SimpleCMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.direct import DIRECT
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.termination.default import DefaultSingleObjectiveTermination

from pymoo_driver import PymooDriver

os.environ["OPENMDAO_REPORTS"] = "none"
TERMINATION = DefaultSingleObjectiveTermination(xtol=1e-6, cvtol=1e-6)


class SphereComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("x", val=0.0)
        self.add_input("y", val=0.0)
        self.add_output("f", val=0.0)
        self.add_output("g", val=0.0)
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        outputs["f"] = x**2 + y**2
        outputs["g"] = 1 - x - y


class AlgorithmFactory:
    def __init__(self, algorithm_cls: Callable, *args, expected_error=None, **kwargs):
        self.algorithm_cls = algorithm_cls
        self.args = args
        self.kwargs = kwargs
        self.expected_error = expected_error

    def __call__(self):
        return self.algorithm_cls(*self.args, **self.kwargs)


working_algorithms = [
    pytest.param(
        AlgorithmFactory(BRKGA, termination=TERMINATION),
        id="BRKGA",
    ),
    pytest.param(
        AlgorithmFactory(DE, termination=TERMINATION),
        id="DE",
    ),
    pytest.param(
        AlgorithmFactory(DIRECT, termination=TERMINATION),
        id="DIRECT",
    ),
    pytest.param(
        AlgorithmFactory(G3PCX, termination=TERMINATION),
        id="G3PCX",
    ),
    pytest.param(
        AlgorithmFactory(NicheGA, termination=TERMINATION, return_all_opt=False),
        id="NicheGA",
    ),
    pytest.param(
        AlgorithmFactory(GA, termination=TERMINATION),
        id="GA",
    ),
    pytest.param(
        AlgorithmFactory(NelderMead, termination=TERMINATION),
        id="NelderMead",
    ),
    pytest.param(
        AlgorithmFactory(PatternSearch, termination=TERMINATION),
        id="PatternSearch",
    ),
    pytest.param(
        AlgorithmFactory(EPPSO, termination=TERMINATION),
        id="EPPSO",
    ),
    pytest.param(
        AlgorithmFactory(PSO, termination=TERMINATION),
        id="PSO",
    ),
    pytest.param(
        AlgorithmFactory(RandomSearch, termination=TERMINATION),
        id="RandomSearch",
    ),
]

known_broken_algorithms = [
    pytest.param(
        AlgorithmFactory(CMAES, termination=TERMINATION),
        id="CMAES",
        marks=pytest.mark.xfail(reason="NumPy 2 incompatibility", strict=True),
    ),
    pytest.param(
        AlgorithmFactory(SimpleCMAES, termination=TERMINATION),
        id="SimpleCMAES",
        marks=pytest.mark.xfail(reason="NumPy 2 incompatibility", strict=True),
    ),
    pytest.param(
        AlgorithmFactory(BIPOPCMAES, termination=TERMINATION),
        id="BIPOPCMAES",
        marks=pytest.mark.xfail(reason="NumPy 2 incompatibility", strict=True),
    ),
    pytest.param(
        AlgorithmFactory(ISRES, termination=TERMINATION),
        id="ISRES",
        marks=pytest.mark.xfail(reason="NumPy 2 incompatibility", strict=True),
    ),
    pytest.param(
        AlgorithmFactory(SRES, termination=TERMINATION),
        id="SRES",
        marks=pytest.mark.xfail(reason="NumPy 2 incompatibility", strict=True),
    ),
    pytest.param(
        AlgorithmFactory(Optuna, termination=TERMINATION),
        id="Optuna",
        marks=pytest.mark.xfail(reason="Problem must define `vars`", strict=True),
    ),
]

algorithm_factories = working_algorithms + known_broken_algorithms


@pytest.mark.parametrize("algorithm_factory", algorithm_factories)
def test_algorithm_converges(algorithm_factory):
    algorithm = algorithm_factory()

    prob = om.Problem()
    model = prob.model

    model.add_subsystem("sphere", SphereComp(), promotes=["*"])
    prob.driver = PymooDriver()
    prob.driver.options["algorithm"] = algorithm
    prob.driver.options["termination"] = algorithm.termination
    prob.driver.options["verbose"] = False

    model.add_design_var("x", lower=-2.0, upper=2.0)
    model.add_design_var("y", lower=-2.0, upper=2.0)
    model.add_objective("f")
    model.add_constraint("g", upper=0.0)

    prob.setup()
    prob.set_val("x", 0.5)
    prob.set_val("y", 0.5)

    prob.run_driver()

    g_val = prob.get_val("g")[0]
    f_val = prob.get_val("f")[0]

    assert g_val <= 1e-6, f"Constraint violated: g = {g_val}"
    assert f_val < 0.6, f"Poor objective: f = {f_val}"

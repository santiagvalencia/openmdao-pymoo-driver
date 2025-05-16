import os
from pathlib import Path

import matplotlib.pyplot as plt
import openmdao.api as om
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.direct import DIRECT
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.core.algorithm import Algorithm
from pymoo.termination.default import DefaultSingleObjectiveTermination

from pymoo_driver.driver import PymooDriver

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


class SphereComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("x", val=0.0)
        self.add_input("y", val=0.0)
        self.add_output("f", val=0.0)
        self.add_output("g", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        outputs["f"] = x**2 + y**2
        outputs["g"] = 1 - x - y


def make_and_run_problem(algorithm: Algorithm, verbose=False):
    algorithm_name = algorithm.__class__.__name__
    problem_name = f"sphere_{algorithm_name}"

    prob = om.Problem(name=problem_name)
    model = prob.model

    prob.options["work_dir"] = SCRIPT_DIR

    model.add_subsystem("sphere", SphereComp(), promotes=["*"])
    prob.driver = PymooDriver()
    prob.driver.options["algorithm"] = algorithm
    prob.driver.options["termination"] = algorithm.termination
    prob.driver.options["verbose"] = verbose

    prob.driver.recording_options["record_objectives"] = True
    prob.driver.add_recorder(om.SqliteRecorder(f"driver_cases_{algorithm_name}.sqlite"))

    model.add_design_var("x", lower=-2.0, upper=2.0)
    model.add_design_var("y", lower=-2.0, upper=2.0)
    model.add_objective("f")
    model.add_constraint("g", upper=0.0)

    prob.setup()
    prob.set_val("x", 0.5)
    prob.set_val("y", 0.5)

    prob.run_driver()
    prob.cleanup()

    print(f"Best solution found using {algorithm_name}:")
    print(f"x = {prob.get_val('x')}")
    print(f"y = {prob.get_val('y')}")
    print(f"f = {prob.get_val('f')}")
    print(f"g = {prob.get_val('g')}")

    cr = om.CaseReader(
        SCRIPT_DIR / f"{problem_name}_out/driver_cases_{algorithm_name}.sqlite"
    )
    driver_cases = list(cr.get_cases("driver"))
    print(driver_cases[0].__dict__)
    iters = [i for i, _ in enumerate(driver_cases, start=1)]
    print(iters)
    objs = [list(case.get_objectives().values())[0] for case in driver_cases]
    print(objs)
    plt.plot(iters, objs)
    plt.show()


def run():
    termination = DefaultSingleObjectiveTermination(xtol=1e-6, cvtol=1e-6)
    algorithms = [
        DE(termination=termination),
        DIRECT(termination=termination),
        G3PCX(termination=termination),
        NicheGA(termination=termination, return_all_opt=False),
        GA(termination=termination),
        NelderMead(termination=termination),
        PatternSearch(termination=termination),
        PSO(termination=termination),
        RandomSearch(termination=termination),
    ]

    for algorithm in algorithms:
        make_and_run_problem(algorithm, verbose=True)
        break


if __name__ == "__main__":
    run()

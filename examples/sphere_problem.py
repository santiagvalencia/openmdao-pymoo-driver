import openmdao.api as om
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderAndMeadTermination, NelderMead
from pymoo.core.algorithm import Algorithm
from pymoo.termination.default import DefaultSingleObjectiveTermination

from pymoo_driver.driver import PymooDriver


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
    prob = om.Problem()
    model = prob.model

    model.add_subsystem("sphere", SphereComp(), promotes=["*"])
    prob.driver = PymooDriver()
    prob.driver.options["algorithm"] = algorithm
    prob.driver.options["termination"] = algorithm.termination
    prob.driver.options["verbose"] = verbose

    model.add_design_var("x", lower=-2.0, upper=2.0)
    model.add_design_var("y", lower=-2.0, upper=2.0)
    model.add_objective("f")
    model.add_constraint("g", upper=0.0)

    prob.setup()
    prob.set_val("x", 0.5)
    prob.set_val("y", 0.5)

    prob.run_driver()

    algorithm_name = algorithm.__repr__().split(" ")[0].split(".")[-1]

    print(f"Best solution found using {algorithm_name}:")
    print(f"x = {prob.get_val('x')}")
    print(f"y = {prob.get_val('y')}")
    print(f"f = {prob.get_val('f')}")
    print(f"g = {prob.get_val('g')}")


def run():
    algorithms = [
        NSGA2(pop_size=40, termination=("n_gen", 10)),
        DE(pop_size=40, termination=("n_gen", 10)),
        GA(
            pop_size=40,
            termination=DefaultSingleObjectiveTermination(xtol=1e-6, cvtol=1e-6),
        ),
        NelderMead(termination=NelderAndMeadTermination()),
    ]

    for algorithm in algorithms:
        make_and_run_problem(algorithm, verbose=True)


if __name__ == "__main__":
    run()

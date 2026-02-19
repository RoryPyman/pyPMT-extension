
from unified_planning.shortcuts import PlanValidator, SequentialSimulator
from unified_planning.io import PDDLWriter
from unified_planning.engines.sequential_simulator import evaluate_quality_metric, evaluate_quality_metric_in_initial_state
from unified_planning.model.metrics import MinimizeSequentialPlanLength

class SMTSequentialPlan:
    def __init__(self, plan, task):
        self.isvalid    = None
        self.cost_value = None
        self.validation_fail_reason = None
        self.plan = plan
        self.task = task
        self._plan_str = None
        self.compiled_names = None  # set by solveUP before lifting; used for --annotate

    def __len__(self):
        """!
        Returns the length of the plan.

        @return the length of the actions in the plan.
        """
        return len(self.plan.actions)

    def __iter__(self):
        """!
        Returns the plan's actions iterator.

        @return an iterator of the actions in the plan.
        """

        return iter(self.plan.actions)

    def __str__(self):
        """!
        Returns the plan as a string in PDDL format.

        @return the plan as a string.
        """
        if self._plan_str is None:
            self._plan_str = PDDLWriter(self.task).get_plan(self.plan)
        return self._plan_str

    def annotated_str(self):
        """!
        Returns the plan with intention/delegation annotations.

        For each action that was produced by the intention compiler
        (i.e. its compiled name contains '-because-'), a comment line
        is appended showing the original reasoning, e.g.:
            (travel aladdin castle mountain)
              ; because: traveller intends married-to

        @return the annotated plan as a string.
        """
        if not self.compiled_names:
            return str(self)

        def _pddl_action(ai):
            name = ai.action.name
            params = " ".join(str(p) for p in ai.actual_parameters)
            return f"({name} {params})" if params else f"({name})"

        def _annotation(compiled_name):
            # compiled_name is the grounded action name, e.g.
            #   "travel-because-traveller-intends-married-to_aladdin_castle_mountain"
            # strip grounding suffix (everything after the first '_')
            base = compiled_name.split("_")[0] if "_" in compiled_name else compiled_name
            if "-because-" not in base:
                return ""
            _, reasoning = base.split("-because-", 1)
            return f"  ; because: {reasoning.replace('-', ' ')}"

        lines = []
        for action_instance, compiled_name in zip(self.plan.actions, self.compiled_names):
            lines.append(_pddl_action(action_instance))
            note = _annotation(compiled_name)
            if note:
                lines.append(note)
        return "\n".join(lines)
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, value) -> bool:
        return self.__hash__() == value.__hash__()

    def cost(self):
        """!
        Computes the cost of the plan.

        @return cost: dictionary containing the cost of the plan.
        """
        # return the cost of the plan if already computed before.
        if self.cost_value is not None:
            return self.cost_value

        quality_metrics = self.task.quality_metrics

        # Since we don't have any metric, then we will use the MinimizeSequentialPlanLength metric.
        quality_metrics = (
            [("makespan", MinimizeSequentialPlanLength())]
            if len(quality_metrics) == 0
            else [(m.expression, m) for m in quality_metrics]
        )

        # First simulate the plan to get the states.
        with SequentialSimulator(problem=self.task) as simulator:
            initial_state = simulator.get_initial_state()
            current_state = initial_state
            states = [current_state]
            for action_instance in self.plan.actions:
                current_state = simulator.apply(current_state, action_instance)
                if current_state is None:
                    assert False, "No cost available since the plan is invalid."
                states.append(current_state)

            # since the plan is already computed, we just want to evaluate the cost of the plan for a given fluent in
            # the state.
            metric_values = {}
            for metricname, metric in quality_metrics:
                metric_value = evaluate_quality_metric_in_initial_state(
                    simulator, metric
                )
                current_state = states[0]
                for next_state, action_instance in zip(states[1:], self.plan.actions):
                    metric_value = evaluate_quality_metric(
                        simulator,
                        metric,
                        metric_value,
                        current_state,
                        action_instance.action,
                        action_instance.actual_parameters,
                        next_state,
                    )
                    current_state = next_state
                metric_values[metricname] = metric_value
        self.cost_value = metric_values
        return self.cost_value
    
    def validate(self):
        """!
        Validates plan (when one is found).

        @param domain: path to PDDL domain file.
        @param problem: path to PDDL problem file.

        @return plan: string containing plan if plan found is valid, None otherwise.
        """
        if self.plan is None or self.task is None:
            self.validation_fail_reason = "No plan or task provided." 
            return None
        if self.isvalid is not None: return self.isvalid
        
        with PlanValidator(name='sequential_plan_validator') as validator:
            validationresult = validator.validate(self.task, self.plan)
        self.validation_fail_reason = validationresult.reason
        self.isvalid = validationresult.status.value == 1 if validationresult else False

        return self.isvalid
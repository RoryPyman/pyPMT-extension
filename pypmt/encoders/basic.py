import networkx as nx
import z3

from copy import deepcopy
from collections import defaultdict

from unified_planning.shortcuts import Effect, EffectKind
from unified_planning.shortcuts import FNode, Fraction
from unified_planning.model.fluent import get_all_fluent_exp

from unified_planning.plans import SequentialPlan
from unified_planning.plans import ActionInstance

from pypmt.encoders.base import Encoder
from pypmt.encoders.utilities import str_repr, str_repr_axiom, flattern_list
from pypmt.modifiers.modifierLinear import LinearModifier
from pypmt.modifiers.modifierParallel import ParallelModifier, MutexSemantics

from pypmt.planner.plan.smt_sequential_plan import SMTSequentialPlan

class EncoderGrounded(Encoder):
    """!
    As its filename implies, it's the most basic encoding you can imagine.  It
    first uses UP to ground the problem, and then implements a state-based
    encoding of Planning as SAT.  Details of the encoding can be found in the
    recent Handbooks of Satisfiability in the chapter written by Rintanen:
    Planning and SAT.

    The classical way of improving the performance of encodings is to allow more
    than one action per step (layer). This class is really a "base class" for 
    two encodings:
        - sequential encoding: Kautz & Selman 1992 for the original encoding
        - ForAll semantics: this implements a generalisation for numeric
        planning of the original work in Kautz & Selman 1996 

    22/10 - RP
    This encoder has been updated to encode axioms, firing them as triggers after each axiom.
    The updated overall structure of the encodign is
    Timestep structure:
    - t=0: Initial state
    - Odd timesteps (1,3,5...): Actions execute
    - Even timesteps (2,4,6...): Axioms fire (if their preconditions hold)
    
    State flow: state(0) → action(1) → state(1) → axioms(2) → state(2) → action(3) → state(3)...
    """

    def __init__(self, name, task, modifier, parallel):
        self.task = task # The UP problem
        self.name = name
        self.modifier = modifier
        self.ctx = z3.Context() # The context where we will store the problem
        self.parallel = parallel
        # cache all fluents in the problem.
        self.all_fluents = flattern_list([list(get_all_fluent_exp(task, f)) for f in task.fluents])
        self._initialize_fluents(task, self.all_fluents)
        self.all_fluents = flattern_list([list(get_all_fluent_exp(task, f)) for f in task.fluents])
        # The main idea here is that we have lists representing
        # the layers (steps) containing the respective variables

        # this is a mapping from the UP ground actions to z3 and back
        self.z3_actions_to_up = dict() # multiple z3 vars point to one grounded fluent
        self.up_actions_to_z3 = defaultdict(list)

        # this is a mapping from the UP ground axioms to z3 and back
        self.z3_axioms_to_up = dict()
        self.up_axioms_to_z3 = defaultdict(list)


        # mapping from up fluent to Z3 var
        self.up_fluent_to_z3 = defaultdict(list)

        # frame index, indexing what actions can modify which fluent
        self.frame_add_action = defaultdict(list)
        self.frame_del_action = defaultdict(list)
        self.frame_num_action = defaultdict(list)
        
        self.frame_add_axiom = defaultdict(list)
        self.frame_del_axiom = defaultdict(list)
        self.frame_num_axiom = defaultdict(list)

        # Store the "raw" formula that we will later instantiate
        self.formula  = defaultdict(list)

        # Store the length of the formula
        self.formula_length = 0

    def __iter__(self):
        return iter(self.task.actions)

    def __len__(self):
        return self.formula_length

    def _initialize_fluents(self, _task, _fluentslist):
        initialized_fluents = list(_task.explicit_initial_values.keys())
        unintialized_fluents = list(filter(lambda x: not x in initialized_fluents, _fluentslist))
        for fe in unintialized_fluents:
            if fe.type.is_bool_type():
                _task.set_initial_value(fe, False) # we need this for plan validator.
            elif fe.type.is_real_type():
                _task.set_initial_value(fe, 0) # we need this for plan validator.
            else:
                raise TypeError

    def get_action_var(self, name, t):
        """!
        Given a str representation of a fluent/action and a timestep,
        returns the respective Z3 var.

        @param name: str representation of a fluent or action
        @param t: the timestep we are interested in
        @returns: the corresponding Z3 variable
        """
        # Actions are at odd timesteps, so map givem timestep to actual timestep
        actual_t = 2 * t + 1
        if actual_t >= len(self.up_actions_to_z3[name]):
            return None
        return self.up_actions_to_z3[name][actual_t]
    
    def get_axiom_var(self, name, t):
        """!
        Given a str representation of a fluent/axiom and a timestep,
        returns the respective Z3 var.

        @param name: str representation of a fluent or axiom
        @param t: the timestep we are interested in
        @returns: the corresponding Z3 variable
        """
        # Axioms are at even timesteps (after t=0), so map logical step to actual timestep
        actual_t = 2 * t + 2
        if actual_t >= len(self.up_axioms_to_z3[name]):
            return None
        return self.up_axioms_to_z3[name][actual_t]

    def _populate_modifiers(self):
        """!
        Populates an index on which grounded actions & axioms can modify which fluents.
        These are used afterwards for encoding the frame.
        """
        # Index what actions can modify
        for action in self.task.actions:
            str_action = str_repr(action)
            for effect in action.effects:
                var_modified = str_repr(effect.fluent)
                condition = effect.condition # get the condition of the effect
                if effect.value.is_true(): # boolean effect
                    self.frame_add_action[var_modified].append((condition, str_action))
                elif effect.value.is_false():
                    self.frame_del_action[var_modified].append((condition, str_action))
                else: # is a numeric or complex expression
                    self.frame_num_action[var_modified].append((condition, str_action))
        
        # Index what axioms can modify
        for axiom in self.task.axioms:
            str_axiom = str_repr_axiom(axiom)
            for effect in axiom.effects:
                var_modified = str_repr(effect.fluent)
                # Axiom conditions are their preconditions
                condition = z3.BoolVal(True, ctx=self.ctx)  # Will be handled in frame encoding
                if effect.value.is_true():
                    self.frame_add_axiom[var_modified].append((condition, str_axiom))
                elif effect.value.is_false():
                    self.frame_del_axiom[var_modified].append((condition, str_axiom))
                else:
                    self.frame_num_axiom[var_modified].append((condition, str_axiom))


    def extract_plan(self, model, horizon):
        """!
        Given a model of the encoding generated by this class and its horizon,
        extract a plan from it. Only actions go in the plan.
        @returns: an instance of a SMTSequentialPlan
        """
        plan = SequentialPlan([])
        if not model: return SMTSequentialPlan(plan, self.task)
        if self.parallel:
            # Linearise plan taking into account step order
            action_map = {action.name: action for action in self}
            for t in range(0, horizon + 1):
                actual_t = 2 * t + 1  # Convert to actual timestep
                active_actions = set()
                for action in self:
                    if actual_t < len(self.up_actions_to_z3[action.name]):
                        if z3.is_true(model[self.up_actions_to_z3[action.name][actual_t]]):
                            active_actions.add(action.name)
                if len(self.modifier.graph.nodes) > 0:
                    sorted_action_names = list(nx.topological_sort(self.modifier.graph.subgraph(active_actions)))[::-1]
                else:
                    sorted_action_names = active_actions
                for action_name in sorted_action_names:
                    plan.actions.append(ActionInstance(action_map.get(action_name)))
        else:
            ## linearize partial-order plan
            for t in range(0, horizon + 1):
                actual_t = 2 * t + 1
                for action in self:
                    if actual_t < len(self.up_actions_to_z3[action.name]):
                        if z3.is_true(model[self.up_actions_to_z3[action.name][actual_t]]):
                            plan.actions.append(ActionInstance(action))
        return SMTSequentialPlan(plan, self.task)

    def encode(self, t):
        """!
        Encode formulas for one logical planning step, an action and an axiom.
        Each logical step creates 2 'actual' timesteps: one for the action, one for the axioms.
        @param t: the current timestep we want the encoding for
        @returns: A dict with the different parts of the formula encoded
        """
        if t == 0:
            self.base_encode()
            return deepcopy(self.formula)

        # Create variables for next action step and axiom step
        action_t = 2 * t + 1
        axiom_t = action_t + 1
        
        self.create_variables(action_t)   # Action timestep
        self.create_variables(axiom_t)    # Axiom timestep

        # Build substitutions for the new timesteps
        list_substitutions_actions = []
        list_substitutions_axioms = []
        list_substitutions_fluents = []
        
        for key in self.up_actions_to_z3.keys():
            list_substitutions_actions.append(
                (self.up_actions_to_z3[key][1],
                 self.up_actions_to_z3[key][action_t]))
        for key in self.up_axioms_to_z3.keys():
            list_substitutions_axioms.append(
                (self.up_axioms_to_z3[key][2],
                 self.up_axioms_to_z3[key][axiom_t])
            )
        for key in self.up_fluent_to_z3.keys():
            list_substitutions_fluents.append(
                (self.up_fluent_to_z3[key][0],
                 self.up_fluent_to_z3[key][action_t-1]))
            list_substitutions_fluents.append(
                (self.up_fluent_to_z3[key][1],
                 self.up_fluent_to_z3[key][action_t]))
            list_substitutions_fluents.append(
                (self.up_fluent_to_z3[key][2],
                 self.up_fluent_to_z3[key][axiom_t]))
 
        encoded_formula = dict()
        encoded_formula['initial']      = self.formula['initial']
        encoded_formula['goal']         = z3.substitute(self.formula['goal'], list_substitutions_fluents)
        encoded_formula['actions']      = z3.substitute(self.formula['actions'], list_substitutions_fluents + list_substitutions_actions)
        encoded_formula['axioms']       = z3.substitute(self.formula['axioms'], list_substitutions_fluents + list_substitutions_axioms)
        encoded_formula['frame_action'] = z3.substitute(self.formula['frame_action'], list_substitutions_fluents + list_substitutions_actions)
        encoded_formula['frame_axiom']  = z3.substitute(self.formula['frame_axiom'], list_substitutions_fluents + list_substitutions_axioms)
        if 'sem' in self.formula.keys():
            encoded_formula['sem'] = z3.substitute(self.formula['sem'], list_substitutions_actions)
        return encoded_formula

    def base_encode(self):
        """!
        Builds the encoding. Populates the formula dictionary class attribute,
        where all the "raw" formulas are stored. Those will later be used by 
        the encode function.
        """
        
        # Check for intentional fluents
        intentional_fluents = [
            f for f in self.task.fluents 
            if 'intends' in f.name or 'delegated' in f.name
        ]
        print(f"Intentional fluents found: {len(intentional_fluents)}")
        for f in intentional_fluents:
            print(f"  - {f.name}: {f.signature}")
        
        # create vars for first transition
        self.create_variables(0)
        self.create_variables(1)
        self.create_variables(2)
        self._populate_modifiers() # do indices

        initial_raw = z3.And(self.encode_initial_state())  # Encode initial state axioms
        goal_raw = z3.And(self.encode_goal_state())  # Encode goal state axioms
        actions_raw = z3.And(self.encode_actions())  # Encode universal axioms
        axioms_raw = z3.And(self.encode_axioms()) if len(self.task.axioms) > 0 else z3.BoolVal(True, ctx=self.ctx)
        frame_action_raw = z3.And(self.encode_frame_action())
        frame_axiom_raw = z3.And(self.encode_frame_axiom())

        # Translate any ASTs returned from other modules into our Z3 context
        self.formula['initial'] = self._translate_ast(initial_raw)
        self.formula['goal'] = self._translate_ast(goal_raw)
        self.formula['actions'] = self._translate_ast(actions_raw)
        self.formula['axioms'] = self._translate_ast(axioms_raw)
        self.formula['frame_action'] = self._translate_ast(frame_action_raw)
        self.formula['frame_axiom'] = self._translate_ast(frame_axiom_raw)

        execution_semantics = self.encode_execution_semantics()
        if len(execution_semantics) > 0:
            # Ensure any ASTs returned by modifiers are in our Z3 context
            exec_in_ctx = self._translate_ast(execution_semantics)
            self.formula['sem'] = z3.And(exec_in_ctx)  # Encode execution semantics (lin/par)

        metrics = self.encode_quality_metrics()
        if metrics:
            print(metrics)
            self.formula['metrics'] = [m["z3_var"] for m in metrics]

    def encode_execution_semantics(self):
        """!
        Encodes execution semantics as specified by the modifier class held.

        @returns: axioms that specify execution semantics.
        """
        action_vars = [self.up_actions_to_z3[key][1] for key in self.up_actions_to_z3.keys()] #TODO-Axiom: Understand
        return self.modifier.encode(self, action_vars)

    def create_variables(self, t):
        """!
        Creates state variables needed in the encoding for step t.

        @param t: the timestep
        """
        # increment the formula lenght
        self.formula_length += 1

        # Create action variables only at odd timesteps
        if t % 2 == 1:
            for grounded_action in self.task.actions:
                key   = str_repr(grounded_action)
                keyt  = str_repr(grounded_action, t)
                act_var = z3.Bool(keyt, ctx=self.ctx)
                self.up_actions_to_z3[key].append(act_var)
                self.z3_actions_to_up[act_var] = key
        else:
            # Pad with None for even timesteps
            for grounded_action in self.task.actions:
                key = str_repr(grounded_action)
                self.up_actions_to_z3[key].append(None)

        # Create axiom variables only at even timesteps (after t=0)
        if t > 0 and t % 2 == 0:
            for grounded_axiom in self.task.axioms:
                key = str_repr_axiom(grounded_axiom)
                keyt = f"{key}_t{t}"
                axiom_var = z3.Bool(keyt, ctx=self.ctx)
                self.up_axioms_to_z3[key].append(axiom_var)
                self.z3_axioms_to_up[axiom_var] = key
        else:
            # Pad with None for odd timesteps and t=0
            for grounded_axiom in self.task.axioms:
                key = str_repr_axiom(grounded_axiom)
                self.up_axioms_to_z3[key].append(None)

        # for fluents
        for fe in self.all_fluents:
            key  = str_repr(fe)
            keyt = str_repr(fe, t)
            if fe.type.is_real_type():
                self.up_fluent_to_z3[key].append(z3.Real(keyt, ctx=self.ctx))
            elif fe.type.is_bool_type():
                self.up_fluent_to_z3[key].append(z3.Bool(keyt, ctx=self.ctx))
            else:
                raise TypeError

    def _translate_ast(self, obj):
        """Ensure a Z3 AST (or list/tuple of ASTs) is in this encoder's context.

        If `obj` is a sequence, translate each element recursively. If it's a
        Z3 AST with a different ctx, translate it into `self.ctx`. Otherwise
        return it unchanged.
        """
        # sequences: translate each element
        if isinstance(obj, (list, tuple, set)):
            return [self._translate_ast(x) for x in obj]

        # Z3 AST-like objects have a `ctx` attribute and `translate` method
        try:
            if hasattr(obj, "ctx") and hasattr(obj, "translate"):
                if obj.ctx != self.ctx:
                    return obj.translate(self.ctx)
                return obj
        except Exception:
            # Be defensive: if anything goes wrong, fall back to returning obj
            return obj

        return obj

    def encode_initial_state(self):
        """!
        Encodes formula defining initial state
        @returns: Z3 formula asserting initial state
        """
        t = 0
        initial = []
        for FNode, initial_value in self.task.initial_values.items():
            fluent = self._expr_to_z3(FNode, t)
            value  = self._expr_to_z3(initial_value, t)
            initial.append(fluent == value)
        
        return initial

    def encode_goal_state(self):
        """!
        Encodes formula defining goal state
        22/7 - RP: Updated to remove t as param

        @returns: Z3 formula asserting propositional and numeric subgoals
        """
        goal = []
        for goal_pred in self.task.goals:
            goal.append(self._expr_to_z3(goal_pred, 2))
        return goal

    def encode_axioms(self):
        """
        Encode axioms that fire as triggers at even timesteps 
        Axioms should check preconditions at t=1 and apply effects at t=2.

        pre -> a
        a -> pre

        @returns: list of Z3 formulas asserting the axioms
        """
        axioms = []
        
        for grounded_axiom in self.task.axioms:
            key = str_repr_axiom(grounded_axiom)
            axiom_var = self.up_axioms_to_z3[key][2]  # Axiom fires at t=2
            
            # Preconditions checked at state after action (t=1)
            axiom_pre = []
            for pre in grounded_axiom.preconditions:
                axiom_pre.append(self._expr_to_z3(pre, 1))
            
            # Effects applied from t=1 to t=2
            axiom_eff = []
            for eff in grounded_axiom.effects:
                axiom_eff.append(self._expr_to_z3_axiom_effect(eff))
            
            axiom_pre = z3.And(axiom_pre) if len(axiom_pre) > 0 else z3.BoolVal(True, ctx=self.ctx)
            axiom_eff = z3.And(axiom_eff) if len(axiom_eff) > 0 else z3.BoolVal(True, ctx=self.ctx)
            axioms.append(z3.Implies(axiom_pre, axiom_var, ctx=self.ctx))
            axioms.append(z3.Implies(axiom_var, axiom_eff, ctx=self.ctx))
        return axioms
    
    def encode_actions(self):
        """!
        Encodes the transition function. That is, the actions.
        22/7 - RP: t removed as parameter as not needed
        a -> Pre
        a -> Eff

        @param t: the timestep
        @returns: list of Z3 formulas asserting the actions
        """
        actions = []
        for grounded_action in self.task.actions:
            key = str_repr(grounded_action)
            action_var = self.up_actions_to_z3[key][1]

            # translate the action precondition
            action_pre = []
            for pre in grounded_action.preconditions:
                action_pre.append(self._expr_to_z3(pre, 0))
            # translate the action effect
            action_eff = []
            for eff in grounded_action.effects:
               action_eff.append(self._expr_to_z3_action_effect(eff))

            # the proper encoding
            action_pre = z3.And(action_pre) if len(action_pre) > 0 else z3.BoolVal(True, ctx=self.ctx)
            actions.append(z3.Implies(action_var, action_pre, ctx=self.ctx))
            action_eff = z3.And(action_eff) if len(action_eff) > 0 else z3.BoolVal(True, ctx=self.ctx)
            actions.append(z3.Implies(action_var, action_eff, ctx=self.ctx))
        return actions

    def encode_frame_action(self):
        """Encode frame axioms for action transitions (t=0 to t=1)."""
        frame = []
        grounded_up_fluents = [f for f, _ in self.task.initial_values.items()]
        
        for grounded_fluent in grounded_up_fluents:
            key = str_repr(grounded_fluent)
            var_t0 = self.up_fluent_to_z3[key][0]
            var_t1 = self.up_fluent_to_z3[key][1]
            
            or_actions = []
            or_actions.extend(self.frame_add_action[key])
            or_actions.extend(self.frame_del_action[key])
            or_actions.extend(self.frame_num_action[key])
            
            if len(or_actions) == 0:
                who_can_change = z3.BoolVal(False, ctx=self.ctx)
            else:
                who_can_change = z3.Or([
                    z3.And(self._expr_to_z3(cond, 0), self.up_actions_to_z3[x][1])
                    for (cond, x) in or_actions
                ])
            
            frame.append(z3.Implies(var_t0 != var_t1, who_can_change, ctx=self.ctx))
        
        return frame

    def encode_frame_axiom(self):
        """Encode frame axioms for axiom transitions (t=1 to t=2)."""
        frame = []
        grounded_up_fluents = [f for f, _ in self.task.initial_values.items()]
        
        for grounded_fluent in grounded_up_fluents:
            key = str_repr(grounded_fluent)
            var_t1 = self.up_fluent_to_z3[key][1]
            var_t2 = self.up_fluent_to_z3[key][2]
            
            or_axioms = []
            or_axioms.extend(self.frame_add_axiom[key])
            or_axioms.extend(self.frame_del_axiom[key])
            or_axioms.extend(self.frame_num_axiom[key])
            
            if len(or_axioms) == 0:
                who_can_change = z3.BoolVal(False, ctx=self.ctx)
            else:
                # For axioms, the condition is implicit in the axiom preconditions
                who_can_change = z3.Or([
                    self.up_axioms_to_z3[x][2] for (_, x) in or_axioms
                ])
            
            frame.append(z3.Implies(var_t1 != var_t2, who_can_change, ctx=self.ctx))
        
        return frame
    
    def encode_quality_metrics(self):
        metrics = []
        
        if not hasattr(self.task, 'quality_metrics') or not self.task.quality_metrics:
            return metrics
        
        for metric in self.task.quality_metrics:
            metric_expr = metric.expression
            z3_metric = self._expr_to_z3(metric_expr, 2)
            
            metrics.append({
                'expression': z3_metric,
                'z3_var': z3_metric
            })
        
        return metrics

    def get_optimization_objective(self, horizon):
        if not hasattr(self.task, 'quality_metrics') or not self.task.quality_metrics:
            return [], []
        
        minimize_objs = []
        maximize_objs = []
        for metric in self.task.quality_metrics:
            metric_expr = metric.expression
            final_t = 2 * horizon + 2
            z3_metric = self._expr_to_z3(metric_expr, final_t)
            if str(metric).startswith('minimize'):
                minimize_objs.append(z3_metric)
            elif str(metric).startswith('maximize'):
                maximize_objs.append(z3_metric)
            else:
                raise ValueError(f"Unknown optimization direction for metric: {metric}")
        
        return minimize_objs, maximize_objs

    def _expr_to_z3(self, expr, t, c=None):
        """!
        Traverses a UP AST in in-order and converts it to a Z3 expression.
        @param expr: The tree expression node. (Can be a value, variable name, or operator)
        @param t: The timestep for the Fluents to be considered 
        @param c: The context, which can be used to take into account free params
        @returns: An equivalent Z3 expression
        """
        if isinstance(expr, int): # A python Integer
            return z3.IntVal(expr, ctx=self.ctx)
        elif isinstance(expr, bool): # A python Boolean
            return z3.BoolVal(expr, ctx=self.ctx)
        elif isinstance(expr, float): 
            return z3.RealVal(expr, ctx=self.ctx)

        elif isinstance(expr, Effect): # A UP Effect
            # This should not be called for effects anymore
            raise ValueError("Use _expr_to_z3_action_effect or _expr_to_z3_axiom_effect for effects")

        elif isinstance(expr, FNode): # A UP FNode ( can be anything really )
            if expr.is_object_exp(): # A UP object
                raise ValueError(f"{expr} should not be evaluated")
            elif expr.is_constant(): # A UP constant
                return expr.constant_value()
            elif expr.is_or():  # A UP or
                return z3.Or([self._expr_to_z3(x, t, c) for x in expr.args])
            elif expr.is_and():  # A UP and
                return z3.And([self._expr_to_z3(x, t, c) for x in expr.args])
            elif expr.is_fluent_exp(): # A UP fluent
                return self.up_fluent_to_z3[str_repr(expr)][t]
            elif expr.is_parameter_exp():
                raise ValueError(f"{expr} should not be evaluated")
            elif expr.is_lt():
                return self._expr_to_z3(expr.args[0], t, c) < self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_le():
                return self._expr_to_z3(expr.args[0], t, c) <= self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_times():
                return self._expr_to_z3(expr.args[0], t, c) * self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_div():
                return self._expr_to_z3(expr.args[0], t, c) / self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_plus():
                return z3.Sum([self._expr_to_z3(x, t, c) for x in expr.args])
            elif expr.is_minus():
                return self._expr_to_z3(expr.args[0], t, c) - self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_not():
                return z3.Not(self._expr_to_z3(expr.args[0], t, c))
            elif expr.is_equals():
                return self._expr_to_z3(expr.args[0], t, c) == self._expr_to_z3(expr.args[1], t, c)
            elif expr.is_implies():
                return z3.Implies(self._expr_to_z3(expr.args[0], t, c), self._expr_to_z3(expr.args[1], t, c))
            else:
                raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")
        elif isinstance(expr, Fraction):
            return z3.RealVal(f"{expr.numerator}/{expr.denominator}", ctx=self.ctx)
        else:
            raise TypeError(f"Unsupported expression: {expr} of type {type(expr)}")
    
    def _expr_to_z3_action_effect(self, expr):
        """Convert action effect (t=0 to t=1)."""
        eff = None
        if expr.kind == EffectKind.ASSIGN:
            eff = self._expr_to_z3(expr.fluent, 1) == self._expr_to_z3(expr.value, 0)
        elif expr.kind == EffectKind.DECREASE:
            eff = self._expr_to_z3(expr.fluent, 1) == self._expr_to_z3(expr.fluent, 0) - self._expr_to_z3(expr.value, 0)
        elif expr.kind == EffectKind.INCREASE:
            eff = self._expr_to_z3(expr.fluent, 1) == self._expr_to_z3(expr.fluent, 0) + self._expr_to_z3(expr.value, 0)
        
        if expr.is_conditional():
            return z3.Implies(self._expr_to_z3(expr.condition, 0), eff, ctx=self.ctx)
        return eff

    def _expr_to_z3_axiom_effect(self, expr):
        """Convert axiom effect (t=1 to t=2)."""
        eff = None
        if expr.kind == EffectKind.ASSIGN:
            eff = self._expr_to_z3(expr.fluent, 2) == self._expr_to_z3(expr.value, 1)
        elif expr.kind == EffectKind.DECREASE:
            eff = self._expr_to_z3(expr.fluent, 2) == self._expr_to_z3(expr.fluent, 1) - self._expr_to_z3(expr.value, 1)
        elif expr.kind == EffectKind.INCREASE:
            eff = self._expr_to_z3(expr.fluent, 2) == self._expr_to_z3(expr.fluent, 1) + self._expr_to_z3(expr.value, 1)
        
        if expr.is_conditional():
            return z3.Implies(self._expr_to_z3(expr.condition, 1), eff, ctx=self.ctx)
        return eff

class EncoderSequential(EncoderGrounded):
    """
    Implementation of the classical sequential encoding of Kautz & Selman 1992
    where each timestep can have exactly one action.
    """
    def __init__(self, task):
        super().__init__("seq", task, LinearModifier(), parallel=False)


class EncoderForall(EncoderGrounded):
    """
    Forall-step encoding, allowing parallelisation in a real-world manner by permitting multiple
    actions per step.
    """
    def __init__(self, task):
        super().__init__("parForall", task, 
                         ParallelModifier(MutexSemantics.FORALL, lazy=False), parallel=True)

class EncoderExists(EncoderGrounded):
    """
    Exists-step encoding allowing a more relaxed parallelisation than forall.
    """
    def __init__(self, task):
        super().__init__("parExists", task, 
                         ParallelModifier(MutexSemantics.EXISTS, lazy=False), parallel=True)

class EncoderForallLazy(EncoderGrounded):
    """
    Lazy Forall-step encoding, initially adds no interference mutexes and determines
    when to add them lazily
    """
    def __init__(self, task):
        super().__init__("parLazyForall", task, 
                         ParallelModifier(MutexSemantics.FORALL, lazy=True), parallel=True)

class EncoderExistsLazy(EncoderGrounded):
    """
    Lazy Exists-step encoding, initially adds no interference mutexes and determines
    when to add them lazily
    """
    def __init__(self, task):
        super().__init__("parLazyExists", task,
                         ParallelModifier(MutexSemantics.EXISTS, lazy=True), parallel=True)

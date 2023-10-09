#!/usr/bin/env python3

from collections import deque
from pydot import Dot, Edge, Node
from collections import deque
from itertools import chain, combinations
from fractions import Fraction
import math
from gurobipy import *
from z3 import *
import numpy as np

"""
follows from the exact formal definition of a NFA
which is a 5-tuple, (Q, Sigma, T, q0, F)
where T : Q x Sigma -> P(Q) (where P(Q) is the powerset of Q)

in this library we include epsilon transitions. e.g., we can
make a transition without consuming a character from a word.
"""
class NFA:
    def __init__(
        self,
        states, # set
        initial_state, # individual state
        acceptance_states, # set
        alphabet, # set 
        transitions=[]): # list of tuples

        self.states = sorted(list(set(states)))
        self.initial_state = initial_state
        self.alphabet = sorted(list(set(alphabet)))
        self.acceptance_states = sorted(list(set(acceptance_states)))
        self.transitions = transitions

        assert self.initial_state in self.states, "initial state not in states"

        # epsilon transitions are just empty transitions
        assert "" not in self.alphabet, "epsilon transitions (defined just as empty transitions, "") not allowed"

        self.alphabet.append('')

        for state in acceptance_states: 
            assert(state in self.states)

        # build the transition function object, e.g., T : State x Alphabet -> State
        self.transition_function_object = {}

        # build the reachability object, e.g., T : State -> Alphabet x State
        # (for each state, output the states that are reachable and by what character they're reachable by)
        self.reachability_object = {}
        self.reachability_object_reverse = {}

        # build state-to-transition objects
        # e.g., T : state -> transition
        self.end_state_transition_mapping = {}
        self.beginning_state_transition_mapping = {}

        for state in self.states:
            self.end_state_transition_mapping[state] = set()
            self.beginning_state_transition_mapping[state] = set()

            self.reachability_object[state] = set()
            self.reachability_object_reverse[state] = set()

            self.transition_function_object[state] = {}
            for char in alphabet:
                self.transition_function_object[state][char] = set()

            self.transition_function_object[state][''] = set()
        
        for (sA, i, sB) in self.transitions:
            # if sA not in self.states : print("DSADSADA",sA)
            assert(sA in self.states)
            assert(sB in self.states)
            assert(i in self.alphabet)

            self.transition_function_object[sA][i].add(sB)
            self.reachability_object[sA].add((sB, i))
            self.reachability_object_reverse[sB].add((sA, i))

            self.end_state_transition_mapping[sB].add((sA, i, sB))
            self.beginning_state_transition_mapping[sA].add((sA, i, sB))

    # state, char -> set of states
    # (e.g., the key feature of a NFA)
    def transition_function(self, state, char):
        assert state in self.states, "transition_function: state called not in NFA"
        assert char in self.alphabet, "transition_function: char not in alphabet"
        return list(self.transition_function_object[state][char])

    # returns a bool if two states are reachable between each other
    # e.g., seeing if there's a path between state1 and state2
    def reachable_between(self, state1, state2):
        assert state1 in self.states, "reachable_between: state1 not in NFA"
        assert state2 in self.states, "reachable_between: state2 not in NFA"
        if state1 == state2 : return True

        def reachable_between_a(self, state1, state2, visited):
            visited_loc = visited.copy()
            adjacent = self.reachability_object[state1]
            ret = False
            for (state, char) in adjacent:
                if state in visited : continue
                if state == state2 : return True
                visited_loc.add(state)
                ret = ret or reachable_between_a(self, state, state2, visited_loc)

            return ret

        return(reachable_between_a(self, state1, state2, set()))
    
    # returns a set of all reachable states from the current state
    # (need to test termination tho)
    def reachable_from(self, state):
        assert state in self.states, "all_reachable_from: state not in NFA"
        queue = []
        queue.append(state)
        visited = set()
        while queue:
            current_state = queue.pop(0)
            reachable = self.reachability_object[current_state]
            visited.add(current_state)
            for (state, char) in reachable:
                if state not in visited : queue.append(state)

        return visited

    # returns a set of all states that can move to current state
    # (need to test termination tho)
    def reachable_to(self, state):
        assert state in self.states, "all_reachable_from: state not in NFA"
        queue = []
        queue.append(state)
        visited = set()
        while queue:
            current_state = queue.pop(0)
            reachable = self.reachability_object_reverse[current_state]
            visited.add(current_state)
            for (state, char) in reachable:
                if state not in visited : queue.append(state)

        return visited

    def maximize_character_z3(self, char, init_state=None, usable_states=set()):
        assert char in self.alphabet

        if init_state == None : init_state = self.initial_state

        if usable_states == set():
            usable_states = self.reachable_from(init_state)
            assert init_state in usable_states
            acceptance_usable = set()
            for state in self.acceptance_states:
                #usable_states = usable_states.intersection(self.reachable_to(state))
                acceptance_usable = acceptance_usable.union(self.reachable_to(state))

            usable_states = usable_states.intersection(acceptance_usable)

        # for each reachable state, determine constraints
        # e.g., how many incoming and outgoing transitions does each state have?

        incoming_transitions = {}
        outgoing_transitions = {}

        for state in usable_states:
            incoming_transitions[state] = set()
            outgoing_transitions[state] = set()

        # create mapping from transitions to variable names
        # ok yeah this is jank bidirectional map whatever
        # could use this: https://stackoverflow.com/questions/1456373/two-way-reverse-map

        ts = []
        for transition in self.transitions:
            if transition[0] in usable_states and transition[2] in usable_states : ts.append(transition)

        transition_mapping = {}
        # transition_mapping_reverse = {}
        transition_labels = ["t" + str(i) for i in range(len(ts))]

        count = 0
        char_transitions = set()
        for (sA, i, sB) in ts:
            transition_mapping[transition_labels[count]] = (sA, i, sB)
            # transition_mapping_reverse[(sA, i, sB)] = transition_labels[count]

            if i == char : char_transitions.add(count)

            outgoing_transitions[sA].add("t" + str(count))
            incoming_transitions[sB].add("t" + str(count))

            count = count + 1

        """
        2. plug flow constraints into a sat solver
        """
        s = Optimize()
        tosolve = {}
        tomax = {}
        for t in transition_labels:
            r = str(t)[1:]
            if int(r) in char_transitions : tomax[t] = Int(str(t))
            tosolve[t] = Int(str(t))

        for t in tosolve.values():
            s.add(t >= 0)
            s.add(t <= len(ts) * 5)
        
        s.maximize(Sum(list(tomax.values())))

        # do sat solving thing by going through states and adding to solver 
        # as well as constraining the variables above 0 (and maybe add a height too?)

        accepting_states_to_solve = set()

        for state in usable_states:

            # for accepting states, outgoing + 1 = incoming 
            # e.g., one more incoming than outgoing
            if state in self.acceptance_states:
                #s.add(Sum([tosolve[t] for t in outgoing_transitions[state]]) + 1 == Sum([tosolve[t] for t in incoming_transitions[state]]) - 0)
                accepting_states_to_solve.add(state)

            # for initial states, outgoing = incoming + 1
            # e.g., one more outgoing than incoming
            elif state == init_state:
                s.add(Sum([tosolve[t] for t in outgoing_transitions[state]]) - 0 == Sum([tosolve[t] for t in incoming_transitions[state]]) + 1)
            else:
            # want to add [all outgoing] == [all incoming] to solver 
                s.add(Sum([tosolve[t] for t in outgoing_transitions[state]]) == Sum([tosolve[t] for t in incoming_transitions[state]]))

        # use or construction to constrain accepting states
        # e.g., if we have 2 accepting states, we only want our parikh image to consider the closer one
        s.add(Or([Sum([tosolve[t] for t in outgoing_transitions[state]]) + 1 == Sum([tosolve[t] for t in incoming_transitions[state]]) - 0 for state in accepting_states_to_solve]))

        if s.check() == sat:
            m = s.model()
            solns = {var: m[var] for var in tosolve.values()}
            
            char_total = {}
            for char in self.alphabet:
                char_total[char] = 0

            """
            3. for each character, sum the sat-outputted values of the transitions whose
                character is the aformentioned
            """
            for transition, value in solns.items():
                sA, i, sB = transition_mapping[str(transition)]
                char_total[i] += int(str(value))

            """
            4. put these values in a vector (or a map), and return
            """

            return char_total

        else:
            return {}

    def minimize_character_z3(self, char, init_state=None, usable_states=set()):
        assert char in self.alphabet

        if init_state == None : init_state = self.initial_state

        if usable_states == set():
            usable_states = self.reachable_from(init_state)
            assert init_state in usable_states
            acceptance_usable = set()
            for state in self.acceptance_states:
                #usable_states = usable_states.intersection(self.reachable_to(state))
                acceptance_usable = acceptance_usable.union(self.reachable_to(state))

            usable_states = usable_states.intersection(acceptance_usable)

        # for each reachable state, determine constraints
        # e.g., how many incoming and outgoing transitions does each state have?

        incoming_transitions = {}
        outgoing_transitions = {}

        for state in usable_states:
            incoming_transitions[state] = set()
            outgoing_transitions[state] = set()

        # create mapping from transitions to variable names
        # ok yeah this is jank bidirectional map whatever
        # could use this: https://stackoverflow.com/questions/1456373/two-way-reverse-map

        ts = []
        for transition in self.transitions:
            if transition[0] in usable_states and transition[2] in usable_states : ts.append(transition)

        transition_mapping = {}
        # transition_mapping_reverse = {}
        transition_labels = ["t" + str(i) for i in range(len(ts))]

        count = 0
        char_transitions = set()
        for (sA, i, sB) in ts:
            transition_mapping[transition_labels[count]] = (sA, i, sB)
            # transition_mapping_reverse[(sA, i, sB)] = transition_labels[count]

            if i == char : char_transitions.add(count)

            outgoing_transitions[sA].add("t" + str(count))
            incoming_transitions[sB].add("t" + str(count))

            count = count + 1

        """
        2. plug flow constraints into a sat solver
        """
        s = Optimize()
        tosolve = {}
        tomin = {}
        for t in transition_labels:
            r = str(t)[1:]
            if int(r) in char_transitions : tomin[t] = Int(str(t))
            tosolve[t] = Int(str(t))

        for t in tosolve.values():
            s.add(t >= 0)
            s.add(t <= len(ts) * 5)
        
        s.minimize(Sum(list(tomin.values())))

        # do sat solving thing by going through states and adding to solver 
        # as well as constraining the variables above 0 (and maybe add a height too?)

        accepting_states_to_solve = set()

        for state in usable_states:

            # for accepting states, outgoing + 1 = incoming 
            # e.g., one more incoming than outgoing
            if state in self.acceptance_states:
                #s.add(Sum([tosolve[t] for t in outgoing_transitions[state]]) + 1 == Sum([tosolve[t] for t in incoming_transitions[state]]) - 0)
                accepting_states_to_solve.add(state)

            # for initial states, outgoing = incoming + 1
            # e.g., one more outgoing than incoming
            elif state == init_state:
                s.add(Sum([tosolve[t] for t in outgoing_transitions[state]]) - 0 == Sum([tosolve[t] for t in incoming_transitions[state]]) + 1)
            else:
            # want to add [all outgoing] == [all incoming] to solver 
                s.add(Sum([tosolve[t] for t in outgoing_transitions[state]]) == Sum([tosolve[t] for t in incoming_transitions[state]]))

        # use or construction to constrain accepting states
        # e.g., if we have 2 accepting states, we only want our parikh image to consider the closer one
        s.add(Or([Sum([tosolve[t] for t in outgoing_transitions[state]]) + 1 == Sum([tosolve[t] for t in incoming_transitions[state]]) - 0 for state in accepting_states_to_solve]))

        if s.check() == sat:
            m = s.model()
            solns = {var: m[var] for var in tosolve.values()}
            
            char_total = {}
            for char in self.alphabet:
                char_total[char] = 0

            """
            3. for each character, sum the sat-outputted values of the transitions whose
                character is the aformentioned
            """
            for transition, value in solns.items():
                sA, i, sB = transition_mapping[str(transition)]
                char_total[i] += int(str(value))

            """
            4. put these values in a vector (or a map), and return
            """

            return char_total

        else:
            return {}
 
 

    def to_single_acceptance_state(self):
        """
        returns a NFA that accepts the same
        langauge as self, but only has a single accepting state
                    
        also, has the side effect of removing dead states via DFS
        """
        s = []
        s.append(self.initial_state)
        discovered=set()

        nfa_initial = self.initial_state
        nfa_alphabet = self.alphabet.copy()
        nfa_alphabet.remove('')
        nfa_acceptance_state = "s" + str(len(self.states) + 1)
        nfa_transitions = self.transitions.copy()

        while s != []:
            v = s.pop()
            if v not in discovered:
                if v in self.acceptance_states:
                    # nfa_transitions+=self.end_state_transition_mapping[v]
                    for (sA, i, sB) in self.end_state_transition_mapping[v]:
                        nfa_transitions.append((sA, i, nfa_acceptance_state))

                    # and, we don't add v to acceptance states of the created nfa
                discovered.add(v)
                reachable = self.reachability_object[v]
                for state, char in reachable : s.append(state)

        nfa_states = set(discovered)
        nfa_states.add(nfa_acceptance_state)

        # very costly solution... TODO: fix this
        tt = []
        for transition in nfa_transitions:
            if transition[0] in nfa_states and transition[2] in nfa_states:
                tt.append(transition)

        return NFA(nfa_states, nfa_initial, [nfa_acceptance_state], nfa_alphabet, tt)


    def get_min_of_char_gurobi(self, char, init_state=None, usable_states=set()):
        assert char in self.alphabet

        # step 1
        nfa = self.to_single_acceptance_state() # O(V+E), via DFS
        if init_state == None : init_state = nfa.initial_state
        nfa_acceptance_state = list(nfa.acceptance_states)[0]

        if usable_states == set():
            usable_states = nfa.reachable_from(init_state).intersection(nfa.reachable_to(nfa_acceptance_state))

        # step 2 and 3: extract flow constaints and make that shit

        incoming_transitions = {}
        outgoing_transitions = {}

        for state in usable_states:
            incoming_transitions[state] = set()
            outgoing_transitions[state] = set()

        ts = []
        for transition in nfa.transitions:
            if transition[0] in usable_states and transition[2] in usable_states : ts.append(transition)

        transition_mapping = {}
        transition_mapping_reverse = {}
        len_ts = len(ts)
        transition_labels = ["t" + str(i) for i in range(len_ts)]

        count = 0

        char_transitions = set()
        for (sA, i, sB) in ts:
            transition_mapping[transition_labels[count]] = (sA, i, sB)
            transition_mapping_reverse[(sA, i, sB)] = transition_labels[count]

            if i == char : char_transitions.add(count)

            outgoing_transitions[sA].add("t" + str(count))
            incoming_transitions[sB].add("t" + str(count))

            count = count + 1

        # put things into Ax = b matrix
        # A format: incoming -outgoing 

        A_matrix = []
        b_matrix = []

        itr = 0
        x_charstates = set()
        for state in usable_states:
            # handle A matrix
            r = [0] * len_ts * 2
            # e.g., for each state, get the input and output transitions, and put them into 
            # the array 
            incoming = incoming_transitions[state]
            outgoing = outgoing_transitions[state]

            for transition in incoming:
                #val = int(transition_mapping_reverse[transition][1:])
                val = int(transition[1:])
                if val in char_transitions : x_charstates.add(itr)
                r[val]+=1

            for transition in outgoing:
                #val = int(transition_mapping_reverse[transition][1:])
                val = int(transition[1:])
                if val in char_transitions : x_charstates.add(itr)
                r[val + len_ts]-=1

            # handle b matrix
            a = [0]
            if state == init_state : a[0] = -1
            elif state == nfa_acceptance_state : a[0] = 1
            else : a[0] = 0

            A_matrix.append(r)
            b_matrix.append(a)
            itr+=1

        model = Model('matrix1')

        A = np.matrix(A_matrix)
        b = np.matrix(b_matrix)

        num_constrs, num_vars = A.shape

        x = model.addVars(num_vars)
        model.update()

        for i in range(num_constrs):
            model.addConstr(quicksum(A[i, j]*x[j] for j in range(num_vars)) == b[i])

        # Specify objective: minimize the sum of x
        model.setObjective(x.sum(), GRB.MINIMIZE)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            solns = {}
            for char in nfa.alphabet : solns[char] = 0
            for v in model.getVars():
                tnum = int(v.varName[1:])
                if tnum >= len_ts : tnum-=len_ts
                solns[transition_mapping["t" + str(tnum)][1]]+= v.x
            return solns

        else : return {}


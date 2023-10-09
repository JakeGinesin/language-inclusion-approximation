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


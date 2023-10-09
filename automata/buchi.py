#!/usr/bin/env python3

from pydot import Dot, Edge, Node

from collections import deque

from z3 import *
from gurobipy import *
import numpy as np

class Buchi:
    def __init__(
        self,
        states,
        initial_state,
        acceptance_states,
        alphabet,
        transitions=[]):

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

    def transition_function(self, state, char):
        assert state in self.states, "transition_function: state called not in NFA"
        assert char in self.alphabet, "transition_function: char not in alphabet"
        return list(self.transition_function_object[state][char])

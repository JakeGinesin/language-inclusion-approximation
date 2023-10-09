#!/usr/bin/env python3

from automata.nfa import NFA
from automata.buchi import Buchi
from z3 import *
from tqdm import tqdm
from utils.algorithms import *

from gurobipy import *
import numpy as np

import heapq
import itertools

import os
import subprocess
import re

def read_ba_to_buchi(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize parameters for the Buchi object
    initial_state = None
    states = set()
    acceptance_states = set()
    transitions = []
    alphabet = set()

    # Parse initial state
    initial_state = lines[0].strip()
    states.add(initial_state)

    # Iterate over each line and parse it
    for line in lines[1:]:
        line = line.strip()

        # Parse transition
        if "->" in line:
            label, transition = line.split(",")
            src, dest = transition.split("->")
            src, dest = src.strip(), dest.strip()

            states.add(src)
            states.add(dest)
            alphabet.add(label)

            transitions.append((src, label, dest))

        # Parse acceptance state
        elif line:
            acceptance_states.add(line)

    return Buchi(states, initial_state, acceptance_states, alphabet, transitions)

# we can always interpret a buchi automata as a NFA, so... there's no problem with this
def read_ba_to_NFA(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize parameters for the Buchi object
    initial_state = None
    states = set()
    acceptance_states = set()
    transitions = []
    alphabet = set()

    # Parse initial state
    initial_state = lines[0].strip()
    states.add(initial_state)

    # Iterate over each line and parse it
    for line in lines[1:]:
        line = line.strip()

        # Parse transition
        if "->" in line:
            label, transition = line.split(",")
            src, dest = transition.split("->")
            src, dest = src.strip(), dest.strip()

            states.add(src)
            states.add(dest)
            alphabet.add(label)

            transitions.append((src, label, dest))

        # Parse acceptance state
        elif line:
            acceptance_states.add(line)

    return NFA(states, initial_state, acceptance_states, alphabet, transitions)


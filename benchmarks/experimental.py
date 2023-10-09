#!/usr/bin/env python3

from automata.nfa import NFA

nfa9_transitions = [
    ('s1', 'a', 's2'),
    ('s2', 'b', 's3'),
    ('s3', 'c', 's1'),
    ('s3', 'a', 's4')
]
nfa9 = NFA(["s1", "s2", "s3", "s4"], "s1", ["s4"], ["a","b","c"], nfa9_transitions)

print(nfa9.minimize_character_z3("a"))

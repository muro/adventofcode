#! /usr/bin/python3

import collections
import itertools
import math
import os
import re
import unittest
from typing import Any, Callable, List, NamedTuple, TypeVar

T = TypeVar('T')

def getInput(filename : str, parser : Callable[[str], T]) -> T:
  with open(os.path.join(os.path.dirname(__file__), 'input', filename)) as f:
    return parser(f.read())

def rotations(data: str) -> List[int]:
    return [int(s[1:]) if s.startswith('R') else -int(s[1:]) for s in data.split('\n')]


def count_zeros(rotations: List[int]) -> int:
    n = 50
    count = 0
    for r in rotations:
       n += r
       n = (n + 100) % 100
       if n == 0:
          count += 1
    return count

Solution = NamedTuple('Solution', [('filename', str), ('parser', Callable[[str], Any]), ('solve', Callable[[Any], Any])])

solutions = [
   Solution('1.txt', rotations, count_zeros)
]

# --------------- Tests --------------- #
class SolutionTest(unittest.TestCase):
  def testProblem1(self):
    check = lambda s: count_zeros(rotations(s))
    self.assertEqual(1, check('\n'.join(["R50", "L50"])))
    self.assertEqual(2, check('\n'.join(["R10"]*20)))
    self.assertEqual(2, check('\n'.join(["L10"]*20)))
    self.assertEqual(0, check('\n'.join(["R10", "L10"]*20)))
    self.assertEqual(0, check('\n'.join(["R10", "L10"]*20)))
    self.assertEqual(3, check('\n'.join(["L68", "L30", "R48", "L5", "R60", "L55", "L1", "L99", "R14", "L82"])))

# unittest.main()


for i in range(len(solutions)):
  s: Solution = solutions[i]
  print('Day %d:' % (i+1),
      s.solve(getInput(s.filename, s.parser)))
#! /usr/bin/python3

import collections
import itertools
import math
import os
import re
import unittest
from typing import Any, Callable, List, NamedTuple, Tuple, TypeVar

T = TypeVar('T')

def getInput(filename : str, parser : Callable[[str], T]) -> T:
  with open(os.path.join(os.path.dirname(__file__), 'input', filename)) as f:
    return parser(f.read())

def rotations(data: str) -> List[int]:
  return [int(s[1:]) if s.startswith('R') else -int(s[1:]) for s in data.split('\n')]


def landing_zeros(rotations: List[int]) -> int:
  return len(list(
    filter(lambda x: x % 100 == 0, itertools.accumulate([50] + rotations))))

def count_rot(n: int, r: int) -> Tuple[int, int]:
  count = 0
  if r >= 100:
    count += abs(r) // 100
    r = r % 100
  elif r <= -100:
    count += abs(r) // 100
    r = r + 100 * count
  if n != 0:
    if r < 0 and -r >= n:
      count += 1
    elif r < 0 and -(r + n) >= 100:
      count += 1
    elif r > 0 and r + n >= 100:
      count += 1
  n += r
  n = (n + 100) % 100
  return (n, count)

def passing_zeros(rotations: List[int]) -> int:
  n = 50
  count = 0
  for r in rotations:
    n, c = count_rot(n, r)
    count += c
  return count

def noop(data: Any) -> Any:
  return None

Solution = NamedTuple('Solution', [
   ('filename', str),
   ('parser', Callable[[str], Any]),
   ('part1', Callable[[Any], Any]),
   ('part2', Callable[[Any], Any])])

solutions = [
   Solution('1.txt', rotations, landing_zeros, passing_zeros)
]

def solve(s: Solution) -> Tuple[Any, Any]:
  return s.part1(getInput(s.filename, s.parser)), s.part2(getInput(s.filename, s.parser))

# --------------- Tests --------------- #
class SolutionTest(unittest.TestCase):
  def testProblem1(self):
    check: Callable[[str], int] = lambda s: landing_zeros(rotations(s))
    check2: Callable[[str], int] = lambda s: passing_zeros(rotations(s))

    self.assertEqual(1, check('\n'.join(["R50", "L50"])))
    self.assertEqual(2, check('\n'.join(["R10"] * 20)))
    self.assertEqual(2, check('\n'.join(["L10"] * 20)))
    self.assertEqual(0, check('\n'.join(["R10", "L10"] * 20)))
    self.assertEqual(0, check('\n'.join(["R10", "L10"] * 20)))

    self.assertEqual((5, 1), count_rot(10, -105))
    self.assertEqual((95, 2), count_rot(10, -115))

    self.assertEqual(10, check2('\n'.join(["R999"])))
    self.assertEqual(10, check2('\n'.join(["R1000"])))
    self.assertEqual(10, check2('\n'.join(["R1001"])))
    self.assertEqual(10, check2('\n'.join(["L999"])))
    self.assertEqual(10, check2('\n'.join(["L1000"])))
    self.assertEqual(10, check2('\n'.join(["L1001"])))

    example = '\n'.join(["L68", "L30", "R48", "L5", "R60", "L55", "L1", "L99", "R14", "L82"])
    self.assertEqual(3, check(example))

    self.assertEqual(6, check2(example))
    self.assertEqual((1132, 6623), solve(solutions[0]))

unittest.main()


for i in range(len(solutions)):
  s: Solution = solutions[i]
  print('Day %d:' % (i+1),
    s.part1(getInput(s.filename, s.parser)),
    s.part2(getInput(s.filename, s.parser)))
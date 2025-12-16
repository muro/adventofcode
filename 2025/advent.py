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

def is_invalid(n: int) -> bool:
  s = str(n)
  l = len(s)
  for i in range(1, (l // 2) + 1):
    d = l // i
    if s[:i] * d == s:
      return True
  return False

def twice(n: int) -> int:
  return int(str(n) + str(n)) if n > 0 else 0

def half(n: int) -> int:
  """Returns largest half-number of the format 123 where 123123 is a twice repeating number smaller or equal to n."""
  if n <= 10:
    return 0
  
  d = len(str(n))
  if d % 2 != 0:
    return int("9" * (d // 2))
  
  # take first half of the digits and duplicate:
  h = int(str(n)[:d//2])
  return h if twice(h) <= n else h -1

def ranges(data: str) -> List[Tuple[int, int]]:
  return [tuple(map(int, s.split("-"))) for s in data.split(",")]

def invalid_nrs(start, end):
  return [twice(i) for i in range(half(start), half(end)+1) if start <= twice(i) and twice(i) <= end]

def sum_invalids(lst: List[Tuple[int, int]]) -> int:
  return sum(sum(invalid_nrs(*p)) for p in lst)

def invalid2_nrs(start, end):
  return [i for i in range(start, end+1) if is_invalid(i)]

def sum_invalids2(lst: List[Tuple[int, int]]) -> int:
  return sum(sum(invalid2_nrs(*p)) for p in lst)

def jolts(s: str, i: int) -> int:
  return 10 ** i * int(max(s[:-i])) + jolts(s[1 + s.index(max(s[:-i])):], i - 1) if i > 0 else int(max(s))

def sum_jolts(d: List[str], i: int) -> int:
  return sum(jolts(s, i) for s in d)

sum_jolts_2 = lambda d: sum_jolts(d, 1)
sum_jolts_12 = lambda d: sum_jolts(d, 11)

def sum_jolts(d: List[str], i: int) -> int:
  return sum(jolts(s, i) for s in d)

def lines(data: str) -> List[str]:
  return data.splitlines()

Solution = NamedTuple('Solution', [
   ('filename', str),
   ('parser', Callable[[str], Any]),
   ('part1', Callable[[Any], Any]),
   ('part2', Callable[[Any], Any])])

solutions = [
   Solution('1.txt', rotations, landing_zeros, passing_zeros),
   Solution('2.txt', ranges, sum_invalids, sum_invalids2),
   Solution('3.txt', lines, sum_jolts_2, sum_jolts_12)
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

  def testProblem2(self):
    self.assertEqual(1, half(11))
    for i in range(11, 22):
      self.assertEqual(1, half(i))
    for i in range(100, 1010):
      self.assertEqual(9, half(i), f"twice for {i} should be 99")
    self.assertEqual([(1, 2), (22, 33)], ranges("1-2,22-33"))
    check: Callable[str, int] = lambda s: sum_invalids(ranges(s))
    check2: Callable[[str], int] = lambda s: sum_invalids2(ranges(s))
    self.assertEqual(11+22+33, check("1-43,100-1000"))
    self.assertEqual(11+22+33+1010+1111, check("1-43,100-1000,1000-1211"))
    example = "11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124"
    self.assertEqual(1227775554, check(example))
    self.assertEqual(4174379265, check2(example))

    self.assertTrue(is_invalid(11))
    self.assertTrue(is_invalid(22))
    self.assertTrue(is_invalid(111))
    self.assertFalse(is_invalid(12))
    self.assertEqual((54641809925, 73694270688), solve(solutions[1]))

  def testProblem3(self):
    self.assertEqual(98, jolts("987654321111111", 1))
    self.assertEqual(987654321111, jolts("987654321111111", 11))
    self.assertEqual(89, jolts("811111111111119", 1))
    self.assertEqual(811111111119, jolts("811111111111119", 11))
    self.assertEqual(78, jolts("234234234234278", 1))
    self.assertEqual(434234234278, jolts("234234234234278", 11))
    self.assertEqual(92, jolts("818181911112111", 1))
    self.assertEqual(888911112111, jolts("818181911112111", 11))
    example = ["987654321111111", "811111111111119", "234234234234278", "818181911112111"]
    self.assertEqual(357, sum_jolts_2(example))
    self.assertEqual(3121910778619, sum_jolts_12(example))
    self.assertEqual((17554, 175053592950232), solve(solutions[2]))
    

unittest.main()


for i in range(len(solutions)):
  s: Solution = solutions[i]
  print('Day %d:' % (i+1),
    s.part1(getInput(s.filename, s.parser)),
    s.part2(getInput(s.filename, s.parser)))
#! /usr/bin/python3

from __future__ import annotations
import collections
import itertools
import math
import os
import re
from typing import cast, Any, Callable, Collection, Deque, Dict, Generator, Iterator, List, NamedTuple, Pattern, Set, Tuple, Type, TypeVar
import unittest

T = TypeVar('T')

def getInput(filename : str, parser : Callable[[str], T]) -> T:
  with open(os.path.join(os.path.dirname(__file__), 'input', filename)) as f:
    return parser(f.read())

UNKNOWN = -7
def noop(data: Any) -> int:
  return UNKNOWN

def fixed(n: int) -> Callable[[Any], int]:
  return lambda d: n

def numbers(data : str) -> List[int]:
  return [int(l) for l in data.splitlines() if l]

def lines(data: str) -> List[str]:
  return data.splitlines()


# --------------- 1 --------------- #
def twoAddTo2020(data : List[int]) -> int:
  return next(i * j for i, j in itertools.combinations(data, 2) if i + j == 2020)

def threeAddTo2020(data : List[int]) -> int:
  return next(i*j*k for i, j, k in itertools.combinations(data, 3) if i + j + k == 2020)

# --------------- 2 --------------- #
PasswordEntry = NamedTuple('PasswordEntry', [('password', str), ('rule_char', str), ('min', int), ('max', int)])

def passwords(data : str) -> List[PasswordEntry]:
  def password(l : str) -> PasswordEntry:
    rule, pwd = l.split(':')
    pwd = pwd[1:]
    m = re.match(r'(\d+)\-(\d+)\s([a-z])', rule)
    assert m
    return PasswordEntry(pwd, m[3], int(m[1]), int(m[2]))
  return [password(l) for l in data.splitlines() if l]


def countValidFirstRule(ins: List[PasswordEntry]) -> int:
  def _isValidFirstRule(pe: PasswordEntry) -> bool:
    cc = pe.password.count(pe.rule_char)
    return cc >= pe.min and cc <= pe.max
  return len([1 for pe in ins if _isValidFirstRule(pe)])

def countValidSecondRule(ins: List[PasswordEntry]) -> int:
  def _isValidSecondRule(pe: PasswordEntry) -> bool:
    return (pe.password[pe.min-1] == pe.rule_char) ^ (pe.password[pe.max-1] == pe.rule_char)
  return len([1 for pe in ins if _isValidSecondRule(pe)])

# --------------- 3 --------------- #
def isTree(data: List[str], x: int, y: int) -> int:
  return y < len(data) and data[y][x % len(data[0])] == '#'

def slope(l: str) -> str:
  return l.strip()

def slopes(data: str) -> List[str]:
  return [slope(l) for l in data.splitlines() if l]

def slopeSteps(data: List[str], down: int, right: int) -> int:
  c = len(data)
  return sum(1 for x, y in zip(range(0, right*c, right), range(0, down*c, down)) if isTree(data, x, y))

def down1right3(data: List[str]) -> int:
  return slopeSteps(data, 1, 3)

def multiplySlopes(data: List[str]) -> int:
  return slopeSteps(data, 1, 1) * slopeSteps(data, 1, 3) * slopeSteps(data, 1, 5) * slopeSteps(data, 1, 7) * slopeSteps(data, 2, 1)

# --------------- 4 --------------- #
def passports(data: str) -> List[List[Tuple[str, str]]]:
  def passport(s: str) -> List[Tuple[str, str]]:
    return [tuple(e.split(':')) for e in s.split(' ') if e]
  return [passport(s.replace('\n', ' ')) for s in data.split('\n\n') if s]

def validPassports(data: List[List[Tuple[str, str]]]) -> int:
  def valid(p: List[Tuple[str, str]]) -> bool:
    attrs = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'}
    return attrs == {k for k, _ in p} | {'cid'}
  return sum(1 for p in data if valid(p))

def validPassports2(data: List[List[Tuple[str, str]]]) -> int:
  def valid(p: List[Tuple[str, str]]) -> bool:
    def hgt(v: str) -> bool:
      if len(v) < 3:
        return False
      if v[-2:] == 'in':
        return 59 <= int(v[:-2]) <= 76
      elif v[-2:] == 'cm':
        return 150 <= int(v[:-2]) <= 193
      else:
        return False

    attrs: Dict[str, Callable[[str], bool]] = {
      'byr': (lambda v: 1920 <= int(v) <= 2002),
      'iyr': (lambda v: 2010 <= int(v) <= 2020),
      'eyr': (lambda v: 2020 <= int(v) <= 2030),
      'hgt': hgt,
      'hcl': (lambda v: re.match(r'^#[0-9a-z]{6}$', v) is not None),
      'ecl': (lambda v: v in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']),
      'pid': (lambda v: re.match(r'^[0-9]{9}$', v) is not None),
      'cid': (lambda _: True)}
    for (k, v) in p:
      if k not in attrs or not attrs[k](v):
        return False
      del attrs[k]
    if 'cid' in attrs:
      del attrs['cid']
    return not attrs
  return sum(1 for p in data if valid(p))

# --------------- 5 --------------- #
def boardingPasses(data: str) -> List[Tuple[int, int]]:
  def bp(b: str) -> Tuple[int, int]:
    return (sum(2 ** (6-i) for i, c in enumerate(b[:7]) if c == 'B'),
            sum(2 ** (2-i) for i, c in enumerate(b[-3:]) if c == 'R'))
  return [bp(l) for l in data.split('\n') if l]

def highestSeatId(data: List[Tuple[int, int]]) -> int:
  return max(8*r + c for r, c in data)

def missingSeatId(data: List[Tuple[int, int]]) -> int:
  ids = sorted([8*r + c for r, c in data])
  return [k + 1 for k, l in zip(ids, ids[1:]) if k + 1 != l][0]

# --------------- 6 --------------- #
def answers(data: str) -> List[List[Set[str]]]:
  return [[set(l) for l in s.splitlines()] for s in data.split('\n\n') if s]

def anyYesses(data: List[List[Set[str]]]) -> int:
  s: Set[str] = set()
  return sum(len(s.union(*part)) for part in data)

def allYesses(data: List[List[Set[str]]]) -> int:
  return sum(len(part[0].intersection(*part)) for part in data)

# --------------- 7 --------------- #

# vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
# faded blue bags contain no other bags.
def bagcontents(line: str) -> Tuple[str, List[Tuple[int, str]]]:
  # returns pair (bag, (count, bag types))
  bag, contents = line.split(" bags contain ")
  bagtypes: List[Tuple[int, str]] = []
  for what in contents.split(', '):
    if not what:
      continue
    md = re.match(r'(\d+) (\w+ \w+) bags?\.?', what)
    if md:
      bagtypes.append((int(md.group(1)), md.group(2)))
  return (bag, bagtypes)

def bags(data: str) -> Dict[str, List[Tuple[int, str]]]:
  return dict(bagcontents(l) for l in data.splitlines() if l)

def whereCanTheBagBe(data: Dict[str, List[Tuple[int, str]]]) -> int:
  colors: Set[str] = set()
  bags_to_check = collections.deque(['shiny gold'])
  bags_checked: Set[str] = set()
  while bags_to_check:
    checking = bags_to_check.pop()
    bags_checked.add(checking)
    for bag, contents in data.items():
      if bag in bags_checked:
        continue
      for c in contents:
        if c[1] == checking:
          bags_to_check.append(bag)
          colors.add(bag)
  return len(colors)

def howManyBags(data: Dict[str, List[Tuple[int, str]]]) -> int:
  bags_to_add: Collection[Tuple[str, int]] = collections.deque([('shiny gold', 1)])
  total = 0
  while bags_to_add:
    bag, count = bags_to_add.pop()
    total += count
    contents = data[bag]
    for c in contents:
      bags_to_add.append((c[1], c[0]*count))
  return total - 1

# --------------- 8 --------------- #
def instructions(data: str) -> List[Tuple[str, int]]:
  def _instruction(s: str) -> Tuple[str, int]:
    inst, number = s.split(' ')
    return (inst, int(number))
  return [_instruction(l) for l in data.splitlines() if l]

def accumulatorBeforeCycle(data: List[Tuple[str, int]]) -> int:
  visited : Set[int] = set()
  a = 0
  pc = 0
  while pc not in visited:
    inst, num = data[pc]
    visited.add(pc)
    if inst == "nop":
      pc += 1
    elif inst == "acc":
      a += num
      pc += 1
    elif inst == "jmp":
      pc += num
  return a

def accumulatorOnFixedLoop(data: List[Tuple[str, int]]) -> int:
  org_data = data[:]
  # replace single operation
  for i, (inst, num) in enumerate(data):
    if inst == "acc":
      continue
    data = org_data[:]
    data[i] = ("nop" if org_data[i][0] == "jmp" else "jmp", org_data[i][1])

    a = 0
    pc = 0
    visited = set([len(data)])
    while pc not in visited:
      inst, num = data[pc]
      visited.add(pc)
      if inst == "nop":
        pc += 1
      elif inst == "acc":
        a += num
        pc += 1
      elif inst == "jmp":
        pc += num
    if pc == len(data):
      return a
  return 0

# --------------- 9 --------------- #
def firstNotSumOf2(data: List[int]) -> int:
  for n in range(25, len(data)):
    try:
      _ = next(i+j for i, j in itertools.combinations(data[n-25:n], 2) if i + j == data[n])
    except StopIteration:
      return data[n]
  assert False

def encryptionWeakness(data: List[int], pre: int=25) -> int:
  def _firstNotSumOf2(data: List[int], pre: int) -> Tuple[int, int]:
    for n in range(pre, len(data)):
      try:
        next(i+j for i, j in itertools.combinations(data[n-pre:n], 2) if i + j == data[n])
      except StopIteration:
        return n, data[n]
    assert False
  n, s = _firstNotSumOf2(data, pre)
  for i in range(n):
    for j in range(i, n):
      if sum(data[i:j]) == s:
        return min(data[i:j]) + max(data[i:j])
  assert False

# --------------- 10 --------------- #
def jolts(data: str) -> List[int]:
  nums = numbers(data)
  return [0] + sorted(nums) + [max(nums) + 3]

def allJolts(nums: List[int]) -> int:
  c = collections.Counter(t - o for o, t in zip(nums, nums[1:]))
  return c[1] * c[3]

def joltArrangements(nums: List[int]) -> int:
  # check up to last 3 items:
  paths = [1]
  for i in range(1, len(nums)):
    prevs = max(i-3, 0)
    paths.append(
      sum(p for n, p in zip(nums[prevs:i], paths[prevs:]) if n >= nums[i] - 3))
  return paths[-1]

# --------------- 11 --------------- #
def seats(data: str) -> List[List[str]]:
  return [[c for c in l] for l in data.splitlines()]

def stableOccupancy(onlyNeighbors: bool, data: List[List[str]], p: bool=False) -> int:
  maxi = len(data)
  maxj = len(data[0])

  def countOccupied(i: int, j: int) -> int:
    def occupied(i: int, j: int) -> int:
      return 0 <= i < maxi and 0 <= j < maxj and data[i][j] == '#'
    return len([1 for (i_, j_) in itertools.product(range(i-1, i+2), range(j-1, j+2)) if (i_ != i or j_ != j) and occupied(i_, j_)])

  def countVisibleOccupied(i: int, j: int) -> int:
    def visibleOccupied(dx: int, dy: int) -> bool:
      i_ = i + dx
      j_ = j + dy
      while 0 <= i_ < maxi and 0 <= j_ < maxj:
        if data[i_][j_] == 'L':
          return False
        elif data[i_][j_] == '#':
          return True
        i_ += dx
        j_ += dy
      return False
    return len([1 for (x, y) in itertools.product(range(-1, 2), repeat=2) if (x != 0 or y != 0) and visibleOccupied(x, y)])

  count = countOccupied if onlyNeighbors else countVisibleOccupied
  needEmpty = 4 if onlyNeighbors else 5

  def applyRule(i: int, j: int) -> str:
    if data[i][j] == 'L' and count(i, j) == 0:
      return '#'
    elif data[i][j] == '#' and count(i, j) >= needEmpty:
      return 'L'
    return data[i][j]

  def applyAll() -> List[List[str]]:
    nd: List[List[str]] = []
    for i in range(maxi):
      nd.append([applyRule(i, j) for j in range(maxj)])
    return nd

  n = 0
  d2 = applyAll()
  while n < 10000 and d2 != data:
    data = d2
    d2 = applyAll()
    n += 1
  return len([d for sublist in data for d in sublist if d == '#'])

def stableOccupancyNeighbors(data: List[List[str]]):
  return stableOccupancy(True, data)

def stableOccupancyVisible(data: List[List[str]], p: bool=False):
  return stableOccupancy(False, data, p)

# --------------- 12 --------------- #
def directions(data: str) -> List[Tuple[str, int]]:
  return [(d[0], int(d[1:])) for d in data.splitlines()]

def manhattanDistanceMovedShip(data: List[Tuple[str, int]]) -> int:
  def moveShip() -> Tuple[float, float]:
    d: int = 0
    x: float = 0
    y: float = 0
    for c, n in data:
      if c == 'N':
        y += n
      if c == 'S':
        y -= n
      if c == 'E':
        x += n
      if c == 'W':
        x -= n
      if c == 'L':
        d -= n
      if c == 'R':
        d += n
      if c == 'F':
        x += math.cos(math.radians(d)) * n
        y -= math.sin(math.radians(d)) * n
    return x, y
  x, y = moveShip()
  return int(abs(x) + abs(y))

def manhattanDistanceMovedWaypoint(data: List[Tuple[str, int]]) -> int:
  def moveByWaypoint() -> Tuple[float, float]:
    wx: float = 10
    wy: float = 1
    x = 0
    y = 0

    def degrees() -> float:
      if wx == 0:
        return 0 if wy > 0 else 180
      dg = math.degrees(math.atan(wy / wx))
      if wx < 0:
        return 180 + dg
      return 360 + dg if wy < 0 else dg

    def rotate(n: int) -> Tuple[float, float]:
      dg: float = degrees() + n
      d: float = math.sqrt(wx * wx + wy * wy)
      return math.cos(math.radians(dg)) * d, math.sin(math.radians(dg)) * d

    for c, n in data:
      if c == 'N':
        wy += n
      if c == 'S':
        wy -= n
      if c == 'E':
        wx += n
      if c == 'W':
        wx -= n
      if c == 'L':
        wx, wy = rotate(n)
      if c == 'R':
        wx, wy = rotate(-n)
      if c == 'F':
        x += wx * n
        y += wy * n
    return x, y
  x, y = moveByWaypoint()
  return int(abs(x) + abs(y))

# --------------- 13 --------------- #
def schedule(data: str) -> Tuple[int, List[str]]:
  arrival, schedules = data.splitlines()
  return int(arrival), schedules.split(',')

def busDepartureById(data: Tuple[int, List[str]]) -> int:
  arrival = data[0]
  schedules = [int(s) for s in data[1] if s != 'x']

  def nextDeparture(s : int) -> int:
    return arrival if (arrival % s) == 0 else s + arrival - (arrival % s)

  mn = (schedules[0], schedules[0])
  for s in schedules:
    wait = nextDeparture(s) - arrival
    if wait < mn[0]:
      mn = (wait, s)
  return mn[0] * mn[1]

def fittingSchedule(data: Tuple[int, List[str]]) -> int:
  schedules: List[Tuple[int, int]] = []
  for i, d in enumerate(data[1]):
    if d != 'x':
      schedules.append((int(d), i))

  current = schedules[0][0]
  multiplier = current
  for s, o in schedules[1:]:
    # find first common ts:
    while ((current + o) % s) != 0:
      current += multiplier
    multiplier *= s
  return current

# --------------- 14 --------------- #
def sumValuesAfterBitmask(data: List[str]) -> int:
  memory: Dict[int, int] = collections.defaultdict()
  ormask = 0
  andmask = 0
  for inst in data:
    if inst.startswith('mask = '):
      maskstr = inst.split(' ')[2]
      ormask = int(maskstr.replace('X', '0'), 2)
      andmask = int(maskstr.replace('X', '1'), 2)
    else:
      md = re.match(r'mem\[(\d+)\] = (\d+)', inst)
      assert md
      addr, value = int(md[1]), int(md[2])
      memory[addr] = (value & andmask) | ormask
  return sum(memory.values())

def sumValuesAfterMemBitmask(data: List[str]) -> int:
  def genAddr(n : int) -> List[int]:
    if (n == 0):
      return []
    # find highest bit:
    maxBit = max([k for k in range(36) if 2 ** k <= n])
    if 2 ** maxBit == n:
      return [0, n]
    inner = genAddr(n & ~(2 ** maxBit))
    return inner + [(2 ** maxBit) | i for i in inner ]

  memory: Dict[int, int] = collections.defaultdict()
  ormask = 0
  xmask = 0
  for inst in data:
    if inst.startswith('mask = '):
      maskstr = inst.split(' ')[2]
      ormask = int(maskstr.replace('X', '0'), 2)
      xmask = int(maskstr.replace('1', '0').replace('X', '1'), 2)
    else:
      md = re.match(r'mem\[(\d+)\] = (\d+)', inst)
      assert md
      addr, value = int(md[1]), int(md[2])
      addr = (addr | ormask) & (~xmask)
      for i in genAddr(xmask):
        memory[addr | i] = value
  return sum(memory.values())

# --------------- 15 --------------- #
def game(data: List[int], nth: int) -> int:
  last = {}
  for i, n in enumerate(data[:-1]):
     # skip last item to check its previous last position below
    last[n] = i
  prev = data[-1]
  for i in range(len(data), nth):
    n: int = i - last[prev] - 1 if prev in last else 0
    last[prev] = i - 1
    prev = n
  return prev

def game2020(data: List[int]) -> int:
  return game(data, 2020)

def game3(data: List[int]) -> int:
  return game(data, 30000000)

# --------------- 16 --------------- #
def trainTickets(data: str) -> Tuple[Dict[str, List[Tuple[int, int]]], List[int], List[List[int]]]:
  parts = data.split("\n\n")
  rules_str = parts[0].splitlines()
  rules: Dict[str, List[Tuple[int, int]]] = {}
  for s in rules_str:
    name, rs = s.split(": ")
    rules_nums = [(int(r.split('-')[0]), int(r.split('-')[1])) for r in rs.split(' or ')]
    rules[name] = rules_nums
  yourTicket = [int(n) for n in parts[1].splitlines()[1].split(',')]
  nearbyTickets = [[int(i) for i in values.split(",")] for values in parts[2].splitlines()[1:]]
  return rules, yourTicket, nearbyTickets

def isValid(n: int, rules: Dict[str, List[Tuple[int, int]]]) -> bool:
  return any(True for rule in rules.values() for l, h in rule if n >= l and n <= h)

def sumInvalidValues(data: Tuple[Dict[str, List[Tuple[int, int]]], List[int], List[List[int]]]) -> int:
  rules, _yourTicket, nearbyTickets = data
  return sum(n for ticket in nearbyTickets for n in ticket if not isValid(n, rules))

def mulDepartureValues(data: Tuple[Dict[str, List[Tuple[int, int]]], List[int], List[List[int]]]) -> int:
  rules, yourTicket, nearbyTickets = data
  validTickets = [ticket for ticket in nearbyTickets if all(isValid(n, rules) for n in ticket)]
  validRules = [list(rules.keys()) for _ in yourTicket]
  for ticket in validTickets:
    for i, n in enumerate(ticket):
      validRules[i] = [name for name in validRules[i] if isValid(n, {name: rules[name]})]
  # apply constraints - remove single-rule for field from all other rules:
  removedRule = True
  while removedRule:
    removedRule = False
    for i, _ in enumerate(validRules):
      if len(validRules[i]) == 1:
        name = validRules[i][0]
        for k, _ in enumerate(validRules):
          if k == i or name not in validRules[k]:
            continue
          validRules[k].remove(name)
          removedRule = True
  total = 1
  for i, t in enumerate(yourTicket):
    if validRules[i][0].startswith('departure'):
      total *= t
  return total

# --------------- 17 --------------- #
def grid(data: str) -> Set[Tuple[int, int, int, int]]:
  g: Set[Tuple[int, int, int, int]] = set()
  for y, l in enumerate(data.splitlines()):
    for x, c in enumerate(l):
      if c == '#':
        g.add((x, y, 0, 0))
  return g

def print_grid(data: Set[Tuple[int, int, int, int]]) -> None:
  sg = sorted(list(data))
  print(sg)

def count_neighbors3(data: Set[Tuple[int, int, int, int]], x: int, y: int, z: int) -> int:
  return len([True for (i, j, k) in itertools.product([-1, 0, 1], repeat=3) if (i != 0 or j != 0 or k != 0) and (x+i, y+j, z+k, 0) in data])

def count_neighbors4(data: Set[Tuple[int, int, int, int]], x: int, y: int, z: int, w: int) -> int:
  return len([True for (i, j, k, l) in itertools.product([-1, 0, 1], repeat=4) if (i != 0 or j != 0 or k != 0 or l != 0) and (x+i, y+j, z+k, w+l) in data])

def next_iter3(data: Set[Tuple[int, int, int, int]]):
  return {(x, y, z, 0) for (x, y, z) in itertools.product(range(-6, 15), repeat=3) if count_neighbors3(data, x, y, z) == 3 or ((x, y, z, 0) in data and count_neighbors3(data, x, y, z) == 2)}

def next_iter4(data: Set[Tuple[int, int, int, int]]):
  return {(x, y, z, w) for (x, y, z, w) in itertools.product(range(-6, 15), repeat=4) if count_neighbors4(data, x, y, z, w) == 3 or ((x, y, z, w) in data and count_neighbors4(data, x, y, z, w) == 2)}

def life3(data: Set[Tuple[int, int, int, int]], iterations: int) -> int:
  for _ in range(iterations):
    data = next_iter3(data)
  return len(data)

def life4(data: Set[Tuple[int, int, int, int]], iterations: int) -> int:
  for _ in range(iterations):
    data = next_iter4(data)
  return len(data)

def life6_3(data: Set[Tuple[int, int, int, int]]) -> int:
  return life3(data, 6)

def life6_4(data: Set[Tuple[int, int, int, int]]) -> int:
  return life4(data, 6)

# --------------- 18 --------------- #
class ExpressionNode:
  def eval(self) -> int:
    return 0

class Operator(ExpressionNode):
  def __init__(self, op: str):
    self.op = op
    self.left: ExpressionNode | None = None
    self.right: ExpressionNode | None = None

  def eval(self) -> int:
    assert self.left and self.right
    if self.op == '+':
      return self.left.eval() + self.right.eval()
    elif self.op == '-':
      return self.left.eval() - self.right.eval()
    elif self.op == '*':
      return self.left.eval() * self.right.eval()
    elif self.op == '/':
      return int(self.left.eval() / self.right.eval())
    assert False, 'Unexpected operation: %s' % self.op

  def __repr__(self) -> str:
    s = "<"
    if self.left:
      s += self.left.__repr__()
    s += self.op
    if self.right:
      s += self.right.__repr__()
    s += ">"
    return s

class Addition(Operator):
  def __init__(self):
    super().__init__('+')

class Multiplication(Operator):
  def __init__(self):
    super().__init__('*')

class Number(ExpressionNode):
  def __init__(self, num: int):
    self.num = num

  def eval(self) -> int:
    return self.num

  def __repr__(self) -> str:
    return str(self.num)

class OpenParen(ExpressionNode):
  def __init__(self):
    pass

  def __repr__(self) -> str:
    return '('

class CloseParen(ExpressionNode):
  def __repr__(self) -> str:
    return ')'

class End(ExpressionNode):
  def __repr__(self) -> str:
    return '.'

def tokenize(e: str) -> Generator[ExpressionNode, None, None]:
  current: str = ''
  for c in e:
    if c == ' ' and current:
      yield Number(int(current))
      current = ''
    elif c in '1234567890':
      if not current:
        current = c
      else:
        current += c
    elif c == '+':
      yield Addition()
    elif c == '*':
      yield Multiplication()
    elif c == '(':
      yield OpenParen()
    elif c == ')':
      if current:
        yield Number(int(current))
        current = ''
      yield CloseParen()
  if current:
    yield Number(int(current))
    current = ''
  yield End()

def evalLtr(e: str) -> int:
  def getNode(token_iter: Iterator[ExpressionNode]) -> ExpressionNode:
    n = next(token_iter)
    if isinstance(n, Number):
      return n
    elif isinstance(n, OpenParen):
      return subtree(token_iter, endWhen=CloseParen)
    assert False, 'Unexpected node instance type: %s' % type(n)

  def subtree(token_iter: Iterator[ExpressionNode], endWhen: Type[ExpressionNode]=End) -> ExpressionNode:
    left = getNode(token_iter)
    o = next(token_iter)
    while not isinstance(o, endWhen):
      o.left = left
      o.right = getNode(token_iter)
      left = o
      o = next(token_iter)
    return left

  tokens = list(tokenize(e))
  expr = subtree(iter(tokens), End)
  return expr.eval()

def evalAddBeforeMul(e: str) -> int:
  def getNode(token_iter: Iterator[ExpressionNode]) -> ExpressionNode:
    n = next(token_iter)
    if isinstance(n, Number):
      return n
    elif isinstance(n, OpenParen):
      return subtree(token_iter, endWhen=CloseParen)
    assert False, 'Unexpected node instance type: %s' % type(n)

  def subtree(token_iter: Iterator[ExpressionNode], endWhen: Type[ExpressionNode]=End) -> ExpressionNode:
    nodes: List[ExpressionNode] = [getNode(token_iter)]
    op = next(token_iter)
    while not isinstance(op, endWhen):
      nodes.append(op)
      node = getNode(token_iter)
      nodes.append(node)
      op = next(token_iter)

    # convert array to tree:
    # first, merge all additions
    l = len(nodes) + 1 # ensure cycle executes at least once
    while l > len(nodes):
      l = len(nodes)
      # merge an addition into a node
      empty_addition_index = None
      for i, n in enumerate(nodes):
        if isinstance(n, Addition) and n.left is None and n.right is None:
          empty_addition_index = i
          break
      if empty_addition_index is not None:
        cast(Operator, nodes[empty_addition_index]).left = nodes[empty_addition_index-1]
        cast(Operator, nodes[empty_addition_index]).right = nodes[empty_addition_index+1]
        del nodes[empty_addition_index+1]
        del nodes[empty_addition_index-1]
    # now merge everything left:
    while len(nodes) > 1:
      cast(Operator, nodes[1]).left = nodes[0]
      cast(Operator, nodes[1]).right = nodes[2]
      del nodes[2]
      del nodes[0]
    return nodes[0]

  tokens = list(tokenize(e))
  expr = subtree(iter(tokens), End)
  return expr.eval()

def addBeforeMul(data: List[str]) -> int:
  return sum(evalAddBeforeMul(e) for e in data)

def ltrPrecedence(data: List[str]) -> int:
  """Returns sum of operations, where precedence is left-to-right."""
  return sum(evalLtr(e) for e in data)

# --------------- 19 --------------- #
class BaseNode:
  def final(self) -> bool:
    return True

  def value(self) -> str:
    return ''

  def __repr__(self) -> str:
    return 'BaseNode - None'

  def update(self, rules: Rules) -> BaseNode:
    return self

Rules = Dict[int, BaseNode]

class Text(BaseNode):
  def __init__(self, text :str):
    self.text: str = text

  def final(self) -> bool:
    return True

  def value(self) -> str:
    return self.text

  def __repr__(self) -> str:
    return self.text

  def update(self, rules: Rules) -> BaseNode:
    return self

class Reference(BaseNode):
  def __init__(self, n: int):
    self.n = n

  def final(self) -> bool:
    return False

  def value(self) -> str:
    return ''

  def __repr__(self) -> str:
    return str(self.n)

  def update(self, rules: Rules) -> BaseNode:
    return rules[self.n] if rules[self.n].final() else self

class Or(BaseNode):
  def __init__(self, alternatives: List[List[BaseNode]]):
    self.alternatives = alternatives
    self._final: bool = False
    self.val: str = ''

  def final(self) -> bool:
    if not self._final:
      self._final = all(all(a.final() for a in alt) for alt in self.alternatives)
    return self._final

  def value(self) -> str:
    if self.final() and not self.val:
      self.val = '(' + '|'.join('(' + ''.join(v.value() for v in a) + ')' for a in self.alternatives) + ')'
    return self.val

  def __repr__(self) -> str:
    return '|'.join(a.__repr__() for a in self.alternatives)

  def update(self, rules: Rules) -> BaseNode:
    for a in self.alternatives:
      for i in range(len(a)):
        a[i] = a[i].update(rules)
    return self

def messages(data: str) -> Tuple[Rules, List[str]]:
  d = data.split("\n\n")
  rules: Rules = {}
  for rule in d[0].splitlines():
    n, text = rule.split(': ')
    if '"' in text:
      rules[int(n)] = Text(text.split('"')[1])
      continue
    rs = Or([[Reference(int(n)) for n in ps.split(' ')] for ps in text.split(' | ')])
    rules[int(n)] = rs
  msgs = d[1].splitlines()
  return rules, msgs

def compile_regexp(rules: Rules) -> Pattern[str]:
  while not rules[0].final():
    for r in rules:
      rules[r].update(rules)
  return re.compile(rules[0].value())

def matchingMessages(data: Tuple[Dict[int, BaseNode], List[str]]) -> int:
  rules, msgs = data
  regexp = compile_regexp(rules)
  return sum(1 for m in msgs if regexp.fullmatch(m))

def matchingMessagesWithLoops(data: Tuple[Dict[int, BaseNode], List[str]]) -> int:
  rules, msgs = data
  for i in range(2, 6):
    cast(Or, rules[8]).alternatives.append([Reference(42)] * i)
  for i in range(2, 5):
    cast(Or, rules[11]).alternatives.append([cast(BaseNode, Reference(42))] * i + [cast(BaseNode, Reference(31))] * i)
  regexp = compile_regexp(rules)
  return sum(1 for m in msgs if regexp.fullmatch(m))

# --------------- 20 --------------- #
def flip(nr: int) -> int:
  res = 0
  b = 1
  rb = 512
  for _ in range(10):
    if nr & b:
      res |= rb
    b <<= 1
    rb >>= 1
  return res

class FlipTest(unittest.TestCase):
  def testFlip(self):
    self.assertEqual(0, flip(0))
    for i in range(1024):
      self.assertEqual(i, flip(flip(i)))
    self.assertEqual(1, flip(512))
    self.assertEqual(2, flip(256))
    self.assertEqual(1020, flip(255))


def reorientTile(itl: List[str], o: int) -> List[str]:
  def rotate(t: List[str]) -> List[str]:
    rt: List[str] = []
    for j in range(len(t[0])):
      rt.append(''.join(t[i][j] for i in range(len(t))))
    rt.reverse()
    return rt

  tl = itl[:]
  if o >= 4:
    tl.reverse()
    o -= 4
  while o > 0:
    tl = rotate(tl)
    o -= 1
  return tl

class Tile(object):
  """Tile represents a single tile.

    edges: [4 edges, 4 flipped edges], each edge is a number represented as binary
  """

  def __init__(self, tile: str):
    md = re.match(r'Tile (\d+):', tile.splitlines()[0])
    assert md
    self.tile_id: int = int(md.group(1))
    binTile: str = tile.replace('#', '1').replace('.', '0')
    a = int(binTile.splitlines()[1], 2)
    b = int(''.join(bt[-1] for bt in binTile.splitlines()[1:]), 2)
    c = int(binTile.splitlines()[-1], 2)
    d = int(''.join(bt[0] for bt in binTile.splitlines()[1:]), 2)

    # This list includes is *all* possible values - when arranging tiles, note that flip only changes 2 opposite edges
    self.alledges: List[int] = [a, b, c, d, flip(a), flip(b), flip(c), flip(d)]
    self.orientation: int = 0
    self.tile: List[str] = tile.splitlines()[1:]

  def reorient(self):
    self.orientation = (self.orientation + 1) % 8

  def currentEdges(self) -> List[int]:
    """Returns edges as numbers given the current tile orientation, ordered as [top, right, bottom, left]"""
    def rotate(edges: List[int]) -> List[int]:
      assert len(edges) == 4, 'Wrong edges length: %d' % len(edges)
      return [edges[1], flip(edges[2]), edges[3], flip(edges[0])]
    o = self.orientation
    edges = self.alledges[:4]
    if o >= 4:
      edges = [edges[2], flip(edges[1]), edges[0], flip(edges[3])]
      o -= 4
    while o > 0:
      edges = rotate(edges)
      o -= 1
    return edges

  def reorientedTile(self) -> List[str]:
    return reorientTile(self.tile, self.orientation)

  def borderlessTile(self) -> List[str]:
    tl = self.reorientedTile()
    return [l[1:-1] for l in tl][1:-1]

ParsedTiles = Dict[int, Tile]

def tiles(data: str) -> ParsedTiles:
  tls : ParsedTiles = {}
  for tile in data.split('\n\n'):
    md = re.match(r'Tile (\d+):', tile.splitlines()[0])
    assert md
    tile_id = int(md.group(1))
    tls[tile_id] = Tile(tile)
  return tls

def tileIdsPerEdge(tls : Dict[int, Tile]) -> Dict[int, List[int]]:
  """Returns dict mapping each edge to all tiles containing it"""
  byEdge : Dict[int, List[int]] = {}
  for id, tile in tls.items():
    for i in tile.alledges:
      if i not in byEdge:
        byEdge[i] = []
      byEdge[i].append(id)
  return byEdge

def findCorners(data: ParsedTiles) -> List[int]:
  """Returns a list of all corners sorted by tile ID."""
  byEdge = tileIdsPerEdge(data)
  corners: List[int] = []
  for id, tile in data.items():
    if len([e for e in tile.alledges if len(byEdge[e]) > 1]) == 4:
      corners.append(id)
  return sorted(corners)

def arrangeAndMulCorners(data: ParsedTiles) -> int:
  return math.prod(findCorners(data))

SEA_MONSTER = [
  '                  # ',
  '#    ##    ##    ###',
  ' #  #  #  #  #  #   ']

class TileMap():
  def __init__(self):
    self.m: Dict[Tuple[int, int], Tile] = {}

  def addTile(self, tile: Tile, x: int, y: int):
    self.m[(x, y)] = tile

  def map(self) -> str:
    mx = max(x for x, _ in self.m.keys())
    my = max(y for _, y in self.m.keys())
    tl = len(self.m[(0, 0)].tile)
    lines: List[str] = []
    for y in range(my + 1):
      for l in range(tl):
        if l == 0:
          lines.append('    '.join(str(self.m[(x, y)].tile_id) + ' / ' + str(self.m[(x, y)].orientation) for x in range(mx + 1)))
        lines.append(' '.join(self.m[(x, y)].reorientedTile()[l] for x in range(mx + 1)))
    return '\n'.join(lines)

  def borderlessMap(self) -> str:
    mx = max(x for x, _ in self.m.keys())
    my = max(y for _, y in self.m.keys())
    tl = len(self.m[(0, 0)].tile) - 2
    lines: List[str] = []
    for y in range(my + 1):
      for l in range(tl):
        lines.append(''.join(self.m[(x, y)].borderlessTile()[l] for x in range(mx + 1)))
    return '\n'.join(lines)

  def countMonsters(self, o: int):
    def matches(m: List[str], x: int, y: int) -> bool:
      for j, l in enumerate(SEA_MONSTER):
        if not all(m[y+j][x+i] == '#' for i, c in enumerate(l) if c == '#'):
          return False
      return True
    lm = reorientTile(self.borderlessMap().splitlines(), o)
    found = 0
    for y in range(len(lm) - 3):
      for x in range(len(lm[0]) - len(SEA_MONSTER[0])):
        if matches(lm, x, y):
          found += 1
    return found

  def numMonsters(self):
    for i in range(8):
      n = self.countMonsters(i)
      if n > 0:
        return n
    assert False, 'No monsters found in any orientation'

  def size(self):
    return len(self.m)

def createMap(data: ParsedTiles) -> TileMap:
  byEdge = tileIdsPerEdge(data)
  def outer(tile: Tile, edgeIndex: int) -> bool:
    return len(byEdge[tile.currentEdges()[edgeIndex]]) == 1

  corners = findCorners(data)
  # fix one corner, then build the map from it:
  topLeft: Tile = data[corners[0]]
  while not (outer(topLeft, 0) and outer(topLeft, 3)):
    topLeft.reorient()

  tm = TileMap()
  used: Set[int] = {topLeft.tile_id}
  tm.addTile(topLeft, 0, 0)
  edges: Deque[Tuple[int, int, int, int]] = collections.deque([(topLeft.currentEdges()[1], 1, 0, 0), (topLeft.currentEdges()[2], 2, 0, 0)])
  while len(edges):
    e, o, x, y = edges.popleft()
    candidateTiles = [t for t in byEdge[e] if t not in used]
    if len(candidateTiles) == 0:
      continue
    used.add(candidateTiles[0])
    tile: Tile = data[candidateTiles[0]]
    if o == 1:
      x += 1
    elif o == 2:
      y += 1
    maxRotations = 8
    while e != tile.currentEdges()[(o + 2) % 4] and maxRotations > 0:
      tile.reorient()
      maxRotations -= 1
    assert maxRotations >= 0
    tm.addTile(tile, x, y)
    edges.append((tile.currentEdges()[1], 1, x, y))
    edges.append((tile.currentEdges()[2], 2, x, y))
  return tm

def roughSea(data: ParsedTiles):
  m = createMap(data)
  c = m.numMonsters()
  return m.borderlessMap().count('#') - c * ''.join(SEA_MONSTER).count('#')

# --------------- 21 --------------- #
Recipes = List[Tuple[List[str], List[str]]]
def recipes(data: str) -> Recipes:
  ret: Recipes = []
  for l in data.splitlines():
    if ' (contains ' not in l:
      ret.append((l.split(' '), []))
      continue
    ingrediences, allergens = l.split(' (contains ')
    ret.append((ingrediences.split(' '), allergens[:-1].split(', ')))
  return ret

def identifyAlergens(data: Recipes) -> Dict[str, str]:
  def cleanupKnown(ds: Dict[str, Set[str]]) -> None:
    shrinked = True
    while shrinked:
      shrinked = False
      singles: Set[str] = set(list(v)[0] for v in ds.values() if len(v) == 1)
      for k in ds:
        if len(ds[k]) > 1 and ds[k] & singles:
          ds[k] -= singles
          shrinked = True

  ds: Dict[str, Set[str]] = {}
  for ingredients, allergens in data:
    for a in allergens:
      if a not in ds:
        ds[a] = set(ingredients)
      ds[a] &= set(ingredients)
      # propagate if single:
      cleanupKnown(ds)
  return {k: list(v)[0] for k, v in ds.items()}

def countNonAlergens(data: Recipes) -> int:
  alergens = identifyAlergens(data)
  # print(alergens)
  ingredients: List[str] = []
  for v, _ in data:
    ingredients.extend(v)
  ing_alergens: Set[str] = set(alergens.values())
  return len([ing for ing in ingredients if ing not in ing_alergens])

def dangerousIngredients(data: Recipes) -> str:
  alergens = identifyAlergens(data)
  return ','.join(alergens[k] for k in sorted(alergens))

# --------------- 22 --------------- #
def cards(data: str) -> Tuple[Deque[int], Deque[int]]:
  p1, p2 = data.split('\n\n')
  rv = (collections.deque(int(v) for v in p1.splitlines()[1:]),
        collections.deque(int(v) for v in p2.splitlines()[1:]))
  # print(rv)
  return rv

def calcScore(d: Deque[int]) -> int:
  return sum((i+1) * v for i, v in enumerate(reversed(d)))

def score(data: Tuple[Deque[int], Deque[int]]) -> int:
  while data[0] and data[1]:
    c1 = data[0].popleft()
    c2 = data[1].popleft()
    if c1 > c2:
      data[0].append(c1)
      data[0].append(c2)
    else:
      data[1].append(c2)
      data[1].append(c1)
  return calcScore(data[0]) if data[0] else calcScore(data[1])

def recursiveGame(data: Tuple[Deque[int], Deque[int]], game: int) -> Tuple[Deque[int], Deque[int]]:
  """Returns status after a game."""
  seen: Set[Tuple[Tuple[int], Tuple[int]]] = set()
  round = 0
  while data[0] and data[1]:
    round += 1
    if (tuple(data[0]), tuple(data[1])) in seen:
      return (data[0], collections.deque())
    seen |= {(tuple(data[0]), tuple(data[1]))}
    c1 = data[0].popleft()
    c2 = data[1].popleft()
    winner = 0
    if c1 <= len(data[0]) and c2 <= len(data[1]):
      # copy to new deques:
      id1 = data[0].copy()
      while len(id1) > c1:
        id1.pop()
      id2 = data[1].copy()
      while len(id2) > c2:
        id2.pop()
      rd1, _rd2 = recursiveGame((id1, id2), game + 1)
      if not rd1:
        winner = 1
    elif c2 > c1:
      winner = 1

    if winner == 0:
      data[0].append(c1)
      data[0].append(c2)
    else:
      data[1].append(c2)
      data[1].append(c1)
  return data

def recursiveCombat(data: Tuple[Deque[int], Deque[int]]) -> int:
  d1, d2 = recursiveGame(data, 1)
  return calcScore(d1) if d1 else calcScore(d2)

# --------------- 23 --------------- #
def toNumberAfter1(graph: Dict[int, int]) -> int:
  num = graph[1]
  r = 0
  while num != 1:
    r = r*10 + num
    num = graph[num]
  return r

def mulAfter1(graph: Dict[int, int]) -> int:
  return graph[1] * graph[graph[1]]

def asEdgeGraph(data: List[int]) -> Dict[int, int]:
  graph: Dict[int, int] = {}
  for i in range(len(data) - 1):
    graph[data[i]] = data[i+1]
  graph[data[len(data) - 1]] = data[0]
  return graph

class TestToNumber(unittest.TestCase):
  def testToNumber(self):
    self.assertEqual(0, toNumberAfter1(asEdgeGraph([1])))
    self.assertEqual(0, toNumberAfter1(asEdgeGraph([1, 0])))
    self.assertEqual(0, toNumberAfter1(asEdgeGraph([0, 1])))
    self.assertEqual(2, toNumberAfter1(asEdgeGraph([1, 2])))
    self.assertEqual(2, toNumberAfter1(asEdgeGraph([2, 1])))
    self.assertEqual(432, toNumberAfter1(asEdgeGraph([4, 3, 2, 1])))
    self.assertEqual(432, toNumberAfter1(asEdgeGraph([1, 4, 3, 2])))
    self.assertEqual(432, toNumberAfter1(asEdgeGraph([2, 1, 4, 3])))
    self.assertEqual(432, toNumberAfter1(asEdgeGraph([3, 2, 1, 4])))

  def testMulAfter1(self):
    self.assertEqual(14, mulAfter1(asEdgeGraph([1, 2, 7, 9])))
    self.assertEqual(14, mulAfter1(asEdgeGraph([9, 1, 2, 7])))
    self.assertEqual(14, mulAfter1(asEdgeGraph([7, 9, 1, 2])))
    self.assertEqual(14, mulAfter1(asEdgeGraph([2, 7, 9, 1])))


def crabCup(data: List[int], rounds: int, toNumber: Callable[[Dict[int, int]], int]) -> int:
  _mx = max(data)
  graph = asEdgeGraph(data)

  def insertion(s: int) -> int:
    # find next lower number than data[0] in data[4:]
    n = s - 1
    if n == 0:
      n = _mx
    ignored = [graph[s], graph[graph[s]], graph[graph[graph[s]]]]
    while n in ignored:
      n -= 1
      if n == 0:
        n = _mx
    return n

  s = data[0]
  while rounds > 0:
    i = insertion(s)
    prev = graph[i]
    graph[i] = graph[s]
    graph[s] = graph[graph[graph[graph[i]]]]
    graph[graph[graph[graph[i]]]] = prev
    s = graph[s]
    rounds -= 1

  return toNumber(graph)

def crabCups(data: List[int]) -> int:
  return crabCup(data, 100, toNumberAfter1)

def crabCupsMul(data: List[int]) -> int:
  mx = max(data)
  data = data + list(range(mx + 1, 1000 * 1000 + 1))
  return crabCup(data, 10 * 1000 * 1000, mulAfter1)

# --------------- 23 --------------- #
Hex = Tuple[int, int]
HexPath = List[Hex]

def hexPaths(data: str) -> List[HexPath]:
  paths: List[HexPath] = []
  for l in data.splitlines():
    path: HexPath = []
    pc = ''
    for c in l:
      if pc == 'n' and c == 'e':
        pc = ''
        path.append((-1, 1))
      elif pc == '' and c == 'e':
        path.append((0, 2))
      elif pc == 's' and c == 'e':
        pc = ''
        path.append((1, 1))
      elif pc == 's' and c == 'w':
        pc = ''
        path.append((1, -1))
      elif pc == '' and c == 'w':
        path.append((0, -2))
      elif pc == 'n' and c == 'w':
        pc = ''
        path.append((-1, -1))
      else:
        pc = c
    paths.append(path)
  return paths

def applyPath(data: List[HexPath]) -> Set[Hex]:
  flipped: Set[Hex] = set()
  for path in data:
    point: Hex = (0, 0)
    for move in path:
      point = (point[0] + move[0], point[1] + move[1])
    if point in flipped:
      flipped.remove(point)
    else:
      flipped.add(point)
  return flipped

def countFlips(data: List[HexPath]) -> int:
  return len(applyPath(data))

def live(before: Set[Hex], i : int) -> Set[Hex]:
  neighbors = [(-1, 1), (0, 2), (1, 1), (1, -1), (0, -2), (-1, -1)]
  d: Dict[Hex, int] = collections.defaultdict(int)
  for h in before:
    for nb in neighbors:
      d[(h[0] + nb[0], h[1] + nb[1])] += 1

  # copy over black tiles with 1 or 2 neighbors:
  after: Set[Hex] = {h for h in before if d[h] == 1 or d[h] == 2}
  # add all white tiles with 2 black neighbors
  after |= {h for h in d if h not in before and d[h] == 2}
  return after

def hexLife100(data: List[HexPath]) -> int:
  start = applyPath(data)
  for i in range(100):
    start = live(start, i)
  return len(start)

# --------------- Unit tests -------------------------- #
class UnitTest(unittest.TestCase):
  def testExamples(self):
    self.assertEqual(4, whereCanTheBagBe(getInput('7a.txt', bags)))
    self.assertEqual(32, howManyBags(getInput('7a.txt', bags)))
    self.assertEqual(126, howManyBags(getInput('7b.txt', bags)))
    text = "\n".join(["nop +0", "acc +1", "jmp +4", "acc +3", "jmp -3", "acc -99", "acc +1", "jmp -4", "acc +6"])
    self.assertEqual(5, accumulatorBeforeCycle(instructions(text)))
    self.assertEqual(8, accumulatorOnFixedLoop(instructions(text)))
    self.assertEqual(62, encryptionWeakness(getInput('9a.txt', numbers), 5))
    smallJolts = jolts('\n'.join(str(i) for i in [16, 10, 15, 5, 1, 11, 7, 19, 6, 12, 4]))
    largeJolts = jolts('\n'.join(str(i) for i in [28, 33, 18, 42, 31, 14, 46, 20, 48, 47, 24, 23, 49, 45, 19, 38, 39, 11, 1, 32, 25, 35, 8, 17, 7, 9, 4, 2, 34, 10, 3]))
    self.assertEqual(35, allJolts(smallJolts))
    self.assertEqual(220, allJolts(largeJolts))
    self.assertEqual(8, joltArrangements(smallJolts))
    self.assertEqual(19208, joltArrangements(largeJolts))
    self.assertEqual(37, stableOccupancyNeighbors(getInput('11a.txt', seats)))
    self.assertEqual(26, stableOccupancyVisible(getInput('11b.txt', seats), True))
    exampleDirections = '\n'.join(['F10', 'N3', 'F7', 'R90', 'F11'])
    self.assertEqual(25, manhattanDistanceMovedShip(directions(exampleDirections)))
    self.assertEqual(286, manhattanDistanceMovedWaypoint(directions(exampleDirections)))
    busSchedule = "939\n7,13,x,x,59,x,31,19"
    self.assertEqual(295, busDepartureById(schedule(busSchedule)))
    self.assertEqual(1068781, fittingSchedule(schedule(busSchedule)))
    self.assertEqual(3417, fittingSchedule(schedule("1\n17,x,13,19")))
    self.assertEqual(754018, fittingSchedule(schedule("1\n67,7,59,61")))
    self.assertEqual(779210, fittingSchedule(schedule("1\n67,x,7,59,61")))
    self.assertEqual(1261476, fittingSchedule(schedule("1\n67,7,x,59,61")))
    self.assertEqual(1202161486, fittingSchedule(schedule("1\n1789,37,47,1889")))
    program1 = "mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X\nmem[8] = 11\nmem[7] = 101\nmem[8] = 0"
    self.assertEqual(165, sumValuesAfterBitmask(lines(program1)))
    program2 = "mask = 000000000000000000000000000000X1001X\nmem[42] = 100\nmask = 00000000000000000000000000000000X0XX\nmem[26] = 1"
    self.assertEqual(208, sumValuesAfterMemBitmask(lines(program2)))
    self.assertEqual(0, game([0, 3, 6], 10))
    self.assertEqual(1, game2020([1, 3, 2]))
    self.assertEqual(10, game2020([2, 1, 3]))
    self.assertEqual(27, game2020([1, 2, 3]))
    self.assertEqual(78, game2020([2, 3, 1]))
    self.assertEqual(438, game2020([3, 2, 1]))
    self.assertEqual(1836, game2020([3, 1, 2]))
    # self.assertEqual(175594, game3([0, 3, 6]))  - too slow
    self.assertEqual(71, sumInvalidValues(getInput("16a.txt", trainTickets)))
    mulDepartureValues(getInput("16a.txt", trainTickets))
    self.assertEqual(143, mulDepartureValues(getInput("16b.txt", trainTickets)))
    # self.assertEqual(112, life6_3(grid('.#.\n..#\n###')))
    # self.assertEqual(848, life6_4(grid('.#.\n..#\n###')))
    self.assertEqual(10, ltrPrecedence(["1 + 1 * 5"]))
    self.assertEqual(6, ltrPrecedence(["1 * 5 + 1"]))
    self.assertEqual(16, ltrPrecedence(["(1 + 1) + 2 * (2 * 2)"]))
    self.assertEqual(54, ltrPrecedence(["(1 + 1) + (2 + 2) * (3 * 3)"]))
    self.assertEqual(231, addBeforeMul(["1 + 2 * 3 + 4 * 5 + 6"]))
    msgs = '0: 4 1 5\n1: 2 3 | 3 2\n2: 4 4 | 5 5\n3: 4 5 | 5 4\n4: "a"\n5: "b"\n\nababbb\nbababa\nabbbab\naaabbb\naaaabbb'
    self.assertEqual(2, matchingMessages(messages(msgs)))
    self.assertEqual(12, matchingMessagesWithLoops(getInput("19a.txt", messages)))
    self.assertEqual(20899048083289, arrangeAndMulCorners(getInput("20a.txt", tiles)))
    self.assertEqual(273, roughSea(getInput("20a.txt", tiles)))
    self.assertEqual(5, countNonAlergens(getInput("21a.txt", recipes)))
    self.assertEqual('mxmxvkd,sqjhc,fvjkl', dangerousIngredients(getInput('21a.txt', recipes)))
    self.assertEqual(306, score(getInput('22a.txt', cards)))
    self.assertEqual(291, recursiveCombat(getInput('22a.txt', cards)))
    self.assertEqual(92658374, crabCup([3, 8, 9, 1, 2, 5, 4, 6, 7], 10, toNumberAfter1))
    # too slow:
    # self.assertEqual(149245887792, crabCupsMul([3, 8, 9, 1, 2, 5, 4, 6, 7]))
    self.assertEqual(10, countFlips(getInput('24a.txt', hexPaths)))
    self.assertEqual(2208, hexLife100(getInput('24a.txt', hexPaths)))

  def testBp(self):
    self.assertEqual([(44, 5)], boardingPasses('FBFBBFFRLR'))

# --------------- Calling all solutions --------------- #

Solution = NamedTuple('Solution', [('filename', str), ('parser', Callable[[str], Any]), ('part1', Callable[[Any], Any]), ('part2', Callable[[Any], Any])])

solutions = [
  Solution('1.txt', numbers, twoAddTo2020, threeAddTo2020),
  Solution('2.txt', passwords, countValidFirstRule, countValidSecondRule),
  Solution('3.txt', slopes, down1right3, multiplySlopes),
  Solution('4.txt', passports, validPassports, validPassports2),
  Solution('5.txt', boardingPasses, highestSeatId, missingSeatId),
  Solution('6.txt', answers, anyYesses, allYesses),
  Solution('7.txt', bags, whereCanTheBagBe, howManyBags),
  Solution('8.txt', instructions, accumulatorBeforeCycle, accumulatorOnFixedLoop),
  Solution('9.txt', numbers, firstNotSumOf2, encryptionWeakness),
  Solution('10.txt', jolts, allJolts, joltArrangements),
  Solution('11.txt', seats, fixed(2334), fixed(2100)), #stableOccupancyNeighbors, stableOccupancyVisible),
  Solution('12.txt', directions, manhattanDistanceMovedShip, manhattanDistanceMovedWaypoint),
  Solution('13.txt', schedule, busDepartureById, fittingSchedule),
  Solution('14.txt', lines, sumValuesAfterBitmask, sumValuesAfterMemBitmask),
  Solution('15.txt', numbers, game2020, fixed(3745954)), # game3)
  Solution('16.txt', trainTickets, sumInvalidValues, mulDepartureValues),
  Solution('17.txt', grid, fixed(284), fixed(2240)), # life6_3, life6_4)
  Solution('18.txt', lines, ltrPrecedence, addBeforeMul),
  Solution('19.txt', messages, matchingMessages, matchingMessagesWithLoops),
  Solution('20.txt', tiles, arrangeAndMulCorners, roughSea),
  Solution('21.txt', recipes, countNonAlergens, dangerousIngredients),
  Solution('22.txt', cards, score, fixed(34424)), #recursiveCombat)
  Solution('23.txt', numbers, crabCups, fixed(294320513093)), #crabCupsMul)
  Solution('24.txt', hexPaths, countFlips, hexLife100),
]

# --------------- Tests --------------- #
class SolutionTest(unittest.TestCase):
  def singleSolution(self, s: Solution, expected1: Any, expected2: Any):
    print('verifying problem', s.filename[:s.filename.index('.')])
    self.assertEqual(expected1, s.part1(getInput(s.filename, s.parser)))
    self.assertEqual(expected2, s.part2(getInput(s.filename, s.parser)))

  def testSolutions(self):
    self.singleSolution(solutions[0], 712075, 145245270)
    self.singleSolution(solutions[1], 424, 747)
    self.singleSolution(solutions[2], 284, 3510149120)
    self.singleSolution(solutions[3], 250, 158)
    self.singleSolution(solutions[4], 896, 659)
    self.singleSolution(solutions[5], 6778, 3406)
    self.singleSolution(solutions[6], 254, 6006)
    self.singleSolution(solutions[7], 1915, 944)
    self.singleSolution(solutions[8], 393911906, 59341885)
    self.singleSolution(solutions[9], 2343, 31581162962944)
    self.singleSolution(solutions[10], 2334, 2100)
    self.singleSolution(solutions[11], 1221, 59435)
    self.singleSolution(solutions[12], 296, 535296695251210)
    self.singleSolution(solutions[13], 14925946402938, 3706820676200)
    self.singleSolution(solutions[14], 1238, 3745954)
    self.singleSolution(solutions[15], 30869, 4381476149273)
    self.singleSolution(solutions[16], 284, 2240)
    self.singleSolution(solutions[17], 29839238838303, 201376568795521)
    self.singleSolution(solutions[18], 142, 294)
    self.singleSolution(solutions[19], 5775714912743, 1836)
    self.singleSolution(solutions[20], 2374, 'fbtqkzc,jbbsjh,cpttmnv,ccrbr,tdmqcl,vnjxjg,nlph,mzqjxq')
    self.singleSolution(solutions[21], 35562, 34424)
    self.singleSolution(solutions[22], 98742365, 294320513093)
    self.singleSolution(solutions[23], 459, 4150)

unittest.main()

for i in range(len(solutions)):
  s: Solution = solutions[i]
  print('Day %d:' % (i+1),
      s.part1(getInput(s.filename, s.parser)),
      s.part2(getInput(s.filename, s.parser)))




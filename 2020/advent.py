#! /usr/bin/python3

import collections
import itertools
import math
import os
import re
import unittest


def getInput(filename, parser):
  with open(os.path.join(os.path.dirname(__file__), 'input', filename)) as f:
    return parser(f.read())

def number(l):
  return int(l)

def numbers(data):
  return [number(l) for l in data.splitlines() if l]

def twoAddTo2020(data):
  return next(i * j for i, j in itertools.combinations(data, 2) if i + j == 2020)

def threeAddTo2020(data):
  return next(i*j*k for i, j, k in itertools.combinations(data, 3) if i + j + k == 2020)

PasswordEntry = collections.namedtuple('PasswordEntry', ['password', 'rule_char', 'min', 'max'])

def passwords(data):
  def password(l):
    rule, pwd = l.split(':')
    pwd = pwd[1:]
    m = re.match('(\d+)\-(\d+)\s([a-z])', rule)
    return PasswordEntry(pwd, m[3], int(m[1]), int(m[2]))
  return [password(l) for l in data.splitlines() if l]


def countValidFirstRule(ins):
  def _isValidFirstRule(pe):
    cc = pe.password.count(pe.rule_char)
    return cc >= pe.min and cc <= pe.max
  return len([1 for pe in ins if _isValidFirstRule(pe)])

def countValidSecondRule(ins):
  def _isValidSecondRule(pe):
    return (pe.password[pe.min-1] == pe.rule_char) ^ (pe.password[pe.max-1] == pe.rule_char)
  return len([1 for pe in ins if _isValidSecondRule(pe)])


def isTree(data, x, y):
  return y < len(data) and data[y][x % len(data[0])] == '#'

def slope(l):
  return l.strip()

def slopes(data):
  return [slope(l) for l in data.splitlines() if l]


def slopeSteps(data, down, right):
  c = len(data)
  return sum(1 for x, y in zip(range(0, right*c, right), range(0, down*c, down)) if isTree(data, x, y))

def down1right3(data):
  return slopeSteps(data, 1, 3)

def multiplySlopes(data):
  return slopeSteps(data, 1, 1) * slopeSteps(data, 1, 3) * slopeSteps(data, 1, 5) * slopeSteps(data, 1, 7) * slopeSteps(data, 2, 1)

def noop(data):
  return '?'

def passports(data):
  def passport(s):
    return [tuple(e.split(':')) for e in s.split(' ') if e]
  return [passport(s.replace('\n', ' ')) for s in data.split('\n\n') if s]

def validPassports(data):
  def valid(p):
    attrs = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'}
    return attrs == {k for k, _ in p} | {'cid'}
  return sum(1 for p in data if valid(p))

def validPassports2(data):
  def valid(p):
    def hgt(v):
      if len(v) < 3:
        return False
      if v[-2:] == 'in':
        return 59 <= int(v[:-2]) <= 76
      elif v[-2:] == 'cm':
        return 150 <= int(v[:-2]) <= 193
      else:
        return False

    attrs = {
      'byr': (lambda v: 1920 <= int(v) <= 2002),
      'iyr': (lambda v: 2010 <= int(v) <= 2020),
      'eyr': (lambda v: 2020 <= int(v) <= 2030),
      'hgt': hgt,
      'hcl': (lambda v: re.match('^#[0-9a-z]{6}$', v)),
      'ecl': (lambda v: v in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']),
      'pid': (lambda v: re.match('^[0-9]{9}$', v)),
      'cid': (lambda _: True)}
    for (k, v) in p:
      if k not in attrs or not attrs[k](v):
        return False
      del attrs[k]
    if 'cid' in attrs:
      del attrs['cid']
    return not attrs
  return sum(1 for p in data if valid(p))

def boardingPasses(data):
  def bp(b):
    return (sum(2 ** (6-i) for i, c in enumerate(b[:7]) if c == 'B'),
            sum(2 ** (2-i) for i, c in enumerate(b[-3:]) if c == 'R'))
  return [bp(l) for l in data.split('\n') if l]

def highestSeatId(data):
  return max(8*r + c for r, c in data)

def missingSeatId(data):
  ids = sorted([8*r + c for r, c in data])
  return [k + 1 for k, l in zip(ids, ids[1:]) if k + 1 != l][0]

def answers(data):
  return [[set(l) for l in s.splitlines()] for s in data.split('\n\n') if s]

def anyYesses(data):
  return sum(len(set().union(*part)) for part in data)

def allYesses(data):
  return sum(len(part[0].intersection(*part)) for part in data)

# vibrant plum bags contain 5 faded blue bags, 6 dotted black bags.
# faded blue bags contain no other bags.
def bagcontents(line):
  # returns pair (bag, (count, bag types))
  bag, contents = line.split(" bags contain ")
  bagtypes = []
  for what in contents.split(', '):
    if not what:
      continue
    md = re.match('(\d+) (\w+ \w+) bags?\.?', what)
    if md:
      bagtypes.append((int(md.group(1)), md.group(2)))
  return (bag, bagtypes)

def bags(data):
  return dict(bagcontents(l) for l in data.splitlines() if l)

def whereCanTheBagBe(data):
  colors = set()
  bags_to_check = collections.deque(['shiny gold'])
  bags_checked = set()
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

def howManyBags(data):
  bags_to_add = collections.deque([('shiny gold', 1)])
  total = 0
  while bags_to_add:
    bag, count = bags_to_add.pop()
    total += count
    contents = data[bag]
    for c in contents:
      bags_to_add.append((c[1], c[0]*count))
  return total - 1

def instructions(data):
  def _instruction(s):
    inst, number = s.split(' ')
    return (inst, int(number))
  return [_instruction(l) for l in data.splitlines() if l]

def accumulatorBeforeCycle(data):
  visited = set()
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

def accumulatorOnFixedLoop(data):
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

def firstNotSumOf2(data):
  for n in range(25, len(data)):
    try:
      next(i+j for i, j in itertools.combinations(data[n-25:n], 2) if i + j == data[n])
    except StopIteration:
      return data[n]

def encryptionWeakness(data, pre=25):
  def _firstNotSumOf2(data, pre):
    for n in range(pre, len(data)):
      try:
        next(i+j for i, j in itertools.combinations(data[n-pre:n], 2) if i + j == data[n])
      except StopIteration:
        return n, data[n]
  n, s = _firstNotSumOf2(data, pre)
  for i in range(n):
    for j in range(i, n):
      if sum(data[i:j]) == s:
        return min(data[i:j]) + max(data[i:j])

def jolts(data):
  nums = numbers(data)
  return [0] + sorted(nums) + [max(nums) + 3]

def allJolts(nums):
  c = collections.Counter(t - o for o, t in zip(nums, nums[1:]))
  return c[1] * c[3]

def joltArrangements(nums):
  # check up to last 3 items:
  paths = [1]
  for i in range(1, len(nums)):
    prevs = max(i-3, 0)
    paths.append(
      sum(p for n, p in zip(nums[prevs:i], paths[prevs:]) if n >= nums[i] - 3))
  return paths[-1]

def seats(data):
  return [[c for c in l] for l in data.splitlines()]

def stableOccupancy(onlyNeighbors, data, p=False):
  maxi = len(data)
  maxj = len(data[0])

  def countOccupied(i, j):
    def occupied(i, j):
      return 0 <= i < maxi and 0 <= j < maxj and data[i][j] == '#'
    return len([1 for (i_, j_) in itertools.product(range(i-1, i+2), range(j-1, j+2)) if (i_ != i or j_ != j) and occupied(i_, j_)])

  def countVisibleOccupied(i, j):
    def visibleOccupied(dx, dy):
      i_ = i + dx
      j_ = j + dy
      while 0 <= i_ < maxi and 0 <= j_ < maxj:
        if data[i_][j_] == 'L':
          return False
        elif data[i_][j_] == '#':
          return True
        i_ += dx
        j_ += dy
    return len([1 for (x, y) in itertools.product(range(-1, 2), repeat=2) if (x != 0 or y != 0) and visibleOccupied(x, y)])

  count = countOccupied if onlyNeighbors else countVisibleOccupied
  needEmpty = 4 if onlyNeighbors else 5

  def applyRule(i, j):
    if data[i][j] == 'L' and count(i, j) == 0:
      return '#'
    elif data[i][j] == '#' and count(i, j) >= needEmpty:
      return 'L'
    return data[i][j]

  def applyAll():
    nd = []
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

def stableOccupancyNeighbors(data):
  return stableOccupancy(True, data)

def stableOccupancyVisible(data, p=False):
  return stableOccupancy(False, data, p)

def fixed(n):
  return lambda d: n

def directions(data):
  return [(d[0], int(d[1:])) for d in data.splitlines()]

def manhattanDistanceMovedShip(data):
  def moveShip():
    d = 0
    x = 0
    y = 0
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

def manhattanDistanceMovedWaypoint(data):
  def moveByWaypoint():
    wx = 10
    wy = 1
    x = 0
    y = 0

    def degrees():
      if wx == 0:
        return 0 if wy > 0 else 180
      dg = math.degrees(math.atan(wy / wx))
      if wx < 0:
        return 180 + dg
      return 360 + dg if wy < 0 else dg

    def rotate(n):
      dg = degrees() + n
      d = math.sqrt(wx * wx + wy * wy)
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

def schedule(data):
  arrival, schedules = data.splitlines()
  return int(arrival), schedules.split(',')

def busDepartureById(data):
  arrival = data[0]
  schedules = [int(s) for s in data[1] if s != 'x']

  def nextDeparture(s : int):
    return arrival if (arrival % s) == 0 else s + arrival - (arrival % s)

  mn = (schedules[0], schedules[0])
  for s in schedules:
    wait = nextDeparture(s) - arrival
    if wait < mn[0]:
      mn = (wait, s)
  return mn[0] * mn[1]

def fittingSchedule(data):
  schedules = []
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

def bitmaskProgram(data):
  return data.splitlines()

def sumValuesAfterBitmask(data):

  memory = collections.defaultdict()
  ormask = 0
  andmask = 0
  for inst in data:
    if inst.startswith('mask = '):
      maskstr = inst.split(' ')[2]
      ormask = int(maskstr.replace('X', '0'), 2)
      andmask = int(maskstr.replace('X', '1'), 2)
    else:
      md = re.match('mem\[(\d+)\] = (\d+)', inst)
      addr, value = int(md[1]), int(md[2])
      memory[addr] = (value & andmask) | ormask
  return sum(memory.values())

def sumValuesAfterMemBitmask(data):
  def genAddr(n : int):
    if (n == 0):
      return []
    # find highest bit:
    maxBit = max([k for k in range(36) if 2 ** k <= n])
    if 2 ** maxBit == n:
      return [0, n]
    inner = genAddr(n & ~(2 ** maxBit))
    return inner + [(2 ** maxBit) | i for i in inner ]

  memory = collections.defaultdict()
  ormask = 0
  xmask = 0
  for inst in data:
    if inst.startswith('mask = '):
      maskstr = inst.split(' ')[2]
      ormask = int(maskstr.replace('X', '0'), 2)
      xmask = int(maskstr.replace('1', '0').replace('X', '1'), 2)
    else:
      md = re.match('mem\[(\d+)\] = (\d+)', inst)
      addr, value = int(md[1]), int(md[2])
      addr = (addr | ormask) & (~xmask)
      for i in genAddr(xmask):
        memory[addr | i] = value
  return sum(memory.values())

def game(data, nth):
  last = {}
  for i, n in enumerate(data[:-1]):
     # skip last item to check its previous last position below
    last[n] = i
  prev = data[-1]
  for i in range(len(data), nth):
    n = i - last[prev] - 1 if prev in last else 0
    last[prev] = i - 1
    prev = n
  return prev

def game2020(data):
  return game(data, 2020)

def game3(data):
  return game(data, 30000000)

def trainTickets(data):
  parts = data.split("\n\n")
  rules_str = parts[0].splitlines()
  rules = {}
  for s in rules_str:
    name, rs = s.split(": ")
    rules_nums = [(int(r.split('-')[0]), int(r.split('-')[1])) for r in rs.split(' or ')]
    rules[name] = rules_nums
  yourTicket = [int(n) for n in parts[1].splitlines()[1].split(',')]
  nearbyTickets = [[int(i) for i in values.split(",")] for values in parts[2].splitlines()[1:]]
  return rules, yourTicket, nearbyTickets

def isValid(n, rules):
  return any(True for rule in rules.values() for l, h in rule if n >= l and n <= h)

def sumInvalidValues(data):
  rules, yourTicket, nearbyTickets = data

  return sum(n for ticket in nearbyTickets for n in ticket if not isValid(n, rules))

def mulDepartureValues(data):
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


def grid(data):
  g = set()
  for y, l in enumerate(data.splitlines()):
    for x, c in enumerate(l):
      if c == '#':
        g.add((x, y, 0, 0))
  return g

def print_grid(data):
  sg = sorted(list(data))
  print(sg)

def count_neighbors3(data, x, y, z):
  return len([True for (i, j, k) in itertools.product([-1, 0, 1], repeat=3) if (i != 0 or j != 0 or k != 0) and (x+i, y+j, z+k, 0) in data])

def count_neighbors4(data, x, y, z, w):
  return len([True for (i, j, k, l) in itertools.product([-1, 0, 1], repeat=4) if (i != 0 or j != 0 or k != 0 or l != 0) and (x+i, y+j, z+k, w+l) in data])

def next_iter3(data):
  return {(x, y, z, 0) for (x, y, z) in itertools.product(range(-6, 15), repeat=3) if count_neighbors3(data, x, y, z) == 3 or ((x, y, z, 0) in data and count_neighbors3(data, x, y, z) == 2)}

def next_iter4(data):
  return {(x, y, z, w) for (x, y, z, w) in itertools.product(range(-6, 15), repeat=4) if count_neighbors4(data, x, y, z, w) == 3 or ((x, y, z, w) in data and count_neighbors4(data, x, y, z, w) == 2)}

def life3(data, iterations):
  for i in range(iterations):
    data = next_iter3(data)
  return len(data)

def life4(data, iterations):
  for i in range(iterations):
    data = next_iter4(data)
  return len(data)

def life6_3(data):
  return life3(data, 6)

def life6_4(data):
  return life4(data, 6)

def expressions(data):
  return data.splitlines()

class ExpressionNode:
  def eval(self):
    return 0

class Operator(ExpressionNode):
  def __init__(self, op):
    self.op = op
    self.left = None
    self.right = None

  def eval(self):
    if self.op == '+':
      return self.left.eval() + self.right.eval()
    elif self.op == '-':
      return self.left.eval() - self.right.eval()
    if self.op == '*':
      return self.left.eval() * self.right.eval()
    if self.op == '/':
      return self.left.eval() / self.right.eval()

  def __repr__(self):
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
  def __init__(self, num):
    self.num = num

  def eval(self):
    return self.num

  def __repr__(self):
    return str(self.num)

class OpenParen(ExpressionNode):
  def __init__(self):
    pass

  def __repr__(self):
    return '('

class CloseParen(ExpressionNode):
  def __repr__(self):
    return ')'

class End(ExpressionNode):
  def __repr__(self):
    return '.'

def tokenize(e):
  current = None
  for c in e:
    if c == ' ' and current:
      yield Number(int(current))
      current = None
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
        current = None
      yield CloseParen()
  if current:
    yield Number(int(current))
    current = None
  yield End()

def evalLtr(e):
  def getNode(token_iter):
    n = next(token_iter)
    if isinstance(n, Number):
      return n
    elif isinstance(n, OpenParen):
      return subtree(token_iter, endWhen=CloseParen)

  def subtree(token_iter, endWhen=End):
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

def evalAddBeforeMul(e):
  def getNode(token_iter):
    n = next(token_iter)
    if isinstance(n, Number):
      return n
    elif isinstance(n, OpenParen):
      return subtree(token_iter, endWhen=CloseParen)

  def subtree(token_iter, endWhen=End):
    nodes = [getNode(token_iter)]
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
        nodes[empty_addition_index].left = nodes[empty_addition_index-1]
        nodes[empty_addition_index].right = nodes[empty_addition_index+1]
        del nodes[empty_addition_index+1]
        del nodes[empty_addition_index-1]
    # now merge everything left:
    while len(nodes) > 1:
      nodes[1].left = nodes[0]
      nodes[1].right = nodes[2]
      del nodes[2]
      del nodes[0]
    return nodes[0]

  tokens = list(tokenize(e))
  expr = subtree(iter(tokens), End)
  return expr.eval()

def addBeforeMul(data):
  return sum(evalAddBeforeMul(e) for e in data)

def ltrPrecedence(data):
  """Returns sum of operations, where precedence is left-to-right."""
  return sum(evalLtr(e) for e in data)

class Text:
  def __init__(self, text):
    self.text = text

  def final(self):
    return True

  def value(self):
    return self.text

  def __repr__(self):
    return self.text

  def update(self, rules):
    return self

class Reference:
  def __init__(self, n):
    self.n = n

  def final(self):
    return False

  def value(self):
    pass

  def __repr__(self):
    return str(self.n)

  def update(self, rules):
    return rules[self.n] if rules[self.n].final() else self

class Or:
  def __init__(self, alternatives):
    self.alternatives = alternatives
    self._final = False
    self.val = None

  def final(self):
    if not self._final:
      self._final = all(all(a.final() for a in alt) for alt in self.alternatives)
    return self._final

  def value(self):
    if self.final() and not self.val:
      self.val = '(' + '|'.join('(' + ''.join(v.value() for v in a) + ')' for a in self.alternatives) + ')'
    return self.val

  def __repr__(self):
    return '|'.join(a.__repr__() for a in self.alternatives)

  def update(self, rules):
    for a in self.alternatives:
      for i in range(len(a)):
        a[i] = a[i].update(rules)
    return self

def messages(data):
  d = data.split("\n\n")
  rules = {}
  for rule in d[0].splitlines():
    n, text = rule.split(': ')
    if '"' in text:
      rules[int(n)] = Text(text.split('"')[1])
      continue
    rs = Or([[Reference(int(n)) for n in ps.split(' ')] for ps in text.split(' | ')])
    rules[int(n)] = rs
  msgs = d[1].splitlines()
  return rules, msgs

def compile_regexp(rules):
  while not rules[0].final():
    for r in rules:
      rules[r].update(rules)
  return re.compile(rules[0].value())

def matchingMessages(data):
  rules, msgs = data
  regexp = compile_regexp(rules)
  return sum(1 for m in msgs if regexp.fullmatch(m))

def matchingMessagesWithLoops(data):
  rules, msgs = data
  for i in range(2, 6):
    rules[8].alternatives.append([Reference(42)] * i)
  for i in range(2, 5):
    rules[11].alternatives.append([Reference(42)] * i + [Reference(31)] * i)
  regexp = compile_regexp(rules)
  return sum(1 for m in msgs if regexp.fullmatch(m))

def flip(nr):
  res = 0
  b = 1
  rb = 512
  for i in range(10):
    if nr & b:
      res |= rb
#    if nr == 255:
#      print("i:", i, "b:", b, "rb:", rb, "nr:", nr, "res:", res)
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


def tiles(data):
  # return id : [4 edges] - each edge is a number represented as binary
  tls = {}
  bin_complement = 2 ** len(data.splitlines()[1])
  print("bin complement: ", bin_complement)
  for tile in data.split('\n\n'):
    tile_id = int(re.match('Tile (\d+):', tile.splitlines()[0]).group(1))
    binTile = tile.replace('#', '1').replace('.', '0')
    a = int(binTile.splitlines()[1], 2)
    b = int(''.join(bt[0] for bt in binTile.splitlines()[1:]), 2)
    c = int(binTile.splitlines()[-1], 2)
    d = int(''.join(bt[-1] for bt in binTile.splitlines()[1:]), 2)

    # This list includes is *all* possible values - when arranging tiles, note that flip only changes 2 opposite edges
    tls[tile_id] = (a, b, c, d, flip(a), flip(b), flip(c), flip(d))
  ids = [i for _, v in tls.items() for i in list(v)]
  c = collections.Counter(ids)
  print('tiles:', len(tls), 'all ids:', len(ids), 'unique ids:', len(set(ids)))
  print('counted:', c.most_common(10))
  print('unique:', len([count for _, count in c.items() if count <= 1]))
  return tls

def arrangeAndMulCorners(data):
  return math.prod(findCorners(data))

def findCorners(data):
  # returns tile ids for the 4 corners
  ids = [i for _, v in data.items() for i in list(v)]
  c = collections.Counter(ids)
  tile_connections = {}
  for tid in data:
    tile_connections[tid] = 0
    for i in list(data[tid]):
      if c[i] > 1:
        tile_connections[tid] += 1
  cs = collections.Counter(tile_connections)
  return [k for k, _ in cs.most_common()[:-5:-1]]

def numMonsters(data):
  # create inverse map: edge id --> tile
  edge_to_tile = {}
  for k, v in data.items():
    for i in list(v):
      if i not in edge_to_tile:
        edge_to_tile[i] = []
      edge_to_tile[i].append(k)
  print(edge_to_tile)
  corner = findCorners(data)[0]
  return 0
  # ids = [i for _, v in data.items() for i in list(v)]
  # c = collections.Counter(ids)
  # tile_connections = {}
  # for tid in data:
  #   tile_connections[tid] = 0
  #   for i in list(data[tid]):
  #     if c[i] > 1:
  #       tile_connections[tid] += 1
  # cs = collections.Counter(tile_connections)
  # return math.prod(k for k, _ in cs.most_common()[:-5:-1])

Solution = collections.namedtuple('Solution', ['filename', 'parser', 'part1', 'part2'])

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
  Solution('14.txt', bitmaskProgram, sumValuesAfterBitmask, sumValuesAfterMemBitmask),
  Solution('15.txt', numbers, game2020, fixed(3745954)), # game3)
  Solution('16.txt', trainTickets, sumInvalidValues, mulDepartureValues),
  Solution('17.txt', grid, fixed(284), fixed(2240)), # life6_3, life6_4)
  Solution('18.txt', expressions, ltrPrecedence, addBeforeMul),
  Solution('19.txt', messages, matchingMessages, matchingMessagesWithLoops),
  Solution('20.txt', tiles, arrangeAndMulCorners, noop)
]

class Test(unittest.TestCase):
  def singleSolution(self, s, expected1, expected2):
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
    self.singleSolution(solutions[19], 5775714912743, '?')

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
    self.assertEqual(165, sumValuesAfterBitmask(bitmaskProgram(program1)))
    program2 = "mask = 000000000000000000000000000000X1001X\nmem[42] = 100\nmask = 00000000000000000000000000000000X0XX\nmem[26] = 1"
    self.assertEqual(208, sumValuesAfterMemBitmask(bitmaskProgram(program2)))
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
    self.assertEqual(-1, numMonsters(getInput("20a.txt", tiles)))

  def testBp(self):
    self.assertEqual([(44, 5)], boardingPasses('FBFBBFFRLR'))

for i in range(len(solutions)):
  s = solutions[i]
  print('Day %d:' % (i+1),
      s.part1(getInput(s.filename, s.parser)),
      s.part2(getInput(s.filename, s.parser)))

unittest.main()
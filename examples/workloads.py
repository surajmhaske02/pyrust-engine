from pyrust_engine import F64Array

def integer_work(n:int):
    total=0
    for i in range(n):
        if i%7==0:
            total+=i//3
        else:
            total+=i%11
    return total

def float_work(n:int,x:float):
    value=x
    for i in range(n):
        value=(value*1.000001+0.0001)/1.000002
    return value

def array_energy(a:F64Array):
    total=0.0
    for i in range(len(a)):
        x=float(a[i])
        total+=x*x
    return total

def scalar_add(a:int,b:int):
    return a+b

import math


def math_work(n: int) -> float:
    # sqrt, sin and log in a loop: needs the 0.5.0 math support.
    total = 0.0
    for i in range(1, n):
        x = float(i)
        total += math.sqrt(x) * math.sin(x) + math.log(x)
    return total


def distance(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def calls_work(n: int) -> float:
    # A helper function called inside the loop: needs 0.5.0 calls between functions.
    total = 0.0
    for i in range(n):
        total += distance(float(i), float(n - i))
    return total


def bits_work(n: int) -> int:
    # Integer hashing with shifts, xor, masks and ** (0.5.0 operators).
    mask = 2 ** 32 - 1
    h = 2166136261
    for i in range(n):
        h = ((h ^ (i & 255)) * 16777619) & mask
        h = h ^ (h >> 13) ^ ((h << 5) & mask)
    return h

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

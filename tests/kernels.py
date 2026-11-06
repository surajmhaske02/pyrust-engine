from pyrust_engine import F64Array

def divmod_mix(a:int,b:int):
    return (a//b)*7+a%b

def stepped(start,stop,step):
    total=0
    for i in range(start,stop,step):
        if i%3==0:
            continue
        if total>10000:
            break
        total+=i
    return total

def while_sum(n):
    total=0
    i=0
    while i<n:
        i+=1
        if i%2==0:
            continue
        if i>50:
            break
        total+=i
    return total

def float_math(x:float,y:float):
    result=x*y
    result=result+x/y
    return result

def sum_squares(a:F64Array):
    total=0.0
    for i in range(len(a)):
        v=float(a[i])
        total+=v*v
    return total

def array_at(a:F64Array,i:int):
    return float(a[i])

def add(a,b):
    return a+b

def annotated(n:int):
    total:int=0
    for i in range(n):
        total+=i
    return total

def bool_logic(a,b):
    result=0
    if a>0 and not (b<0):
        result=1
    if a<0 or b>10:
        result+=2
    return result

def min_literal(n):
    return -9223372036854775808+n

def nested_control(n):
    total=0
    for i in range(n):
        for j in range(i, -1, -1):
            if j==2:
                continue
            if j==1:
                break
            total+=j
    return total

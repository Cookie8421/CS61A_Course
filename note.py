
# False Value in Python: False, 0, '', None (More to come)
# True Value in Python: Anything else

# 三目运算符：>>> x = 0
#                     >>> abs(1/x if x != 0 else 0)

# Naming Tips
# Names typically don’t matter for correctness but they matter a lot for composition
# Names should convey the meaning or purpose of the values to which they are bound.
# The type of value bound to the name is best documented in a function's docstring.
# Function names typically convey their effect (print), their behavior (triple), or the value returned (abs).
# Repeated compound expressions
# Meaningful parts of complex expressions
# Names can be long if they help document your code
# Names can be short if they represent generic quantities: counts, arbitrary functions, arguments to mathematical operations, etc.
# n, k, i - Usually integers
# x, y, z - Usually real numbers
# f, g, h - Usually functions

"""
week 1:

a = 1
b = 2
b, a = a+b, b
print(a)  #2
print(b) #3

# Importing and arithmetic with call expressions
from operator import add, mul
add(1, 2)
mul(4, 6)
mul(add(4, mul(4, 6)), add(3, 5))
add(4, mul(9, mul(add(4, mul(4, 6)), add(3, 5))))

# Objects
# Note: Download from http://composingprograms.com/shakespeare.txt
shakes = open('shakespeare.txt')
text = shakes.read().split()
len(text)
text[:25]
text.count('the')
text.count('thou')
text.count('you')
text.count('forsooth')
text.count(',')

# Sets
words = set(text)
len(words)
max(words)
max(words, key=len)

# Reversals
'draw'[::-1]
{w for w in words if w == w[::-1] and len(w)>4}
{w for w in words if w[::-1] in words and len(w) == 4}
{w for w in words if w[::-1] in words and len(w) > 6}

# Imports
from math import pi
pi * 71 / 223
from math import sin
sin(pi/2)

# Function values
max
max(3, 4)
f = max
f
f(3, 4)
max = 7
f(3, 4)
f(3, max)
f = 2
# f(3, 4)
__builtins__.max

# User-defined functions
from operator import add, mul

def square(x):
    return mul(x, x)

square(21)
square(add(2, 5))
square(square(3))

def sum_squares(x, y):
    return add(square(x), square(y))
sum_squares(3, 4)
sum_squares(5, 12)

# area function
def area():
    return pi * radius * radius
area()
radius = 20
area()
radius = 10
area()

# Name conflicts
def square(square):
    return mul(square, square)
square(4)


# Print
-2
print(-2)
'Go Bears'
print('Go Bears')
print(1, 2, 3)
None
print(None)
x = -2
x
x = print(-2)
x
print(print(1), print(2))

# Addition/Multiplication
2 + 3 * 4 + 5
(2 + 3) * (4 + 5)

# Division
618 / 10
618 // 10
618 % 10
from operator import truediv, floordiv, mod
floordiv(618, 10)
truediv(618, 10)
mod(618, 10)

# Approximation
5 / 3
5 // 3
5 % 3

# Multiple return values
def divide_exact(n, d):
    return n // d, n % d
quotient, remainder = divide_exact(618, 10)

# Dostrings, doctests, & default arguments
def divide_exact(n, d=10):
    Return the quotient and remainder of dividing N by D.

    >>> quotient, remainder = divide_exact(618, 10)
    >>> quotient
    61
    >>> remainder
    8

    return floordiv(n, d), mod(n, d)

# Return

def end(n, d):
    Print the final digits of N in reverse order until D is found.

    >>> end(34567, 5)
    7
    6
    5
    
    while n > 0:
        last, n = n % 10, n // 10
        print(last)
        if d == last:
            return None

def search(f):
    #Return the smallest non-negative integer x for which f(x) is a true value.
    x = 0
    while True:
        if f(x):
            return x
        x += 1

def is_three(x):
    Return whether x is three.

    >>> search(is_three)
    3

    return x == 3

def square(x):
    return x * x

def positive(x):
    A function that is 0 until square(x)-100 is positive.

    >>> search(positive)
    11

    return max(0, square(x) - 100)

def invert(f):
    Return a function g(y) that returns x such that f(x) == y.

    >>> sqrt = invert(square)
    >>> sqrt(16)
    4

    return lambda y: search(lambda x: f(x) == y)

# Control

def if_(c, t, f):
    if c:
        t
    else:
        f

from math import sqrt

def real_sqrt(x):
    Return the real part of the square root of x.

    >>> real_sqrt(4)
    2.0
    >>> real_sqrt(-4)
    0.0

    if x > 0:
        return sqrt(x)
    else:
        return 0.0
    if_(x > 0, sqrt(x), 0.0)

# Control Expressions

def has_big_sqrt(x):
    #Return whether x has a big square root.

    >>> has_big_sqrt(1000)
    True
    >>> has_big_sqrt(100)
    False
    >>> has_big_sqrt(0)
    False
    >>> has_big_sqrt(-1000)
    False
    
    return x > 0 and sqrt(x) > 10

def reasonable(n):
    Is N small enough that 1/N can be represented?

    >>> reasonable(100)
    True
    >>> reasonable(0)
    True
    >>> reasonable(-100)
    True
    >>> reasonable(10 ** 1000)
    False
    
    return n == 0 or 1/n != 0.0

from math import pi, sqrt

def area_square(r):
    Return the area of a square with side length R.
    return r * r

def area_circle(r):
    Return the area of a circle with radius R.
    return r * r * pi

def area_hexagon(r):
    Return the area of a regular hexagon with side length R.
    return r * r * 3 * sqrt(3) / 2

def area(r, shape_constant):
    Return the area of a shape from length measurement R.
    assert r > 0, 'A length must be positive'
    return r * r * shape_constant

def area_square(r):
    return area(r, 1)

def area_circle(r):
    return area(r, pi)

def area_hexagon(r):
    return area(r, 3 * sqrt(3) / 2)

# Functions as arguments

def sum_naturals(n):
    Sum the first N natural numbers.

    >>> sum_naturals(5)
    15

    total, k = 0, 1
    while k <= n:
        total, k = total + k, k + 1
    return total

def sum_cubes(n):
    Sum the first N cubes of natural numbers.

    >>> sum_cubes(5)
    #225

    total, k = 0, 1
    while k <= n:
        total, k = total + pow(k, 3), k + 1
    return total

def identity(k):
    return k

def cube(k):
    return pow(k, 3)

def summation(n, term):
    Sum the first N terms of a sequence.

    >>> summation(5, cube)
    225

    total, k = 0, 1
    while k <= n:
        total, k = total + term(k), k + 1
    return total

from operator import mul

def pi_term(k):
    return 8 / mul(k * 4 - 3, k * 4 - 1)

summation(1000000, pi_term)


# Local function definitions; returning functions

def make_adder(n):
    Return a function that takes one argument K and returns K + N.

    >>> add_three = make_adder(3)
    >>> add_three(4)
    7

    def adder(k):
        return k + n
    return adder

make_adder(2000)(19)


# Example: Sound

from wave import open
from struct import Struct
from math import floor

frame_rate = 11025

def encode(x):
    Encode float x between -1 and 1 as two bytes.
    (See https://docs.python.org/3/library/struct.html)
    
    i = int(16384 * x)
    return Struct('h').pack(i)

def play(sampler, name='song.wav', seconds=2):
    Write the output of a sampler function as a wav file.
    (See https://docs.python.org/3/library/wave.html)
    
    out = open(name, 'wb')
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(frame_rate)
    t = 0
    while t < seconds * frame_rate:
        sample = sampler(t)
        out.writeframes(encode(sample))
        t = t + 1
    out.close()

def tri(frequency, amplitude=0.3):
    A continuous triangle wave.
    period = frame_rate // frequency
    def sampler(t):
        saw_wave = t / period - floor(t / period + 0.5)
        tri_wave = 2 * abs(2 * saw_wave) - 1
        return amplitude * tri_wave
    return sampler

c_freq, e_freq, g_freq = 261.63, 329.63, 392.00

play(tri(e_freq))

def note(f, start, end, fade=.01):
    Play f for a fixed duration.
    def sampler(t):
        seconds = t / frame_rate
        if seconds < start:
            return 0
        elif seconds > end:
            return 0
        elif seconds < start + fade:
            return (seconds - start) / fade * f(t)
        elif seconds > end - fade:
            return (end - seconds) / fade * f(t)
        else:
            return f(t)
    return sampler

play(note(tri(e_freq), 1, 1.5))

def both(f, g):
    return lambda t: f(t) + g(t)

c = tri(c_freq)
e = tri(e_freq)
g = tri(g_freq)
low_g = tri(g_freq / 2)

play(both(note(e, 0, 1/8), note(low_g, 1/8, 3/8)))

play(both(note(c, 0, 1), both(note(e, 0, 1), note(g, 0, 1))))

def mario(c, e, g, low_g):
    z = 0
    song = note(e, z, z + 1/8)
    z += 1/8
    song = both(song, note(e, z, z + 1/8))
    z += 1/4
    song = both(song, note(e, z, z + 1/8))
    z += 1/4
    song = both(song, note(c, z, z + 1/8))
    z += 1/8
    song = both(song, note(e, z, z + 1/8))
    z += 1/4
    song = both(song, note(g, z, z + 1/4))
    z += 1/2
    song = both(song, note(low_g, z, z + 1/4))
    return song

def mario_at(octave):
    c = tri(octave * c_freq)
    e = tri(octave * e_freq)
    g = tri(octave * g_freq)
    low_g = tri(octave * g_freq / 2)
    return mario(c, e, g, low_g)

play(both(mario_at(1), mario_at(1/2)))


# Composition

def compose1(f, g):
    Return a function that composes f and g.

    f, g -- functions of a single argument

    def h(x):
        return f(g(x))
    return h

def triple(x):
    return 3 * x

squiple = compose1(square, triple)
tripare = compose1(triple, square)
squadder = compose1(square, make_adder(2))



# Currying

from operator import add, mul

def curry2(f):
    Curry a two-argument function.

    >>> m = curry2(add)
    >>> add_three = m(3)
    >>> add_three(4)
    7
    >>> m(2)(1)
    3
    
    def g(x):
        def h(y):
            return f(x, y)
        return h
    return g

week 2:

# Functional arguments

def apply_twice(f, x):
    >>> apply_twice(square, 2)
    16
    >>> from math import sqrt
    >>> apply_twice(sqrt, 16)
    2.0

    return f(f(x))

def square(x):
    return x * x

result = apply_twice(square, 2)

# Functional return values

def make_adder(n):
    Return a function that takes one argument k and returns k + n.

    >>> add_three = make_adder(3)
    >>> add_three(4)
    7
    
    def adder(k):
        return k + n
    return adder

# Lexical scope and returning functions

def f(x, y):
    return g(x)

def g(a):
    return a + y

# This expression causes an error because y is not bound in g.
# f(1, 2)

# Composition

def compose1(f, g):
    Return a function that composes f and g.

    f, g -- functions of a single argument
    
    def h(x):
        return f(g(x))
    return h

def triple(x):
    return 3 * x

squiple = compose1(square, triple)
tripare = compose1(triple, square)
squadder = compose1(square, make_adder(2))

# Lambda expressions

x = 10
square = x * x
square = lambda x: x * x
square(4)


# Self Reference

def print_all(k):
    #Print all arguments of repeated calls.

    >>> f = print_all(1)(2)(3)(4)(5)
    1
    2
    3
    4
    5
    
    print(k)
    return print_all

def print_sums(n):
    Print all sums of arguments of repeated calls.

    >>> f = print_sums(1)(2)(3)(4)(5)
    1
    3
    6
    10
    15

    print(n)
    def next_sum(k):
        return print_sums(n+k)
    return next_sum

def print_sums(n):
    #Print all sums of arguments of repeated calls.

    >>> f = print_sums(1)(2)(3)(4)(5)
    1
    3
    6
    10
    15

    print(n)
    def next_sum(k, j):
        return print_sums(n+k+j)
    return next_sum



def split(n):
    Split a positive integer into all but its last digit and its last digit.
    return n // 10, n % 10

def sum_digits(n):
    Return the sum of the digits of positive integer n.

    >>> sum_digits(9)
    9
    >>> sum_digits(18117)
    18
    >>> sum_digits(9437184)
    36
    >>> sum_digits(11408855402054064613470328848384)
    126
    
    if n < 10:
        return n
    else:
        all_but_last, last = split(n)
        return sum_digits(all_but_last) + last

# Iteration vs recursion

def fact_iter(n):
    total, k = 1, 1
    while k <= n:
        total, k = total * k, k + 1
    return total

def fact(n):
    if n == 0:
        return 1
    else:
        return n * fact(n-1)

# Luhn algorithm: mutual recursion

def luhn_sum(n):
    Return the digit sum of n computed by the Luhn algorithm.

    >>> luhn_sum(2)
    2
    >>> luhn_sum(12)
    4
    >>> luhn_sum(42)
    10
    >>> luhn_sum(138743)
    30
    >>> luhn_sum(5105105105105100) # example Mastercard
    20
    >>> luhn_sum(4012888888881881) # example Visa
    90
    >>> luhn_sum(79927398713) # from Wikipedia
    70
    
    if n < 10:
        return n
    else:
        all_but_last, last = split(n)
        return luhn_sum_double(all_but_last) + last

def luhn_sum_double(n):
    #Return the Luhn sum of n, doubling the last digit.
    all_but_last, last = split(n)
    luhn_digit = sum_digits(2 * last)
    if n < 10:
        return luhn_digit
    else:
        return luhn_sum(all_but_last) + luhn_digit

# Converting iteration to recursion

def sum_digits_iter(n):
    #Sum digits iteratively.

    >>> sum_digits_iter(11408855402054064613470328848384)
    126
    
    digit_sum = 0
    while n > 0:
        n, last = split(n)
        digit_sum = digit_sum + last
    return digit_sum

def sum_digits_rec(n, digit_sum):
    #Sum digits using recursion, based on iterative version.

    >>> sum_digits_rec(11408855402054064613470328848384, 0)
    126
    
    if n == 0:
        return digit_sum
    else:
        n, last = split(n)
        return sum_digits_rec(n, digit_sum + last)

# Ordering

def cascade(n):
    #Print a cascade of prefixes of n.

    >>> cascade(1234)
    1234
    123
    12
    1
    12
    123
    1234
    
    if n < 10:
        print(n)
    else:
        print(n)
        cascade(n//10)
        print(n)

def cascade2(n):
    #Print a cascade of prefixes of n.
    print(n)
    if n >= 10:
        cascade(n//10)
        print(n)

def inverse_cascade(n):
    #Print an inverse cascade of prefixes of n.
    
    >>> inverse_cascade(1234)
    1
    12
    123
    1234
    123
    12
    1
    
    grow(n)
    print(n)
    shrink(n)

def f_then_g(f, g, n):
    if n:
        f(n)
        g(n)

grow = lambda n: f_then_g(grow, print, n//10)
shrink = lambda n: f_then_g(print, shrink, n//10)


# Tree recursion

def fib(n):
    #Compute the nth Fibonacci number.

    >>> fib(8)
    21

    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-2) + fib(n-1)


#Knapsack

def knap(n, k):
    if n == 0:
        return k == 0
    with_last = knap(n // 10, k - n % 10)
    without_last = knap(n // 10, k)
    return with_last or without_last


#Count Partitions

def count_partitions(n, m):
    #Count the partitions of n using parts up to size m.

    >>> count_partitions(6, 4)
    9
    >>> count_partitions(10, 10)
    42

    if n == 0:
        return 1
    elif n < 0:
        return 0
    elif m == 0:
        return 0
    else:
        with_m = count_partitions(n-m, m)
        without_m = count_partitions(n, m-1)
        return with_m + without_m

def knap(n, k):
    if n == 0:
        return k == 0
    with_last = knap(n // 10, k - n % 10)
    without_last = knap(n // 10, k)
    return with_last or without_last


#Binary Print

def all_nums(k):
    def h(k, prefix):
        if k == 0:
            print(prefix)
            return
        h(k - 1, prefix * 10)
        h(k - 1, prefix * 10 + 1)
    h(k, 0)


week 3:

# Lists

odds = [41, 43, 47, 49]
len(odds)
odds[1]
odds[0] - odds[3] + len(odds)
odds[odds[3]-odds[2]]

# Containers

digits = [1, 8, 2, 8]
1 in digits
'1' in digits
[1, 8] in digits
[1, 2] in [[1, 2], 3]

# For statements

def count_while(s, value):
    #Count the number of occurrences of value in sequence s.
    >>> count_while(digits, 8)
    2

    total, index = 0, 0
    while index < len(s):
        if s[index] == value:
            total = total + 1
        index = index + 1
    return total

def count_for(s, value):
    #Count the number of occurrences of value in sequence s.

    >>> count_for(digits, 8)
    2

    total = 0
    for elem in s:
        if elem == value:
            total = total + 1
    return total


def count_same(pairs):
    #Return how many pairs have the same element repeated.

    >>> pairs = [[1, 2], [2, 2], [2, 3], [4, 4]]
    >>> count_same(pairs)
    2

    same_count = 0
    for x, y in pairs:
        if x == y:
            same_count = same_count + 1
    return same_count


# Ranges

list(range(5, 8))   # >>>[5, 6, 7]
list(range(4))  # >>>[0, 1, 2, 3]
len(range(4))   # >>>4

def sum_below(n):
    total = 0
    for i in range(n):
        total += i
    return total

def cheer():
    for _ in range(3):
        print('Go Bears!')
    >>>Go Bears!
    >>>Go Bears!
    >>>Go Bears!
    >>>None


# List comprehensions

odds = [1, 3, 5, 7, 9]
[x+1 for x in odds]
[x for x in odds if 25 % x == 0]

#"Unpacking" a list
a = [1, 2, 3, 4]
b, c, d, e = a
print(b)

from operator import getitem
>>> getitem(pair, 0)
1
>>> getitem(pair, 1)
2

def divisors(n):
    #Return the integers that evenly divide n.

    >>> divisors(1)
    [1]
    >>> divisors(4)
    [1, 2]
    >>> divisors(12)
    [1, 2, 3, 4, 6]
    >>> [n for n in range(1, 1000) if sum(divisors(n)) == n]
    [1, 6, 28, 496]
    
    return [1] + [x for x in range(2, n) if n % x == 0]

    >>> print([1,2,3] + sum([[4,5,6]],[]))
    [1, 2, 3, 4, 5, 6]


# Rational arithmetic

def add_rational(x, y):
    #The sum of rational numbers X and Y.
    nx, dx = numer(x), denom(x)
    ny, dy = numer(y), denom(y)
    return rational(nx * dy + ny * dx, dx * dy)

def mul_rational(x, y):
    #The product of rational numbers X and Y.
    return rational(numer(x) * numer(y), denom(x) * denom(y))

def rationals_are_equal(x, y):
    #True iff rational numbers X and Y are equal.
    return numer(x) * denom(y) == numer(y) * denom(x)

def print_rational(x):
    #Print rational X.
    print(numer(x), "/", denom(x))


# Constructor and selectors

def rational(n, d):
    #A representation of the rational number N/D.
    return [n, d]

def numer(x):
    #Return the numerator of rational number X.
    return x[0]

def denom(x):
    #Return the denominator of rational number X.
    return x[1]


# Improved specification

from fractions import gcd
def rational(n, d):
    #A representation of the rational number N/D.
    g = gcd(n, d)
    return [n//g, d//g]

def numer(x):
    #Return the numerator of rational number X in lowest terms and having
    the sign of X.
    return x[0]

def denom(x):
    #Return the denominator of rational number X in lowest terms and positive.
    return x[1]


# Functional implementation

def rational(n, d):
    #A representation of the rational number N/D.
    g = gcd(n, d)
    n, d = n//g, d//g
    def select(name):
        if name == 'n':
            return n
        elif name == 'd':
            return d
    return select

def numer(x):
    #Return the numerator of rational number X in lowest terms and having
    #the sign of X.
    return x('n')

def denom(x):
    #Return the denominator of rational number X in lowest terms and positive.
    return x('d')

# Dicts

def dict_demos():
    numerals = {'I': 1, 'V': 5, 'X': 10}
    numerals['X']
    numerals.values()
    list(numerals.values())
    sum(numerals.values())
    dict([(3, 9), (4, 16), (5, 25)])
    numerals.get('X', 0)
    numerals.get('X-ray', 0)
    {x: x*x for x in range(3,6)}

    {1: 2, 1: 3}
    {[1]: 2}
    {1: [2]}

Debugging:
1.Assertions: Use
    assert isinstance(x, int)
2.Testing: Doctests
    ○To run: python3 -m doctest file.py
3.Print Debugging
    ○print(“Debug: x=”, x)
4.Interactive Debugging: REPL
    To use, run
    ○ python3 -i file.py
    ○ then run whatever python commands you want
     OK integration:
    ○ python3 ok -q whatever -i
    ○ Starts out already having run code for that question
5.PythonTutor
    You can also step through your code line
    by line on PythonTutor:
    ○ Just copy your code into tutor.cs61a.org
     Ok integration:
    ○ python ok -q whatever --trace
6.Error Message Patterns




week 4:
def make_withdraw(balance):
    def withdraw(amount):
        nonlocal balance
        if amount > balance:
            return 'Insufficient funds'
        balance = balance - amount
        return balance
    return withdraw

withdraw = make_withdraw(100)

print(withdraw(50))
print(withdraw(20))

def make_withdraw_list(balance):
    b = [balance]
    def withdraw(amount):
        if amount > b[0]:
            return 'Insufficient funds'
        b[0] = b[0] - amount
        return b[0]
    return withdraw

withdraw = make_withdraw_list(100)

print(withdraw(50))
print(withdraw(20))



def make_adder(n):
    def add(x):
        nonlocal n
        n = n + 2
        return x + n
    return add
add = make_adder(2)

print(add(3))
print(add(3))



nonlocal
nonlocal是 Python3 新增的作用域关键词。
Python对闭包的支持一直不是很完美，在 Python2 中，闭包函数可以读取到父级函数的变量，但是无法修改变量的值，
为此，我们经常要把变量声明为global全局变量，这样就打破了闭包的性质。



def f(x):
    x = 4
    def g(y):
        def h(z):
            nonlocal x
            x = x + 1
            return x + y + z
        return h
    return g
a = f(1)
b = a(2)
print(b(3) + b(4))
为了解决这个问题，Python3 引入了nonlocal，如上例代码，我们使用声明了nonlocal n之后，就可以正常操作。



def oski(bear):
    def cal(berk):
        nonlocal bear
        if bear(berk) == 0:
            return [berk+1, berk-1]
        bear = lambda ley: berk-ley
        return [berk, cal(berk)]
    return cal(2)
print(oski(abs))
声明为nonlocal的变量如果变更，环境内的变量就改变了，会影响下次的调用



def make_withdraw_list(balance):
    b = [balance]
    def withdraw(amount):
        if amount > b[0]:
            return 'Insufficient funds'
        b[0] = b[0] - amount
        return b[0]
    return withdraw

withdraw = make_withdraw_list(100)
print(withdraw(25))
print(withdraw(25))
print(withdraw(25))
Only objects of mutable types can change: lists & dictionaries



def f(s = []):
    s.append(3)
    return len(s)
print(f())
print(f())
print(f())

for t, r in zip("a b c d", "a d"):
    print(t, r)



Higher Order Functions (Self Reference)
def print_delayed(x):
    def delay_print(y):
        print(x)
        return print_delayed(y)
    return delay_print

f = print_delayed(1)
f = f(2)
f = f(3)


def print_n(n):
    def inner_print(x):
        if n <= 0:
            print("done")
        else:
            print(x)
        return print_n(n-1)
    return inner_print

g = print_n(1)

g("first")("second")("third")

print('berr' == 'berry')


"""


#!/usr/bin/python
# -*- coding: UTF-8 -*-
from hw.hw01.hw01.hw01 import a_plus_abs_b
from hw.hw01.hw01.hw01 import two_of_three
from hw.hw01.hw01.hw01 import largest_factor
from hw.hw01.hw01.hw01 import if_function
from hw.hw01.hw01.hw01 import with_if_statement
from hw.hw01.hw01.hw01 import with_if_function
from hw.hw01.hw01.hw01 import hailstone

print(a_plus_abs_b(2, 3))

print(a_plus_abs_b(2, -3))

print(two_of_three(1, 2, 3))

print(two_of_three(5, 3, 1))

print(two_of_three(10, 2, 8))

print(two_of_three(5, 5, 5))

print(largest_factor(15))

print(largest_factor(80))

print(largest_factor(13))

print(if_function(True, 2, 3))

print(if_function(False, 2, 3))

print(if_function(3==2, 3+2, 3-2))

print(if_function(3>2, 3+2, 3-2))

result = with_if_statement()
print(result)

result = with_if_function()
print(result)

a = hailstone(10)
print(a)

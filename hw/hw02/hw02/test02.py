#!/usr/bin/python
# -*- coding: UTF-8 -*-
from operator import add, mul, sub
from hw02 import product
from hw02 import square
from hw02 import identity
from hw02 import triple
from hw02 import increment
from hw02 import accumulate
from hw02 import summation_using_accumulate
from hw02 import product_using_accumulate
from hw02 import make_repeater
from hw02 import zero
from hw02 import successor
from hw02 import one
from hw02 import two
from hw02 import church_to_int
from hw02 import add_church
from hw02 import mul_church
from hw02 import pow_church


print(product(3, identity))

print(product(5, identity))

print(product(3, square))

print(product(5, square))

print(product(3, increment))

print(product(3, triple))

print(accumulate(add, 0, 5, identity))

print(accumulate(add, 11, 5, identity))

print(accumulate(add, 11, 0, identity))

print(accumulate(add, 11, 3, square))

print(accumulate(mul, 2, 3, square))

print(accumulate(lambda x, y: x + y + 1, 2, 3, square))

print(accumulate(lambda x, y: 2 * (x + y), 2, 3, square))

print(accumulate(lambda x, y: (x + y) % 17, 19, 20, square))

print(summation_using_accumulate(5, square))

print(summation_using_accumulate(5, triple))

print(product_using_accumulate(4, square))

print(product_using_accumulate(6, triple))

print(make_repeater(triple, 5)(1))

print(make_repeater(square, 2)(5))

print(make_repeater(square, 4)(5))

print(make_repeater(square, 0)(5))

print(church_to_int(zero))

print(church_to_int(one))

print(church_to_int(two))
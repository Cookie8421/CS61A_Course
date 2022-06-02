#!/usr/bin/python
# -*- coding: UTF-8 -*-
from operator import add, mul, sub
from hw.hw02.hw02.hw02 import product
from hw.hw02.hw02.hw02 import square
from hw.hw02.hw02.hw02 import identity
from hw.hw02.hw02.hw02 import triple
from hw.hw02.hw02.hw02 import increment
from hw.hw02.hw02.hw02 import accumulate
from hw.hw02.hw02.hw02 import summation_using_accumulate
from hw.hw02.hw02.hw02 import product_using_accumulate
from hw.hw02.hw02.hw02 import make_repeater

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

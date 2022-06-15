
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
def square(x):
    return x * x
def make_adder(n):
    def add(x):
        return x + n
    return add
def compose1(f, g):
    def h(x):
        return f(g(x))
    return h

print(compose1(square, make_adder(2))(3))
 25


def pirate(arggg):
    print("matey")
    def plunder(arggg):
        return arggg
    return plunder
print(pirate(pirate))
print( pirate(pirate(pirate)) (5) (7) )

# Example: Sound

from wave import open
from struct import Struct
from math import floor

frame_rate = 11025

def encode(x):
    i = int(16384 * x)
    return Struct('h').pack(i)
    #Encode float x between -1 and 1 as two bytes. (See https://docs.python.org/3/library/struct.html)


def play(sampler, name='song.wav', seconds=2):
    #Write the output of a sampler function as a wav file.(See https://docs.python.org/3/library/wave.html)

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
    #A continuous triangle wave.
    period = frame_rate // frequency
    def sampler(t):
        saw_wave = t / period - floor(t / period + 0.5)
        tri_wave = 2 * abs(2 * saw_wave) - 1
        return amplitude * tri_wave
    return sampler

c_freq, e_freq, g_freq = 261.63, 329.63, 392.00

play(tri(e_freq))

def note(f, start, end, fade=.01):
    #Play f for a fixed duration.
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
"""



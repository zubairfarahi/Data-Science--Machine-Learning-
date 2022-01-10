"""
    Ex. 12: Scrieti un decorator pt f care sa scrie outputul lui f intr-un
    fisier, "output12.data".
    Observatii: f nu e la fel ca la ex 11.
"""

import sys

def dec(func):
    def wrapper(*args):
        f = open('output12.data', 'w')
        f.write(*args)
        func(*args)
        f.close()
    return wrapper


# decorate me
@dec
def f(x):
    print(x)


f("hello world")
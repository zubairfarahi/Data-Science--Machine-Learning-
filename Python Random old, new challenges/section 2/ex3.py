"""
    Ex. 11: Scrieti un decorator care sa logheze outputul unei functii
    pe care o decoreaza, intr-un fisier "output11.data".
    https://www.w3schools.com/python/python_file_write.asp
    Functia decorata este f.
"""


def dec(func):
    def wrapper():
        f = open("output11.data", 'w')
        f.write(func())
        f.close()

    return wrapper


# decorate me
@dec
def f():
    return "CMI"



f()
"""
    Ex. 13: Scrieti un decorator care sa modifice modul de functionare
    al functiei f. Puteti alege voi cum. Momentan, f intoarce 'cmi', un exemplu
    ar fi sa intoarca 'CmI' dupa aplicarea decoratorului.
"""


def dec(func):
    def wrapper():
        x = func()[0:1].upper() + func()[1:len(func()) - 1] + func()[len(func()) - 1:].upper()
        print(x)


    return wrapper

# decoarate me
@dec
def f():
    return 'zubair'



f()
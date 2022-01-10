"""
    Ex. 17: Scrieti un decorator care scrie outputul unei functii intr-un fisier
    "output17.data", dar sa nu suprascrie fisierul daca scriptul e rulat de
    mai multe ori, iar contentul nou sa fie pe o noua linie.
    Scrieti o functie f care sa primeasca un intreg (x) ca parametru si sa
    genereze un string aleator din x litere.
    Decorati f cu decoratorul de mai sus.
    Exemplu:
        la prima rulare, x = 3, stringul generat = 'cmi', fisierul arata asa:
            cmi
        la a doua rulare, x = 6, stringul generat = 'cmicmi', fisierul arata:
            cmi
            cmicmi
        la a treia rulare, x = 1, stringul generat = 'b', fisierul arata asa:
            cmi
            cmicmi
            b
"""

import random
def dec(func):
    def wrapper(*args):
       with open('output17.data','a') as f:      
           f.write(f"{func(*args)} \n")
           f.close()

    return wrapper

@dec
def f(x):
    newStr = ''
    for i in range(0, x):
        newStr += random.choice(['a','c','m','i','u','z','s','w'])
    return newStr


x = int(input())
f(x)
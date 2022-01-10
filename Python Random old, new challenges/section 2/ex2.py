"""
    Ex. 6: Scrieti o functie cu un singur parametru (string) care
    intoarce un string cu toate literele stringului primit +1 (adica urmatoarea
    litera din alfabet)
    Raspuns:
        - func('aabbcc')
            ---> 'bbccdd'
"""

def getString(str):
    return "".join([chr(ord(x) + 1) for x in str])



print(getString("aabbcc"))
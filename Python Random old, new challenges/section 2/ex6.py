"""
    Ex. 16: Scrieti o functie upper care sa intoarca un text uppercase complet,
    primind un parametru my_str (string).
    --> f('cmi') --> 'CMI'
    Scrieti o functie lower care sa intoarca un text lowercase complet,
    primind un parametru my_str (string).
    --> f('CMI') --> 'cmi'
    Veti primi un input de la tastatura, un string.
    Scrieti o alta functie call_changers, care sa primeasca o functie ca si
    parametru, iar daca inputul are un numar par de caractere, va printa inputul
    cu uppercase, altfel, va printa inputut lowercase.
    Exemplu:
        - veti primi input: 'ceva'
            ---> CEVA
        - veti primi input: 'cEVa1'
            ---> ceva1
"""

def call_changers(func):
    def wrapper(*args):
        print(func(*args))
    return wrapper

@call_changers
def upper(my_str):
    return my_str.upper()
@call_changers
def lower(my_str):
    return my_str.lower()




val = input()
if len(val) % 2 == 0:
    upper(val)
else: 
    lower(val)

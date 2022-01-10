"""
    Ex. 4: Below you have 2 functions:
        - add_prefix -> add a received prefix as a parameter to a string
        also received as a parameter
        - generate_random_str -> generates a random string of size X,
        X being a parameter
        You will need to write a function that adds a suffix (add_suffix),
        but that suffix must not contain any letter in the prefix.
        You will receive the suffix and the prefix as input from the keyboard, as well
        X, which is the length of the random string.
        If the suffix has a letter that the prefix has,
        you will ask for a new suffix, until a correct one is given, or until
        has been tried 3 times. The 4th time you will print the string without a suffix.
    Rezultatul ar trebui sa arate asa:
        - pentru prefix = 'bla', sufix = 'cmi', x = 3 si un string aleator 'lol'
            ---> 'blalolcmi'
        - pentru prefix = 'bla', sufix = (pe rand, 'ba', 'la', 'bla') x = 3
        si un string aleator 'lol'
            ---> 'blalol'
    Orice e neclar, ma intrebati pe discord la orice ora, fara probleme.
"""
import random



def add_suffix(pfx):
    i = 0
    suf = input("Enter new suffix\n")
    while i < 3:
        l1 = [*suf, *pfx]
        l2 = set(l1)
        if len(l1) == len(l2):
            return suf
        suf = input(f"plz Enter new {i} suffix\n")
        i += 1
            
    return ""

def add_prefix(pfx, rand_str):
    
    return pfx + rand_str + add_suffix(pfx) 


# Nu am spus ca stringul generat aleator trebuie sa contina toate literele
def generate_random_str(str_length):
    rand_str = ''
    while str_length:
        str_length -= 1
        rand_str += random.choice(['a', 'x', 'c', 'm', 'i'])
    print(f"The generated string is {rand_str}")
    return rand_str


prefix = input('Give me an prefix\n')
x = int(input('Give me a number to generate the random string\n'))

print(add_prefix(prefix, generate_random_str(x)))
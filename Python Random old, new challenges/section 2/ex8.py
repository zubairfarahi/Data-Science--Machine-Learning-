"""
    Ex. 19: Scrieti o functie care primeste un string ca si parametru,
    creeaza un fisier cu numele parametrului primit (.json) si scrie in el
    un dictionar de 4 elemente aleatoare unde key = int, iar value = string,
    iar stringul sa aiba intre 3 si 6 caractere si key sa fie intre 0 si 10.
    Exemplu:
        f('ceva')
        ---> generez ceva.json ca si fisier
        ---> generez un dictionar
            {
                1: 'blabla',
                5: 'cmi',
                7: 'cmi22',
                10: 'balqef'
            }
"""
import random

def make_random_string():
    str = ""
    for i in range(0, random.randrange(3,6)):
        str += random.choice(['a', 'x', 'c', 'm', 'i','d','y','u','e'])
    
    return str
            
def build_dic():
    dic = {}
    for i in random.sample(range(0,10),4):
        dic[i] = make_random_string()
    return dic

def write_in_jason(parm):
    dic = {1: 'hi', 2:'hello'}
    with open(parm+".json", 'a+') as f:
        f.writelines(str(build_dic())+"\n")
        f.close()



write_in_jason("file")
    
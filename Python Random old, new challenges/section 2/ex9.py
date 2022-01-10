"""
    Ex. 20: Deschideti fisierul .json creat la exercitiul anterior, cititi
    continutul si returnati un dictionar (dictionarul de acolo).
    Toate astea le veti face intr-o functie read_from_file(file), unde
    file este numele fisierului primit dat ca parametru.
"""

import ast
def read_from_file(file):
    lines = []
    with open(file, "r+") as r:
        lines = r.readlines()
        r.close()
    
    for x in lines:
        print(ast.literal_eval(x))


read_from_file(file="file.json")
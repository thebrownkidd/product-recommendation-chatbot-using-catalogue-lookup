import pandas as pd

def MakeList(cat_string):
    cats_i = cat_string[1:-1].split(',')
    cats_list = []
    for cat in cats_i:
        cats_list.append(cat.strip().strip("'"))
    return cats_list

# initialise the script for use as a module
if __name__ == "__main__":
    pass
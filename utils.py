import pandas as pd
from PIL import Image
import requests
from io import BytesIO

def MakeList(cat_string):
    cats_i = cat_string[1:-1].split(',')
    cats_list = []
    for cat in cats_i:
        cats_list.append(cat.strip().strip("'"))
    return cats_list

def GetImgFromLink(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

if __name__ == "__main__":
    pass
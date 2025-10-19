# ...existing code...
from utils import GetImgFromLink, MakeList
import pandas as pd
import json
import os
import imghdr
import shutil
import requests
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
os.makedirs("images", exist_ok=True)

df = pd.read_csv("catalogue.csv")

# select columns as a copy to avoid SettingWithCopyWarning
df_to_keep = df.loc[:, ["uniq_id", "images", "categories"]].copy()
df_to_keep.loc[:, "categories"] = df_to_keep["categories"].apply(MakeList)
df_to_keep.loc[:, "images"] = df_to_keep["images"].apply(MakeList)

def _save_bytes_image(data: bytes, dest_path: str):
    with open(dest_path, "wb") as f:
        f.write(data)

def _ext_from_bytes(data: bytes, fallback="jpg"):
    ext = imghdr.what(None, data)
    return ext if ext else fallback

def _download_and_save(link: str, uid: str, idx: int):
    # try GetImgFromLink first
    try:
        res = GetImgFromLink(link)
    except Exception:
        res = None

    # If GetImgFromLink returned a filesystem path
    if isinstance(res, str) and os.path.exists(res):
        ext = os.path.splitext(res)[1].lstrip(".") or "jpg"
        dest = os.path.join("images", f"{uid}_{idx}.{ext}")
        shutil.copy(res, dest)
        return dest

    # If returned a PIL Image
    if isinstance(res, Image.Image):
        ext = "png"
        dest = os.path.join("images", f"{uid}_{idx}.{ext}")
        try:
            res.save(dest)
            return dest
        except Exception:
            pass

    # If returned bytes-like
    if isinstance(res, (bytes, bytearray)):
        ext = _ext_from_bytes(res)
        dest = os.path.join("images", f"{uid}_{idx}.{ext}")
        _save_bytes_image(res, dest)
        return dest

    # If returned a file-like object (BytesIO)
    if hasattr(res, "read"):
        try:
            data = res.read()
            if isinstance(data, str):
                data = data.encode()
            ext = _ext_from_bytes(data)
            dest = os.path.join("images", f"{uid}_{idx}.{ext}")
            _save_bytes_image(data, dest)
            return dest
        except Exception:
            pass

    # Fallback: download via requests
    try:
        r = requests.get(link, timeout=10)
        r.raise_for_status()
        data = r.content
        ext_guess = os.path.splitext(link)[1].lstrip(".")
        ext = _ext_from_bytes(data, fallback=ext_guess or "jpg")
        dest = os.path.join("images", f"{uid}_{idx}.{ext}")
        _save_bytes_image(data, dest)
        return dest
    except Exception:
        return None

# make a dictionary with uid as key and image paths and categories as values
img_cat_dict = {}
for index, row in tqdm(df_to_keep.iterrows(), total=len(df_to_keep), desc="rows", unit="row"):
    uid = str(row["uniq_id"])
    img_links = row["images"] or []
    categories = row["categories"]
    saved_imgs = []
    for i, link in enumerate(tqdm(img_links, desc=f"images {uid}", leave=False, unit="img")):
        if not link:
            continue
        saved = _download_and_save(link, uid, i)
        if saved:
            saved_imgs.append(saved)
    img_cat_dict[uid] = {
        "images": saved_imgs,
        "categories": categories
    }
with open("img_cat.json", "w", encoding="utf-8") as f:
    json.dump(img_cat_dict, f, ensure_ascii=False, indent=4)
# ...existing code...
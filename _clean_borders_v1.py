import os
import numpy as np
from glob import glob
import cv2

THRESH = 8.0
for p in glob('./some_dataset/*.png'):
    img = np.array(cv2.imread(p, cv2.IMREAD_COLOR_RGB))
    new_img = img.copy()
    h, w = img.shape[:2]
    row_mid = h//2

    uniform_rows = []
    for row_idx in range(h):
        row = img[row_idx]
        if np.var(row) <= THRESH:
            uniform_rows.append(row_idx)
    col_mid = w//2
    uniform_cols = []
    for col_idx in range(w):
        col = img[:, col_idx]
        if np.var(col) <= THRESH:
            uniform_cols.append(col_idx)
    row_min = 0
    row_max = -1
    if len(uniform_rows) > 0:
        uniform_rows = np.array(uniform_rows, dtype=np.uint32)
        row_upper = uniform_rows[uniform_rows < row_mid]
        row_lower = uniform_rows[uniform_rows > row_mid]
        if len(row_upper) > 0:
            row_min = row_upper.max()
        if len(row_lower) > 0:
            row_max = row_lower.min()
    col_min = 0
    col_max = -1
    if len(uniform_cols) > 0:
        uniform_cols = np.array(uniform_cols, dtype=np.uint32)
        col_upper = uniform_cols[uniform_cols < col_mid]
        col_lower = uniform_cols[uniform_cols > col_mid]
        if len(col_upper) > 0:
            col_min = col_upper.max()
        if len(col_lower) > 0:
            col_max = col_lower.min()
    print(f'[ {row_min} : {row_max} , {col_min} : {col_max} ]')
    if col_max == -1:
        col_max = w
    if row_max == -1:
        row_max = h
    new_img = new_img[row_min:row_max, col_min:col_max]
    new_path = p.replace('renamed', 'cleaned')
    cv2.imwrite(new_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    print(f"Wrote: {new_path}")
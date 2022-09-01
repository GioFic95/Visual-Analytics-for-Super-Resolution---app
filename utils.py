from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing


def get_df(csv: Path, types_dict: Dict[str, type]) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=types_dict)
    df.query("mask == False", inplace=True)
    return df


def compute_pca(df: pd.DataFrame, num_cols: List[str]):
    d_std = preprocessing.StandardScaler().fit_transform(df[num_cols])
    pca = PCA(n_components=4)
    d_pca = pca.fit_transform(d_std)
    df['pca_x'] = d_pca[:, 0]
    df['pca_y'] = d_pca[:, 1]


if __name__ == '__main__':
    types = {"name": str, "ssim": float, "psnr_rgb": float, "psnr_y": float, "lpips": float,
             "type": str, "mask": bool, "category": str}
    df = get_df(Path("./assets/test_results_all_isb.csv"), types)
    num_cols = ["ssim", "psnr_rgb", "psnr_y", "lpips"]
    compute_pca(df, num_cols)
    print(df)

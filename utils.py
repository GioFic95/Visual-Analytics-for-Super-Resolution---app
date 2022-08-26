from typing import Dict
from pathlib import Path
import pandas as pd


def get_df(csv: Path, types_dict: Dict[str, type]) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=types_dict)
    # df.query("mask == False", inplace=True)
    return df


def make_query(queries: dict, avg: bool = False) -> str:
    if avg:
        query_list = [v for k, v in queries.items() if v != "" and k != "parallel"]
    else:
        query_list = [v for v in queries.values() if v != ""]
    query = " & ".join(query_list)
    print("query:", query)
    return query


def adapt_benchmark(benchmark_path: Path, out_csv_path: Path):
    out_df = pd.DataFrame(columns=["filename", "MS-SSIM", "PSNR", "category", "quality", "size"])
    for size in ["1K", "SD"]:
        for pipeline in ["pipeline_1", "pipeline_2", "pipeline_3", "pipeline_4", "pipeline_5", "pipeline_6"]:
            input_path = benchmark_path / size / pipeline / f"{pipeline}_benchmark.csv"
            in_df = pd.read_csv(input_path)[["filename", "PSNR", "MS-SSIM", "quality"]]
            in_df["size"] = [size for _ in range(len(in_df))]
            in_df["category"] = [pipeline for _ in range(len(in_df))]
            out_df = pd.concat([out_df, in_df])
            print(out_df.filename)
    out_df["filename"] = out_df.apply(lambda x: f"{x.filename:0>5}_{x['size']}_{x.quality}_{x.category}.png", axis=1)
    print(out_df.filename)
    out_df.sort_values(["filename"], inplace=True)
    print(out_df.filename)
    out_df.rename(columns={"filename": "name"}, inplace=True)
    print(out_df.name)
    out_df.to_csv(out_csv_path, index=False)


def all_to_avg(all_df: pd.DataFrame) -> pd.DataFrame:
    res = all_df.copy()
    res["category"] = res.apply(lambda x: f"{x.category}_{x['size']}_{x.quality}", axis=1)
    res = res.groupby(by="category").agg({"MS-SSIM": "mean", "PSNR": "mean", "quality": "first", "size": "first"})\
        .reset_index()\
        .rename(columns={"category": "name"})
    return res


if __name__ == '__main__':
    bp = Path("H:/Drive condivisi/Underwater Computer Vision/Super-Resolution/Benchmark")
    ocp = Path(bp / "all.csv")
    adapt_benchmark(bp, ocp)

    df = get_df(bp / "all.csv",
                {"name": str, "MS-SSIM": float, "PSNR": float, "quality": str, "size": str, "category": str})
    print(all_to_avg(df))
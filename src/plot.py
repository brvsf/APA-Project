import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_results(path="results.csv"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(r["n"])
            rows.append({"n": n, "algorithm": "Dijkstra", "time": float(r["dijkstra_mean"])})
            rows.append({"n": n, "algorithm": "Binary Heap", "time": float(r["heap_mean"])})
            rows.append({"n": n, "algorithm": "BMSSP", "time": float(r["duan_mean"])})
    return pd.DataFrame(rows)

def main():
    df = load_results("results.csv")

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.1
    )

    plt.figure(figsize=(7.5, 4.5))

    sns.lineplot(
        data=df,
        x="n",
        y="time",
        hue="algorithm",
        marker="o",
        linewidth=1.3,
        markersize=5,
        palette="tab10"
    )

    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("SSSP Runtime Comparison")

    plt.yscale("log")

    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    main()

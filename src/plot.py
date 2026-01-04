import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "Dijkstra": "tab:blue",
    "Binary Heap": "tab:orange",
    "BMSSP": "tab:green",
}

def load_results(path="results.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("n")
    return df

def plot_means(df, out_path="benchmark_means.png"):
    plt.figure(figsize=(7.5, 4.5))

    plt.plot(
        df["n"], df["dijkstra_mean"],
        "o-", linewidth=1.3, markersize=4,
        color=COLORS["Dijkstra"], label="Dijkstra"
    )
    plt.plot(
        df["n"], df["heap_mean"],
        "o-", linewidth=1.3, markersize=4,
        color=COLORS["Binary Heap"], label="Binary Heap"
    )
    plt.plot(
        df["n"], df["duan_mean"],
        "o-", linewidth=1.3, markersize=4,
        color=COLORS["BMSSP"], label="BMSSP"
    )

    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("SSSP Runtime Comparison — Mean")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, facecolor="white")
    plt.show()

def plot_stds(df, out_path="benchmark_stds.png"):
    plt.figure(figsize=(7.5, 4.5))

    plt.plot(
        df["n"], df["dijkstra_std"],
        "o-", linewidth=1.3, markersize=4,
        color=COLORS["Dijkstra"], label="Dijkstra"
    )
    plt.plot(
        df["n"], df["heap_std"],
        "o-", linewidth=1.3, markersize=4,
        color=COLORS["Binary Heap"], label="Binary Heap"
    )
    plt.plot(
        df["n"], df["duan_std"],
        "o-", linewidth=1.3, markersize=4,
        color=COLORS["BMSSP"], label="BMSSP"
    )

    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Runtime standard deviation (seconds)")
    plt.title("SSSP Runtime Comparison — Standard Deviation")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, facecolor="white")
    plt.show()

def main():
    df = load_results("results.csv")
    plot_means(df)
    plot_stds(df)

if __name__ == "__main__":
    main()

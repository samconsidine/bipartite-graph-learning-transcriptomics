"""Script to make some plots for the report"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_lmplot(df: pd.DataFrame):
    sns.lmplot(
        data=df,
        x="Num Genes",
        y="Accuracy",
        hue="Experiment",
        size=7,
        aspect=1.4,
    )
    #plt.title(sys.argv[2])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of genes included (most variable)')
    #plt.tight_layout()
    plt.show()


def load_data(fname: str) -> pd.DataFrame:
    df = pd.read_csv(fname)
    df = df.melt(id_vars=["Num Genes"], var_name="Experiment", value_name="Accuracy")
    df = df.loc[df['Accuracy'] > 0.5]

    return df

def main():
    df = load_data(sys.argv[1])
    make_lmplot(df)

if __name__ == "__main__":
    main()

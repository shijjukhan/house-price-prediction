import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairwise_relationships(data):
    sns.pairplot(data)
    plt.show()

def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()


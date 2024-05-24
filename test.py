import click
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator

@click.command()
def main():
    # List all .xlsx files in the current directory
    files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    if not files:
        click.echo("No Excel files found in the current directory.")
        return

    click.echo("Available Excel files:")
    for i, file in enumerate(files):
        click.echo(f"{i}: {file}")

    file_index = click.prompt('Select the file to process by entering the index number', type=int)
    if file_index < 0 or file_index >= len(files):
        click.echo("Invalid selection.")
        return

    file = files[file_index]
    df = pd.read_excel(file)
    display_columns(df)

    ignore_columns, label_columns = get_column_selections(len(df.columns))

    x_columns = [i for i in range(len(df.columns)) if i not in ignore_columns and i not in label_columns]
    X = df.iloc[:, x_columns]
    labels = df.iloc[:, label_columns] if label_columns else None

    X = preprocess_data(X)

    click.echo("\nX-block Data (First 5 rows):")
    click.echo(X.head())

    if labels is not None:
        click.echo("\nLabels (First 5 rows):")
        click.echo(labels.head())
    else:
        click.echo("\nNo label columns selected.")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plot_pca_results(X_pca)

    num_clusters_choice = click.prompt("Choose how to determine the number of clusters:\n1: User input\n2: Elbow method", type=int)
    if num_clusters_choice == 1:
        num_clusters = click.prompt("Enter the number of clusters for K-means", type=int, default=3)
    elif num_clusters_choice == 2:
        num_clusters = determine_num_clusters(X)
    else:
        click.echo("Invalid choice. Defaulting to 3 clusters.")
        num_clusters = 3

    if X.shape[0] > 1:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        labels_kmeans = kmeans.labels_

        plot_kmeans_results(X_pca, labels_kmeans)
    else:
        click.echo("\nNot enough samples for clustering.")

def display_columns(df):
    click.echo("Columns in the dataset:")
    for i, col in enumerate(df.columns):
        click.echo(f"{i}: {col}")

def get_column_selections(num_columns):
    ignore_columns = click.prompt("Enter column numbers to ignore (comma-separated)", type=str)
    label_columns = click.prompt("Enter column numbers that are labels (comma-separated)", type=str, default='')

    ignore_columns = [int(i) for i in ignore_columns.split(',') if i.isdigit() and 0 <= int(i) < num_columns]
    label_columns = [int(i) for i in label_columns.split(',') if i.isdigit() and 0 <= int(i) < num_columns]

    return ignore_columns, label_columns

def preprocess_data(X):
    click.echo("\nPreprocessing Options:")
    click.echo("1: Autoscaling (Standardization)")
    click.echo("2: Normalization (Min-Max Scaling)")
    click.echo("3: No preprocessing")

    choice = click.prompt("Choose a preprocessing option (1, 2, or 3)", type=int)

    if choice == 1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        click.echo("\nData has been autoscaled.")
        return pd.DataFrame(X_scaled, columns=X.columns)
    elif choice == 2:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        click.echo("\nData has been normalized.")
        return pd.DataFrame(X_scaled, columns=X.columns)
    else:
        click.echo("\nNo preprocessing applied.")
        return X

def plot_pca_results(X_pca):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], color='b', alpha=0.5)
    plt.title('PCA Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

def plot_kmeans_results(X_pca, labels_kmeans):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis')
    plt.title('K-means Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

def determine_num_clusters(X):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, algorithm='lloyd')
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.show()

    knee_locator = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    num_clusters = knee_locator.elbow
    click.echo(f"\nOptimal number of clusters determined using the elbow method: {num_clusters}")
    return num_clusters

if __name__ == "__main__":
    main()


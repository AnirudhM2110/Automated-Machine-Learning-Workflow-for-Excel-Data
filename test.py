import click
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator


@click.command()
def main():
    files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    if not files:
        click.echo("No Excel files found in the current directory.")
        return

    click.echo("Available Excel files:")
    for i, file in enumerate(files, start=1):
        click.echo(f"{i}: {file}")

    file_index = click.prompt('Select the file to process by entering the index number', type=int)
    if file_index < 1 or file_index > len(files):
        click.echo("Invalid selection.")
        return

    file = files[file_index - 1]

    excel_file = pd.ExcelFile(file)
    sheet_names = excel_file.sheet_names
    if len(sheet_names) > 1:
        click.echo("Available sheets:")
        for i, sheet in enumerate(sheet_names, start=1):
            click.echo(f"{i}: {sheet}")

        sheet_index = click.prompt('Select the sheet to process by entering the index number', type=int)
        if sheet_index < 1 or sheet_index > len(sheet_names):
            click.echo("Invalid selection.")
            return

        sheet_name = sheet_names[sheet_index - 1]
    else:
        sheet_name = sheet_names[0]

    df = pd.read_excel(file, sheet_name=sheet_name)
    display_columns(df)

    ignore_columns, label_columns = get_column_selections(len(df.columns))

    x_columns = [i for i in range(len(df.columns)) if i not in ignore_columns and i not in label_columns]
    X = df.iloc[:, x_columns]
    labels = df.iloc[:, label_columns] if label_columns else None

    if not check_numeric_columns(X):
        click.echo(
            "Non-numeric data detected in the selected columns for analysis. Please select only numeric columns.")
        return

    X = preprocess_data(X)

    click.echo("\nX-block Data :")
    click.echo(X)

    if labels is not None:
        click.echo("\nLabels :")
        click.echo(labels)
    else:
        click.echo("\nNo label columns selected.")

    pca, X_pca, num_components = select_pca_components(X)

    save_pca_results(X_pca, pca, X.columns)

    click.echo("\nPlease close the plot window to proceed.")
    plot_pca_results(X_pca, pca)

    num_clusters_choice = click.prompt(
        "\nChoose how to determine the number of clusters:\n1: User input\n2: Elbow method\nChoose an option", type=int)
    if num_clusters_choice == 1:
        num_clusters = click.prompt("Enter the number of clusters for K-means", type=int)
    elif num_clusters_choice == 2:
        num_clusters = determine_num_clusters(X)
    else:
        click.echo("Invalid choice. Defaulting to 3 clusters.")
        num_clusters = 3

    if X.shape[0] > 1:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X_pca)
        labels_kmeans = kmeans.labels_

        click.echo("\nPlease close the plot window to proceed.")
        plot_kmeans_combinations(X_pca, labels_kmeans)
    else:
        click.echo("\nNot enough samples for clustering.")


def display_columns(df):
    click.echo("Columns in the dataset:")
    for i, col in enumerate(df.columns, start=1):
        click.echo(f"{i}: {col}")


def get_column_selections(num_columns):
    ignore_columns = click.prompt("Enter column numbers to ignore (space separated)", type=str)
    label_columns = click.prompt("Enter column numbers that is label (ignore if doing unsupervised learning)", type=str,
                                 default='')

    ignore_columns = [int(i) - 1 for i in ignore_columns.split(' ') if i.isdigit() and 1 <= int(i) <= num_columns]
    label_columns = [int(i) - 1 for i in label_columns.split(' ') if i.isdigit() and 1 <= int(i) <= num_columns]

    return ignore_columns, label_columns


def check_numeric_columns(X):
    non_numeric_columns = X.select_dtypes(exclude=[float, int]).columns
    if len(non_numeric_columns) > 0:
        click.echo(f"Non-numeric columns detected: {', '.join(non_numeric_columns)}")
        return False
    return True


def preprocess_data(X):
    click.echo("\nPreprocessing Options:")
    click.echo("1: Autoscaling (Standardization)")
    click.echo("2: Normalization (Min-Max Scaling, includes autoscaling)")
    click.echo("3: No preprocessing")

    choice = click.prompt("Choose a preprocessing option (1, 2, or 3)", type=int)

    if choice == 1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        click.echo("\nData has been autoscaled.")
        return pd.DataFrame(X_scaled, columns=X.columns)
    elif choice == 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        min_max_scaler = MinMaxScaler()
        X_normalized = min_max_scaler.fit_transform(X_scaled)
        click.echo("\nData has been normalized (Min-Max Scaling after autoscaling).")
        return pd.DataFrame(X_normalized, columns=X.columns)
    else:
        click.echo("\nNo preprocessing applied.")
        return X


def select_pca_components(X, variance_threshold=0.9):
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    click.echo(f"\nNumber of PCA components selected: {num_components}")
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X)

    return pca, X_pca, num_components


def plot_pca_results(X_pca, pca):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], color='b', alpha=0.5)
    plt.title('PCA Results: PC1 vs PC2')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% Variance)')
    plt.grid(True)
    plt.savefig('figures/pca_1_vs_2.png')
    plt.show()


def plot_kmeans_combinations(X_pca, labels_kmeans):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    combinations = [(0, 1), (0, 2), (1, 2)]
    for (i, j) in combinations:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, i], X_pca[:, j], c=labels_kmeans, cmap='viridis')
        plt.title(f'K-means Clustering: PC{i+1} vs PC{j+1}')
        plt.xlabel(f'Principal Component {i+1}')
        plt.ylabel(f'Principal Component {j+1}')
        plt.grid(True)
        plt.savefig(f'figures/kmeans_{i+1}_vs_{j+1}.png')
        plt.show()


def determine_num_clusters(X):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, algorithm='lloyd')
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    if not os.path.exists('figures'):
        os.makedirs('figures')

    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.savefig('figures/elbow_method.png')
    plt.show()


    knee_locator = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    num_clusters = knee_locator.elbow
    click.echo(f"\nOptimal number of clusters determined using the elbow method: {num_clusters}")
    return num_clusters


def save_pca_results(X_pca, pca, columns):
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    pca_df.to_excel('pca_transformed_data.xlsx', index=False)
    click.echo("\nPCA-transformed data saved to 'pca_transformed_data.xlsx'.")

    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=columns)
    loadings.to_excel('pca_loadings.xlsx')
    click.echo("PCA loadings saved to 'pca_loadings.xlsx'.")


if __name__ == "__main__":
    main()

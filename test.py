import click
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from kneed import KneeLocator
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_decomposition import PLSRegression




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

    analysis_type = click.prompt(
        "\nChoose the type of analysis:\n1: Unsupervised Learning\n2: Supervised Learning\nChoose an option",
        type=int)

    if analysis_type == 1:
        # Apply PCA
        pca, X_pca = apply_pca(X, n_components=5)
        save_pca_results(X_pca, pca, X.columns)
        click.echo("\nPlease close the plot window to proceed.")
        plot_pca_results(X_pca, pca)

        # Run the original unsupervised clustering code
        clustering_method = click.prompt(
            "\nChoose the clustering method:\n1: K-means\n2: DBSCAN\n3: Hierarchical Clustering\nChoose an option",
            type=int)

        labels_clustering = None  # Initialize labels_clustering

        if clustering_method == 1:
            labels_clustering = interactive_kmeans_clustering(X_pca)
        elif clustering_method == 2:
            labels_clustering = interactive_dbscan_clustering(X_pca)
        elif clustering_method == 3:
            labels_clustering = interactive_hierarchical_clustering(X_pca)
        else:
            click.echo("Invalid choice.")

        if labels_clustering is not None:
            save_clustering_results(df, labels_clustering)
    elif analysis_type == 2:
        # Run the supervised learning code with Random Forest
        if labels is None or labels.empty:
            click.echo("Supervised learning requires label columns. Please select label columns.")
            return
        supervised_learning(X, labels)
    else:
        click.echo("Invalid choice.")
        return

def display_columns(df):
    click.echo("Columns in the dataset:")
    for i, col in enumerate(df.columns, start=1):
        click.echo(f"{i}: {col}")


def get_column_selections(num_columns):
    ignore_columns_input = click.prompt(
        "Enter column numbers to ignore (space separated, type 'none' to include all columns)", type=str)

    if ignore_columns_input.lower() == 'none':
        ignore_columns = []
    else:
        ignore_columns = [int(i) - 1 for i in ignore_columns_input.split(' ') if i.isdigit()]

        invalid_ignore_columns = [i for i in ignore_columns if i < 0 or i >= num_columns]
        if invalid_ignore_columns:
            click.echo("Error, choice not in sheet.")
            return get_column_selections(num_columns)

    label_columns_input = click.prompt(
        "Enter column numbers that is label (ignore if doing unsupervised learning)", type=str, default='')
    label_columns = [int(i) - 1 for i in label_columns_input.split(' ') if i.isdigit()]

    invalid_label_columns = [i for i in label_columns if i < 0 or i >= num_columns]
    if invalid_label_columns:
        click.echo("Error, choice not in sheet.")
        return get_column_selections(num_columns)

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


def apply_pca(X, n_components=5):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    click.echo(f"\nPCA applied with {n_components} components.")

    return pca, X_pca


def plot_pca_results(X_pca, pca):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    explained_variance = pca.explained_variance_ratio_

    max_limit = max(np.max(X_pca[:, 0]), np.max(X_pca[:, 1])) + 1

    plt.figure(figsize=(8, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], color='b', alpha=0.5)
    plt.title('PCA Results: PC1 vs PC2',fontsize=20)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}% Variance)',fontsize=16)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}% Variance)',fontsize=16)
    plt.grid(False)
    plt.xlim(-max_limit, max_limit)
    plt.ylim(-max_limit, max_limit)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('figures/pca_1_vs_2.png')
    plt.show()



def plot_clustering_combinations(X_pca, labels_clustering, method_name):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    unique_labels = np.unique(labels_clustering)
    palette = sns.color_palette("viridis", as_cmap=True)

    combinations = [(0, 1), (0, 2), (1, 2)]
    for (i, j) in combinations:
        max_limit = max(np.max(X_pca[:, i]), np.max(X_pca[:, j])) + 1

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(X_pca[:, i], X_pca[:, j], c=labels_clustering, cmap=palette)
        plt.title(f'{method_name} Clustering: PC{i + 1} vs PC{j + 1}',fontsize=20)
        plt.xlabel(f'Principal Component {i + 1}',fontsize=16)
        plt.ylabel(f'Principal Component {j + 1}',fontsize=16)
        plt.grid(False)
        handles, labels = scatter.legend_elements()
        plt.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=16,title_fontsize=18)
        plt.xlim(-max_limit, max_limit)
        plt.ylim(-max_limit, max_limit)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        for label in unique_labels:
            if label != -1:  # Skip outliers
                cluster_data = X_pca[labels_clustering == label]
                if len(cluster_data) > 1:
                    confidence_ellipse(cluster_data[:, i], cluster_data[:, j], plt.gca(),
                                       edgecolor=plt.cm.viridis(label / len(unique_labels)),
                                       facecolor=plt.cm.viridis(label / len(unique_labels)), alpha=0.2)

        plt.tight_layout()
        plt.savefig(f'figures/{method_name.lower()}_{i + 1}_vs_{j + 1}.png')
        plt.show()



def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', edgecolor='blue', alpha=0.2, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radii.
    facecolor : str
        The color of the ellipse.
    edgecolor : str
        The color of the ellipse edge.
    alpha : float
        The transparency level for the fill color.
    **kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, edgecolor=edgecolor,
                      alpha=alpha, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * 1.3*n_std
    scale_y = np.sqrt(cov[1, 1]) * 1.3*n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def determine_num_clusters(X_pca):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, algorithm='lloyd')
        kmeans.fit(X_pca)
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
    click.echo(f"\nOptimal number of clusters determined by elbow method: {num_clusters}")
    return num_clusters


def interactive_dbscan_clustering(X_pca):
    while True:
        eps = click.prompt("Enter the epsilon value for DBSCAN (suggested range: 0.1 to 10)", type=float)
        min_samples = click.prompt("Enter the minimum number of samples for DBSCAN", type=int)

        if X_pca.shape[0] > 1:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels_clustering = dbscan.fit_predict(X_pca)

            num_clusters = len(set(labels_clustering)) - (1 if -1 in labels_clustering else 0)
            click.echo(f"\nNumber of clusters obtained from DBSCAN: {num_clusters}")
            click.echo(f"\nOutliers are represented by -1 in the cluster column of clustering_results.xlsx ")

            click.echo("\nPlease close the plot window to proceed.")
            plot_clustering_combinations(X_pca, labels_clustering, "DBSCAN")

            proceed = click.prompt("Are you satisfied with the clustering result? (yes/no)", type=str)
            if proceed.lower() == 'yes':
                return labels_clustering  # Return the labels_clustering to save the results
        else:
            click.echo("\nNot enough samples for clustering.")
            break


def interactive_kmeans_clustering(X_pca):
    while True:
        num_clusters_choice = click.prompt(
            "\nChoose how to determine the number of clusters for K-means:\n1: User input\n2: Elbow method\nChoose an option",
            type=int)
        if num_clusters_choice == 1:
            num_clusters = click.prompt("Enter the number of clusters for K-means", type=int)
        elif num_clusters_choice == 2:
            num_clusters = determine_num_clusters(X_pca)
        else:
            click.echo("Invalid choice. Defaulting to 3 clusters.")
            num_clusters = 3

        if X_pca.shape[0] > 1:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(X_pca)
            labels_clustering = kmeans.labels_

            click.echo("\nPlease close the plot window to proceed.")
            plot_clustering_combinations(X_pca, labels_clustering, "K-means")

            proceed = click.prompt("Are you satisfied with the clustering result? (yes/no)", type=str)
            if proceed.lower() == 'yes':
                return labels_clustering  # Return the labels_clustering to save the results
        else:
            click.echo("\nNot enough samples for clustering.")
            break


def interactive_hierarchical_clustering(X_pca):
    while True:
        num_clusters = click.prompt("Enter the number of clusters for Hierarchical Clustering", type=int)

        if X_pca.shape[0] > 1:
            hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
            labels_clustering = hierarchical.fit_predict(X_pca)

            click.echo("\nPlease close the plot window to proceed.")
            plot_clustering_combinations(X_pca, labels_clustering, "Hierarchical ")

            proceed = click.prompt("Are you satisfied with the clustering result? (yes/no)", type=str)
            if proceed.lower() == 'yes':
                return labels_clustering
        else:
            click.echo("\nNot enough samples for clustering.")
            break


def save_pca_results(X_pca, pca, columns):
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(X_pca.shape[1])])
    pca_df.to_excel('pca_transformed_data.xlsx', index=False)
    click.echo("\nPCA-transformed data saved to 'pca_transformed_data.xlsx'.")

    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(X_pca.shape[1])], index=columns)
    loadings.to_excel('pca_loadings.xlsx')
    click.echo("PCA loadings saved to 'pca_loadings.xlsx'.")


def save_clustering_results(df, labels_clustering):
    clustering_results = pd.DataFrame(df.iloc[:, 0])
    clustering_results['Cluster'] = labels_clustering

    clustering_results.to_excel('clustering_results.xlsx', index=False)
    click.echo("\nClustering results saved to 'clustering_results.xlsx'.")


def supervised_learning(X, y):
    # Ensure y is a 1D array
    y = y.values.ravel() if hasattr(y, 'values') else np.ravel(y)

    # Save the original minimum value of y
    y_min = np.min(y)

    # Adjust target labels to start from 0
    y = y - y_min

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Prompt the user to choose between Random Forest, XGBoost, SVM, and PLS-DA
    model_choice = click.prompt(
        "\nChoose the supervised learning algorithm:\n1: Random Forest\n2: XGBoost\n3: SVM\n4: PLS-DA\nChoose an option",
        type=int)

    if model_choice == 1:
        click.echo("You selected Random Forest.")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == 2:
        click.echo("You selected XGBoost.")
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_choice == 3:
        click.echo("You selected SVM.")
        model = SVC(kernel='rbf', random_state=42)
    elif model_choice == 4:
        click.echo("You selected PLS-DA.")
        # PLS-DA specific code: Use LabelBinarizer to convert y_train into a binary matrix
        lb = LabelBinarizer()
        Y_train_bin = lb.fit_transform(y_train)

        # Initialize and train PLS-DA
        n_components = 2  # You can adjust this based on your dataset
        model = PLSRegression(n_components=n_components)
        model.fit(X_train, Y_train_bin)

        # Make predictions
        Y_pred_continuous = model.predict(X_test)
        y_pred = np.argmax(Y_pred_continuous, axis=1)
    else:
        click.echo("Invalid choice. Defaulting to Random Forest.")
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    if model_choice != 4:
        # Fit the model for Random Forest, XGBoost, and SVM
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Change the labels back to their original values
    y_test = y_test + y_min
    y_pred = y_pred + y_min

    # Output the results
    click.echo("\nSupervised Learning Results:")
    click.echo("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    click.echo("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    click.echo(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

    click.echo("\nClassification Report:")
    click.echo(classification_report(y_test, y_pred))

    # Save the results
    model_name = {1: 'random_forest', 2: 'xgboost', 3: 'svm', 4: 'pls_da'}.get(model_choice, 'random_forest')
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    results.to_excel(f'{model_name}_results.xlsx', index=False)
    click.echo(f"\nResults saved to '{model_name}_results.xlsx'.")

if __name__ == "__main__":
    main()
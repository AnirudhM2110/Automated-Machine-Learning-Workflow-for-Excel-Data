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

    # Always use the first column as label
    labels = df.iloc[:, 0]
    df_data = df.iloc[:, 1:]

    display_columns(df_data)

    ignore_columns, class_columns = get_column_selections(len(df_data.columns))

    x_columns = [i for i in range(len(df_data.columns)) if i not in ignore_columns and i not in class_columns]
    X = df_data.iloc[:, x_columns]
    classes = df_data.iloc[:, class_columns] if class_columns else None

    if not check_numeric_columns(X):
        click.echo(
            "Non-numeric data detected in the selected columns for analysis. Please select only numeric columns.")
        return

    X = preprocess_data(X)

    click.echo("\nX-block Data :")
    click.echo(X)

    if classes is not None:
        click.echo("\nClass Data:")
        click.echo(classes)
    else:
        click.echo("\nNo class columns selected.")

    analysis_type = click.prompt(
        "\nChoose the type of analysis:\n1: Unsupervised Learning\n2: Supervised Learning\nChoose an option",
        type=int)

    if analysis_type == 1:
        # Apply PCA
        pca, X_pca = apply_pca(X, n_components=5)
        save_pca_results(X_pca, pca, X.columns, labels)
        click.echo("\nPlease close the plot window to proceed.")
        plot_pca_results(X_pca, pca)

        # Run the unsupervised clustering code
        clustering_method = click.prompt(
            "\nChoose the clustering method:\n1: K-means\n2: DBSCAN\n3: Hierarchical Clustering\nChoose an option",
            type=int)

        cluster_labels = None

        if clustering_method == 1:
            cluster_labels = interactive_kmeans_clustering(X_pca)
        elif clustering_method == 2:
            cluster_labels = interactive_dbscan_clustering(X_pca)
        elif clustering_method == 3:
            cluster_labels = interactive_hierarchical_clustering(X_pca)
        else:
            click.echo("Invalid choice.")

        if cluster_labels is not None:
            save_clustering_results(labels, cluster_labels)
    elif analysis_type == 2:
        if classes is None or classes.empty:
            click.echo("Supervised learning requires class columns. Please select class columns.")
            return
        supervised_learning(X, classes, labels)
    else:
        click.echo("Invalid choice.")
        return


def display_columns(df):
    click.echo("\nAvailable columns for analysis (The first column is automatically assigned as label):")
    for i, col in enumerate(df.columns, start=1):
        click.echo(f"{i}: {col}")

def get_column_selections(num_columns):
    class_columns_input = click.prompt(
        "Enter column numbers for class/target variables (ignore if doing unsupervised learning)", type=str, default='')
    class_columns = [int(i) - 1 for i in class_columns_input.split(' ') if i.isdigit()]

    invalid_class_columns = [i for i in class_columns if i < 0 or i >= num_columns]
    if invalid_class_columns:
        click.echo("Error, choice not in sheet.")
        return get_column_selections(num_columns)

    ignore_columns_input = click.prompt(
        "Enter column numbers to ignore (space separated)", type=str, default='none')

    if ignore_columns_input.lower() == 'none':
        ignore_columns = []
    else:
        ignore_columns = [int(i) - 1 for i in ignore_columns_input.split(' ') if i.isdigit()]

        invalid_ignore_columns = [i for i in ignore_columns if i < 0 or i >= num_columns]
        if invalid_ignore_columns:
            click.echo("Error, choice not in sheet.")
            return get_column_selections(num_columns)

    return ignore_columns, class_columns

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

    # Add 1 to labels to start from 1 instead of 0 (except for DBSCAN outliers which are -1)
    plot_labels = np.where(labels_clustering == -1, -1, labels_clustering + 1)
    unique_labels = np.unique(plot_labels)
    palette = sns.color_palette("viridis", as_cmap=True)

    combinations = [(0, 1), (0, 2), (1, 2)]
    for (i, j) in combinations:
        max_limit = max(np.max(X_pca[:, i]), np.max(X_pca[:, j])) + 1

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(X_pca[:, i], X_pca[:, j], c=plot_labels, cmap=palette)
        plt.title(f'{method_name} Clustering: PC{i + 1} vs PC{j + 1}',fontsize=20)
        plt.xlabel(f'Principal Component {i + 1}',fontsize=16)
        plt.ylabel(f'Principal Component {j + 1}',fontsize=16)
        plt.grid(False)

        # Create custom legend labels
        legend_labels = [f'Cluster {label}' if label != -1 else 'Outliers'
                        for label in unique_labels]
        handles = scatter.legend_elements()[0]
        plt.legend(handles, legend_labels, title="Clusters",
                  bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=16, title_fontsize=18)

        plt.xlim(-max_limit, max_limit)
        plt.ylim(-max_limit, max_limit)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        for label in unique_labels:
            if label != -1:  # Skip outliers
                cluster_data = X_pca[plot_labels == label]
                if len(cluster_data) > 1:
                    confidence_ellipse(cluster_data[:, i], cluster_data[:, j], plt.gca(),
                                    edgecolor=plt.cm.viridis((label-1)/(len(unique_labels)-1) if -1 in unique_labels else label/len(unique_labels)),
                                    facecolor=plt.cm.viridis((label-1)/(len(unique_labels)-1) if -1 in unique_labels else label/len(unique_labels)),
                                    alpha=0.2)

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

            proceed = click.prompt("Are you satisfied with the clustering result? (y/n)", type=str)
            if proceed.lower() == 'y':
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

            proceed = click.prompt("Are you satisfied with the clustering result? (y/n)", type=str)
            if proceed.lower() == 'y':
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

            proceed = click.prompt("Are you satisfied with the clustering result? (y/n)", type=str)
            if proceed.lower() == 'y':
                return labels_clustering
        else:
            click.echo("\nNot enough samples for clustering.")
            break


def save_pca_results(X_pca, pca, columns, labels):
    # Create DataFrame with labels
    pca_df = pd.DataFrame({'Label': labels})
    # Add PCA components
    pca_components = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(X_pca.shape[1])])
    pca_df = pd.concat([pca_df, pca_components], axis=1)

    pca_df.to_excel('pca_transformed_data.xlsx', index=False)
    click.echo("\nPCA-transformed data saved to 'pca_transformed_data.xlsx'.")

    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(X_pca.shape[1])], index=columns)
    loadings.to_excel('pca_loadings.xlsx')
    click.echo("PCA loadings saved to 'pca_loadings.xlsx'.")


def save_clustering_results(labels, cluster_labels):
    # Add 1 to cluster labels (except for DBSCAN outliers which are -1)
    adjusted_clusters = np.where(cluster_labels == -1, -1, cluster_labels + 1)

    clustering_results = pd.DataFrame({
        'Label': labels,
        'Cluster': adjusted_clusters
    })
    clustering_results.to_excel('clustering_results.xlsx', index=False)
    click.echo("\nClustering results saved to 'clustering_results.xlsx'.")


def supervised_learning(X, classes, labels):
    """
    Perform supervised learning with interactive hyperparameter selection,
    probability predictions, and iterative result review.
    """
    # Validate that we have a classification task
    unique_values = classes.nunique()
    if not all(unique_values <= 10):
        click.echo("Error: The selected class columns do not appear to be suitable for classification.")
        return None

    while True:
        # Get user-selected models with hyperparameters
        selected_models = get_model_selection(is_classification=True)

        if not selected_models:
            click.echo("No models selected. Exiting.")
            return None

        # Split the data
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
            X, classes, labels, test_size=0.2, random_state=42
        )

        # Prepare to store overall results
        overall_results = {}

        def classification_metrics(y_true, y_pred):
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Classification Report': classification_report(y_true, y_pred, zero_division=0),
                'Confusion Matrix': confusion_matrix(y_true, y_pred)
            }

        # For each class column, train and evaluate selected classifiers
        for col in classes.columns:
            click.echo(f"\n--- Classifier Analysis for: {col} ---")

            # Ensure y is a 1D array
            y_train_col = y_train[col].values.ravel()
            y_test_col = y_test[col].values.ravel()

            # Save the original minimum value of y
            y_train_min = np.min(y_train_col)
            y_test_min = np.min(y_test_col)

            # Adjust target labels to start from 0
            y_train_col = y_train_col - y_train_min
            y_test_col = y_test_col - y_test_min

            # Get unique classes
            unique_classes = np.unique(y_test_col + y_test_min)

            # Store results for this column
            column_results = {}

            for clf_name, classifier in selected_models:
                click.echo(f"\nTraining {clf_name} Classifier")

                # Special handling for PLS-DA
                if clf_name == 'PLS-DA':
                    # One-hot encode the target for PLS-DA
                    lb = LabelBinarizer()
                    y_train_encoded = lb.fit_transform(y_train_col)
                    y_test_encoded = lb.transform(y_test_col)

                    # Train the PLS-DA model
                    plsda = classifier
                    plsda.fit(X_train, y_train_encoded)

                    # Predict probabilities
                    y_pred_proba = plsda.predict(X_test)
                    y_pred = np.argmax(y_pred_proba, axis=1)

                    # Create results DataFrame WITHOUT probability columns
                    results = pd.DataFrame({
                        'Label': labels_test,
                        'Actual': y_test_col + y_test_min,  # Convert back to original labels
                        'Predicted': y_pred + y_test_min  # Convert back to original labels
                    })

                # Handling for classifiers with predict_proba method
                elif hasattr(classifier, 'predict_proba'):
                    # Train the classifier
                    classifier.fit(X_train, y_train_col)

                    # Predict labels and probabilities
                    y_pred = classifier.predict(X_test)
                    y_pred_proba = classifier.predict_proba(X_test)

                    # Create results DataFrame with probabilities for each class
                    results = pd.DataFrame({
                        'Label': labels_test,
                        'Actual': y_test_col + y_test_min,  # Convert back to original labels
                        'Predicted': y_pred + y_test_min  # Convert back to original labels
                    })

                    # Add probability columns
                    for i, cls in enumerate(unique_classes):
                        results[f'Prob_Class_{cls}'] = y_pred_proba[:, i]

                # Fallback for classifiers without probability prediction
                else:
                    # Train the classifier
                    classifier.fit(X_train, y_train_col)

                    # Predict labels
                    y_pred = classifier.predict(X_test)

                    # Create results DataFrame without probabilities
                    results = pd.DataFrame({
                        'Label': labels_test,
                        'Actual': y_test_col + y_test_min,  # Convert back to original labels
                        'Predicted': y_pred + y_test_min  # Convert back to original labels
                    })

                # Save results to Excel
                results.to_excel(f'{clf_name.lower().replace(" ", "_")}_classification_results_{col}.xlsx', index=False)
                click.echo(
                    f"\nResults saved to '{clf_name.lower().replace(' ', '_')}_classification_results_{col}.xlsx'")

                # Calculate and display metrics
                metrics = classification_metrics(y_test_col, y_pred)

                click.echo(f"\nAccuracy: {metrics['Accuracy']:.4f}")
                click.echo("\nClassification Report:")
                click.echo(metrics['Classification Report'])

                # Plot Confusion Matrix
                plot_confusion_matrix(
                    y_test_col,
                    y_pred,
                    classes=unique_classes,
                    title=f'{clf_name} Confusion Matrix - {col}'
                )

                # Feature importance
                if clf_name in ['Random Forest', 'XGBoost']:
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': classifier.feature_importances_
                    }).sort_values('importance', ascending=False)

                    click.echo("\nFeature Importance:")
                    click.echo(feature_importance)
                elif clf_name == 'PLS-DA':
                    # For PLS-DA, use the PLS weights as a proxy for feature importance
                    pls_weights = np.abs(plsda.coef_).mean(axis=0)
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': pls_weights
                    }).sort_values('importance', ascending=False)

                    click.echo("\nPLS-DA Feature Weights:")
                    click.echo(feature_importance)
                elif clf_name == 'Support Vector Machine':
                    click.echo("\nFeature importance not directly available for SVM.")
                else:
                    click.echo("\nFeature importance not available for this model.")

                # Store results
                column_results[clf_name] = {
                    'metrics': metrics,
                    'predictions': results,
                    'hyperparameters': classifier.get_params()
                }

            # Store overall results
            overall_results[col] = column_results

        # Ask user if they're satisfied with the results
        save_model_statistics(overall_results)

        proceed = click.prompt("Are you satisfied with the classification results? (y/n)", type=str)
        if proceed.lower() == 'y':
            break
        else:
            click.echo("\nRerunning supervised learning with new model selection.")

    return overall_results

def get_model_selection(is_classification):
    """
    Interactively get model selection from the user for classification only.
    """
    available_models = {
        1: ('PLS-DA', PLSRegression()),
        2: ('Support Vector Machine', SVC(random_state=42)),
        3: ('XGBoost', XGBClassifier(n_estimators=100, random_state=42)),
        4: ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))
    }

    # Display available models
    click.echo("\nAvailable Classification Models:")
    for key, (name, _) in available_models.items():
        click.echo(f"{key}: {name}")

    # Allow multiple selections
    selections = click.prompt(
        "Enter the numbers of models you want to use (space-separated)",
        type=str
    )

    # Process selections
    selected_models = []
    for sel in selections.split():
        try:
            model_num = int(sel)
            if model_num in available_models:
                selected_models.append(available_models[model_num])
            else:
                click.echo(f"Invalid selection: {model_num}")
        except ValueError:
            click.echo(f"Invalid input: {sel}")

    return selected_models


def get_model_selection(is_classification):
    available_models = {
        1: ('PLS-DA', PLSRegression(n_components=2)),
        2: ('Support Vector Machine', SVC(random_state=42)),
        3: ('XGBoost', XGBClassifier(random_state=42)),
        4: ('Random Forest', RandomForestClassifier(random_state=42))
    }

    click.echo("\nAvailable Classification Models:")
    for key, (name, _) in available_models.items():
        click.echo(f"{key}: {name}")

    selections = click.prompt("Enter the numbers of models you want to use (space-separated)", type=str)
    selected_models = []

    for sel in selections.split():
        try:
            model_num = int(sel)
            if model_num in available_models:
                model_name, base_model = available_models[model_num]
                click.echo(f"\nConfiguring hyperparameters for {model_name}")
                hyperparameters = get_model_hyperparameters(model_name)

                if model_name == 'PLS-DA':
                    model = PLSRegression(**hyperparameters)
                elif model_name == 'Support Vector Machine':
                    model = SVC(**hyperparameters)
                elif model_name == 'XGBoost':
                    model = XGBClassifier(**hyperparameters)
                else:  # Random Forest
                    model = RandomForestClassifier(**hyperparameters)

                selected_models.append((model_name, model))
        except ValueError:
            click.echo(f"Invalid input: {sel}")

    return selected_models
def plot_confusion_matrix(y_true, y_pred, classes, title):
    """
    Plot a confusion matrix using seaborn and matplotlib

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - classes: Unique class labels
    - title: Title for the plot
    """
    # Ensure the 'figures' directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Use seaborn to create a more visually appealing confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    # Save the plot
    filename = f'figures/{title.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(filename)
    plt.close()

    click.echo(f"\nConfusion matrix plot saved to '{filename}'")


def save_model_statistics(overall_results, output_file='model_statistics.txt'):
    """
    Save detailed model statistics to a text file, now including hyperparameters

    Parameters:
    - overall_results: Dictionary containing results from all models and columns
    - output_file: Name of the output file
    """
    with open(output_file, 'w') as f:
        for col, column_results in overall_results.items():
            f.write(f"{'=' * 50}\n")
            f.write(f"MODEL STATISTICS FOR COLUMN: {col}\n")
            f.write(f"{'=' * 50}\n\n")

            for model_name, model_data in column_results.items():
                f.write(f"\n--- {model_name} Model Statistics ---\n")

                # Hyperparameters
                f.write("\nHyperparameters:\n")
                for param, value in model_data['hyperparameters'].items():
                    f.write(f"{param}: {value}\n")

                # Metrics
                metrics = model_data['metrics']
                f.write("\nMetrics:\n")
                f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")

                # Classification Report
                f.write("\nClassification Report:\n")
                f.write(str(metrics['Classification Report']) + "\n")

                # Confusion Matrix
                f.write("\nConfusion Matrix:\n")
                f.write(str(metrics['Confusion Matrix']) + "\n")

                # Predictions Analysis
                predictions = model_data['predictions']
                f.write("\nPrediction Analysis:\n")
                f.write(f"Total Samples: {len(predictions)}\n")
                correct_predictions = (predictions['Actual'] == predictions['Predicted']).sum()
                f.write(f"Correctly Predicted: {correct_predictions}\n")
                f.write(f"Incorrectly Predicted: {len(predictions) - correct_predictions}\n")

                # Prediction Error Analysis
                misclassified = predictions[predictions['Actual'] != predictions['Predicted']]
                f.write("\nMisclassification Details:\n")
                f.write(str(misclassified) + "\n")

                f.write("\n" + "-" * 50 + "\n")

    click.echo(f"\nModel statistics saved to '{output_file}'")


def get_model_hyperparameters(model_name):
    """
    Interactively get hyperparameters for different classification models.
    """
    hyperparameters = {}

    if model_name == 'PLS-DA':
        n_components = click.prompt("Enter number of components (n_components)", type=int, default=2)
        hyperparameters = {
            'n_components': n_components
        }

    elif model_name == 'Support Vector Machine':
        kernel = click.prompt("Choose kernel (linear/rbf/poly/sigmoid)",
                              type=click.Choice(['linear', 'rbf', 'poly', 'sigmoid']),
                              default='rbf')
        c_value = click.prompt("Enter regularization parameter C (default: 1.0)", type=float, default=1.0)
        gamma = click.prompt("Enter kernel coefficient (auto/scale/float value, default: 'scale')",
                             type=str, default='scale')
        hyperparameters = {
            'kernel': kernel,
            'C': c_value,
            'gamma': 'scale' if gamma == 'scale' or gamma == 'auto' else float(gamma),
            'random_state': 42
        }

    elif model_name == 'XGBoost':
        n_estimators = click.prompt("Enter number of boosting rounds (default: 100)", type=int, default=100)
        learning_rate = click.prompt("Enter learning rate (default: 0.1)", type=float, default=0.1)
        max_depth = click.prompt("Enter max depth of trees (default: 6)", type=int, default=6)
        hyperparameters = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }

    elif model_name == 'Random Forest':
        n_estimators = click.prompt("Enter number of trees (default: 100)", type=int, default=100)
        max_depth = click.prompt("Enter max depth of trees (default: 0)",
                                 type=int, default=0)
        min_samples_split = click.prompt("Enter min samples to split (default: 2)",
                                         type=int, default=2)
        hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': None if max_depth == 0 else max_depth,
            'min_samples_split': min_samples_split,
            'random_state': 42
        }

    return hyperparameters




if __name__ == "__main__":
    main()

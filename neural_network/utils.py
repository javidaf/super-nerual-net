from neural_network import NeuralNetwork
import numpy as np
from sklearn.model_selection import KFold
from itertools import product
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def grid_search_nn(
    train_features,
    targets_one_hot,
    search_params=None,
    fixed_params=None,
    k_folds=5,
    epochs=200,
    random_state=42,
):
    """
    Flexible grid search for neural network parameters

    Args:
        search_params: Dict of parameters to search over, e.g.
            {'hidden_layers': [[3], [4]], 'activations': ['relu', 'sigmoid']}
        fixed_params: Dict of fixed parameters to use
    """
    default_params = {
        "hidden_layers": [[3]],
        "activations": ["relu"],
        "learning_rate": 0.01,
        "optimizer": "adam",
        "output_activation": "softmax",
        "use_regularization": False,
        "lambda_": 0.01,
        "classification_type": "multiclass",
        "initializer": "he",
    }

    if fixed_params:
        default_params.update(fixed_params)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    accuracies = []

    search_params = search_params or {}
    param_names = list(search_params.keys())
    param_values = list(search_params.values())

    for params in product(*param_values):
        current_params = default_params.copy()
        current_params.update(dict(zip(param_names, params)))

        fold_accuracies = []

        for train_idx, val_idx in kf.split(train_features):
            X_train_fold = train_features[train_idx]
            y_train_fold = targets_one_hot[train_idx]
            X_val_fold = train_features[val_idx]
            y_val_fold = targets_one_hot[val_idx]

            NN = NeuralNetwork(
                input_size=train_features.shape[1],
                hidden_layers=current_params["hidden_layers"],
                output_size=3,
                learning_rate=current_params["learning_rate"],
                optimizer=current_params["optimizer"],
                output_activation=current_params["output_activation"],
                hidden_activation=current_params["activations"],
                use_regularization=current_params["use_regularization"],
                lambda_=current_params["lambda_"],
                classification_type=current_params["classification_type"],
                initializer=current_params["initializer"],
            )

            NN.train_classifier(X_train_fold, y_train_fold, epochs=epochs)
            fold_accuracies.append(NN.accuracy_score(X_val_fold, y_val_fold))

        accuracies.append(np.mean(fold_accuracies))

    return accuracies, param_names, param_values


def save_neural_network(model, filepath):
    """Save neural network model to file"""
    model_state = {
        "params": model.params,
        "weights": model.weights,
        "biases": model.biases,
    }
    with open(filepath, "wb") as f:
        pickle.dump(model_state, f)


def load_neural_network(filepath):
    """Load neural network model from file"""
    with open(filepath, "rb") as f:
        model_state = pickle.load(f)

    # Recreate the model using the stored parameters
    model = NeuralNetwork(**model_state["params"])
    model.weights = model_state["weights"]
    model.biases = model_state["biases"]
    return model


def plot_accuracy_heatmap(
    accuracies, param1, param2, param1_label, param2_label, title
):
    matrix = np.array(accuracies).reshape(len(param1), len(param2))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=param2,
        yticklabels=param1,
        cmap="viridis",
        annot_kws={"size": 20},
        cbar_kws={"label": "Accuracy"},
        linewidths=0.5,
        linecolor="black",
    )
    plt.xlabel(param2_label, fontsize=12)
    plt.ylabel(param1_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()

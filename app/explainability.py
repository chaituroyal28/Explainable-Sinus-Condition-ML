import numpy as np


def get_pipeline_parts(model):
    """Get the preprocessing transformer and final estimator."""
    if not hasattr(model, "named_steps"):
        raise ValueError("Saved model must be a scikit-learn Pipeline.")

    steps = model.named_steps

    preprocessor = None

    for step in steps.values():
        if hasattr(step, "get_feature_names_out"):
            preprocessor = step

    estimator = list(steps.values())[-1]

    if preprocessor is None:
        raise ValueError("Preprocessing transformer not found.")

    return preprocessor, estimator


def get_original_feature(feature_name):
    """Map transformed feature names to original survey features."""

    feature_name = str(feature_name)

    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "", 1)

    if feature_name.startswith("cat__"):
        name = feature_name.replace("cat__", "", 1)

        categorical_features = [
            "Gender",
            "Regular Sinus Issues",
            "Most Problematic Season",
            "Triggered by External Factors",
            "Known Allergy",
            "Doctor Consulted (Past Year)",
            "Headaches or Facial Pain",
            "Medication Frequency"
        ]

        for feature in categorical_features:
            if name.startswith(feature + "_"):
                return feature

    return feature_name


def get_selected_category(feature, input_df):
    """Return the actual category selected by the user."""

    if feature in input_df.columns:
        return str(input_df.iloc[0][feature])

    return ""


def explain_prediction(model, input_df, top_n=8):
    """
    Generate a live explanation for the tuned Decision Tree.

    This version does NOT require SHAP, because the local Windows
    environment blocks the scikit-learn DLL required by SHAP.

    The explanation is based on the Decision Tree's prediction path
    and feature changes along that path.
    """

    preprocessor, estimator = get_pipeline_parts(model)

    # Transform the user's input using the exact model preprocessing
    transformed = preprocessor.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    feature_names = preprocessor.get_feature_names_out()

    # ---------------------------------------------------------
    # Decision Tree path
    # ---------------------------------------------------------

    if not hasattr(estimator, "tree_"):
        return []

    tree = estimator.tree_

    node_indicator = estimator.decision_path(transformed)
    leaf_id = estimator.apply(transformed)

    path_nodes = node_indicator.indices[
        node_indicator.indptr[0]:
        node_indicator.indptr[1]
    ]

    # ---------------------------------------------------------
    # Calculate feature contribution from the prediction path
    # ---------------------------------------------------------

    contributions = {}

    for node_id in path_nodes:

        # Leaf nodes do not contain a split
        if node_id == leaf_id[0]:
            continue

        feature_index = tree.feature[node_id]

        if feature_index < 0:
            continue

        threshold = tree.threshold[node_id]

        value = transformed[0, feature_index]

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        parent_probability = tree.value[node_id][0]

        if parent_probability.sum() == 0:
            continue

        parent_yes_probability = (
            parent_probability[1] /
            parent_probability.sum()
        )

        child_id = (
            left_child
            if value <= threshold
            else right_child
        )

        child_probability = tree.value[child_id][0]

        if child_probability.sum() == 0:
            continue

        child_yes_probability = (
            child_probability[1] /
            child_probability.sum()
        )

        # Change in probability of the Yes class
        contribution = (
            child_yes_probability -
            parent_yes_probability
        )

        transformed_feature = str(
            feature_names[feature_index]
        )

        original_feature = get_original_feature(
            transformed_feature
        )

        if original_feature not in contributions:
            contributions[original_feature] = 0.0

        contributions[original_feature] += contribution

    # ---------------------------------------------------------
    # If no split contribution exists
    # ---------------------------------------------------------

    if not contributions:
        return []

    # ---------------------------------------------------------
    # Sort by absolute contribution
    # ---------------------------------------------------------

    sorted_features = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    top_features = sorted_features[:top_n]

    max_abs = max(
        abs(value)
        for _, value in top_features
    )

    # ---------------------------------------------------------
    # Prepare frontend response
    # ---------------------------------------------------------

    results = []

    for feature, value in top_features:

        value = float(value)

        direction = (
            "Supports Yes"
            if value >= 0
            else "Supports No"
        )

        strength = (
            abs(value) / max_abs * 100
            if max_abs > 0
            else 0
        )

        category = get_selected_category(
            feature,
            input_df
        )

        results.append({
            "feature": feature,
            "value": round(value, 6),
            "direction": direction,
            "strength": round(strength, 2),
            "category": category
        })

    return results
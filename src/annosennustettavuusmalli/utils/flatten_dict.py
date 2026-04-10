"""
Tekijä: Akseli Leino
"""


def flatten_dict(d: dict, parent_key="", sep="."):
    """Flattens a nested dictionary into a single-level dictionary with keys representing the path to each value.

    Example:
        Input: {"a": 1, "b": {"c": 2, "d": 3}}

        Output: {"a": 1, "b.c": 2, "b.d": 3}

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The base key to use for the current level of recursion. Defaults to "".
        sep (str, optional): The separator to use between keys. Defaults to ".".

    Returns:
        dict: A flattened dictionary where nested keys are concatenated with the specified separator.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

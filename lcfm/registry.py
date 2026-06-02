DATASETS = {}
MODELS = {}
METHODS = {}
SOLVERS = {}
METRICS = {}


def register(registry, name):
    def decorator(obj):
        if name in registry:
            raise ValueError(f"Duplicate registry entry: {name}")
        registry[name] = obj
        return obj

    return decorator


def get(registry, name):
    try:
        return registry[name]
    except KeyError as exc:
        options = ", ".join(sorted(registry))
        raise KeyError(f"Unknown entry '{name}'. Available: {options}") from exc

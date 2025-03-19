import argparse
import yaml


def join(loader: yaml.Loader, node: yaml.Node) -> str:
    """
    Custom YAML tag to join sequences into a single string.
    """
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments, especially the configuration file path.

    Returns:
    - argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse arguments and return parser object"
    )

    parser.add_argument(
        "-C",
        "--config_path",
        type=str,
        default="configs/test_config.yaml",
        help="Path to the configuration file. If not provided, takes the default path.",
    )

    return parser.parse_args()


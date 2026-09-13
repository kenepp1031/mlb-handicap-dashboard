"""Consistent provider failures without leaking request URLs or credentials."""
import requests


def get_json(url: str, error_type: type[RuntimeError], **kwargs):
    try:
        response = requests.get(url, timeout=12, **kwargs)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise error_type(f"Data feed unavailable ({type(error).__name__}). Try refreshing.") from error

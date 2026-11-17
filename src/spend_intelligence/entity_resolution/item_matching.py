from ..preprocessing.cleaning import canonical_item
from ..preprocessing.normalization import stable_key


def item_key(description=None, code=None) -> str:
    return stable_key("item", canonical_item(description, code))

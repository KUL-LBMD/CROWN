import os
import pickle
import functools
import gemmi

def normalize(s: str) -> str:
    return s.replace("\\", "")

from src.config import DATA_DIR
CCD_CACHE_PATH = os.path.join(
    DATA_DIR,
    "mmCIF",
    "ccd",
    "components_heavy.pkl",
)

def build_ccd_cache(force: bool = False) -> dict:
    if not force and os.path.exists(CCD_CACHE_PATH):
        with open(CCD_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    doc = gemmi.cif.read(
        os.path.join(os.path.dirname(CCD_CACHE_PATH), "components.cif")
    )

    heavy = {}

    for block in doc:
        table = block.find(
            "_chem_comp_atom.",
            ["atom_id", "type_symbol"],
        )

        if not table:
            continue

        heavy[block.name] = frozenset(
            normalize(row[0])
            for row in table
            if row[1] not in ("H", "D")
            and row[0] != "OXT"
        )

    os.makedirs(os.path.dirname(CCD_CACHE_PATH), exist_ok=True)

    with open(CCD_CACHE_PATH, "wb") as f:
        pickle.dump(heavy, f, protocol=pickle.HIGHEST_PROTOCOL)

    return heavy

@functools.lru_cache(maxsize=1)
def _load_ccd_cache():
    if os.path.exists(CCD_CACHE_PATH):
        with open(CCD_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    return build_ccd_cache()

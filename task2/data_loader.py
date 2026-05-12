import numpy as np
import os


def load_tsp_file(file_path):
    """
    Parse a TSPLIB file (.tsp) and extract city coordinates.

    The function reads the file line by line, identifies the dimension,
    skips the header metadata, and reads the NODE_COORD_SECTION.

    Parameters
    ----------
    file_path : str
        The relative or absolute path to the .tsp file.

    Returns
    -------
    coords : ndarray
        An (N, 2) array of float coordinates (x, y).
    info   : dict
        Metadata about the problem (name, type, dimension).
    """
    coords = []
    info = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    reading_coords = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ":" in line and not reading_coords:
            key, value = line.split(":", 1)
            info[key.strip().lower()] = value.strip()
            continue

        if line.startswith("NODE_COORD_SECTION"):
            reading_coords = True
            continue

        if line.startswith("EOF"):
            break

        if reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append([x, y])

    return np.array(coords), info


def get_available_instances(directory):
    """
    List all available .tsp files in a given directory.

    Parameters
    ----------
    directory : str
        Path to the folder containing .tsp files (e.g., 'tsplib/').

    Returns
    -------
    files : list
        A list of filenames ending with .tsp.
    """
    return [f for f in os.listdir(directory) if f.endswith(".tsp")]
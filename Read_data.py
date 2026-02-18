import numpy as np

def read_data(path, dtype=np.complex64, count=2_000_000):
    """
    Reads complex data from a binary file.

    Parameters:
    path (str): The path to the binary file.
    dtype: The data type of the complex numbers (default is np.complex64).
    count (int): The number of complex numbers to read (default is 2,000,000).

    Returns:
    np.ndarray: An array of complex numbers read from the file.
    """
    try:
        data = np.fromfile(path, dtype=dtype, count=count)
        print(f"Successfully read from {path}!")
        return data
    except Exception as e:
        print(f"Error reading data from {path}: {e}")
        return None
from load_data import load_anns
import matplotlib.pyplot as plt
import numpy as np


anns = load_anns('data/annotations.xml')
print(anns["annotations"].keys())
print(anns["annotations"]["image"][0])
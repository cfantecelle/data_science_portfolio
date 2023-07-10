############################################################################################################################

########################################
## Author: Carlos Henrique Fantecelle ##
########################################

# This module contains useful functions
# I have built or gathered along. When
# the latter is the case, I indicated
# the author in the description of said
# function.

##################
## Dependencies ##                                                                                                          
##################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import textwrap


############################################################################################################################

def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, 
    millions and thousands) such as 4500 into 
    4.5K and also appropriately turns 4000 
    into 4K (no zero after the decimal).

    Author: https://dfrieds.com/
    """

    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 2)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 2)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 2)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 2)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    # index_of_decimal = new_tick_format.find(".")

    # if index_of_decimal != -1:
    #     value_after_decimal = new_tick_format[index_of_decimal+1]
    #     if value_after_decimal == "0":
    #         # remove the 0 after the decimal point since it's not needed
    #         new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format


# Defining some code to wrap labels (Code by: Ted Petrou @ Dunder Data)
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
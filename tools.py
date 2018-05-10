import numpy as np
import math


def increase_size_list(small_list, new_size):
    size_small_list = len(small_list)

    if new_size>size_small_list:
        big_list = np.zeros((new_size))
        next_small_idx = 0
        next_big_idx = 0

        for idx in range(new_size):
            if next_big_idx <= idx:
                big_list[idx] = small_list[next_small_idx]
                next_small_idx += 1
                next_big_idx = math.floor(next_small_idx / size_small_list * new_size)
            else:
                big_list[idx] = big_list[idx-1]

        return big_list
    else:
        return small_list

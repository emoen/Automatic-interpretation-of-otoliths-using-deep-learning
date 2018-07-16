import numpy as np

def dataframe_to_nparray_for_pairs_to_singles(a_pairs_dataframe, pair_to_keep = 'both'):
    """ takes a pandas dataframe of pairs of otoliths and 
    returns numpy array of each image vector a, and b, each age y, and yield
    and each index in x_otolith_small
    >>> import pandas as pd
    >>> data = np.array([np.arange(5)]*3) 
    >>> data = np.vstack( (data, [[0,2,4,6,8], [1,3,5,7,9]]) ).T  
    >>> df_ = pd.DataFrame(data, columns=['image_vector_a', 'image_vector_b', 'y', 'idx_a', 'idx_b'])
    >>> x, y, idx_ = dataframe_to_nparray_for_pairs_to_singles( df_ )
    >>> x
    array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    >>> idx_
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    x, y, idx_xy = [],[],[]
    if pair_to_keep == 'both':
        for a_index, a_row in a_pairs_dataframe.iterrows():
            x.append(a_row['image_vector_a'])
            x.append(a_row['image_vector_b'])
            y.append(a_row['y'])
            y.append(a_row['y'])
            idx_xy.append(a_row['idx_a'])
            idx_xy.append(a_row['idx_b'])
    elif pair_to_keep == 'left':
        for a_index, a_row in a_pairs_dataframe.iterrows():
            x.append(a_row['image_vector_a'])
            y.append(a_row['y'])
            idx_xy.append(a_row['idx_a'])
    elif pair_to_keep == 'right':
        for a_index, a_row in a_pairs_dataframe.iterrows():
            x.append(a_row['image_vector_b'])
            y.append(a_row['y'])
            idx_xy.append(a_row['idx_b'])
    
    return np.asarray(x), np.asarray(y), idx_xy
    
def dataframe_to_nparray( singles ):
    """
    >>> import pandas as pd
    >>> data = np.array([np.arange(5)]*3).T
    >>> df_ = pd.DataFrame(data, columns=['image_vector', 'y', 'idx'])
    >>> x, y, idx_ = dataframe_to_nparray( df_ )    
    >>> idx_
    [0, 1, 2, 3, 4]
    """
    x, y, idx_xy = [],[],[]
    for a_index, a_row in singles.iterrows():
        x.append(a_row['image_vector'])
        y.append(a_row['y'])
        idx_xy.append(a_row['idx'])
    
    return np.asarray(x), np.asarray(y), idx_xy
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()    
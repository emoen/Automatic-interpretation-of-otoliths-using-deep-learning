import pandas as pd

def find_pairs_of_otoliths( x_otolith_rescaled, x, y ):
    """ given array of paths to images given in x - finds pairs where a pair
        is the same filename ending with _a or _b before file-extension

    >>> from shrink_img import shrink_img_rgb
    >>> x = ['./batch1/2006_02098sep1_t01a.jpg', './batch1/2006_02098sep1_t01b.jpg', './batch1/2006_02098sep1_t02a.jpg']
    >>> y = [1,1,3]
    >>> x_otolith_rescaled = shrink_img_rgb( (299,299,3), x)
    >>> pairs, not_pairs = find_pairs_of_otoliths( x_otolith_rescaled, x, y)
    >>> len(pairs)
    1
    >>> len(not_pairs)
    1
    >>> pairs.loc[0,'y']
    1
    >>> pairs.loc[0,'filename_a']
    '2006_02098sep1_t01a'
    >>> pairs.loc[0,'filename_b']
    '2006_02098sep1_t01b'
    >>> pairs.loc[0,'idx_a']
    0
    >>> pairs.loc[0,'idx_b']
    1
    >>> not_pairs.loc[0,'y']
    3
    >>> not_pairs.loc[0, 'filename']
    '2006_02098sep1_t02a'


    """
    assert len(x_otolith_rescaled) == len(x)
    assert len(x) == len(y)

    file_ext = '.jpg'
    sub_len_file_ext = len(file_ext) * -1

    not_pairs = pd.DataFrame(columns=['y', 'filename', 'image_vector', 'idx']) # is not used for val or test so doesnt need idx
    pairs = pd.DataFrame(columns=['y', 'y_pred_mean', 'filename_a', 'filename_b', 'image_vector_a', 'image_vector_b', 'idx_a', 'idx_b'], dtype=int)
    filenames = [f.split('/')[-1][0:sub_len_file_ext] for f in x] #gets only filename parth of path - including a or b
    for index, afilename in enumerate(filenames):
        if not afilename in pairs['filename_a'].values and not afilename in pairs['filename_b'].values:
            last_letter = afilename[-1:]
            without_last_letter = afilename[0:-1]
            idx_other_file = []
            if last_letter == 'a':
                idx_other_file = [index for index, word in enumerate(filenames) if without_last_letter+'b' in word ]
            elif last_letter == 'b':
                idx_other_file = [index for index, word in enumerate(filenames) if without_last_letter+'a' in word ]
            idx_afilename = []
            if len(idx_other_file) > 0:
                idx_afilename = [index for index, word in enumerate(filenames) if without_last_letter in word ]
            if (last_letter == 'a' or last_letter == 'b') and len(idx_afilename) > 1:
                a_file = filenames[ idx_afilename[0] ]
                b_file = filenames[ idx_afilename[1] ]
                if a_file[-1:] != 'a':
                    tmp = a_file
                    a_file = b_file
                    b_file = tmp
                    tmp_idx_b = idx_afilename[0]
                    idx_afilename[0] = idx_afilename[1]
                    idx_afilename[1] = tmp_idx_b
                x_otolith_rescaled_a = x_otolith_rescaled[idx_afilename[0]]
                x_otolith_rescaled_b = x_otolith_rescaled[idx_afilename[1]]
                #extra_dim_x_otolith_rescaled_a = np.expand_dims(x_otolith_rescaled_a, axis=0)
                #extra_dim_x_otolith_rescaled_b = np.expand_dims(x_otolith_rescaled_b, axis=0)
                pairs = pairs.append({'y': y[idx_afilename[0]], 'filename_a': a_file, 'filename_b': b_file, \
                 'image_vector_a': x_otolith_rescaled_a, 'image_vector_b': x_otolith_rescaled_b,
                                      'idx_a': int(idx_afilename[0]), 'idx_b': int(idx_afilename[1]), ' y_pred_mean':None}, ignore_index=True)
            else:
                not_pairs = not_pairs.append({'y':y[index], 'filename':afilename, \
                    'image_vector':x_otolith_rescaled[index], 'idx': int(index)}, ignore_index=True)
    pairs = pairs.astype(dtype={'y': 'int64', 'filename_a': 'object', 'filename_b': 'object', 'image_vector_a': 'object', \
               'image_vector_b': 'object', 'idx_a': 'int64', 'idx_b': 'int64', 'y_pred_mean': 'object'})
    not_pairs = not_pairs.astype(dtype={'y': 'int64', 'filename': 'object', 'image_vector': 'object', 'idx': 'int64'})
    
    assert not(set(not_pairs['idx']) < set(pairs['idx_a']))
    assert not(set(not_pairs['idx']) < set(pairs['idx_b']))

    return pairs, not_pairs

if __name__ == '__main__':
    import doctest
    doctest.testmod()


def filter_data(images, labels, *choices):
    '''
    INPUT
    images: np array of images
    labels: np array consisting of 2 state labels (2 letter abbrev) for images
    *choices: items in labels to filter

    OUTPUT
    np array: filtered images
    np array: filtered labels of choices, but converted to integers starting with 0 for first of choices, 1 for second, etc.
    compare_type: string showing each choice selected separated by "_"

    filter images and lables per *choices, e.g.:
    images = [1, 2, 1, 3, 1]
    labels = ['CO', 'WA', 'CO', 'NM', 'CO']
    filter_data(images, labels, 'CO', 'NM')
    returns [1, 1, 3, 1], [0, 0, 1, 0,], 'CO_NM'
    '''
    images_list = []
    labels_list = []
    compare_type = ""
    choice_int = -1
    for choice in choices:
        compare_type += choice + "_"
    compare_type = compare_type[:-1] #drop last "_" on compare_type

    for image, label in zip(images, labels):
            choice_int = -1
            for choice in choices:
                choice_int += 1
                if label == choice:
                    images_list.append(image)
                    labels_list.append(choice_int)

    return np.array(images_list), np.array(labels_list), compare_type

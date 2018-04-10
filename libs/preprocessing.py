import numpy as np

# ToDo: fourth input variable for data-augmentation
# f.e. angle: shifting, rotating and scaling images, as well as augmenting grey values
def crop2D(arr, pos, size):
    """Crop 2D out of one image."""
    pos = np.array(pos)
    end = pos[:2] + np.array([size,size])
    if (np.array(arr.shape[:2])-end).min() >= 0:
        result = arr[pos[0]:end[0], pos[1]:end[1], pos[2],:]
    else:
        print('Invalid arguments, returning None.')
        result = None
    return result

def crop3D(arr, pos, size):
    """Crop 3D out of images."""
    pos = np.array(pos)
    end = pos + np.array([size,size,size])
    if (np.array(arr.shape[:3])-end).min() >= 0:
        result = arr[pos[0]:end[0], pos[1]:end[1], pos[2]:end[2] ,:]
    else:
        print('Invalid arguments, returning None.')
        result = None
    return result

def crop(dim, arr, pos, cropsize):
    """Decide which crop-method to use."""
    if dim == 2: result = crop2D(arr, pos, cropsize)
    elif dim == 3: result = crop3D(arr, pos, cropsize)
    else: raise 'Unknown mode.'
    return result

def augmentation(dim, patient, label, samples_size, cropsize_X, cropsize_Y, border):  #here data augmentation is just crop
    """Data augmentation for training samples."""
    size_x, size_y, size_z = patient.shape[:3]              #sample size: how many augmentations for one patient
    pos_x_X = np.random.randint(border, size_x-cropsize_X-border, samples_size)         #cropsize_x = 32
    pos_y_X = np.random.randint(border, size_y-cropsize_X-border, samples_size)         # cropsize_y if not out_res = 32
    pos_z_X = np.random.randint(border, size_z-cropsize_X-border, samples_size)         # border = 20, distance in pixel between crops
    shape_X1 = (samples_size,) + dim*(cropsize_X,)          # so its a 32 patch size with a stride of 20
                                                        #[1,32,32,32]
    dist = (cropsize_X-cropsize_Y) // 2
    pos_x_Y = pos_x_X + dist
    pos_y_Y = pos_y_X + dist
    if dim == 2: pos_z_Y = pos_z_X
    elif dim == 3:  pos_z_Y = pos_z_X + dist
    else: raise "Invalid mode."
    shape_Y1 = (samples_size,) + dim*(cropsize_Y,)
    
    channels = patient.shape[-1]
    shape_X2 = shape_X1 + (channels,)
    samples_X = np.ndarray(shape_X2)
    
    classes = label.shape[-1]
    shape_Y2 = shape_Y1 + (classes,)
    samples_Y = np.ndarray(shape_Y2)
    for i, pos in enumerate(zip(pos_x_X, pos_y_X, pos_z_X)):
        samples_X[i,...] = crop(dim, patient, pos, cropsize_X)
    for i, pos in enumerate(zip(pos_x_Y, pos_y_Y, pos_z_Y)):
        samples_Y[i,...] = crop(dim, label, pos, cropsize_Y)

    pos_X = [np.array([pos_x,pos_y,pos_z]) for pos_x,pos_y,pos_z in zip(pos_x_X,pos_y_X,pos_z_X)]
    #  normalize position
    max_pos = np.array([size_x-cropsize_X-border, size_y-cropsize_X-border, size_z-cropsize_X-border])
    pos_X = [np.divide(pos, max_pos) for pos in pos_X]
    
    return samples_X, pos_X, samples_Y

def get_meshgrid(patient, density, cropsize, border):
    """Get meshgrid to generate validation samples."""
    x=np.linspace(border, patient.shape[0]-cropsize-border, density)
    y=np.linspace(border, patient.shape[1]-cropsize-border, density)
    z=np.linspace(border, patient.shape[2]-cropsize-border, density)
    
    result = (np.array(np.meshgrid(x,y,z)).reshape(3,density**3).T).astype('int')
    return result


def get_valid(mode, patients_valid, labels_valid, density, cropsize_X, cropsize_Y, border):
    """Get validation data."""
    X_valid = []
    Y_valid = []
    dim = int(mode[0])

    pos_feed = []
    for i, (patient, label) in enumerate(zip(patients_valid, labels_valid)):
        positions_X = get_meshgrid(patient, density, cropsize_X, border)

        # normalize postion
        size_x, size_y, size_z = patient.shape[:3]
        max_pos = np.array([size_x-cropsize_X-border, size_y-cropsize_X-border, size_z-cropsize_X-border])
        positions_X_normalized = [np.divide(pos_X, max_pos) for pos_X in positions_X]

        pos_feed.append(positions_X_normalized)

        dist = (cropsize_X - cropsize_Y) // 2
        if dim == 2: delta = [dist, dist, 0]
        elif dim == 3: delta = [dist, dist, dist]
        else: raise "Invalid mode."
        delta = np.array(delta)
        positions_Y = np.array([pos_X + delta for pos_X in positions_X])

        for pos_X, pos_Y in zip(positions_X, positions_Y):
            X_valid.append(crop(dim, patient, pos_X, cropsize_X))
            Y_valid.append(crop(dim, label, pos_Y, cropsize_Y))

    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)

    pos_feed = np.reshape(np.array(pos_feed), (-1,3))

    return X_valid, pos_feed, Y_valid




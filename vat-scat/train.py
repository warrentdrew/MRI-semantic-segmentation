from libs.training import generator_valid
from utils import generator_train
import libs.preprocessing as preprocessing
import keras
import keras.backend as K
from utils import batch_num

def fit(model,
        patients_train,
        data_valid,
        epochs,
        batch_size,
        patient_buffer_capacity,
        batches_per_shift,
        density,
        border,
        callbacks,
        mult_inputs=False,
        empty_patient_buffer=False):
    """Perform training on given model and datasets."""  # returns a history object
    batches_per_train_epoch = batches_per_shift * len(patients_train)

    if mult_inputs:  # multi input stands for adding pos information
        cropsize_X = model.get_input_shape_at(0)[0][
            1]  # get_input_shape_at(0)， 0 stands for the 0th node(multi input situation),
        # model shape [batch_size, w, h, slice, channels] output size cropsize * cropsize
    else:
        cropsize_X = model.get_input_shape_at(0)[1]

    cropsize_Y = model.get_output_shape_at(-1)[
        1]  # get_output_shape_at(-1) stands for the last output of the network, this means the output layer
    dim = len(model.get_output_shape_at(-1)) - 2  # this means 3D > output 5 dim, 2D > output 4 dim
    mode = str(dim) + 'D'

    patients_valid = []
    labels_valid = []
    for patient_valid, label_valid in data_valid:
        patients_valid.append(patient_valid)
        labels_valid.append(label_valid)

    '''
    use the revised verion of generator_train in utils
    generator_train(patient_list, capacity, batch_size, cropsize_X, cropsize_Y, batches_per_load, border, mult_inputs)
    '''
    gen_train = generator_train(patients_list = patients_train,
                                capacity=patient_buffer_capacity,
                                batches_per_shift=batches_per_shift,
                                batch_size=batch_size,
                                cropsize_X=cropsize_X,
                                cropsize_Y=cropsize_Y,
                                batches_per_load = batches_per_shift,
                                border=border)

    X_valid, pos_X, Y_valid = preprocessing.get_valid(mode=mode,
                                                      patients_valid=patients_valid,
                                                      labels_valid=labels_valid,
                                                      density=density,
                                                      cropsize_X=cropsize_X,
                                                      cropsize_Y=cropsize_Y,
                                                      border=border)
    gen_valid = generator_valid(X_valid=X_valid,
                                pos_X=pos_X,
                                Y_valid=Y_valid,
                                batch_size=batch_size,
                                mult_inputs=mult_inputs)

    steps_valid = Y_valid.shape[0] // batch_size
    histObj = model.fit_generator(generator=gen_train,
                                  epochs=epochs,
                                  steps_per_epoch=batches_per_train_epoch,
                                  validation_data=gen_valid,
                                  validation_steps=steps_valid,
                                  max_q_size=(batches_per_shift) + 1,
                                  callbacks=callbacks)
    return histObj
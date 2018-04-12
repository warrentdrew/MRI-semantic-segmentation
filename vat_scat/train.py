#from libs.training import generator_valid
from utils import data_generator
#import libs.preprocessing as preprocessing
import keras
import keras.backend as K
from utils import batch_num

def fit(model,
        patients_train,
        patients_valid,
        epochs,
        batch_size,
        patient_buffer_capacity,
        batches_per_shift,
        steps_valid,
        border,
        callbacks,
        mult_inputs=False):
    """Perform training on given model and datasets."""  # returns a history object
    batches_per_train_epoch = batches_per_shift * len(patients_train)

    if mult_inputs:  # multi input stands for adding pos information
        cropsize_X = model.get_input_shape_at(0)[0][1]  # get_input_shape_at(0)， 0 stands for the 0th node(multi input situation),
        # model shape [batch_size, w, h, slice, channels] output size cropsize * cropsize
    else:
        cropsize_X = model.get_input_shape_at(0)[1]


    cropsize_Y = model.get_output_shape_at(-1)[1]  # get_output_shape_at(-1) stands for the last output of the network, this means the output layer

    print('cropsize_x:', cropsize_X)
    print('cropsize_y:', cropsize_Y)

    '''
    use the revised verion of generator_train in utils
    data_generator(patient_list, capacity, batch_size, cropsize_X, cropsize_Y, batches_per_load, border, mult_inputs)
    '''
    gen_train = data_generator(patient_list = patients_train,
                                capacity=patient_buffer_capacity,
                                batch_size=batch_size,
                                cropsize_X=cropsize_X,
                                cropsize_Y=cropsize_Y,
                                batches_per_load = batches_per_shift,
                                border=border,
                                mult_inputs= mult_inputs)

    gen_valid = data_generator(patient_list = patients_valid,
                                capacity=patient_buffer_capacity,
                                batch_size=batch_size,
                                cropsize_X=cropsize_X,
                                cropsize_Y=cropsize_Y,
                                batches_per_load = batches_per_shift,
                                border = border,
                                mult_inputs= mult_inputs)

    # steps_valid = Y_valid.shape[0] // batch_size
    histObj = model.fit_generator(generator=gen_train,
                                  epochs=epochs,
                                  steps_per_epoch=batches_per_train_epoch,
                                  validation_data=gen_valid,
                                  validation_steps=steps_valid,
                                  max_q_size=(batches_per_shift) + 1,
                                  callbacks=callbacks)
    return histObj
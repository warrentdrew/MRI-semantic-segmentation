
import random
import numpy as np
import keras
import util
import preprocessing

class PatientBuffer():
    """Loading patients dynamically in RAM while training."""
    def __init__(self, patients, capacity, batch_size, cropsize_X, cropsize_Y, dim, verbose=False):
        """Constructor."""
        self.patients = patients                #patient buffer capacity is 30  patient is a list which represents the buffer
        self.batch_size = batch_size            #batch_size = 48
        self.cropsize_X = cropsize_X
        self.cropsize_Y = cropsize_Y
        self.dim = dim
        self.verbose = verbose
        
        if len(patients) < capacity:
            capacity = len(patients)

        self.samples_per_patient = batch_size // capacity
        self.sh_register = util.ShiftRegister(capacity=capacity)
        self.patient_pointer = capacity
        
        for patient in self.patients[:capacity]:
            self.sh_register.shift(patient.get_slices(verbose=verbose), verbose=True)       #shift means to add a new dicom array(input) and its corresponding label (a set),this jus means onw patient, pointer move to the next item
                                                                                            #this shift is the shift in the util.py
            #so the buffer will only contain 30 patients
        
        self.build_batch_buffer()
        
    def shift(self):
        """Fill shift-register of PatientBuffer."""
        capacity = self.sh_register.capacity

        if self.patient_pointer == capacity - 1:               #when pointer points to the last elemant in the buffer, drop all the elements in the buffer and reset pointer to the first one
            for i in range(capacity - 1):
                self.patients[capacity - i - 2].drop()          #drop is implemented in Patient class, slice counter-1
            self.patients[len(self.patients) - 1].drop()
            random.shuffle(self.patients)
            for patient in self.patients[:capacity]:
                self.sh_register.shift(patient.get_slices(verbose=self.verbose), verbose=True)
        else:
            self.sh_register.shift(self.patients[self.patient_pointer].get_slices(verbose=self.verbose))
            self.patients[self.patient_pointer - capacity].drop()
        
        self.patient_pointer = (self.patient_pointer + 1) % len(self.patients)
     
    def drop_all(self):
        for i in range(len(self.patients)):
            self.patients[i].drop()   
        
    def build_batch_buffer(self):
        channels, classes = [x.shape[-1] for x in self.sh_register.content[0]]      # here x = (arrayDicom, arraylabels), x.shape[-1] = arrayDicim.shape[-1], arraylabels.shape[-1]
        
        shape_X = (self.batch_size,) + self.dim * (self.cropsize_X,) + (channels,)
        shape_Y = (self.batch_size,) + self.dim * (self.cropsize_Y,) + (classes,)
        
        self.batch_X = np.ndarray(shape_X)
        self.batch_Y = np.ndarray(shape_Y)
    
    def sample_batch(self, border):
        positions = []
        end = 0
        for patient_train_slices, label_train_slices in self.sh_register.content:
            start = end
            end += self.samples_per_patient
            self.batch_X[start:end,...], pos, self.batch_Y[start:end,...] = preprocessing.augmentation(self.dim,                #sample_per_patient = 1
                                                                                                       patient_train_slices,
                                                                                                       label_train_slices,
                                                                                                       self.samples_per_patient,
                                                                                                       self.cropsize_X,
                                                                                                       self.cropsize_Y,
                                                                                                       border)
            positions += pos
             
        for i in range(end,self.batch_size):
            rnd = np.random.randint(self.sh_register.capacity)
            patient_train_slices, label_train_slices = [arr for arr in self.sh_register.content[rnd]]          #change a set arraydicom, niidicom into a list
            self.batch_X[i,...], pos, self.batch_Y[i,...] = preprocessing.augmentation(self.dim,
                                                                             patient_train_slices,
                                                                             label_train_slices,
                                                                             1,
                                                                             self.cropsize_X,
                                                                             self.cropsize_Y,
                                                                             border)
            positions += pos
        positions = np.array(positions)
        return (self.batch_X, positions, self.batch_Y)

# ToDo: 
# data augmentation
def generator_train(mode,
                    patients_train,
                    patient_buffer_capacity,
                    batches_per_shift,
                    batch_size,
                    cropsize_X,
                    cropsize_Y,
                    border,
                    mult_inputs,
                    epochs,
                    steps_per_epoch,
                    empty_patient_buffer):
    """Generater for feeding training batches to GPU."""
    patient_buffer = PatientBuffer(patients=patients_train,
                                   capacity=patient_buffer_capacity,
                                   batch_size=batch_size,
                                   cropsize_X=cropsize_X,
                                   cropsize_Y=cropsize_Y,
                                   dim=int(mode[0]))
    counter = 0
    batch_counter = 0
    final = steps_per_epoch * epochs
    while True:
        batch_X, pos, batch_Y = patient_buffer.sample_batch(border=border)
        counter += 1
        if ((len(patients_train) > patient_buffer_capacity) and (counter == batches_per_shift)):
            counter = 0
            patient_buffer.shift()
        
        batch_X_dict = {'input_X' : batch_X}
        batch_Y_dict = {'output_Y' : batch_Y}
        if mult_inputs:
            batch_X_dict['input_position'] = pos

        if empty_patient_buffer:
            batch_counter += 1
            if batch_counter == final:
                #print("Drop all!")
                patient_buffer.drop_all()

        yield batch_X_dict, batch_Y_dict

def generator_valid(X_valid, pos_X, Y_valid, batch_size, mult_inputs):
    """Generater for feeding validation batches to GPU."""
    size = Y_valid.shape[0]
    limit = size - batch_size
    end = size+1
    while True:
        if end > limit:
            end = 0
        start = end
        end += batch_size
        batch_X = X_valid[start:end]
        batch_Y = Y_valid[start:end]

        batch_X_dict = {'input_X' : batch_X}
        batch_Y_dict = {'output_Y' : batch_Y}
        if mult_inputs:
            batch_pos_X = pos_X[start:end]
            batch_X_dict['input_position'] = batch_pos_X
      
        yield batch_X_dict, batch_Y_dict

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
    """Perform training on given model and datasets."""       #returns a history object
    batches_per_train_epoch = batches_per_shift * len(patients_train)
    
    if mult_inputs:                                             #multi input stands for adding pos information
        cropsize_X = model.get_input_shape_at(0)[0][1]          #get_input_shape_at(0)， 0 stands for the 0th node(multi input situation),
                                                                # model shape [batch_size, w, h, slice, channels] output size cropsize * cropsize
    else:
        cropsize_X = model.get_input_shape_at(0)[1]
    
    cropsize_Y = model.get_output_shape_at(-1)[1]           #get_output_shape_at(-1) stands for the last output of the network, this means the output layer
    dim = len(model.get_output_shape_at(-1)) - 2                # this means 3D > output 5 dim, 2D > output 4 dim
    mode = str(dim) + 'D'
    
    patients_valid = []
    labels_valid = []
    for patient_valid, label_valid in data_valid:
        patients_valid.append(patient_valid)
        labels_valid.append(label_valid)
    
        
    gen_train = generator_train(mode=mode,
                                patients_train=patients_train,
                                patient_buffer_capacity=patient_buffer_capacity,
                                batches_per_shift=batches_per_shift,
                                batch_size=batch_size,
                                cropsize_X=cropsize_X,
                                cropsize_Y=cropsize_Y,
                                border=border,
                                mult_inputs=mult_inputs,
                                epochs=epochs,
                                steps_per_epoch=batches_per_train_epoch,
                                empty_patient_buffer=empty_patient_buffer)
    
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

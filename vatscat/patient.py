'''
Setting for importing the path
'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from libs.util import Patient
from utils import load_data, merge_labels
'''
patient class for AT are inherited from the original patient class 
'''
class Patient_AT(Patient):
    def __init__(self, patient_path, forget_slices):
        Patient.__init__(self, dicom_dirs=None, nii_dirs=None, forget_slices=forget_slices)
        self.patient_path = patient_path

    def load_slices(self, verbose=False):
        """Load patients data  in numpy arrays."""

        train = load_data(self.patient_path)['img']
        train_shape = train.shape + (1,)
        train = train.reshape(train_shape)  # change the shape into (192, 256, 105, 1)
        label = merge_labels(self.patient_path)

        return (train, label)




def load_correct_patient(train_path, validation_path, test_path, forget_slices):
    '''
    preparation for the fit function
    1. creating the Patient instances and store then in the ring buffer
    2. creating the Patients matrix for the validation data
    :param train_path: 
    :param validation_path: 
    :param test_path: 
    :return: 
    '''
    patients_train = []
    patients_val = []
    patients_test = []

    for pat_train in train_path:
        patient_train = Patient_AT(patient_path= pat_train, forget_slices = forget_slices)
        patients_train.append(patient_train)
    for pat_val in validation_path:
        patient_val = Patient_AT(patient_path= pat_val, forget_slices = forget_slices)
        patients_val.append(patient_val)
    for pat_test in test_path:
        patient_test = Patient_AT(patient_path= pat_test, forget_slices = forget_slices)
        patients_val.append(patient_test)

    # get slices for validation data
    patients_val_slices = []
    for patient in patients_val:
        slices = patient.get_slices(verbose=False)  #slice is a tuple of (data, label)
        patients_val_slices.append(slices)

    return (patients_train, patients_val, patients_test, patients_val_slices)

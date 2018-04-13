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
        label = merge_labels(self.patient_path)

        return (train, label)

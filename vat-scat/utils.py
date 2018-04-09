'''
module for preprocessing
The data are stored in the .mat format 
Each patient folder contains a file named "rework.mat" which contains:
- img: MR image
- P_BG: background mask/label
- P_LT: lean tissue mask/label
- P_AT: (subcutaneous) adipose tissue mask/label
- P_VAT: visceral adipose tissue mask/label
- info: MR DICOM acquisition parameter and patient info
'''

import scipy.io as sio
import numpy as np
import time

file_path = '/med_data/Segmentation/AT/1_5T/PLIS_3609_GK/rework.mat'

mat = sio.loadmat(file_path)
print('img shape:', mat['img'].dtype)
print('SCAT shape:', mat['P_AT'].shape)
print('VAT shape:', mat['P_VAT'].shape)
print('bg shape:', mat['P_BG'].shape)

'''
use ring buffer to load data
each space for the ring buffer contains a Patient object
so its necessary to include the patient class in the new implementation 
'''

class Patient():
    """Creates patient."""

    def __init__(self, patient_path, forget_slices=False):
        """"Constructor."""
        self.patient_path = patient_path
        self.slices = None
        self.shape = None
        self.prediction = None
        self.forget_slices = forget_slices
        self.slice_counter = 0

    def get_slices(self, count=True, verbose=False):
        """Take loaded slices if already available or load slices if not."""
        if count:
            self.slice_counter += 1  # this slices counter actually stands for the num of patients
        if self.slices is None:
            self.slices = self.load_slices(verbose=verbose)
            # print("slice-counter: " + str(self.slice_counter))
            # print("load slices!")
        return self.slices

    def drop(self):
        """Drop slices for PatientBuffer in training."""
        # if self.slices == None:
        # print("Slices are already None!")
        if self.slice_counter != 0:
            self.slice_counter -= 1
            if self.slice_counter == 0 and self.forget_slices:
                self.slices = None
                # print("drop!")

    def get_prediction(self, model):
        """"Take prediction if already available or predict if not."""
        if self.prediction is None:
            self.prediction = self.predict_patient(model)
        return self.prediction

    def load_slices(self, verbose=False):
        """Load patients data (dicom- and nii-files) in numpy arrays."""
        # measure time
        t0 = time.time()
        decomp = False

        channels = len(self.dicom_dirs)
        classes = len(self.nii_dirs) + 1

        # onehot-encoding for validation-data
        onehot_enc = lambda arr: np.eye(classes)[arr]

        # get reference for patient:
        DirRef = del0002(os.listdir(self.dicom_dirs[0]))
        PathRef = os.path.join(self.dicom_dirs[0], DirRef[
            0])  # dicom_dirs = train_path_dcm = (a path in) train_paths_dcm = DicomPathsList[cut:] = pickle.load('patients.pkl')
        RefDs = dicom.read_file(PathRef)
        # Load dimensions:
        ConstPixelDims = (int(RefDs.Rows),
                          int(RefDs.Columns),
                          len(DirRef),
                          channels)
        # Load spacing values (in mm):
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),
                             float(RefDs.PixelSpacing[1]),
                             float(RefDs.SliceThickness))

        ArrayDicom = np.zeros(ConstPixelDims, dtype='float16')

        # walk through channels:
        for ch_index, ch_path in enumerate(self.dicom_dirs):
            lstFilesDCM = del0002(os.listdir(ch_path))
            # avoid double entries:
            lstFilesDCM = list(set(lstFilesDCM))
            # sort in ascending order:
            lstFilesDCM.sort()
            lstPathsDCM = list(map(lambda file_name: os.path.join(ch_path, file_name), lstFilesDCM))
            if verbose: print(" Loading " + ch_path + "...")
            # loop through all the DICOM files
            for z, pathDCMch in enumerate(lstPathsDCM):
                # read the files
                ds = dicom.read_file(pathDCMch)
                # store the raw image data
                try:
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T
                except NotImplementedError:
                    # decompress dicom file
                    decomp = True
                    decomp_filename = '/tmp/curr_patient_decomp.dcm'
                    cmd = ['/opt/gdcm-2.8.3/gdcmbin/bin/gdcmconv', '--raw', '--quiet', '-o', decomp_filename, '-i',
                           pathDCMch]
                    cmd_complete = subprocess.call(cmd)
                    ds = dicom.read_file(decomp_filename)
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T

        ArrayNii = np.zeros(ConstPixelDims[:3], dtype='uint8')
        # walk through classes/labels:
        for cl_index, label_path in enumerate(self.nii_dirs):
            label = nib.load(label_path)
            ArrayNii += (cl_index + 1) * np.array(label.dataobj)

        # onehot encoding:
        ArrayNii = onehot_enc(ArrayNii)

        # normalize ArrayDicom:
        mx = [np.max(ArrayDicom[..., ch]) for ch in range(channels)]
        ArrayDicom = np.divide(ArrayDicom, mx)

        # print needed time in seconds
        nd_time = time.time() - t0
        if verbose:
            if decomp:
                print("Needed time: " + str(nd_time) + "s." + " Files had to be decompressed.")
            else:
                print("Needed time: " + str(nd_time) + "s.")

        return (ArrayDicom, ArrayNii)

    def save_decomp_slices(self, verbose=False):
        """Load patients data (dicom- and nii-files) in numpy arrays."""

        # measure time
        t0 = time.time()
        decomp = False

        channels = len(self.dicom_dirs)
        classes = len(self.nii_dirs) + 1
        # onehot-encoding for validation-data
        onehot_enc = lambda arr: np.eye(classes)[arr]

        # get reference for patient:
        DirRef = del0002(os.listdir(self.dicom_dirs[0]))
        PathRef = os.path.join(self.dicom_dirs[0], DirRef[0])
        RefDs = dicom.read_file(PathRef)
        # Load dimensions:
        ConstPixelDims = (int(RefDs.Rows),
                          int(RefDs.Columns),
                          len(DirRef),
                          channels)
        # Load spacing values (in mm):
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),
                             float(RefDs.PixelSpacing[1]),
                             float(RefDs.SliceThickness))

        ArrayDicom = np.zeros(ConstPixelDims, dtype='float16')

        # walk through channels:
        for ch_index, ch_path in enumerate(self.dicom_dirs):
            lstFilesDCM = del0002(os.listdir(ch_path))
            # avoid double entries:
            lstFilesDCM = list(set(lstFilesDCM))
            # sort in ascending order:
            lstFilesDCM.sort()
            lstPathsDCM = list(map(lambda file_name: os.path.join(ch_path, file_name), lstFilesDCM))
            if verbose: print(" Loading " + ch_path + "...")
            # loop through all the DICOM files
            for z, pathDCMch in enumerate(lstPathsDCM):
                # read the files
                ds = dicom.read_file(pathDCMch)
                # store the raw image data
                try:
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T
                except NotImplementedError:
                    # decompress dicom file
                    decomp = True
                    cmd = ['/opt/gdcm-2.8.3/gdcmbin/bin/gdcmconv', '--raw', '--quiet', '-o', pathDCMch, '-i', pathDCMch]
                    cmd_complete = subprocess.call(cmd)

                    # print needed time in seconds
                    # nd_time = time.time() - t0
                    # if verbose:
                    #    if decomp:
                    #        print("Needed time: " + str(nd_time) + "s." + " Files had to be decompressed.")
                    #    else:
                    #        print("Needed time: " + str(nd_time) + "s.")

    def plot_patient_slices(self, ch, dim, alpha):
        """Function for plotting patient with label."""
        # get patient data
        dicoms, niis = self.get_slices(count=False)
        niis = np.argmax(niis, axis=-1)
        x, y, z = dicoms.shape[:3]
        # arrays for plotting
        dicoms = (dicoms).astype('float32')
        niis = (niis).astype('float32')

        # channel: fat = 0, water = 1
        chNr = 0
        if ch == 'water':
            chNr = 1

        # plot cuts in each dim
        fig = plt.figure(figsize=(18, 7))
        gs = gridspec.GridSpec(1, 3,
                               width_ratios=[x / y, z / y, x / z],
                               height_ratios=[1])
        ax1 = plt.subplot(gs[0])
        plt.axis('off')
        # plt.xlabel('x-axis')
        # plt.ylabel('y-axis')
        ax1.imshow(np.fliplr(np.rot90(dicoms[:, :, dim[2], chNr], axes=(1, 0))), interpolation='none', cmap='gray')
        ax1.imshow(np.fliplr(np.rot90(colorize(niis)[:, :, dim[2], :], axes=(1, 0))), interpolation='none', alpha=alpha)
        ax2 = plt.subplot(gs[1])
        plt.axis('off')
        # plt.xlabel('z-axis')
        # plt.ylabel('y-axis')
        ax2.imshow(dicoms[dim[0], :, :, chNr], interpolation='none', cmap='gray')
        ax2.imshow(colorize(niis)[dim[0], :, :, :], interpolation='none', alpha=alpha)
        ax3 = plt.subplot(gs[2])
        plt.axis('off')
        # plt.xlabel('x-axis')
        # plt.ylabel('z-axis')
        ax3.imshow(np.fliplr(np.rot90(dicoms[:, dim[1], :, chNr], axes=(1, 0))), interpolation='none', cmap='gray')
        ax3.imshow(np.fliplr(np.rot90(colorize(niis)[:, dim[1], :, :], axes=(1, 0))), interpolation='none', alpha=alpha)

        return fig

    def predict_patient(self, model):
        """Prediction of patient with special model."""
        dicoms, _ = self.get_slices(count=False)
        dishape = dicoms.shape
        outshape = model.get_output_shape_at(-1)[1:]
        inshape = model.internal_input_shapes
        if len(inshape) == 1:
            feedpos = False
        elif len(inshape) > 1:
            feedpos = True
        else:
            raise 'no input'
        inshape = inshape[0][1:]
        prediction_shape = dicoms.shape[:3] + (outshape[-1],)
        prediction = np.zeros(prediction_shape)
        prediction[:, :, :, 0] = np.ones(dicoms.shape[:3])
        deltas = tuple((np.array(inshape[:3]) - np.array(outshape[:3])) // 2)
        input_dict = {}
        for x in range(0, dishape[0] - inshape[0], outshape[0]):
            for y in range(0, dishape[1] - inshape[1], outshape[1]):
                for z in range(0, dishape[2] - inshape[2], outshape[2]):

                    input_dict['input_X'] = np.expand_dims(dicoms[x: x + inshape[0],
                                                           y: y + inshape[1],
                                                           z: z + inshape[2], :],
                                                           axis=0)
                    if feedpos:
                        (size_x, size_y, size_z) = dishape[:3]
                        cropsize_X = inshape[0]
                        # hardcoded, see train-script
                        border = 20
                        max_pos = np.array(
                            [size_x - cropsize_X - border, size_y - cropsize_X - border, size_z - cropsize_X - border])
                        pos = np.array([x, y, z]) / max_pos
                        input_dict['input_position'] = np.expand_dims(pos, axis=0)

                    prediction[x + deltas[0]: x + deltas[0] + outshape[0],
                    y + deltas[1]: y + deltas[1] + outshape[1],
                    z + deltas[2]: z + deltas[2] + outshape[2], :] = model.predict(input_dict)
        return prediction

    def heatmap(self, model, depth, cls=0):
        """Heatmap plot of one class."""
        prediction = self.get_prediction(model)
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(prediction[:, :, depth, cls], axes=(1, 0))), cmap='coolwarm')
        plt.show()
        return fig

    def plot_prediction_vs_ground_truth(self, depth, model):
        """Plot prediction (left) vs ground truth (right)."""
        _, label = self.get_slices(count=False)
        labeled_slice = np.argmax(label[:, :, depth, :], axis=-1)
        # prediction = np.argmax(predict_patient(patient,model)[:,:,depth,:], axis=-1)
        prediction = self.get_prediction(model)
        prediction = np.argmax(prediction[:, :, depth, :], axis=-1)
        fig = plt.figure(dpi=100)
        fig.add_subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(prediction, axes=(1, 0))), interpolation='none', cmap='gray')
        fig.add_subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(labeled_slice, axes=(1, 0))), interpolation='none', cmap='gray')
        return fig

    def plot_prediction_on_patient(self, model, ch, dim, alpha):
        """Plot prediction on patient."""
        # get patient data
        dicoms, niis = self.get_slices(count=False)
        x, y, z = dicoms.shape[:3]
        # get prediction data
        # prediction = patient.get_prediction(model)
        prediction = np.argmax(self.get_prediction(model), axis=-1)

        # arrays for plotting
        dicoms = (dicoms).astype('float32')
        prediction = (prediction).astype('float32')

        # channel: fat = 0, water = 1
        chNr = 0
        if ch == 'water':
            chNr = 1

        # plot cuts in each dim
        fig = plt.figure(figsize=(18, 7))
        gs = gridspec.GridSpec(1, 3,
                               width_ratios=[x / y, z / y, x / z],
                               height_ratios=[1])
        ax1 = plt.subplot(gs[0])
        plt.axis('off')
        # plt.xlabel('x-axis')
        # plt.ylabel('y-axis')
        ax1.imshow(np.fliplr(np.rot90(dicoms[:, :, dim[2], chNr], axes=(1, 0))), interpolation='none', cmap='gray')
        ax1.imshow(np.fliplr(np.rot90(colorize(prediction)[:, :, dim[2], :], axes=(1, 0))), interpolation='none',
                   alpha=alpha)
        ax2 = plt.subplot(gs[1])
        plt.axis('off')
        # plt.xlabel('z-axis')
        # plt.ylabel('y-axis')
        ax2.imshow(dicoms[dim[0], :, :, chNr], interpolation='none', cmap='gray')
        ax2.imshow(colorize(prediction)[dim[0], :, :, :], interpolation='none', alpha=alpha)
        ax3 = plt.subplot(gs[2])
        plt.axis('off')
        # plt.xlabel('x-axis')
        # plt.ylabel('z-axis')
        ax3.imshow(np.fliplr(np.rot90(dicoms[:, dim[1], :, chNr], axes=(1, 0))), interpolation='none', cmap='gray')
        ax3.imshow(np.fliplr(np.rot90(colorize(prediction)[:, dim[1], :, :], axes=(1, 0))), interpolation='none',
                   alpha=alpha)
        return fig



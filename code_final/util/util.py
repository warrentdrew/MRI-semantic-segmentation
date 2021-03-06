import os
import time
import subprocess
import numpy as np
import pickle

import dicom
import nibabel as nib
import nrrd
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import gridspec

# dictionary for channels
dictChName_KORA = {'fat':'F',
                   'water':'W'}
dictChName_NAKO = {'fat':'DIXF',
                   'water':'DIXW'}

def findCh_dir(dataset, channel, lst_dir):
    """Find channel direction."""
    if dataset == 'KORA':
        n = int(channel[-1])
        name = channel[:-1]
        ch_dir = list(filter(lambda dir_name: dir_name[-4]==dictChName_KORA[name], lst_dir))[n]
    elif dataset == 'NAKO':
        ch_dir = [d for d in lst_dir if ((dicom.read_file(d)).ScanOptions == dictChName_NAKO[channel])]
    return ch_dir
 
def del0002(lst_dcm):
    """Delete list's "0002"-entries."""
    return list(filter(lambda dcm_name: len(dcm_name)==16, lst_dcm))

def check_patients(dataset, path_dicoms, path_labels, lst_ch, lst_cl, path_to_gdcm, path_save=None, verbose=False):
    """Save directories to valid patients on hard disk."""
    lst_ch.sort() # sort channels: fat = 0, water = 1
    lst_cl.sort() # sort classes: liver = 1, spleen = 2 (background = 0)

    channels = len(lst_ch)
    classes = len(lst_cl) + 1

    if dataset == 'KORA':
        lst_files = os.listdir(path_dicoms)
        lst_files.sort()
        patients = list(filter(lambda file_name: file_name[:4]=="KORA", lst_files)) # choose only patients

    elif dataset == 'NAKO':
        lst_files = os.listdir(path_labels)
        lst_files.sort()
        patients = [file_name[:4] for file_name in lst_files if file_name[4] == 'L'] # choose patients by existing labels (L for 'Liver')
  
    lstDicomPaths = []
    lstLabelsPath = []

    for patient in patients:
        if dataset == 'KORA':
            PathDicom = os.path.join(path_dicoms, patient, "study")
            PathLabel = os.path.join(path_dicoms, patient, patient+"_ROI")
            # get reference for patient:
            lstChDirNames = os.listdir(PathDicom)
            lstChDirNames.sort()
            PathRef = os.path.join(PathDicom, lstChDirNames[1])
            DirRef = del0002(os.listdir(PathRef))
        elif dataset == 'NAKO':
            PathDicom = os.path.join(os.path.join(path_dicoms, '100' + patient + '30'),
                                     os.listdir(os.path.join(path_dicoms, '100' + patient + '30'))[0])
            PathLabel = os.path.join(path_labels, patient)
            # get reference for patient:
            PathRef = PathDicom
            DirRef = os.listdir(PathRef) 
            # sort dicoms with series (0-9) and slice number (0000-9999) in ascending order
            DirRef.sort(key=lambda x: int(x[-5:])) 

        brk = False
        try: 
            PathRef = os.path.join(PathRef, DirRef[0])
        except IndexError:
            if verbose:
                print("Invalid reference, skipping patient " + patient + ".")
            brk = True
        if brk: continue
    
        RefDs = dicom.read_file(PathRef)
        # load dimensions:
        if dataset == 'KORA': 
            ConstPixelDims = (int(RefDs.Rows),
                              int(RefDs.Columns),
                              len(DirRef),
                              channels)
        elif dataset == 'NAKO': 
            ConstPixelDims = (int(RefDs.Columns),
                              int(RefDs.Rows),
                              len(DirRef)//4, # due to the fact that all channels are in one folder
                              channels)
        
        # load spacing values (in mm):
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), 
                             float(RefDs.PixelSpacing[1]),
                             float(RefDs.SliceThickness))
   
        ArrayDicom = np.zeros(ConstPixelDims, dtype='float16')
        DicomPathsList = [0]*channels
        if dataset == 'NAKO': lst_dirs = list(map(lambda file_name: os.path.join(PathDicom, file_name), DirRef))
        # walk through channels:
        for ch_index, ch in enumerate(lst_ch):
            if dataset == 'KORA':
                try: ch_dir = findCh_dir(dataset='KORA', channel=ch, lst_dir=lstChDirNames)
                except IndexError:
                    if verbose:
                        print('Channel does not exits, skipping patient ' + patient + '.')
                    brk = True
                    break 
                ch_path = os.path.join(PathDicom, ch_dir)
                DicomPathsList[ch_index] = ch_path
                lstFilesDCM = del0002(os.listdir(ch_path))
                lstFilesDCM = list(set(lstFilesDCM)) # avoid double entries
                lstFilesDCM.sort() # sort in ascending order
                lstPathsDCM = list(map(lambda file_name: os.path.join(ch_path, file_name), lstFilesDCM))
                if verbose: print('Loading ' + ch_path + '...')
            elif dataset == 'NAKO': 
                lstPathsDCM = findCh_dirs(dataset='NAKO', channel=ch, lst_dir=lst_dirs) # search for right channel
                DicomPathsList[ch_index] = lstPathsDCM
                if verbose: print('Loading ' + ch + ' channel ' + 'of patient' + patient[:-1] + '...')
            # loop through all the DICOM files
            for z, pathDCMch in enumerate(lstPathsDCM):
                ds = dicom.read_file(pathDCMch) # read the files
                # store the raw image data
                brk = False
                try:
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T
                except NotImplementedError:
                    # decompress dicom file
                    decomp_filename = '/tmp/curr_patient_decomp.dcm'
                    cmd = [path_to_gdcm, '--raw', '--quiet', '-o', decomp_filename, '-i', pathDCMch]                    
                    cmd_complete = subprocess.call(cmd)
                    ds = dicom.read_file(decomp_filename)
                    try:
                        ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T
                    except Exception:
                        if verbose:
                            print("DCM corrupted, skipping patient" + patient + ".")
                        brk = True
                        break
                except ValueError:
                        if verbose:
                            print("Invalid array shapes, skipping patient " + patient + ".")
                        brk = True
                        break
            if brk: break
                
        if brk: continue

        if dataset == 'KORA': 
            try:
                lbl_switch = (os.listdir(PathLabel)[-1][:6] == 'label_')
            except OSError:
                    PathLabel = os.path.join(path_dicoms, patient, patient + "_ROi")
                    try: 
                        lbl_switch = (os.listdir(PathLabel)[-1][:6] == 'label_')
                    except Exception: 
                            if verbose:
                                print("Invalid path to labels,skipping patient " + patient + '.')
                            brk= True
            except IndexError:
                    if verbose:
                        print("No label, skipping patient " + patient + '.')
                    brk = True
            if brk: continue
            ArrayLabel = np.zeros(ConstPixelDims[:3], dtype='uint8')

        elif dataset == 'NAKO': 
            ArrayDicom = np.swapaxes(ArrayDicom, axis1=1, axis2=2)
            ArrayLabel = np.swapaxes(np.zeros(ConstPixelDims[:3], dtype='uint8'), axis1=1, axis2=2)
        LabelPathsList = [0]*(classes-1)
        # walk through classes/labels:
        for cl_index, cl in enumerate(lst_cl):
            if dataset == 'KORA':
                label_path = os.path.join(PathLabel, lbl_switch * 'label_' + cl + '2d.nii')
            elif dataset == 'NAKO':
                label_path = PathLabel + cl.capitalize() + '.nrrd'

            brk = False            
            try:
                if dataset == 'KORA': label = nib.load(label_path)
                elif dataset == 'NAKO': 
                    label, _ = nrrd.read(label_path)
                    label = np.swapaxes(np.flip(label[:,:,:,0], axis=2), axis1=1, axis2=2)
            except Exception:
                if verbose:
                    print('No label for class ' + cl + ', skipping patient ' + patient + '.'.format(cl))
                brk = True
                break            
            try:
                if dataset == 'KORA': label = np.array(label.dataobj)
                ArrayLabel += (cl_index + 1) * label
            except ValueError:
                if verbose:
                    print("Invalid label-dimension, skipping patient " + patient + '.')
                brk = True
                break
            LabelPathsList[cl_index] = label_path
    
        if ArrayLabel.max() >= classes:
            if verbose: print("Ambiguous labeling, skipping patient " + patient + '.')
            brk = True

        if brk: continue

        # append results:
        lstDicomPaths.append(DicomPathsList)
        lstLabelsPath.append(LabelPathsList)
    
    # save paths on hard disk with pickle
    if path_save is None:
        with open(os.path.join(path_dicoms,'patients.pkl'), 'wb') as dicoms:
            pickle.dump(lstDicomPaths, dicoms)
        with open(os.path.join(path_dicoms,'labels.pkl'), 'wb') as labels:
            pickle.dump(lstLabelsPath, labels)
    else: 
        with open(os.path.join(path_save,'patients.pkl'), 'wb') as dicoms:
            pickle.dump(lstDicomPaths, dicoms)
        with open(os.path.join(path_save,'labels.pkl'), 'wb') as labels:
            pickle.dump(lstLabelsPath, labels)
    
    amount_patients = len(lstDicomPaths)
    amount_labels = len(lstLabelsPath)
    if verbose:
        print('Loaded {} patients and {} labels and saved on hard disk.'.format(amount_patients, amount_labels))


def colorize(prediction, colors={0 : np.array([0,0,0]),
                                 1 : np.array([1,0,0.2]),
                                 2 : np.array([0,1,0.2])}):
    """Colorize for patient-plots."""
    pred_picture = np.zeros(shape= prediction.shape + (3,))
    for x , row in enumerate(prediction):
        for y, col in enumerate(row):
            for z, prd in enumerate(col):
                pred_picture[x,y,z,:] = colors[prd]
    return pred_picture
        

class Patient():
    """Creates patient."""
    def __init__(self, dicom_dirs, label_dirs, forget_slices=False): 
        """"Constructor."""
        self.dicom_dirs = dicom_dirs
        self.label_dirs = label_dirs
        self.slices = None
        self.shape = None
        self.prediction = None
        self.forget_slices = forget_slices
        self.slice_counter = 0

    def get_slices(self, count=True, verbose=False):
        """Take loaded slices if already available or load slices if not."""
        if count:        
            self.slice_counter += 1
        if self.slices is None:
            self.slices = self.load_slices(verbose=verbose)
        return self.slices

    def drop(self):
        """Drop slices for PatientBuffer in training."""
        if self.slice_counter != 0:
            self.slice_counter -= 1
            if self.slice_counter == 0 and self.forget_slices:
                self.slices = None
    
    def get_prediction(self, model, batch_size):
        """"Take prediction if already available or predict if not."""
        if self.prediction is None:
            self.prediction = self.predict_patient(model, batch_size)
        return self.prediction

    def load_slices(self, path_to_gdcm='gdcmconv', verbose=False):
        """Load patients data (dicom- and label-files) in numpy arrays."""
        if len(np.shape(self.dicom_dirs)) == 1:
           dataset = 'KORA'
        elif len(np.shape(self.dicom_dirs)) == 2:
            dataset = 'NAKO'

        channels = len(self.dicom_dirs)
        classes = len(self.label_dirs)+1

        onehot_enc = lambda arr: np.eye(classes)[arr]

        if dataset == 'KORA':
            DirRef = del0002(os.listdir(self.dicom_dirs[0]))
            PathRef = os.path.join(self.dicom_dirs[0], DirRef[0])
            RefDs = dicom.read_file(PathRef) # get reference for patient
            ConstPixelDims = (int(RefDs.Rows), # Load dimensions
                              int(RefDs.Columns),
                              len(DirRef),
                              channels)
        elif dataset == 'NAKO':
            DirRef = self.dicom_dirs[0]
            #DirRef.sort(key=lambda x: int(x[-5:]))
            RefDs = dicom.read_file(DirRef[0]) # get reference for patient
            ConstPixelDims = (int(RefDs.Columns), # Load dimensions
                              int(RefDs.Rows),
                              len(DirRef),
                              channels)
        
        # Load spacing values (in mm):
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), 
                             float(RefDs.PixelSpacing[1]),
                             float(RefDs.SliceThickness))
        
        ArrayDicom = np.zeros(ConstPixelDims, dtype='float16')

        # walk through channels:
        for ch_index, ch_path in enumerate(self.dicom_dirs):
            if dataset == 'KORA': 
                lstFilesDCM = del0002(os.listdir(ch_path))  # avoid double entries
                lstFilesDCM = list(set(lstFilesDCM)) # sort in ascending order
                lstFilesDCM.sort()
                lstPathsDCM = list(map(lambda file_name: os.path.join(ch_path, file_name), lstFilesDCM))
                if verbose: print(" Loading " + ch_path + "...")
            elif dataset == 'NAKO': 
                lstPathsDCM = ch_path
                lstPathsDCM.sort(key=lambda x: int(x[-5:]))
                if verbose: print(' Loading channel ' + ch_index + ' of ' + ch_path[0][0:55] + '...')   
            # loop through all the DICOM files
            for z, pathDCMch in enumerate(lstPathsDCM):
                # read the files
                ds = dicom.read_file(pathDCMch)
                # store the raw image data
                try:
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T
                except NotImplementedError:
                    # decompress dicom file
                    decomp_filename = '/tmp/curr_patient_decomp.dcm'
                    cmd = [path_to_gdcm, '--raw', '--quiet', '-o', decomp_filename, '-i', pathDCMch]
                    cmd_complete = subprocess.call(cmd)
                    ds = dicom.read_file(decomp_filename)
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T

        if dataset == 'KORA': ArrayLabel = np.zeros(ConstPixelDims[:3], dtype='uint8')
        elif dataset == 'NAKO': 
            ArrayDicom = np.swapaxes(ArrayDicom, axis1=1, axis2=2)
            ArrayLabel = np.swapaxes(np.zeros(ConstPixelDims[:3], dtype='uint8'), axis1=1, axis2=2)
        LabelPathsList = [0]*(classes-1)    
    
        # walk through classes/labels:
        for cl_index, label_path in enumerate(self.label_dirs):
            if dataset == 'KORA': 
                label = nib.load(label_path)
                label = np.array(label.dataobj)
            elif dataset == 'NAKO':
                label, _ = nrrd.read(label_path)
                label = np.swapaxes(np.flip(label[:,:,:,0], axis=2), axis1=1, axis2=2)
            ArrayLabel += (cl_index + 1) * label
        
        # onehot encoding:
        ArrayLabel = onehot_enc(ArrayLabel)

        # normalize ArrayDicom:
        mx = [np.max(ArrayDicom[...,ch]) for ch in range(channels)]
        ArrayDicom = np.divide(ArrayDicom, mx)

        return (ArrayDicom, ArrayLabel)

    def save_decomp_slices(self,  path_to_gdcm, verbose=False):
        """Load patients data (dicom- and label-files) in numpy arrays."""
        if len(np.shape(self.dicom_dirs)) == 1:
           dataset = 'KORA'
        elif len(np.shape(self.dicom_dirs)) == 2:
            dataset = 'NAKO'

        channels = len(self.dicom_dirs)
        classes = len(self.label_dirs)+1

        onehot_enc = lambda arr: np.eye(classes)[arr]

        if dataset == 'KORA':
            DirRef = del0002(os.listdir(self.dicom_dirs[0]))
            PathRef = os.path.join(self.dicom_dirs[0], DirRef[0])
            RefDs = dicom.read_file(PathRef) # get reference for patient
            ConstPixelDims = (int(RefDs.Rows), # Load dimensions
                              int(RefDs.Columns),
                              len(DirRef),
                              channels)
        elif dataset == 'NAKO':
            DirRef = self.dicom_dirs[0]
            RefDs = dicom.read_file(DirRef[0]) # get reference for patient
            ConstPixelDims = (int(RefDs.Columns), # Load dimensions
                              int(RefDs.Rows),
                              len(DirRef),
                              channels)
        
        # Load spacing values (in mm):
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), 
                             float(RefDs.PixelSpacing[1]),
                             float(RefDs.SliceThickness))
        
        ArrayDicom = np.zeros(ConstPixelDims, dtype='float16')

        # walk through channels:
        for ch_index, ch_path in enumerate(self.dicom_dirs):
            if dataset == 'KORA': 
                lstFilesDCM = del0002(os.listdir(ch_path))  # avoid double entries
                lstFilesDCM = list(set(lstFilesDCM)) # sort in ascending order
                lstFilesDCM.sort()
                lstPathsDCM = list(map(lambda file_name: os.path.join(ch_path, file_name), lstFilesDCM))
                if verbose: print(" Loading " + ch_path + "...")
            elif dataset == 'NAKO': 
                lstPathsDCM = ch_path
                lstPathsDCM.sort(key=lambda x: int(x[-5:]))
                if verbose: print(' Loading channel ' + ch_index + ' of ' + ch_path[0][:55] + '...')    
            # loop through all the DICOM files
            for z, pathDCMch in enumerate(lstPathsDCM):
                # read the files
                ds = dicom.read_file(pathDCMch)
                # store the raw image data
                try:
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T
                except NotImplementedError:
                    # decompress dicom file
                    cmd = [path_to_gdcm, '--raw', '--quiet', '-o', pathDCMch, '-i', pathDCMch]
                    cmd_complete = subprocess.call(cmd)

    def plot_patient_slices(self, ch, dim, alpha):
        """Function for plotting patient with label."""
        # get patient data
        dicoms, labels = self.get_slices(count=False)
        labels = np.argmax(labels, axis=-1)
        x, y, z = dicoms.shape[:3]
        # arrays for plotting
        dicoms = (dicoms).astype('float32')
        labels = (labels).astype('float32')
        
        # channel: fat = 0, water = 1
        chNr = 0
        if ch == 'water':
            chNr = 1

        # plot cuts in each dim
        fig = plt.figure(figsize=(18,7))
        gs = gridspec.GridSpec(1,3,
                               width_ratios=[x/y, z/y, x/z],
                               height_ratios=[1])
        ax1 = plt.subplot(gs[0])
        plt.axis('off')
        #plt.xlabel('x-axis')
        #plt.ylabel('y-axis')
        ax1.imshow(np.fliplr(np.rot90(dicoms[:,:,dim[2],chNr], axes=(1,0))), interpolation='none', cmap='gray')
        ax1.imshow(np.fliplr(np.rot90(colorize(labels)[:,:,dim[2],:], axes=(1,0))), interpolation='none', alpha=alpha)
        ax2 = plt.subplot(gs[1])
        plt.axis('off')
        #plt.xlabel('z-axis')
        #plt.ylabel('y-axis')
        ax2.imshow(dicoms[dim[0],:,:,chNr], interpolation='none', cmap='gray')
        ax2.imshow(colorize(labels)[dim[0],:,:,:], interpolation='none', alpha=alpha)
        ax3 = plt.subplot(gs[2])
        plt.axis('off')
        #plt.xlabel('x-axis')
        #plt.ylabel('z-axis')
        ax3.imshow(np.fliplr(np.rot90(dicoms[:,dim[1],:,chNr], axes=(1,0))), interpolation='none', cmap='gray')
        ax3.imshow(np.fliplr(np.rot90(colorize(labels)[:,dim[1],:,:], axes=(1,0))), interpolation='none', alpha=alpha)

        return fig

    def plot_patient_coronal(self, ch, dim_z, alpha):
        """Function for plotting patient with label."""
        # get patient data
        dicoms, labels = self.get_slices(count=False)
        labels = np.argmax(labels, axis=-1)
        x, y, z = dicoms.shape[:3]
        # arrays for plotting
        dicoms = (dicoms).astype('float32')
        labels = (labels).astype('float32')
        
        # channel: fat = 0, water = 1
        chNr = 0
        if ch == 'water':
            chNr = 1
        
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(dicoms[:,:,dim_z,chNr], axes=(1,0))), interpolation='none', cmap='gray')
        plt.imshow(np.fliplr(np.rot90(colorize(labels)[:,:,dim_z,:], axes=(1,0))), interpolation='none', alpha=alpha)

        return fig

    def predict_patient(self, model, batch_size):
        """Prediction of patient with special model."""
        dicoms, _ = self.get_slices(count=False)
        dishape = dicoms.shape
        outshape = model.get_output_shape_at(-1)[1:]
        inshape = model.internal_input_shapes
        if len(inshape)==1:
            feedpos = False
        elif len(inshape)>1:
            feedpos = True
        else:
            raise 'no input'
        inshape = inshape[0][1:]
        dim = len(inshape) - 1
        prediction_shape = dicoms.shape[:3] + (outshape[-1],)
        prediction = np.zeros(prediction_shape)
        prediction[:,:,:,0] = np.ones(dicoms.shape[:3]) # set all voxels in ch 0 as background
        deltas = tuple((np.array(inshape[:3]) - np.array(outshape[:3])) // 2)
        input_dict = {}

        if dim == 2: 
            inshape[2] = 0
            outshape[2] = 1
        
        xs = range(0, dishape[0] - inshape[0], outshape[0])
        ys = range(0, dishape[1] - inshape[1], outshape[1])
        zs = range(0, dishape[2] - inshape[2], outshape[2])

        amount_crops = len(xs)*len(ys)*len(zs)
        input_X = np.zeros(shape=(amount_crops,inshape[0],inshape[1],inshape[2],dishape[-1]))
        input_P = np.zeros(shape=(amount_crops,3))
        coordinates = np.zeros(shape=(amount_crops,3), dtype='int')

        index = 0
        for x in xs:
            for y in ys:
                for z in zs:
                    if dim == 2:
                        i_X = np.expand_dims(dicoms[x : x + inshape[0],
                                                    y : y + inshape[1],
                                                    z,:],
                                                    axis=0)
                    if dim == 3:
                        i_X = np.expand_dims(dicoms[x : x + inshape[0],
                                             y : y + inshape[1],
                                             z : z + inshape[2],:],
                                             axis=0)
                    if feedpos:
                        (size_x, size_y, size_z) = dishape[:3]
                        cropsize_X = inshape[0]
                        border = 20 # hardcoded, see train-script
                        max_pos = np.array([size_x-cropsize_X-border, 
                                            size_y-cropsize_X-border, 
                                            size_z-cropsize_X-border])
                        pos = np.array([x,y,z]) / max_pos
                        i_P = np.expand_dims(pos, axis=0)
                        input_P[index,:] = i_P
     
                    input_X[index,:,:,:] = i_X
                    coordinates[index,:] = [x,y,z]
                    index += 1

        input_dict['input_X'] = input_X
        if feedpos: input_dict['input_position'] = input_P
                    
        # generate batches for parallel computation of prediction
        preds = model.predict(input_dict, batch_size)
            
        for i, pred in enumerate(preds):
            if dim == 2:
                prediction[coordinates[i][0] + deltas[0] : coordinates[i][0] + deltas[0] + outshape[0],
                           coordinates[i][1] + deltas[1] : coordinates[i][1] + deltas[1] + outshape[1],
                           coordinates[i][2],:] = pred
            elif dim == 3:
                prediction[coordinates[i][0] + deltas[0] : coordinates[i][0] + deltas[0] + outshape[0],
                           coordinates[i][1] + deltas[1] : coordinates[i][1] + deltas[1] + outshape[1],
                           coordinates[i][2] + deltas[2] : coordinates[i][2] + deltas[2] + outshape[2],:] = pred

        return prediction

    def heatmap(self, model, depth, batch_size, cls=0):
        """Heatmap plot of one class."""
        prediction = self.get_prediction(model, batch_size)
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(prediction[:,:,depth,cls], axes=(1,0))), cmap='coolwarm')
        plt.show()
        return fig

    def plot_prediction_vs_ground_truth(self, depth, model, batch_size):
        """Plot prediction (left) vs ground truth (right)."""
        _, label  = self.get_slices(count=False)
        labeled_slice = np.argmax(label[:,:,depth,:], axis=-1)
        prediction = self.get_prediction(model, batch_size)
        prediction =  np.argmax(prediction[:,:,depth,:], axis=-1)
        fig = plt.figure(dpi=100)
        fig.add_subplot(1,2,1)
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(prediction, axes=(1,0))), interpolation='none', cmap='gray')
        fig.add_subplot(1,2,2)
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(labeled_slice, axes=(1,0))), interpolation='none', cmap='gray')
        return fig

    def plot_prediction_on_patient(self, model, batch_size, ch, dim, alpha):
        """Plot prediction on patient."""
        # get patient data
        dicoms, _ = self.get_slices(count=False)
        x, y, z = dicoms.shape[:3]
        # get prediction data
        prediction = np.argmax(self.get_prediction(model, batch_size), axis=-1)
           
        # arrays for plotting
        dicoms = (dicoms).astype('float32')
        prediction = (prediction).astype('float32')

        # channel: fat = 0, water = 1
        chNr = 0
        if ch == 'water':
            chNr = 1

        # plot cuts in each dim
        fig = plt.figure(figsize=(18,7))
        gs = gridspec.GridSpec(1,3,
                               width_ratios=[x/y, z/y, x/z],
                               height_ratios=[1])
        ax1 = plt.subplot(gs[0])
        plt.axis('off')
        #plt.xlabel('x-axis')
        #plt.ylabel('y-axis')
        ax1.imshow(np.fliplr(np.rot90(dicoms[:,:,dim[2],chNr], axes=(1,0))), interpolation='none', cmap='gray')
        ax1.imshow(np.fliplr(np.rot90(colorize(prediction)[:,:,dim[2],:], axes=(1,0))), interpolation='none', alpha=alpha)
        ax2 = plt.subplot(gs[1])
        plt.axis('off')
        #plt.xlabel('z-axis')
        #plt.ylabel('y-axis')
        ax2.imshow(dicoms[dim[0],:,:,chNr], interpolation='none', cmap='gray')
        ax2.imshow(colorize(prediction)[dim[0],:,:,:], interpolation='none', alpha=alpha)
        ax3 = plt.subplot(gs[2])
        plt.axis('off')
        #plt.xlabel('x-axis')
        #plt.ylabel('z-axis')
        ax3.imshow(np.fliplr(np.rot90(dicoms[:,dim[1],:,chNr], axes=(1,0))), interpolation='none', cmap='gray')
        ax3.imshow(np.fliplr(np.rot90(colorize(prediction)[:,dim[1],:,:], axes=(1,0))), interpolation='none', alpha=alpha)
        return fig

    def plot_prediction_on_patient_coronal(self, model, batch_size, ch, dim_z, alpha):
        """Plot prediction on patient."""
        # get patient data
        dicoms, _ = self.get_slices(count=False)
        x, y, z = dicoms.shape[:3]
        # get prediction data
        prediction = np.argmax(self.get_prediction(model, batch_size), axis=-1)
           
        # arrays for plotting
        dicoms = (dicoms).astype('float32')
        prediction = (prediction).astype('float32')

        # channel: fat = 0, water = 1
        chNr = 0
        if ch == 'water':
            chNr = 1

        fig = plt.figure()
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(dicoms[:,:,dim_z,chNr], axes=(1,0))), interpolation='none', cmap='gray')
        plt.imshow(np.fliplr(np.rot90(colorize(prediction)[:,:,dim_z,:], axes=(1,0))), interpolation='none', alpha=alpha)
        
        return fig

def split(k, i, lst, perc):
    idxs = set(range(k)) - {i}
    ys = [lst[j] for j in idxs]
    result = []
    for y in ys:
        result += y[0:int(len(y)*perc)]
    return (lst[i], result)    
           
def load_correct_patients(path, 
                          patients_to_take, 
                          forget_slices=False, 
                          cut=None, 
                          k=None,
                          perc=0, 
                          iteration=None,
                          last_val_patients=None,
                          verbose=False):
    """Load valid patient-paths from hard disk and generate patient-objects."""
    with open(os.path.join(path,'patients.pkl'), 'rb') as patients:
        DicomPathsList = pickle.load(patients)
    with open(os.path.join(path,'labels.pkl'), 'rb') as labels:
        LabelPathsList = pickle.load(labels)

    # take patients for testing
    test_paths_dcm = DicomPathsList[:patients_to_take]
    test_paths_label = LabelPathsList[:patients_to_take]

    DicomPathsList = DicomPathsList[patients_to_take:]
    LabelPathsList = LabelPathsList[patients_to_take:]
    
    patients_num = len(DicomPathsList)
    if k is not None:
        # k-fold cross-validation
        steps = int(patients_num/k)
        ks_dicom = [DicomPathsList[(i*steps):((i*steps)+steps)] for i in range(k)]
        ks_label   = [LabelPathsList[(i*steps):((i*steps)+steps)] for i in range(k)]
        val_paths_dcm, train_paths_dcm = split(k=k, i=iteration, lst=ks_dicom, perc=perc)
        val_paths_label, train_paths_label = split(k=k, i=iteration, lst=ks_label, perc=perc)
        
    else:
        # cut into validation- and training-data
        cut = int(cut*patients_num)
        val_paths_dcm = DicomPathsList[:cut]
        val_paths_label = LabelPathsList[:cut]
        train_paths_dcm = DicomPathsList[cut:]
        train_paths_label = LabelPathsList[cut:]

    # drop last validation data
    if last_val_patients is not None:
        for patient in last_val_patients:
            #print(' drop validation patients')
            patient.drop()

    # generate patient-objects
    patients_test  = []
    patients_val   = []
    patients_train = []
   
    for test_path_dcm, test_path_label in zip(test_paths_dcm, test_paths_label):
        patient = Patient(test_path_dcm, test_path_label, forget_slices=forget_slices)
        patients_test.append(patient)
    for val_path_dcm, val_path_label in zip(val_paths_dcm, val_paths_label):
        patient = Patient(val_path_dcm, val_path_label, forget_slices=forget_slices)
        patients_val.append(patient)
    for train_path_dcm, train_path_label in zip(train_paths_dcm, train_paths_label):
        patient = Patient(train_path_dcm, train_path_label, forget_slices=forget_slices)
        patients_train.append(patient)
        
    # get slices for validation data
    patients_val_slices = []
    for patient in patients_val:
        slices = patient.get_slices(verbose=verbose)
        patients_val_slices.append(slices)

    return (patients_test, patients_train, patients_val, patients_val_slices)


class ShiftRegister():
    """Creates shift register for "first in first out queue" purpose."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.content = capacity * [0]
        self.pointer = 0
    
    def shift(self, new_data):
        self.content[self.pointer] = new_data
        self.pointer = (self.pointer + 1) % self.capacity
    
    def read(self):
        return self.content

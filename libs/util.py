import os
import time
import subprocess
import numpy as np
import pickle

import dicom
import nibabel as nib
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import gridspec

# dictionary for channels
dictChName = {'fat':'F',
              'water':'W'}

def findCh_dir(channel, lst_dir):               #lst_dir is lstChDirNames: KoRA/KORA123456/study/1_vibe_dixon_F5      list
    """Find channel direction."""
    n = int(channel[-1])   # n = 0 or 1
    name = channel[:-1]     # name = fat or water
    ch_dir = list(filter(lambda dir_name: dir_name[-4]==dictChName[name], lst_dir))[n]          #filter out the names of files in lst_dir whose 4th last num is "F" or "W"
    return ch_dir                                       #ch_dir is KoRA/KORA123456/study/1_vibe_dixon_F56
 
def del0002(lst_dcm):
    """Delete list's "0002"-entries."""
    return list(filter(lambda dcm_name: len(dcm_name)==16, lst_dcm))

def check_patients(path, lst_ch, lst_cl, path_save=None, verbose=False):
    """Save directories to valid patients on hard disk."""
    # sort channels: fat = 0, water = 1
    lst_ch.sort()
    # sort classes: liver = 1, spleen = 2 (background = 0)
    lst_cl.sort()

    channels = len(lst_ch)
    classes = len(lst_cl) + 1

    lst_files = os.listdir(path)
    lst_files.sort()
    # choose only patients
    patients = list(filter(lambda file_name: file_name[:4]=="KORA", lst_files))

    lstDicomPaths = []
    lstNiiPaths = []

    for patient in patients:
        PathDicom = os.path.join(path, patient, "study")
        PathNii = os.path.join(path, patient, patient+"_ROI")

        # get reference for patient:
        lstChDirNames = os.listdir(PathDicom)
        lstChDirNames.sort()
        PathRef = os.path.join(PathDicom,lstChDirNames[1])      #lstChDirNames: KoRA/KORA123456/study/1_vibe_dixon_F5     list
        DirRef = del0002(os.listdir(PathRef))                      #PATHREF: KORA/KORA1223454/STUDY/T1_vibe_dixon_F56/

        brk = False
        try: 
            PathRef = os.path.join(PathRef,DirRef[0])           #pathref = KoRA/KORA123456/study/T1_vibe_dixon_F56/
        except IndexError:
            if verbose:
                print("Invalid reference, skipping patient " + patient + ".")
            brk = True
        if brk: continue
    
        RefDs = dicom.read_file(PathRef)   #this is only read out for loading dims
        # load dimensions:
        ConstPixelDims = (int(RefDs.Rows),
                          int(RefDs.Columns),
                          len(DirRef),
                          channels)
        # load spacing values (in mm):
        ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), 
                             float(RefDs.PixelSpacing[1]),
                             float(RefDs.SliceThickness))
   
        ArrayDicom = np.zeros(ConstPixelDims, dtype='float16')
        DicomPathsList = [0]*channels           #channels = 3
        # walk through channels:
        for ch_index,ch in enumerate(lst_ch):
            try:
                ch_dir = findCh_dir(ch, lstChDirNames)          #ch_dir is 1_vibe_dixon_F56
            except IndexError:
                if verbose:
                    print("Channel does not exits, skipping patient " + patient + ".")
                brk = True
                break 
            ch_path = os.path.join(PathDicom,ch_dir)            #pathDicom = KoRA/KORA123456/study/
            DicomPathsList[ch_index] = ch_path                  #ch_path = KoRA/KORA123456/study/1_vibe_dixon_F_56       DicomPathsList[0] = KoRA/KORA123456/study/1_vibe_dixon_F56
                                                                # DicomPathsList[1] = KoRA/KORA123456/study/1_vibe_dixon_W_56
            lstFilesDCM = del0002(os.listdir(ch_path))
            # avoid double entries:
            lstFilesDCM = list(set(lstFilesDCM))
            # sort in ascending order:
            lstFilesDCM.sort()                                  #lstFilesDCM = KoRA/KORA123456/study/1_vibe_dixon_F56/asb.dcm    list without same name and sorted
            lstPathsDCM = list(map(lambda file_name: os.path.join(ch_path, file_name), lstFilesDCM))     # for each file in lstFilesDCM, join path KoRA/KORA123456/study/1_vibe_dixon_F56/asb.dcm   list
            if verbose: print("Loading " + ch_path + "...")
            # loop through all the DICOM files
            for z, pathDCMch in enumerate(lstPathsDCM):         #this read out all the files in the lstFilesDCM = KoRA/KORA123456/study/1_vibe_dixon_F56
                # read the files
                ds = dicom.read_file(pathDCMch)
                # store the raw image data
                brk = False
                try:
                    ArrayDicom[:, :, z, ch_index] = ds.pixel_array.T    # add each file as a slice in the 3d image, (in each channel)
                except NotImplementedError:
                    # decompress dicom file
                    decomp_filename = '/tmp/curr_patient_decomp.dcm'
                    cmd = ['/opt/gdcm-2.8.3/gdcmbin/bin/gdcmconv', '--raw', '--quiet', '-o', decomp_filename, '-i', pathDCMch]
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

        try:
            lbl_switch = (os.listdir(PathNii)[-1][:6] == 'label_')      #PathNii = KoRA/KORA123456/study/xxx_ROI/....  find the last one and see if it start with label_
        except OSError:
                PathNii = os.path.join(path, patient, patient + "_ROi")
                try: 
                    lbl_switch = (os.listdir(PathNii)[-1][:6] == 'label_')
                except Exception: 
                        if verbose:
                            print("Invalid path to labels,skipping patient " + patient + '.')
                        brk= True
        except IndexError:
                if verbose:
                    print("No label, skipping patient " + patient + '.')
                brk = True
        if brk: continue
        ArrayNii = np.zeros(ConstPixelDims[:3], dtype='uint8')
        NiiPathsList = [0]*(classes-1)
        # walk through classes/labels:
        for cl_index, cl in enumerate(lst_cl):
            label_path = os.path.join(PathNii, lbl_switch * 'label_' + cl + '2d.nii')       #lbl_switch is a boolean boolean * 'label_' means start with label or not
            brk = False            
            try:
                label = nib.load(label_path)            # this need some change because the new label is in h5py
            except Exception:
                if verbose:
                    print("No label, skipping patient " + patient + ".")
                brk = True
                break            
            try:
                ArrayNii += (cl_index + 1) * np.array(label.dataobj)
            except ValueError:
                if verbose:
                    print("Invalid label-dimension, skipping patient " + patient + '.')
                brk = True
                break
            NiiPathsList[cl_index] = label_path                   #NiiPathsList[0] = liver2d.nii NiiPathsList[1] = spleen2d.nii
    
        if ArrayNii.max() >= classes:
            if verbose: print("Ambiguous labeling, skipping patient " + patient + '.')
            brk = True

        if brk: continue

        # append results:
        lstDicomPaths.append(DicomPathsList)
        lstNiiPaths.append(NiiPathsList)
    
    # save paths on hard disk with pickle
    if path_save is None:
        with open(os.path.join(path,'patients.pkl'), 'wb') as dicoms:
            pickle.dump(lstDicomPaths, dicoms)                      #lstDicomPaths is a list of DicomPathsList(contain all channels path of all patients in turn)
        with open(os.path.join(path,'labels.pkl'), 'wb') as niis:
            pickle.dump(lstNiiPaths, niis)
    else: 
        with open(os.path.join(path_save,'patients.pkl'), 'wb') as dicoms:
            pickle.dump(lstDicomPaths, dicoms)
        with open(os.path.join(path_save,'labels.pkl'), 'wb') as niis:
            pickle.dump(lstNiiPaths, niis)
    
    amount_patients = len(lstDicomPaths)
    amount_labels = len(lstNiiPaths)
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
    def __init__(self, dicom_dirs, nii_dirs, forget_slices=False): 
        """"Constructor."""
        self.dicom_dirs = dicom_dirs
        self.nii_dirs = nii_dirs
        self.slices = None
        self.shape = None
        self.prediction = None
        self.forget_slices = forget_slices
        self.slice_counter = 0

    def get_slices(self, count=True, verbose=False):
        """Take loaded slices if already available or load slices if not."""
        if count:        
            self.slice_counter += 1                 #this slices counter actually stands for the num of patients
        if self.slices is None:
            self.slices = self.load_slices(verbose=verbose)
            #print("slice-counter: " + str(self.slice_counter))
            #print("load slices!")
        return self.slices

    def drop(self):
        """Drop slices for PatientBuffer in training."""
        #if self.slices == None:
            #print("Slices are already None!")
        if self.slice_counter != 0:
            self.slice_counter -= 1
            if self.slice_counter == 0 and self.forget_slices:
                self.slices = None
                #print("drop!")
    
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
        classes = len(self.nii_dirs)+1

        # onehot-encoding for validation-data
        onehot_enc = lambda arr: np.eye(classes)[arr]

        # get reference for patient:
        DirRef = del0002(os.listdir(self.dicom_dirs[0]))
        PathRef = os.path.join(self.dicom_dirs[0],DirRef[0])     #dicom_dirs = train_path_dcm = (a path in) train_paths_dcm = DicomPathsList[cut:] = pickle.load('patients.pkl')
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
                    cmd = ['/opt/gdcm-2.8.3/gdcmbin/bin/gdcmconv', '--raw', '--quiet', '-o', decomp_filename, '-i', pathDCMch]
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
        mx = [np.max(ArrayDicom[...,ch]) for ch in range(channels)]
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
        classes = len(self.nii_dirs)+1
        # onehot-encoding for validation-data
        onehot_enc = lambda arr: np.eye(classes)[arr]

        # get reference for patient:
        DirRef = del0002(os.listdir(self.dicom_dirs[0]))
        PathRef = os.path.join(self.dicom_dirs[0],DirRef[0])
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
        #nd_time = time.time() - t0
        #if verbose:
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
        fig = plt.figure(figsize=(18,7))
        gs = gridspec.GridSpec(1,3,
                               width_ratios=[x/y, z/y, x/z],
                               height_ratios=[1])
        ax1 = plt.subplot(gs[0])
        plt.axis('off')
        #plt.xlabel('x-axis')
        #plt.ylabel('y-axis')
        ax1.imshow(np.fliplr(np.rot90(dicoms[:,:,dim[2],chNr], axes=(1,0))), interpolation='none', cmap='gray')
        ax1.imshow(np.fliplr(np.rot90(colorize(niis)[:,:,dim[2],:], axes=(1,0))), interpolation='none', alpha=alpha)
        ax2 = plt.subplot(gs[1])
        plt.axis('off')
        #plt.xlabel('z-axis')
        #plt.ylabel('y-axis')
        ax2.imshow(dicoms[dim[0],:,:,chNr], interpolation='none', cmap='gray')
        ax2.imshow(colorize(niis)[dim[0],:,:,:], interpolation='none', alpha=alpha)
        ax3 = plt.subplot(gs[2])
        plt.axis('off')
        #plt.xlabel('x-axis')
        #plt.ylabel('z-axis')
        ax3.imshow(np.fliplr(np.rot90(dicoms[:,dim[1],:,chNr], axes=(1,0))), interpolation='none', cmap='gray')
        ax3.imshow(np.fliplr(np.rot90(colorize(niis)[:,dim[1],:,:], axes=(1,0))), interpolation='none', alpha=alpha)

        return fig

    def predict_patient(self, model):
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
        prediction_shape = dicoms.shape[:3] + (outshape[-1],)
        prediction = np.zeros(prediction_shape)
        prediction[:,:,:,0] = np.ones(dicoms.shape[:3])
        deltas = tuple((np.array(inshape[:3]) - np.array(outshape[:3])) // 2)
        input_dict = {}
        for x in range(0,dishape[0] - inshape[0], outshape[0]):
            for y in range(0,dishape[1] - inshape[1], outshape[1]):
                for z in range(0,dishape[2] - inshape[2], outshape[2]):
                
                    input_dict['input_X'] = np.expand_dims(dicoms[x : x + inshape[0],
                                                                  y : y + inshape[1],
                                                                  z : z + inshape[2],:],
                                                           axis=0)
                    if feedpos:
                        (size_x, size_y, size_z) = dishape[:3]
                        cropsize_X = inshape[0]
                        # hardcoded, see train-script
                        border = 20
                        max_pos = np.array([size_x-cropsize_X-border, size_y-cropsize_X-border, size_z-cropsize_X-border])
                        pos = np.array([x,y,z]) / max_pos
                        input_dict['input_position'] = np.expand_dims(pos, axis=0)
                        
                    prediction[x + deltas[0] : x + deltas[0] + outshape[0],
                               y + deltas[1] : y + deltas[1] + outshape[1],
                               z + deltas[2] : z + deltas[2] + outshape[2],:] = model.predict(input_dict)
        return prediction

    def heatmap(self, model, depth, cls=0):
        """Heatmap plot of one class."""
        prediction = self.get_prediction(model)
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(prediction[:,:,depth,cls], axes=(1,0))), cmap='coolwarm')
        plt.show()
        return fig

    def plot_prediction_vs_ground_truth(self, depth, model):
        """Plot prediction (left) vs ground truth (right)."""
        _, label  = self.get_slices(count=False)
        labeled_slice = np.argmax(label[:,:,depth,:], axis=-1)
        #prediction = np.argmax(predict_patient(patient,model)[:,:,depth,:], axis=-1)
        prediction = self.get_prediction(model)
        prediction =  np.argmax(prediction[:,:,depth,:], axis=-1)
        fig = plt.figure(dpi=100)
        fig.add_subplot(1,2,1)
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(prediction, axes=(1,0))), interpolation='none', cmap='gray')
        fig.add_subplot(1,2,2)
        plt.axis('off')
        plt.imshow(np.fliplr(np.rot90(labeled_slice, axes=(1,0))), interpolation='none', cmap='gray')
        return fig

    def plot_prediction_on_patient(self, model, ch, dim, alpha):
        """Plot prediction on patient."""
        # get patient data
        dicoms, niis = self.get_slices(count=False)
        x, y, z = dicoms.shape[:3]
        # get prediction data
        #prediction = patient.get_prediction(model)
        prediction = np.argmax(self.get_prediction(model), axis=-1)
           
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

def split(k, i, lst, perc):
    idxs = set(range(k)) - {i}
    ys = [lst[j] for j in idxs]
    result = []
    for y in ys:
        result += y[0:int(len(y)*perc)]
    return (lst[i], result)    
           
def load_correct_patients(path, 
                          patients_to_take, 
                          forget_slices = False, 
                          cut=None, 
                          k=None,
                          perc=0.25, 
                          iteration=None,
                          last_val_patients=None,
                          verbose=False):
    """Load valid patient-paths from hard disk and generate patient-objects."""
    with open(os.path.join(path,'patients.pkl'), 'rb') as patients:
        DicomPathsList = pickle.load(patients)
    with open(os.path.join(path,'labels.pkl'), 'rb') as labels:
        NiiPathsList = pickle.load(labels)

    # take patients for testing
    test_paths_dcm = DicomPathsList[:patients_to_take]
    test_paths_nii = NiiPathsList[:patients_to_take]

    DicomPathsList = DicomPathsList[patients_to_take:]
    NiiPathsList = NiiPathsList[patients_to_take:]
    
    patients_num = len(DicomPathsList)
    if k is not None:
        # k-fold cross-validation
        steps = int(patients_num/k)
        ks_dicom = [DicomPathsList[(i*steps):((i*steps)+steps)] for i in range(k)]
        ks_nii   = [NiiPathsList[(i*steps):((i*steps)+steps)] for i in range(k)]
        val_paths_dcm, train_paths_dcm = split(k=k, i=iteration, lst=ks_dicom, perc=perc)
        val_paths_nii, train_paths_nii = split(k=k, i=iteration, lst=ks_nii, perc=perc)
        
    else:
        # cut into validation- and training-data
        cut = int(cut*patients_num)
        val_paths_dcm = DicomPathsList[:cut]
        val_paths_nii = NiiPathsList[:cut]
        train_paths_dcm = DicomPathsList[cut:]
        train_paths_nii = NiiPathsList[cut:]

    # drop last validation data
    if last_val_patients is not None:
        for patient in last_val_patients:
            #print(' drop validation patients')
            patient.drop()

    # generate patient-objects
    patients_test  = []
    patients_val   = []
    patients_train = []
   
    for test_path_dcm, test_path_nii in zip(test_paths_dcm, test_paths_nii):
        patient = Patient(test_path_dcm, test_path_nii, forget_slices=forget_slices)
        patients_test.append(patient)
    for val_path_dcm, val_path_nii in zip(val_paths_dcm, val_paths_nii):
        patient = Patient(val_path_dcm, val_path_nii, forget_slices=forget_slices)
        patients_val.append(patient)
    for train_path_dcm, train_path_nii in zip(train_paths_dcm, train_paths_nii):
        patient = Patient(train_path_dcm, train_path_nii, forget_slices=forget_slices)          #train_path_dicom = (a path in) train_paths_dcm = DicomPathsList[cut:] = pickle.load('patients.pkl')
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
    
    def shift(self, new_data, verbose=False):
        #if verbose:
           # print(" Fill PatientBuffer. Shifting: ",self.pointer)
        self.content[self.pointer] = new_data
        self.pointer = (self.pointer + 1) % self.capacity
    
    def read(self):
        return self.content





import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
from pandas import DataFrame
from util.util_at import load_data

def get_tablePos(info, SliceLocation, slices):
    tablelist = []
    # get iPos
    iPos = int(info['FrameOfReferenceUID'][0][0][0].split('.')[-1])

    # adjust slicelocation according to location
    if info['SoftwareVersion'][0][0][0].find('syngo') != -1:
        if info['SoftwareVersion'][0][0][0] == 'syngo MR B17':
            tableStart = SliceLocation + 5000
        else:
            if iPos == 0:
                iPos = 5000
            tableStart = iPos + SliceLocation
    else:
        tableStart = SliceLocation

    for i in range(slices):
        tablelist.append(tableStart + i * info['SliceThickness'][0][0][0][0])

    tablePos = np.array(tablelist)
    return tablePos


def get_patient_info(patient):
    old_path = os.path.join(os.path.join('/med_data/Segmentation/AT', patient.get_patient_id()), 'rework.mat')
    print('old_path:', old_path)
    pat_data = load_data(old_path)
    info = pat_data['info']
    SeriesNumber = pat_data['info']['SeriesNumber'][0][0]
    InstanceNumber = pat_data['info']['InstanceNumber'][0][0][0][0]
    SliceLocation = pat_data['info']['SliceLocation'][0][0][0][0]
    PixelSpacing = pat_data['info']['PixelSpacing'][0][0]
    SpacingBetweenSlices = pat_data['info']['SpacingBetweenSlices'][0][0]

    return info, SeriesNumber, InstanceNumber, SliceLocation, PixelSpacing, SpacingBetweenSlices


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def export_result(patient, model, border, add_pos=False, save_dir='/home/d1251/no_backup/d1251/quantitative'):
    '''
    only export result for one patient
    '''
    print("export result ...")
    img, label = patient.get_slices(count=False)  # img shape x,y,z,1
    (h, w, slices, _) = img.shape

    prediction = patient.predict_patient(model, border, add_pos)
    prediction = np.argmax(prediction, axis=-1)  # prediction has shape [1, 256, 192, S, 4]
    prediction = get_one_hot(prediction, 4)
    P_LT = prediction[..., 1]
    P_VAT = prediction[..., 2]
    P_SCAT = prediction[..., 3]
    lt = np.squeeze(np.sum(P_LT, axis=(0, 1)))  # shape [S,]
    lt = np.expand_dims(lt, axis=1)  # shape [S,1]
    vat = np.squeeze(np.sum(P_VAT, axis=(0, 1)))
    vat = np.expand_dims(vat, axis=1)
    scat = np.squeeze(np.sum(P_SCAT, axis=(0, 1)))
    scat = np.expand_dims(scat, axis=1)
    tissues = np.concatenate([np.concatenate([lt + scat, scat], axis=1), vat], axis=1)  # shape (S,3)
    # get img dim, dim consist of [SInfo.PixelSpacing; SInfo.SpacingBetweenSlices]'
    info, SeriesNumber, InstanceNumber, SliceLocation, PixelSpacing, SpacingBetweenSlices = get_patient_info(patient)

    # The info struct is the DICOM header of the first 2D image slice
    # dim should have dimension [3, S]
    dim = np.repeat(np.concatenate([PixelSpacing, SpacingBetweenSlices], axis=0), repeats=slices,
                    axis=1)  # dim has shape [3,], shape [3, S]
    AUnit = np.repeat(np.expand_dims(np.transpose(dim[0, :] ** 2), axis=1), repeats=3, axis=1)  # shape [S,3]
    tissues = (tissues * AUnit / 100).astype(np.int64)

    instance_list = []
    for i in range(slices):
        instance_list.append(InstanceNumber + i)

    instance = np.array(instance_list)
    instance = np.expand_dims(instance, axis=1)
    series = np.repeat(SeriesNumber, repeats=slices, axis=0)
    tablePos = get_tablePos(info, SliceLocation, slices)
    tablePos = np.expand_dims(tablePos, axis=1)
    export = np.concatenate((instance, tissues, tablePos, series), axis=1)
    df = DataFrame(export)
    savepath = os.path.join(save_dir, 'export_{}.xls'.format(patient.get_patient_id().replace('/', '-')))
    df.to_excel(savepath, sheet_name='sheet1', index=False)

    # TODO create joint
    # TODO GUI for retrieving hip, shoulder, wrist, heel
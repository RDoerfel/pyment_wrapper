#%%
import argparse
import os
import pandas as pd
import numpy as np
from shutil import rmtree
from pyment.utils.preprocessing import convert_mgz_to_nii_gz
from pyment.utils.preprocessing import reorient2std
from pyment.utils.preprocessing import flirt
from pyment.utils.preprocessing import crop_mri
from pyment.data import NiftiDataset
from pyment.data import NiftiGenerator
from pyment.models import RegressionSFCN
import csv

#%%
def get_brainmask_from_fs(result_dir,fs_dir,subject):
    if not os.path.isdir(os.path.join(result_dir,'brainmasks')):
        os.makedirs(os.path.join(result_dir,'brainmasks'))

    brainmask = os.path.join(fs_dir, subject, 'mri', 'brainmask.mgz')
    brainmask = os.path.abspath(brainmask)
    if not os.path.isfile(brainmask):
        print(f'Skipping {subject}. Missing brainmask')
        
    target = os.path.join(result_dir, 'brainmasks', f'{subject}.mgz')
        
    # Check if the symlink already exists, and if so overwrite it
    if os.path.exists(target) or os.path.islink(target):
        os.remove(target)
        
    os.symlink(brainmask, target)

def get_niigz_from_mgz(result_dir,subject):
    if not os.path.isdir(os.path.join(result_dir,'nifti')):
        os.makedirs(os.path.join(result_dir,'nifti'))

    src_mgz = os.path.join(result_dir,'brainmasks',f"{subject}.mgz")
    dest_nii_ngz = os.path.join(result_dir,'nifti',f"{subject}.nii.gz")

    if os.path.isfile(dest_nii_ngz):
        print(f'Skipping {subject}: Nifti already exists')
        return

    convert_mgz_to_nii_gz(src_mgz,dest_nii_ngz)

def transform2std(result_dir,subject):
    if not os.path.isdir(os.path.join(result_dir,'reoriented')):
        os.makedirs(os.path.join(result_dir,'reoriented'))

    src_nifti = os.path.join(result_dir,'nifti',f"{subject}.nii.gz")    
    dest_reoriented = os.path.join(result_dir,'reoriented',f"{subject}.nii.gz")

    if os.path.isfile(dest_reoriented):
        print(f'Skipping {subject}: Reoriented already exists')
        return

    reorient2std(src_nifti,dest_reoriented)

def apply_flirt(result_dir,subject):
    mni_template = os.path.join('/','usr','local','fsl','data','linearMNI','MNI152lin_T1_1mm_brain.nii.gz')

    if not os.path.isdir(os.path.join(result_dir,'mni152')):
        os.makedirs(os.path.join(result_dir,'mni152'))

    src_reoriented = os.path.join(result_dir,'reoriented',f"{subject}.nii.gz")
    dest_mni = os.path.join(result_dir,'mni152',f"{subject}.nii.gz")

    if os.path.isfile(dest_mni):
        print(f'Skipping {subject}: MNI already exists')
        return

    flirt(src_reoriented,dest_mni,template=mni_template)

def crop_image(result_dir,subject):
    if not os.path.isdir(os.path.join(result_dir,'preprocessed')):
        os.makedirs(os.path.join(result_dir,'preprocessed'))
        
    bounds = ((6, 173), (2, 214), (0, 160))
    
    src_mni = os.path.join(result_dir,'mni152',f"{subject}.nii.gz")
    dest_preprocessed = os.path.join(result_dir,'preprocessed',f"{subject}.nii.gz")
    if os.path.isfile(dest_preprocessed):
        print(f'Skipping {subject}: Cropped lready exists')
        return

    crop_mri(src_mni, dest_preprocessed, bounds)

def save_labels(result_dir,subject,sex,age):
    if sex == 'Female':
        sex = 'F'
    elif sex == 'Male':
        sex = 'M'	
    dfSubject = pd.DataFrame({'id':[subject],'sex':[sex],'age':[age]})
    dfSubject.to_csv(os.path.join(result_dir, 'preprocessed', f'{subject}_labels.csv'), index=False)

def labels_to_dataset(result_dir,subject):
    dataset = NiftiDataset.from_folder(root=os.path.join(result_dir,'preprocessed'), 
                                    images='',
                                    labels=f'{subject}_labels.csv', 
                                    target='id')
    return dataset

def dataset_to_generator(dataset):
    preprocessor = lambda x: x/255
    batch_size = 1
    return NiftiGenerator(dataset, preprocessor=preprocessor, batch_size=batch_size)

def predict_from_generator(generator):
    model = RegressionSFCN(weights='brain-age')

    generator.reset()
    preds, labels = model.predict(generator, return_labels=True)
    preds = preds.squeeze()
    return preds.item()

def save_results(result_dir,subject,predicted_age):
    header = ['File','predicted_age',]
    data = [subject,predicted_age]
    result_name = f"{subject}_brain_predicted_age.csv"
    result_file = os.path.join(result_dir,result_name)
    with open(result_file, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)

def predict_age(result_dir,subject,sex,age):
    save_labels(result_dir,subject,sex,age)
    dataset = labels_to_dataset(result_dir,subject)  
    generator = dataset_to_generator(dataset)
    predicted_age = predict_from_generator(generator)
    save_results(result_dir,subject,predicted_age)

def preprocess(fs_dir, result_dir, subject):
    get_brainmask_from_fs(result_dir,fs_dir,subject)
    get_niigz_from_mgz(result_dir,subject)
    transform2std(result_dir,subject)
    apply_flirt(result_dir,subject)
    crop_image(result_dir,subject)

def cleanup_1():
    print("Not cleaning up.")

def cleanup_2(result_dir):
    print("Cleaning up temporary files.")   
    rmtree(os.path.join(result_dir, 'brainmasks'))
    rmtree(os.path.join(result_dir, 'nifti'))
    rmtree(os.path.join(result_dir, 'reoriented'))
    rmtree(os.path.join(result_dir, 'mni152'))

def cleanup_3(result_dir):
    print("Cleaning up everything.")   
    cleanup_2(result_dir)
    rmtree(os.path.join(result_dir, 'preprocessed'))

def cleanup_results(result_dir,level):
    if level == 1:
        cleanup_1()
    elif level == 2:
        cleanup_2(result_dir)
    elif level == 3:
        cleanup_3(result_dir)
    else:
        print(f"Level {level} not in options.")

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Estimates brain age for a fs processed subject.'))
    subparsers = parser.add_subparsers(dest='command',
                                       title='Subcommands',
                                       description='Subcommands for preprocessing, prediction, and cleanup.',
                                       help='Usage: \n' 
                                            'pyment_prediction preproc -fs <path> -r <path> -id <subject_id> \n'
                                            'pyment_prediction predict -r <path> -id <subject_id> -a <age> -s <sex>\n'
                                            'pyment_prediction cleanup -r <path> -l <level>')

    preproc = subparsers.add_parser('preproc')
    preproc.add_argument('-fs', '--fs_dir', 
                        required=True,
                        dest='fs_dir',
                        help=('Directory containing fresurfer results.' 
                              'Should be processed with recon-all and'
                              'folder name should be subjid'))
    preproc.add_argument('-r', '--result_dir', 
                        required=True,
                        dest='result_dir',
                        help=('Directory where results should be stored'))
    preproc.add_argument('-id', '--subject_id', 
                        required=True,
                        dest='subject_id', 
                        help=('Subject identifier, same as folder name fo fs processed subjects'))

    predict = subparsers.add_parser('predict')
    predict.add_argument('-r', '--result_dir', 
                        dest="result_dir", 
                        required=True,
                        help=('Directory where results should be stored'))
    predict.add_argument('-id', '--subject_id', 
                        dest="subject_id", 
                        required=True,
                        help=('Subject identifier, same as folder name fo fs processed subjects'))               
    predict.add_argument('-a', '--age', 
                        required=True, 
                        dest="age", 
                        help=('Age of the subject'))
    predict.add_argument('-s', '--sex', 
                        required=True, 
                        dest="sex", 
                        help=('Sex of the subject'))
    
    cleanup = subparsers.add_parser('cleanup')
    cleanup.add_argument('-r', '--result_dir', 
                        dest="result_dir", 
                        required=True,
                        help=('Directory where results should be stored'))
    cleanup.add_argument('-l', '--level',
                        type=int, 
                        dest="level", 
                        required=True,
                        help=('Level for cleanup: 3 = all folders including preprocessed, 2 = excluding preprocessed, 1 = no cleanup'))

    args = parser.parse_args()

    if args.command == 'cleanup':
        print(f"Running {args.command} on level {args.level}")
        level = args.level
        result_dir = args.result_dir    
        cleanup_results(result_dir,level)

    elif args.command == 'preproc':
        fs_dir = args.fs_dir   
        result_dir = args.result_dir
        subject_id = args.subject_id       
        print(f"Running {args.command} for subject {subject_id}")
        preprocess(fs_dir, result_dir, subject_id)

    elif args.command == 'predict':
        result_dir = args.result_dir
        subject_id = args.subject_id
        age = args.age
        sex = args.sex
        print(f"Running {args.command} for subject {subject_id}")
        predict_age(result_dir,subject_id,sex,age)

    else:
        project_dir = os.path.join('/indirect','users','rubendorfel','git','brain-age')
        pyment_dir = os.path.join(project_dir,'3Party','pyment-public')
        data_dir = os.path.join(project_dir,'data')
        subject_dir = os.path.join(data_dir,'CimbiId_10459')
        fs_dir = os.path.join(data_dir,'recon')

        result_dir = os.path.join(project_dir,'results','test')
        subject = 'f475'
        sex = 'female'
        age = 37.28679
        print(f"python pyment_prediction.py cleanup -r {result_dir} -l 1")
        print(f"python pyment_prediction.py preproc -fs {fs_dir} -r {result_dir} -id {subject}")
        print(f"python pyment_prediction.py predict -r {result_dir} -id {subject} -a {age} -s {sex}")


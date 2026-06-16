import os
import zipfile
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.uid import generateUID

# List of DICOM tags that may contain Protected Health Information (PHI)
PHI_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
    "OtherPatientIDs", "OtherPatientNames", "InstitutionName", "ReferringPhysicianName"
]

def check_phi_in_folder(folder_path: str):
    """Scans a folder for potential PHI in DICOM headers."""
    print(f"🔎 Checking folder for PHI: {folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                filepath = os.path.join(root, file)
                try:
                    dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                    print(f"📁 Sample File: {file}")
                    for tag in PHI_TAGS:
                        if hasattr(dcm, tag):
                            value = getattr(dcm, tag)
                            if value not in ["", None]:
                                print(f"⚠️ {tag}: {value}")
                    break
                except Exception as e:
                    print(f"Error: Could not read {file}. {e}")
                    break

def anonymize_dicom_file(in_path: str, out_path: str):
    """Removes PHI and generates new UIDs for a single DICOM file."""
    ds = pydicom.dcmread(in_path)
    ds.PatientName = ""
    ds.PatientID = ""
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.InstitutionName = ""
    ds.ReferringPhysicianName = ""
    
    ds.StudyInstanceUID = generateUID()
    ds.SeriesInstanceUID = generateUID()
    ds.SOPInstanceUID = generateUID()
    ds.remove_private_tags()
    ds.save_as(out_path)

def anonymize_folder(src_folder: str, dst_folder: str):
    """Iterates through a folder and anonymizes all DICOM files."""
    os.makedirs(dst_folder, exist_ok=True)
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".dcm"):
                in_path = os.path.join(root, file)
                rel_path = os.path.relpath(in_path, src_folder)
                out_path = os.path.join(dst_folder, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                try:
                    anonymize_dicom_file(in_path, out_path)
                except Exception as e:
                    print(f"❌ Error anonymizing {file}: {e}")

def zip_folder(folder_path: str, zip_path: str):
    """Compresses a folder into a zip file."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

class MedicalImageDataset(Dataset):
    """Custom Dataset for loading Amyloid and FDG PET pairs."""
    def __init__(self, amyloid_paths, fdg_paths):
        self.amyloid_paths = amyloid_paths
        self.fdg_paths = fdg_paths

    def __len__(self):
        return len(self.amyloid_paths)

    def __getitem__(self, idx):
        amyloid_dcm = pydicom.dcmread(self.amyloid_paths[idx])
        amyloid_img = apply_voi_lut(amyloid_dcm.pixel_array, amyloid_dcm)
        
        fdg_dcm = pydicom.dcmread(self.fdg_paths[idx])
        fdg_img = apply_voi_lut(fdg_dcm.pixel_array, fdg_dcm)
        
        # Normalize to [0, 1]
        def normalize(image):
            img_min, img_max = image.min(), image.max()
            if img_max - img_min == 0:
                return np.zeros_like(image)
            return (image - img_min) / (img_max - img_min)
            
        amyloid_img = normalize(amyloid_img)
        fdg_img = normalize(fdg_img)
        
        # Convert to tensor and add channel dimension [C, D, H, W]
        amyloid_tensor = torch.FloatTensor(amyloid_img).unsqueeze(0)
        fdg_tensor = torch.FloatTensor(fdg_img).unsqueeze(0)
        
        return amyloid_tensor, fdg_tensor

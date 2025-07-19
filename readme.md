This folder contains the code to run the Cervical Cancer Radiomic Temporal Analysis

The full dataset is available at: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=112591753

'cervical_cancer_preprocess.py' preprocesses the medical images.

'cervical_cancer_texture.py' extracts the 2D and 3D texture features.

'cervical_cancer_2D_GLCM_final.py' predicts the treatment response label with 2D radiomics + zero & first order features

'cervical_cancer_3d_GLCM.py' predicts the treatment response label with 3D radiomics + zero & first order features

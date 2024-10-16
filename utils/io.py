import numpy as np
import nibabel as nib
from nibabel.gifti import gifti

def save_gifti_surface(v, f, save_path,
                       surf_hemi='CortexLeft',
                       surf_type='GrayWhite',
                       geom_type='Anatomical'):

    """
    - surf_hemi: ['CortexLeft', 'CortexRight']
    - surf_type: ['GrayWhite', 'Pial', 'MidThickness']
    - geom_type: ['Anatomical', 'VeryInflated', 'Spherical', 'Inflated']
    """
    v = v.astype(np.float32)
    f = f.astype(np.int32)

    # meta data
    v_meta_dict = {'AnatomicalStructurePrimary': surf_hemi,
                   'AnatomicalStructureSecondary': surf_type,
                   'GeometricType': geom_type,
                   'Name': '#1'}
    f_meta_dict = {'Name': '#2'}

    v_meta = gifti.GiftiMetaData()
    f_meta = gifti.GiftiMetaData()
    v_meta = v_meta.from_dict(v_meta_dict)
    f_meta = f_meta.from_dict(f_meta_dict)

    # new gifti image
    gii_surf = gifti.GiftiImage()

    gii_surf_v = gifti.GiftiDataArray(v, intent='pointset', meta=v_meta)
    gii_surf_f = gifti.GiftiDataArray(f, intent='triangle', meta=f_meta)
    gii_surf.add_gifti_data_array(gii_surf_v)
    gii_surf.add_gifti_data_array(gii_surf_f)

    nib.save(gii_surf, save_path)
import os
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class BSRTestFolderDataset(BaseSRDataset):
    """General paired image folder dataset for image restoration.

    The dataset loads lq (Low Quality) , gt (Ground-Truth) image pairs and kernel,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "folder mode", which needs to specify the lq folder path and gt
    folder path, each folder containing the corresponding images.
    Image lists will be generated automatically. You can also specify the
    filename template to match the lq and gt pairs.

    For example, we have two folders with the following structures:

    ::

        data_root
        ├── lq
        │   ├── 0001_x4.png
        │   ├── 0002_x4.png
        ├── gt
        │   ├── 0001.png
        │   ├── 0002.png
        ├── kernel
        │   ├── 0001.mat
        │   ├── 0002.mat

    then, you need to set:

    .. code-block:: python

        lq_folder = data_root/lq
        gt_folder = data_root/gt
        kernel_folder = data_root/kernel
        filename_tmpl = '{}_x4'

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        kernel_folder (str | :obj: 'Path'): Path to kernel list file
        pipeline (List[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 kernel_folder,
                 pipeline,
                 scale,
                 test_mode=False,
                 filename_tmpl='{}'):
        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.kernel_folder = str(kernel_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for SR dataset.

        It loads the LQ, GT and Kernel path from folders.

        Returns:
            dict: Returned dict for LQ, GT and Kernel.
        """
        data_infos = []
        lq_paths = self.scan_folder(self.lq_folder)
        gt_paths = self.scan_folder(self.gt_folder)
        kernel_paths = os.listdir(self.kernel_folder)

        assert len(lq_paths) == len(gt_paths) and len(kernel_paths) == len(gt_paths), (
            f'gt, lq and kernel datasets have different number of images: '
            f'{len(lq_paths)}, {len(gt_paths)}, {len(kernel_paths)}.')

        for gt_path in gt_paths:
            basename, ext = osp.splitext(osp.basename(gt_path))
            ids = basename.split('_')[1]
            lq_basename = 'im_'+ids
            kernel_basename = 'kernel_'+ids
            lq_path = osp.join(self.lq_folder,
                               (f'{self.filename_tmpl.format(lq_basename)}'
                                f'{ext}'))
            kernel_path = osp.join(self.kernel_folder, kernel_basename + '.mat')

            # if kernel_path not in kernel_paths:
            #     print(f'{kernel_path} is not in kernel_paths.')

            # assert lq_path in lq_paths, f'{lq_path} is not in lq_paths.'
            # assert kernel_path in kernel_paths, f'{kernel_path} is not in kernel_paths.'

            data_infos.append(dict(lq_path=lq_path, gt_path=gt_path, kernel_path=kernel_path))
        return data_infos

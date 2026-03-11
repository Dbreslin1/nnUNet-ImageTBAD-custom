import numpy as np
from typing import Union

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class ImageTBADBodyCropPreprocessor(DefaultPreprocessor):
    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str]
    ):
        data = data.astype(np.float32)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:]
            seg = np.copy(seg)

        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
        properties['shape_before_cropping'] = data.shape[1:]

        print("USING ImageTBADBodyCropPreprocessor")

        data, seg, bbox = self.body_crop_keep_full_z(data, seg)

        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing

        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        data = self._normalize(
            data,
            seg,
            configuration_manager,
            plans_manager.foreground_intensity_properties_per_channel
        )

        data = configuration_manager.resampling_fn_data(
            data, new_shape, original_spacing, target_spacing
        )
        seg = configuration_manager.resampling_fn_seg(
            seg, new_shape, original_spacing, target_spacing
        )

        if seg is not None:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels

            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            properties['class_locations'] = self._sample_foreground_locations(
                seg, collect_for_this, verbose=self.verbose
            )

            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)

        return data, seg, properties

    def body_crop_keep_full_z(self, data, seg):
        # assumes shape (C, Z, Y, X)
        body_mask = data[0] > -600

        xy_projection = body_mask.any(axis=0)
        coords = np.argwhere(xy_projection)

        if len(coords) == 0:
            bbox = [[0, data.shape[1]], [0, data.shape[2]], [0, data.shape[3]]]
            return data, seg, bbox

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1

        z_min = 0
        z_max = data.shape[1]

        data = data[:, z_min:z_max, y_min:y_max, x_min:x_max]
        if seg is not None:
            seg = seg[:, z_min:z_max, y_min:y_max, x_min:x_max]

        bbox = [[z_min, z_max], [y_min, y_max], [x_min, x_max]]
        return data, seg, bbox
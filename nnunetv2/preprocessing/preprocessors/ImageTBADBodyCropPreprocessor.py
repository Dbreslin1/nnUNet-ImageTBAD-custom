import numpy as np
from typing import Union
from scipy.ndimage import binary_fill_holes, label

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class ImageTBADBodyCropPreprocessor(DefaultPreprocessor):
    """
    Conservative CTA-aware preprocessing for ImageTBAD:
    1. Build a body mask from the CT volume
    2. Keep only the largest connected body component
    3. Crop in XY with a generous margin
    4. Trim only clearly empty z-slices at top/bottom
    5. Mark outside-body voxels as -1 in seg (ignored during training)
    6. Suppress data voxels outside the body mask
    7. Let nnU-Net keep its own CT normalization + resampling
    """

    BODY_THRESHOLD = -600     # body vs air
    XY_MARGIN = 40            # keep thoracic context
    Z_MARGIN = 8              # keep superior/inferior context
    OUTSIDE_VALUE = -1000.0   # air-like value outside the body

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
        has_seg = seg is not None
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:]
            seg = np.copy(seg)

        # transpose into nnU-Net internal axis order
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
        properties['shape_before_cropping'] = data.shape[1:]

        # body_crop_trim_and_suppress always returns a valid seg (creates one if None)
        data, seg, bbox, body_mask = self.body_crop_trim_and_suppress(data, seg)

        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing

        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # Normalize BEFORE resampling (matches default nnU-Net order)
        data = self._normalize(
            data,
            seg,
            configuration_manager,
            plans_manager.foreground_intensity_properties_per_channel
        )

        # Standard nnU-Net resampling
        data = configuration_manager.resampling_fn_data(
            data, new_shape, original_spacing, target_spacing
        )
        seg = configuration_manager.resampling_fn_seg(
            seg, new_shape, original_spacing, target_spacing
        )

        # Foreground sampling only for training cases (where seg was provided)
        if has_seg:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = (
                label_manager.foreground_regions
                if label_manager.has_regions
                else label_manager.foreground_labels
            )

            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            properties['class_locations'] = self._sample_foreground_locations(
                seg, collect_for_this, verbose=self.verbose
            )

            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)

        # Always cast seg dtype (seg is guaranteed non-None after body_crop_trim_and_suppress)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)

        return data, seg, properties

    def body_crop_trim_and_suppress(self, data, seg):
        """
        Assumes data shape (C, Z, Y, X).
        Always returns a valid seg (creates one from body mask if input seg is None).
        Outside-body voxels are marked as -1 in seg so training loss ignores them.
        """
        image = data[0]

        # 1) rough body mask
        body_mask = image > self.BODY_THRESHOLD

        # 2) fill holes slice-wise for more stable body region
        filled_mask = np.zeros_like(body_mask, dtype=bool)
        for z in range(body_mask.shape[0]):
            filled_mask[z] = binary_fill_holes(body_mask[z])

        # 3) keep largest connected component in 3D
        labeled, num = label(filled_mask)
        if num > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest_label = np.argmax(sizes)
            body_mask = labeled == largest_label
        else:
            body_mask = filled_mask

        # fallback: if mask is empty, skip cropping/suppression
        if not np.any(body_mask):
            bbox = [[0, data.shape[1]], [0, data.shape[2]], [0, data.shape[3]]]
            if seg is None:
                seg = np.zeros((1,) + data.shape[1:], dtype=np.int8)
            return data, seg, bbox, body_mask

        # 4) XY crop with margin
        xy_projection = body_mask.any(axis=0)   # (Y, X)
        xy_coords = np.argwhere(xy_projection)

        y_min, x_min = xy_coords.min(axis=0)
        y_max, x_max = xy_coords.max(axis=0) + 1

        y_min = max(0, y_min - self.XY_MARGIN)
        x_min = max(0, x_min - self.XY_MARGIN)
        y_max = min(data.shape[2], y_max + self.XY_MARGIN)
        x_max = min(data.shape[3], x_max + self.XY_MARGIN)

        # 5) trim only empty z-slices at the edges
        z_projection = body_mask.any(axis=(1, 2))
        z_coords = np.where(z_projection)[0]

        z_min = max(0, int(z_coords.min()) - self.Z_MARGIN)
        z_max = min(data.shape[1], int(z_coords.max()) + 1 + self.Z_MARGIN)

        # crop image and seg
        data = data[:, z_min:z_max, y_min:y_max, x_min:x_max]
        if seg is not None:
            seg = seg[:, z_min:z_max, y_min:y_max, x_min:x_max]

        # crop body mask to same region
        body_mask = body_mask[z_min:z_max, y_min:y_max, x_min:x_max]

        # Mark outside-body voxels as -1 in seg (matches nnU-Net's nonzero mask convention).
        # This prevents training loss from being computed on air/table/padding voxels.
        if seg is not None:
            seg[(seg == 0) & (~body_mask[None])] = -1
        else:
            seg = np.where(body_mask[None], np.int8(0), np.int8(-1))

        # 6) suppress outside-body voxels in data
        for c in range(data.shape[0]):
            data[c][~body_mask] = self.OUTSIDE_VALUE

        bbox = [[z_min, z_max], [y_min, y_max], [x_min, x_max]]
        return data, seg, bbox, body_mask
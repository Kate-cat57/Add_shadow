import os
import time
import random
import numpy as np
#@TODO: debug to check that packages from requirements.txt are installed

import supervisely_lib as sly

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

my_app = sly.AppService()


@my_app.callback("add_shadow")
@sly.timeit
def add_shadow(api: sly.Api, task_id, context, state, app_logger):
    src_project = api.project.get_info_by_id(PROJECT_ID)

    src_meta_json = api.project.get_meta(PROJECT_ID)
    src_meta = sly.ProjectMeta.from_json(src_meta_json)
    src_datasets = api.dataset.get_list(src_project.id)

    dst_project_name = src_project.name + "_shadow"
    dst_project = api.project.create(WORKSPACE_ID, dst_project_name)

    dst_meta = src_meta
    api.project.update_meta(dst_project.id, dst_meta.to_json())

    for dataset in src_datasets:
        src_images = api.image.get_list(dataset.id)
        dst_dataset = api.dataset.create(dst_project.id, dataset.name + "_shadow")

        for src_image in src_images:

            img = api.image.download_np(src_image.id)

            ann_info = api.annotation.download(src_image.id)
            ann = sly.Annotation.from_json(ann_info.annotation, src_meta)

            mask_src = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            mask_shifted = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

            for label in ann.labels:
                if not label.geometry.geometry_name() == "bitmap":
                    continue
                mask = label.geometry.data

                # random
                shift_row = random.randint(-1 * mask.shape[0] // 2, mask.shape[0] // 2)
                shift_col = random.randint(-1 * mask.shape[1] // 2, mask.shape[1] // 2)

                src_row = label.geometry.origin.row
                src_col = label.geometry.origin.col
                print(src_row, src_col)
                # --------------
                # negative case
                # row
                if shift_row < 0 and src_row + shift_row < 0:
                    shift_row = src_row * -1

                # col
                if shift_col < 0 and src_col + shift_col < 0:
                    shift_col = src_col * -1

                # positive + 0 case
                # row
                if shift_row >= 0 and mask.shape[0] + shift_row + src_row > mask_src.shape[0]:
                    shift_row = mask_src.shape[0] - mask.shape[0] - src_row

                # col
                if shift_col >= 0 and mask.shape[1] + shift_col + src_col > mask_src.shape[1]:
                    shift_col = mask_src.shape[1] - mask.shape[1] - src_col
                # ----------
                mask_src[src_row:src_row + mask.shape[0], src_col:src_col + mask.shape[1]] = mask
                mask_shifted[src_row + shift_row:src_row + mask.shape[0] + shift_row,
                src_col + shift_col:src_col + mask.shape[1] + shift_col] = mask

            # ------------

            temp_shadow = np.logical_xor(mask_shifted, mask_src)
            shadow = np.logical_and(mask_shifted, temp_shadow)

            # invert and convert shadow matrix
            invert_shadow = np.invert(shadow)
            convert_shadow = invert_shadow.astype(float)

            convert_shadow[convert_shadow == 0] = 0.8  # add opacity

            img[:, :, 0] = img[:, :, 0] * convert_shadow
            img[:, :, 1] = img[:, :, 1] * convert_shadow
            img[:, :, 2] = img[:, :, 2] * convert_shadow

            dst_image = api.image.upload_np(dst_dataset.id, src_image.name, img, src_image.meta)
            api.annotation.upload_ann(dst_image.id, ann)

    my_app.stop()


def main():
    api = sly.Api.from_env()
    my_app.run(initial_events=[{"command": "add_shadow"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)

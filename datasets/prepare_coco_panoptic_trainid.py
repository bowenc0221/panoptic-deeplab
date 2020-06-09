import copy
import json
import os


def convert_to_trainid(input_filename, output_filename):
    """
    Convert coco panoptic segmentation dataset to trainid.
    Args:
        input_filename (str): path to the coco panoptic json file.
        output_filename (str): path to the coco panoptic trainid json file.
    """
    with open(input_filename, "r") as f:
        coco_json = json.load(f)

    coco_anns = coco_json.pop('annotations')
    coco_cats = coco_json.pop('categories')
    coco_trainid_json = copy.deepcopy(coco_json)

    coco_train_id_to_eval_id = [coco_cat['id'] for coco_cat in coco_cats]
    coco_eval_id_to_train_id = {v: k for k, v in enumerate(coco_train_id_to_eval_id)}

    new_cats = []
    for coco_cat in coco_cats:
        coco_cat['id'] = coco_eval_id_to_train_id[coco_cat['id']]
        new_cats.append(coco_cat)
    coco_trainid_json['categories'] = new_cats

    new_anns = []
    for coco_ann in coco_anns:
        segments_info = coco_ann.pop('segments_info')
        new_segments_info = []
        for segment_info in segments_info:
            segment_info['category_id'] = coco_eval_id_to_train_id[segment_info['category_id']]
            new_segments_info.append(segment_info)
        coco_ann['segments_info'] = new_segments_info
        new_anns.append(coco_ann)
    coco_trainid_json['annotations'] = new_anns

    with open(output_filename, "w") as f:
        json.dump(coco_trainid_json, f)
    print("{} is converted to trainid and stored in {}.".format(input_filename, output_filename))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), 'coco', 'annotations')
    for s in ['panoptic_train2017', 'panoptic_val2017']:
        print("Start converting {} to trainid.".format(s))
        convert_to_trainid(
            os.path.join(dataset_dir, "{}.json".format(s)),
            os.path.join(dataset_dir, "{}_trainId.json".format(s)),
        )

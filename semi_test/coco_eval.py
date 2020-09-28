import torch
import os
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import check_expected_results
from maskrcnn_benchmark.data import make_data_loader



# def do_coco_evaluation(
#     dataset,
#     predictions,
#     box_only,
#     output_folder,
#     iou_types,
#     expected_results,
#     expected_results_sigma_tol,
# ):
#     logger = logging.getLogger("maskrcnn_benchmark.inference")

#     logger.info("Evaluating bbox proposals")
#     areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
#     res = COCOResults("box_proposal")
#     for limit in [100, 1000]:
#         for area, suffix in areas.items():
#             stats = evaluate_box_proposals(
#                 predictions, dataset, area=area, limit=limit
#             )
#             key = "AR{}@{:d}".format(suffix, limit)
#             res.results["box_proposal"][key] = stats["ar"].item()
#     logger.info(res)
#     check_expected_results(res, expected_results, expected_results_sigma_tol)
#     if output_folder:
#         torch.save(res, os.path.join(output_folder, "box_proposals.pth"))


def main():
    config_file = './semi_test/retinanet_R-50-FPN_1x_semi.yaml'
    output_folder = './output_folder'
    str_pth = "./model_path"

    cfg.merge_from_file(config_file)

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    dataset = data_loaders_val[0].dataset

    range_start = 2
    range_end = 3

    predictions = torch.load(str_pth)[range_start:range_end]
    dataset_imgs = {}
    id_to_img_map = {}
    
    for _prediction in predictions:
        ind_sort = _prediction.get_field('scores').argsort(descending=True)[:12]
        _prediction.bbox = _prediction.bbox[ind_sort]
        _prediction.extra_fields['scores'] = _prediction.extra_fields['scores'][ind_sort]
        _prediction.extra_fields['labels'] = _prediction.extra_fields['labels'][ind_sort]
        _prediction = _prediction.resize([640,478])



     
    iCount = 0
    for i in range(range_start,range_end):
        _id = dataset.id_to_img_map[i]
        id_to_img_map[iCount] = _id
        dataset_imgs[_id] = dataset.coco.imgs[_id]
        iCount += 1

    dataset.coco.imgs = dataset_imgs
    dataset.id_to_img_map = id_to_img_map

    extra_args = dict(
        box_only=False,
        iou_types=('bbox',),
        expected_results=[],
        expected_results_sigma_tol=4
        )

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    r=evaluate(dataset=dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)
    check_expected_results(r, [], 4)
  
    print(r[0])


if __name__ == "__main__":
    main()
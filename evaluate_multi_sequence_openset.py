import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import auc, average_precision_score, roc_curve
from tqdm import tqdm


def convert_to_builtin_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_point_cloud(bin_path: Path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3], points[:, 3]


def load_raw_labels(label_path: Path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF
    inst_labels = labels >> 16
    return sem_labels.astype(np.int32), inst_labels.astype(np.int32)


def load_learning_map(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    learning_map = {int(k): int(v) for k, v in data["learning_map"].items()}
    learning_map_inv = {int(k): int(v) for k, v in data["learning_map_inv"].items()}
    raw_label_names = {int(k): v for k, v in data["labels"].items()}
    return learning_map, learning_map_inv, raw_label_names


def map_raw_gt_to_train_id(raw_gt: np.ndarray, learning_map: dict) -> np.ndarray:
    mapped = np.zeros_like(raw_gt, dtype=np.int32)
    for raw_id, train_id in learning_map.items():
        mapped[raw_gt == raw_id] = train_id
    return mapped


class SequenceEvaluator:
    """
    Internal evaluation protocol:

    GT:
      0..18  known classes
      100    outlier
      -1     ignore

    Closed-set predictions:
      0..18  known classes
      19     dummy

    Open-set predictions:
      0..18  known classes
      19     dummy
      100    predicted outlier
    """

    def __init__(
        self,
        learning_map: dict,
        num_known_classes: int = 19,
        outlier_label: int = 100,
        dummy_label: int = 19,
        min_eval_distance: float = 2.5,
        max_eval_distance: float = 50.0,
        min_num_outlier_points: int = 5,
        raw_ignore_ids=None,
        raw_outlier_ids=None,
    ):
        self.learning_map = learning_map
        self.num_known_classes = num_known_classes
        self.outlier_label = outlier_label
        self.dummy_label = dummy_label
        self.min_eval_distance = min_eval_distance
        self.max_eval_distance = max_eval_distance
        self.min_num_outlier_points = min_num_outlier_points

        if raw_ignore_ids is None:
            raw_ignore_ids = [0, 2, 52, 99]
        if raw_outlier_ids is None:
            raw_outlier_ids = [1]

        self.raw_ignore_ids = set(raw_ignore_ids)
        self.raw_outlier_ids = set(raw_outlier_ids)

        self.internal_outlier_index = self.num_known_classes
        self.num_eval_classes = self.num_known_classes + 1

        self.reset()

    def reset(self):
        self.all_anomaly_scores = []
        self.all_anomaly_targets = []
        self.threshold_at_tpr95 = None

        self.closed_conf = np.zeros(
            (self.num_known_classes, self.num_known_classes + 1),
            dtype=np.int64,
        )

        self.open_conf = np.zeros(
            (self.num_eval_classes, self.num_eval_classes + 1),
            dtype=np.int64,
        )

    def _distance_mask(self, points: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(points, axis=1)
        return (distances >= self.min_eval_distance) & (
            distances <= self.max_eval_distance
        )

    def build_eval_gt_from_raw(self, raw_gt: np.ndarray) -> np.ndarray:
        train_gt = map_raw_gt_to_train_id(raw_gt, self.learning_map)
        eval_gt = np.full_like(train_gt, fill_value=-1, dtype=np.int32)

        ignore_mask = np.isin(raw_gt, list(self.raw_ignore_ids))
        outlier_mask = np.isin(raw_gt, list(self.raw_outlier_ids))

        eval_gt[outlier_mask] = self.outlier_label

        known_mask = (train_gt >= 1) & (train_gt <= self.num_known_classes)
        fill_mask = known_mask & (~ignore_mask) & (~outlier_mask)
        eval_gt[fill_mask] = train_gt[fill_mask] - 1

        return eval_gt

    def _to_internal_gt_index(self, gt_eval_value: int) -> int:
        if 0 <= gt_eval_value < self.num_known_classes:
            return gt_eval_value
        if gt_eval_value == self.outlier_label:
            return self.internal_outlier_index
        raise ValueError(f"Unexpected GT eval value: {gt_eval_value}")

    def _to_internal_open_pred_col(self, pred_value: int) -> int:
        if 0 <= pred_value < self.num_known_classes:
            return pred_value
        if pred_value == self.outlier_label:
            return self.internal_outlier_index
        if pred_value == self.dummy_label:
            return self.num_eval_classes
        return self.num_eval_classes

    def _build_anomaly_binary_targets(self, gt_eval: np.ndarray) -> np.ndarray:
        binary = np.full_like(gt_eval, fill_value=-1, dtype=np.int32)
        binary[(gt_eval >= 0) & (gt_eval < self.num_known_classes)] = 0
        binary[gt_eval == self.outlier_label] = 1
        return binary

    def update_anomaly_metrics(self, points, anomaly_scores, gt_eval):
        if not (len(points) == len(anomaly_scores) == len(gt_eval)):
            raise ValueError("Length mismatch among points, anomaly_scores, gt_eval")

        valid_dist = self._distance_mask(points)
        binary_targets = self._build_anomaly_binary_targets(gt_eval)
        valid_mask = valid_dist & (binary_targets != -1)

        y_true = binary_targets[valid_mask]
        y_score = anomaly_scores[valid_mask]

        if np.sum(y_true == 1) < self.min_num_outlier_points:
            return

        self.all_anomaly_targets.append(y_true)
        self.all_anomaly_scores.append(y_score)

    def compute_anomaly_metrics_and_threshold(self):
        if len(self.all_anomaly_scores) == 0:
            return {}

        y_true = np.concatenate(self.all_anomaly_targets, axis=0)
        y_score = np.concatenate(self.all_anomaly_scores, axis=0)

        ap = average_precision_score(y_true=y_true, y_score=y_score)
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        auroc = auc(fpr, tpr)

        fpr95 = np.nan
        thr95 = np.nan

        for tpr_val, fpr_val, thr in zip(tpr, fpr, thresholds):
            if tpr_val >= 0.95:
                fpr95 = fpr_val
                thr95 = thr
                break

        self.threshold_at_tpr95 = thr95

        return {
            "AUROC": auroc * 100.0,
            "FPR@95TPR": fpr95 * 100.0 if not np.isnan(fpr95) else None,
            "AP": ap * 100.0,
            "threshold_at_TPR95": thr95 if not np.isnan(thr95) else None,
        }

    def update_segmentation_metrics(
        self,
        points,
        pred_closed,
        anomaly_scores,
        gt_eval,
        threshold,
    ):
        if not (
            len(points)
            == len(pred_closed)
            == len(anomaly_scores)
            == len(gt_eval)
        ):
            raise ValueError(
                "Length mismatch among points, pred_closed, anomaly_scores, gt_eval"
            )

        valid_dist = self._distance_mask(points)
        valid_mask = valid_dist & (gt_eval != -1)

        gt_valid = gt_eval[valid_mask]
        pred_closed_valid = pred_closed[valid_mask].copy()
        scores_valid = anomaly_scores[valid_mask]

        closed_gt_mask = (gt_valid >= 0) & (gt_valid < self.num_known_classes)
        gt_closed = gt_valid[closed_gt_mask]
        pred_closed_eval = pred_closed_valid[closed_gt_mask]

        for g_ext, p in zip(gt_closed, pred_closed_eval):
            g = int(g_ext)
            p = int(p)

            if 0 <= p < self.num_known_classes:
                self.closed_conf[g, p] += 1
            elif p == self.dummy_label:
                self.closed_conf[g, self.num_known_classes] += 1
            else:
                self.closed_conf[g, self.num_known_classes] += 1

        pred_open = pred_closed_valid.copy()
        pred_open[scores_valid > threshold] = self.outlier_label

        for g_ext, p_ext in zip(gt_valid, pred_open):
            g_internal = self._to_internal_gt_index(int(g_ext))
            p_internal = self._to_internal_open_pred_col(int(p_ext))
            self.open_conf[g_internal, p_internal] += 1

    def compute_closed_set_metrics(self):
        ious = np.full(self.num_known_classes, np.nan, dtype=np.float64)

        for i in range(self.num_known_classes):
            tp = self.closed_conf[i, i]
            fp = np.sum(self.closed_conf[:, i]) - tp
            fn = np.sum(self.closed_conf[i, :]) - tp
            denom = tp + fp + fn

            if denom > 0:
                ious[i] = tp / denom

        miou = np.nanmean(ious) * 100.0 if np.any(~np.isnan(ious)) else np.nan

        return {
            "mIoU": miou if not np.isnan(miou) else None,
            "per_class_IoU": {
                str(i): ious[i] * 100.0 if not np.isnan(ious[i]) else None
                for i in range(self.num_known_classes)
            },
            "dummy_prediction_count": int(
                np.sum(self.closed_conf[:, self.num_known_classes])
            ),
        }

    def compute_open_set_metrics(self):
        ious = np.full(self.num_eval_classes, np.nan, dtype=np.float64)

        for i in range(self.num_eval_classes):
            tp = self.open_conf[i, i]
            fp = np.sum(self.open_conf[:, i]) - tp
            fn = np.sum(self.open_conf[i, :]) - tp
            denom = tp + fp + fn

            if denom > 0:
                ious[i] = tp / denom

        known_ious = ious[: self.num_known_classes]

        known_miou = (
            np.nanmean(known_ious) * 100.0
            if np.any(~np.isnan(known_ious))
            else np.nan
        )

        outlier_iou = (
            ious[self.internal_outlier_index] * 100.0
            if not np.isnan(ious[self.internal_outlier_index])
            else np.nan
        )

        return {
            "open_mIoU_over_known_classes": (
                known_miou if not np.isnan(known_miou) else None
            ),
            "outlier_IoU": outlier_iou if not np.isnan(outlier_iou) else None,
            "per_class_open_IoU": {
                str(i): known_ious[i] * 100.0
                if not np.isnan(known_ious[i])
                else None
                for i in range(self.num_known_classes)
            },
            "dummy_prediction_count": int(
                np.sum(self.open_conf[:, self.num_eval_classes])
            ),
        }


def collect_sequence_files(data_root: Path, pred_root: Path, seq: str):
    seq_dir = data_root / seq
    pred_seq_dir = pred_root / seq

    lidar_dir = seq_dir / "velodyne"
    label_dir = seq_dir / "labels"

    if not lidar_dir.is_dir():
        raise RuntimeError(f"Missing velodyne dir: {lidar_dir}")
    if not label_dir.is_dir():
        raise RuntimeError(f"Missing labels dir: {label_dir}")
    if not pred_seq_dir.is_dir():
        raise RuntimeError(f"Missing prediction dir: {pred_seq_dir}")

    lidar_files = sorted(lidar_dir.glob("*.bin"))

    if len(lidar_files) == 0:
        raise RuntimeError(f"No .bin files found in {lidar_dir}")

    return seq_dir, pred_seq_dir, label_dir, lidar_files


def load_closed_prediction(path: Path, pred_label_offset: int):
    pred = np.loadtxt(path).astype(np.int32)

    if pred_label_offset != 0:
        pred = pred - pred_label_offset

    return pred


def main(args):
    learning_map, learning_map_inv, raw_label_names = load_learning_map(args.yaml)

    overall_eval = SequenceEvaluator(
        learning_map=learning_map,
        num_known_classes=args.num_known_classes,
        outlier_label=args.outlier_label,
        dummy_label=args.dummy_label,
        min_eval_distance=args.min_eval_distance,
        max_eval_distance=args.max_eval_distance,
        min_num_outlier_points=args.min_num_outlier_points,
        raw_ignore_ids=args.raw_ignore_ids,
        raw_outlier_ids=args.raw_outlier_ids,
    )

    per_seq_results = {}

    if args.fixed_threshold is None:
        print("Pass 1: computing anomaly metrics and thresholds")

        seq_evals = {}

        for seq in args.sequences:
            seq_evals[seq] = SequenceEvaluator(
                learning_map=learning_map,
                num_known_classes=args.num_known_classes,
                outlier_label=args.outlier_label,
                dummy_label=args.dummy_label,
                min_eval_distance=args.min_eval_distance,
                max_eval_distance=args.max_eval_distance,
                min_num_outlier_points=args.min_num_outlier_points,
                raw_ignore_ids=args.raw_ignore_ids,
                raw_outlier_ids=args.raw_outlier_ids,
            )

        for seq in args.sequences:
            _, pred_seq_dir, label_dir, lidar_files = collect_sequence_files(
                args.data_root,
                args.pred_root,
                seq,
            )

            for pcd_file in tqdm(lidar_files, desc=f"Pass1 {seq}"):
                frame_id = pcd_file.stem

                points, _ = load_point_cloud(pcd_file)
                raw_gt, _ = load_raw_labels(label_dir / f"{frame_id}.label")
                gt_eval = seq_evals[seq].build_eval_gt_from_raw(raw_gt)

                anomaly_file = (
                    pred_seq_dir
                    / f"{args.anomaly_prefix}{frame_id}{args.pred_ext}"
                )
                anomaly_scores = np.loadtxt(anomaly_file).astype(np.float32)

                if len(anomaly_scores) != len(points):
                    raise ValueError(
                        f"Length mismatch in sequence {seq}, frame {frame_id}: "
                        f"{len(points)} points vs {len(anomaly_scores)} anomaly scores"
                    )

                seq_evals[seq].update_anomaly_metrics(
                    points,
                    anomaly_scores,
                    gt_eval,
                )

                overall_eval.update_anomaly_metrics(
                    points,
                    anomaly_scores,
                    gt_eval,
                )

        overall_anomaly = overall_eval.compute_anomaly_metrics_and_threshold()

        if (
            "threshold_at_TPR95" not in overall_anomaly
            or overall_anomaly["threshold_at_TPR95"] is None
        ):
            raise RuntimeError("Failed to determine overall threshold_at_TPR95.")

        overall_threshold = overall_anomaly["threshold_at_TPR95"]

        per_seq_anomaly = {}

        for seq in args.sequences:
            per_seq_anomaly[seq] = seq_evals[
                seq
            ].compute_anomaly_metrics_and_threshold()

    else:
        overall_threshold = args.fixed_threshold

        overall_anomaly = {
            "AUROC": None,
            "FPR@95TPR": None,
            "AP": None,
            "threshold_at_TPR95": overall_threshold,
        }

        per_seq_anomaly = {
            seq: {
                "AUROC": None,
                "FPR@95TPR": None,
                "AP": None,
                "threshold_at_TPR95": overall_threshold,
            }
            for seq in args.sequences
        }

    print("Pass 2: computing segmentation metrics")

    per_seq_evals_seg = {}

    for seq in args.sequences:
        per_seq_evals_seg[seq] = SequenceEvaluator(
            learning_map=learning_map,
            num_known_classes=args.num_known_classes,
            outlier_label=args.outlier_label,
            dummy_label=args.dummy_label,
            min_eval_distance=args.min_eval_distance,
            max_eval_distance=args.max_eval_distance,
            min_num_outlier_points=args.min_num_outlier_points,
            raw_ignore_ids=args.raw_ignore_ids,
            raw_outlier_ids=args.raw_outlier_ids,
        )

    for seq in args.sequences:
        _, pred_seq_dir, label_dir, lidar_files = collect_sequence_files(
            args.data_root,
            args.pred_root,
            seq,
        )

        if args.use_overall_threshold_for_each_sequence:
            seq_threshold = overall_threshold
        else:
            seq_threshold = per_seq_anomaly[seq].get("threshold_at_TPR95", None)

        if seq_threshold is None:
            seq_threshold = overall_threshold

        for pcd_file in tqdm(lidar_files, desc=f"Pass2 {seq}"):
            frame_id = pcd_file.stem

            points, _ = load_point_cloud(pcd_file)
            raw_gt, _ = load_raw_labels(label_dir / f"{frame_id}.label")
            gt_eval = per_seq_evals_seg[seq].build_eval_gt_from_raw(raw_gt)

            closed_file = (
                pred_seq_dir
                / f"{args.closed_prefix}{frame_id}{args.pred_ext}"
            )
            anomaly_file = (
                pred_seq_dir
                / f"{args.anomaly_prefix}{frame_id}{args.pred_ext}"
            )

            pred_closed = load_closed_prediction(
                closed_file,
                pred_label_offset=args.pred_label_offset,
            )
            anomaly_scores = np.loadtxt(anomaly_file).astype(np.float32)

            if len(pred_closed) != len(points):
                raise ValueError(
                    f"Length mismatch in sequence {seq}, frame {frame_id}: "
                    f"{len(points)} points vs {len(pred_closed)} closed predictions"
                )

            if len(anomaly_scores) != len(points):
                raise ValueError(
                    f"Length mismatch in sequence {seq}, frame {frame_id}: "
                    f"{len(points)} points vs {len(anomaly_scores)} anomaly scores"
                )

            per_seq_evals_seg[seq].update_segmentation_metrics(
                points=points,
                pred_closed=pred_closed,
                anomaly_scores=anomaly_scores,
                gt_eval=gt_eval,
                threshold=seq_threshold,
            )

            overall_eval.update_segmentation_metrics(
                points=points,
                pred_closed=pred_closed,
                anomaly_scores=anomaly_scores,
                gt_eval=gt_eval,
                threshold=overall_threshold,
            )

    for seq in args.sequences:
        closed_metrics = per_seq_evals_seg[seq].compute_closed_set_metrics()
        open_metrics = per_seq_evals_seg[seq].compute_open_set_metrics()

        if args.use_overall_threshold_for_each_sequence:
            threshold_used = overall_threshold
        else:
            threshold_used = per_seq_anomaly[seq].get("threshold_at_TPR95", None)
            if threshold_used is None:
                threshold_used = overall_threshold

        per_seq_results[seq] = {
            "threshold_used_for_segmentation": threshold_used,
            "anomaly_segmentation": per_seq_anomaly[seq],
            "closed_set_segmentation": closed_metrics,
            "open_set_segmentation": open_metrics,
        }

    overall_results = {
        "threshold_used_for_segmentation": overall_threshold,
        "anomaly_segmentation": overall_anomaly,
        "closed_set_segmentation": overall_eval.compute_closed_set_metrics(),
        "open_set_segmentation": overall_eval.compute_open_set_metrics(),
    }

    results = {
        "sequences": args.sequences,
        "label_protocol": {
            "gt_space": {
                "known_classes": "0..18",
                "outlier": args.outlier_label,
                "ignore": "masked out before evaluation",
            },
            "pred_space_closed_internal": {
                "known_classes": "0..18",
                "dummy": args.dummy_label,
            },
            "pred_space_open_internal": {
                "known_classes": "0..18",
                "dummy": args.dummy_label,
                "predicted_outlier": args.outlier_label,
            },
            "pred_label_offset": args.pred_label_offset,
            "raw_ignore_ids": args.raw_ignore_ids,
            "raw_outlier_ids": args.raw_outlier_ids,
        },
        "per_sequence": per_seq_results,
        "overall": overall_results,
    }

    print(json.dumps(results, indent=4, default=convert_to_builtin_types))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4, default=convert_to_builtin_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-sequence point-wise open-set evaluation"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing sequence folders.",
    )

    parser.add_argument(
        "--pred-root",
        type=Path,
        required=True,
        help="Root containing prediction folders.",
    )

    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="SemanticKITTI yaml with labels / learning_map / learning_map_inv.",
    )

    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        required=True,
        help="Sequence IDs to evaluate.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multi_sequence_openset_results.json"),
    )

    parser.add_argument("--closed-prefix", type=str, default="sem_preds_")
    parser.add_argument("--anomaly-prefix", type=str, default="")
    parser.add_argument("--pred-ext", type=str, default=".txt")

    parser.add_argument("--num-known-classes", type=int, default=19)
    parser.add_argument("--outlier-label", type=int, default=100)
    parser.add_argument("--dummy-label", type=int, default=19)

    parser.add_argument(
        "--pred-label-offset",
        type=int,
        default=0,
        help=(
            "Offset for closed-set prediction labels. "
            "Use 1 if predictions are 1..20 and should be converted to 0..19."
        ),
    )

    parser.add_argument("--min-eval-distance", type=float, default=2.5)
    parser.add_argument("--max-eval-distance", type=float, default=50.0)
    parser.add_argument("--min-num-outlier-points", type=int, default=5)

    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="If provided, skip threshold selection and use this threshold everywhere.",
    )

    parser.add_argument(
        "--use-overall-threshold-for-each-sequence",
        action="store_true",
        help=(
            "If set, per-sequence segmentation metrics also use the overall threshold. "
            "Otherwise each sequence uses its own threshold from Pass 1."
        ),
    )

    parser.add_argument(
        "--raw-ignore-ids",
        type=int,
        nargs="+",
        default=[0, 2, 52, 99],
        help="Raw GT labels to ignore during evaluation.",
    )

    parser.add_argument(
        "--raw-outlier-ids",
        type=int,
        nargs="+",
        default=[1],
        help="Raw GT labels treated as OOD/outlier.",
    )

    args = parser.parse_args()
    main(args)



"""
python evaluate_multi_sequence_openset.py \
  --data-root "data/semantickitti/dataset/sequences" \
  --pred-root "ood_val/" \
  --yaml "/semantic-kitti.yaml" \
  --sequences 411 421 431 441 451 412 422 432 442 452 \
  --output results_10seq_val_energy.json \
  --closed-prefix ""\ # prefix of the semanticpredictions .txt file.
  --anomaly-prefix energy_ \ # prefix of the anomaly scoring .txt file
  --pred-label-offset 1 # 1 if the prediction are in range 1 - 20. change to 0 if the predictions are in range 0-19


 val_weather_scenes = {
 "Day" : ["411", "412"],
 "Night" : ["421", "422"],
 "Rain" : ["441", "442"],
 "Cloudy" : ["431", "432"],
 "Snow" : ["451", "452"]
 } 


test_weather_scenes = {
 "Day" : ["413", "414"],
 "Night" : ["423", "424"],
 "Rain" : ["443", "444"],
 "Cloudy" : ["433", "434"],
 "Snow" : ["453", "454"]
 } 

"""

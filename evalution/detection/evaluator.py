from evaluation.detection.iou import calculate_iou
from evaluation.detection.metrics import DetectionMetrics


class DetectionEvaluator:

    def __init__(self, iou_threshold=0.5):

        self.iou_threshold = iou_threshold

    def evaluate(self, predictions, ground_truth):

        tp = 0
        fp = 0
        matched = set()

        for pred in predictions:

            best_iou = 0
            best_idx = -1

            for idx, gt in enumerate(ground_truth):

                if idx in matched:
                    continue

                iou = calculate_iou(
                    pred.bbox,
                    gt
                )

                if iou > best_iou:

                    best_iou = iou
                    best_idx = idx

            if best_iou >= self.iou_threshold:

                tp += 1
                matched.add(best_idx)

            else:

                fp += 1

        fn = len(ground_truth) - len(matched)

        precision = DetectionMetrics.precision(tp, fp)
        recall = DetectionMetrics.recall(tp, fn)
        f1 = DetectionMetrics.f1(precision, recall)

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

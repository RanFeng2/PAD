import os.path as osp
from copy import deepcopy
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

from PIL import Image
from prettytable import PrettyTable
import numpy as np
import torch
from mmengine.logging import MMLogger, print_log
from mmseg.evaluation import IoUMetric as MMSEG_IoUMetric

from mmseg.registry import METRICS

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


@METRICS.register_module()
class CustomIoUMetric(MMSEG_IoUMetric):

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                self.results.append(
                    [
                    self.intersect_and_union(pred_label, label, num_classes, self.ignore_index),
                    self.calculate_accumulated_matrix(pred_label, label, num_classes, self.ignore_index),   # cal accumulated matrix
                    ]
                )

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        # collect the results returned by the process() function
        results = tuple(zip(*results))
        assert len(results) == 2   # (self.intersect_and_union, self.calculate_accumulated_matrix)

        intersect_and_union = results[0]
        accumulated_matrix = results[1]

        intersect_and_union = tuple(zip(*intersect_and_union))
        assert len(intersect_and_union) == 4
        accumulated_matrix = tuple(zip(*accumulated_matrix))
        assert len(accumulated_matrix) == 2

        # calculate the metrics
        ## accumulate intersect_and_union
        total_area_intersect = sum(intersect_and_union[0])
        total_area_union = sum(intersect_and_union[1])
        total_area_pred_label = sum(intersect_and_union[2])
        total_area_label = sum(intersect_and_union[3])

        ## kappa_matrix
        # confusion_matrix = torch.stack(accumulated_matrix[0])
        # confusion_matrix = torch.sum(confusion_matrix, dim=0)
        kappa_matrix = torch.tensor(accumulated_matrix[1])
        # nanmask = ~torch.isnan(kappa_matrix)
        # kappa_matrix = torch.where(nanmask, kappa_matrix, torch.tensor(0.0, device=kappa_matrix.device, dtype=kappa_matrix.dtype))
        # kappa_matrix = torch.sum(kappa_matrix) / torch.sum(nanmask).clamp(min=1)
        
        ## calculate the metrics based on the accumulated matrix
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, kappa_matrix,
            self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table (total metrics)
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc' or key == 'Kappa':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics.pop('Kappa', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics
    
    @staticmethod
    def calculate_accumulated_matrix(pred_label: torch.tensor, label: torch.tensor, 
                                   num_classes: int, ignore_index: int) -> torch.Tensor:
        """
        Calculate confusion matrix and remove the row and column for the ignore_index, if valid.
        
        Args:
            pred_label (torch.tensor): Predicted labels.
            label (torch.tensor): True labels.
            num_classes (int): Number of classes.
            ignore_index (int): Label index to ignore.
        
        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes-1, num_classes-1) if ignore_index is valid.
                        Otherwise, returns confusion matrix of shape (num_classes, num_classes).
        """
        
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]
        
        pred_label = pred_label.cpu().numpy()
        label = label.cpu().numpy()
        
        valid_labels = list(range(num_classes))

        # calculate the confusion matrix
        cm = sklearn_confusion_matrix(label, pred_label, labels=valid_labels)

        # calculate Kappa
        N = cm.sum()
        p0 = np.diag(cm).sum() / N
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (N * N)
        if 1 - pe <= 0 or p0 < pe:
            kappa = np.nan  # will be ignored
        else:
            kappa = (p0 - pe) / (1 - pe)
        
        cm = torch.tensor(cm)
        kappa = torch.tensor(kappa)

        return cm, kappa
    
    @staticmethod
    def total_area_to_metrics(total_area_intersect: torch.Tensor,
                              total_area_union: torch.Tensor,
                              total_area_pred_label: torch.Tensor,
                              total_area_label: torch.Tensor,
                              kappa_matrix: torch.Tensor,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'Kappa']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall
            elif metric == 'Kappa':
                ret_metrics['Kappa'] = kappa_matrix

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics


if __name__ == '__main__':
    metric = CustomIoUMetric(
        ignore_index=255,
        iou_metrics=['mIoU', 'mFscore', 'Kappa'],
    )
    
    # set the dataset meta information
    metric.dataset_meta = {'classes': ['background', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6']}
    
    # create simulated data
    H, W = 256, 256
    num_classes = 7
    batch_size = 6
    
    pred_logits = [torch.randn(1, num_classes, H, W) for _ in range(batch_size)]
    pred_labels = [pred_logits[i].argmax(dim=1) for i in range(batch_size)]
    
    gt_labels = [torch.randint(0, num_classes, (1, H, W), dtype=torch.long) for _ in range(batch_size)]
    
    data_samples = [{
        'seg_logits': {'data': pred_logits[i]},
        'pred_sem_seg': {'data': pred_labels[i]},
        'gt_sem_seg': {'data': gt_labels[i]},
    } for i in range(batch_size)]
    
    # process the data
    for i in range(batch_size):
        metric.process(None, [data_samples[i]])
    
    # calculate the metrics
    metrics = metric.compute_metrics(metric.results)

    print("Computed metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


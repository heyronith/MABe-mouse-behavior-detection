import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FScoreCalculator:
    """Calculate F-score for MABe competition (per-video → per-lab → global)"""

    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Beta parameter for F-beta score (1.0 = F1, 0.5 = F0.5, 2.0 = F2)
        """
        self.beta = beta

    def calculate_f_score(self,
                         predictions: List[Dict],
                         ground_truth: List[Dict],
                         iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate F-score following MABe competition format

        Args:
            predictions: List of predicted segments
            ground_truth: List of ground truth segments
            iou_threshold: IoU threshold for matching

        Returns:
            Dictionary with f_score, precision, recall (per-video, per-lab, global)
        """
        # Group by video
        pred_by_video = self._group_by_video(predictions)
        gt_by_video = self._group_by_video(ground_truth)

        video_scores = {}

        # Calculate per-video F-scores
        for video_id in set(pred_by_video.keys()) | set(gt_by_video.keys()):
            pred_segments = pred_by_video.get(video_id, [])
            gt_segments = gt_by_video.get(video_id, [])

            scores = self._calculate_video_f_score(pred_segments, gt_segments, iou_threshold)
            video_scores[video_id] = scores

        # Group by lab and calculate per-lab scores
        lab_scores = self._calculate_lab_scores(video_scores)

        # Calculate global scores
        global_scores = self._calculate_global_scores(video_scores)

        return {
            'video_scores': video_scores,
            'lab_scores': lab_scores,
            'global_scores': global_scores
        }

    def _group_by_video(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Group segments by video_id"""
        by_video = {}
        for segment in segments:
            video_id = segment['video_id']
            if video_id not in by_video:
                by_video[video_id] = []
            by_video[video_id].append(segment)
        return by_video

    def _calculate_video_f_score(self,
                               pred_segments: List[Dict],
                               gt_segments: List[Dict],
                               iou_threshold: float) -> Dict[str, float]:
        """Calculate F-score for a single video"""

        # Convert to format for matching
        pred_intervals = [(s['start_frame'], s['stop_frame']) for s in pred_segments]
        gt_intervals = [(s['start_frame'], s['stop_frame']) for s in gt_segments]

        # Match predictions to ground truth
        tp, fp, fn = self._match_segments(pred_intervals, gt_intervals, iou_threshold)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f_score = 0.0
        else:
            f_score = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def _match_segments(self,
                       pred_intervals: List[Tuple[int, int]],
                       gt_intervals: List[Tuple[int, int]],
                       iou_threshold: float) -> Tuple[int, int, int]:
        """Match predicted segments to ground truth using IoU"""

        tp = 0
        fp = len(pred_intervals)
        fn = len(gt_intervals)

        # Simple greedy matching
        matched_gt = set()

        for pred_start, pred_end in pred_intervals:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_start, gt_end) in enumerate(gt_intervals):
                if gt_idx in matched_gt:
                    continue

                iou = self._calculate_iou((pred_start, pred_end), (gt_start, gt_end))

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                fp -= 1
                matched_gt.add(best_gt_idx)
                fn -= 1

        return tp, fp, fn

    def _calculate_iou(self, interval1: Tuple[int, int], interval2: Tuple[int, int]) -> float:
        """Calculate IoU between two intervals"""
        start1, end1 = interval1
        start2, end2 = interval2

        # Intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)

        # Union
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = max(0, union_end - union_start)

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_lab_scores(self, video_scores: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate per-lab scores"""
        # Get lab_id from video_id (assuming format: lab_video)
        lab_scores = {}

        for video_id, scores in video_scores.items():
            # Extract lab_id from video_id (first part before underscore)
            lab_id = video_id.split('_')[0]

            if lab_id not in lab_scores:
                lab_scores[lab_id] = {'scores': [], 'count': 0}

            lab_scores[lab_id]['scores'].append(scores['f_score'])
            lab_scores[lab_id]['count'] += 1

        # Calculate mean per lab
        for lab_id in lab_scores:
            scores = lab_scores[lab_id]['scores']
            lab_scores[lab_id]['mean_f_score'] = np.mean(scores)
            lab_scores[lab_id]['video_count'] = lab_scores[lab_id]['count']
            del lab_scores[lab_id]['scores']
            del lab_scores[lab_id]['count']

        return lab_scores

    def _calculate_global_scores(self, video_scores: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate global scores across all videos"""
        if not video_scores:
            return {'f_score': 0.0, 'precision': 0.0, 'recall': 0.0}

        all_f_scores = [scores['f_score'] for scores in video_scores.values()]

        # Global F-score is mean of per-video F-scores
        global_f_score = np.mean(all_f_scores)

        # Calculate weighted precision and recall
        total_tp = sum(scores['tp'] for scores in video_scores.values())
        total_fp = sum(scores['fp'] for scores in video_scores.values())
        total_fn = sum(scores['fn'] for scores in video_scores.values())

        global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        return {
            'f_score': global_f_score,
            'precision': global_precision,
            'recall': global_recall,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }


class LeaveOneLabOutValidator:
    """Leave-One-Lab-Out cross-validation for MABe"""

    def __init__(self, f_score_calculator: FScoreCalculator):
        self.f_score_calculator = f_score_calculator

    def validate(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """
        Perform Leave-One-Lab-Out validation

        Args:
            predictions: All predictions
            ground_truth: All ground truth

        Returns:
            Validation results with worst-lab and global scores
        """
        # Group by lab
        pred_by_lab = self._group_by_lab(predictions)
        gt_by_lab = self._group_by_lab(ground_truth)

        lab_results = {}

        # Leave-one-lab-out validation
        all_labs = sorted(set(pred_by_lab.keys()) | set(gt_by_lab.keys()))

        for test_lab in all_labs:
            # Training labs (all except test_lab)
            train_labs = [lab for lab in all_labs if lab != test_lab]

            # Get predictions and ground truth for test lab
            test_pred = pred_by_lab.get(test_lab, [])
            test_gt = gt_by_lab.get(test_lab, [])

            # Calculate scores for test lab
            scores = self.f_score_calculator.calculate_f_score(test_pred, test_gt)
            lab_results[test_lab] = scores['global_scores']['f_score']

        # Calculate validation metrics
        lab_f_scores = list(lab_results.values())

        results = {
            'global_f_score': np.mean(lab_f_scores),
            'worst_lab_f_score': np.min(lab_f_scores),
            'best_lab_f_score': np.max(lab_f_scores),
            'lab_f_scores': lab_results,
            'n_labs': len(all_labs)
        }

        return results

    def _group_by_lab(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Group segments by lab_id"""
        by_lab = {}
        for segment in segments:
            lab_id = segment['lab_id']
            if lab_id not in by_lab:
                by_lab[lab_id] = []
            by_lab[lab_id].append(segment)
        return by_lab


def decode_predictions(probs: np.ndarray,
                      threshold: float = 0.5,
                      min_duration: int = 6,
                      max_gap: int = 9) -> List[Dict]:
    """
    Decode model predictions into segments

    Args:
        probs: (n_frames, n_behaviors) probability array
        threshold: Probability threshold for detection
        min_duration: Minimum segment duration in frames
        max_gap: Maximum gap between segments to merge

    Returns:
        List of segment dictionaries
    """
    segments = []

    for behavior_idx in range(probs.shape[1]):
        # Get predictions for this behavior
        behavior_probs = probs[:, behavior_idx]
        predictions = (behavior_probs > threshold).astype(int)

        # Find continuous segments
        behavior_segments = find_continuous_segments(predictions, min_duration, max_gap)

        # Convert to segment format
        for start, end in behavior_segments:
            # Map behavior index back to behavior name (would need behavior mapping)
            behavior_name = f"behavior_{behavior_idx}"

            segments.append({
                'agent_id': 'unknown',  # Would need to map from model output
                'target_id': 'unknown',
                'action': behavior_name,
                'start_frame': int(start),
                'stop_frame': int(end)
            })

    return segments


def find_continuous_segments(binary_signal: np.ndarray,
                           min_duration: int = 6,
                           max_gap: int = 9) -> List[Tuple[int, int]]:
    """
    Find continuous segments in binary signal with gap filling

    Args:
        binary_signal: Binary array (0s and 1s)
        min_duration: Minimum segment duration
        max_gap: Maximum gap to fill between segments

    Returns:
        List of (start, end) tuples
    """
    segments = []
    n_frames = len(binary_signal)

    i = 0
    while i < n_frames:
        if binary_signal[i] == 1:
            # Start of segment
            start = i

            # Find end of segment
            while i < n_frames and binary_signal[i] == 1:
                i += 1
            end = i

            # Check if segment meets minimum duration
            if end - start >= min_duration:
                segments.append((start, end))
        else:
            i += 1

    # Gap filling: merge segments that are close together
    if max_gap > 0:
        merged_segments = []
        for segment in segments:
            if not merged_segments:
                merged_segments.append(list(segment))
            else:
                last_segment = merged_segments[-1]
                if segment[0] - last_segment[1] <= max_gap:
                    # Merge segments
                    last_segment[1] = segment[1]
                else:
                    merged_segments.append(list(segment))

        segments = [(s[0], s[1]) for s in merged_segments]

    return segments

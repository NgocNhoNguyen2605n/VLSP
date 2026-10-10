"""ETA theo thời gian thực và log chỉ từ worker được bật."""

import contextlib
import io
import unittest
from unittest.mock import patch

from baseline.progress import BatchProgress
from baseline.runtime import print_training_config


class ProgressTests(unittest.TestCase):
    def test_eta_updates_one_line_and_finishes_with_one_newline(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(output, 'isatty', return_value=True), patch('baseline.progress.time.monotonic',
                                                      side_effect=[100, 100, 177, 632]):
            with BatchProgress('Epoch 1/5', 76, remaining_label='Còn epoch') as progress:
                progress.update(11)
                progress.update(76)
        text = output.getvalue()
        self.assertIn('Batch 0/76 | Còn epoch ~--:--:--', text)
        self.assertIn('Batch 11/76 | Còn epoch ~00:07:35', text)
        self.assertIn('Batch 76/76 | Còn epoch ~00:00:00', text)
        self.assertEqual(text.count('\n'), 1)

    def test_captured_output_has_complete_lines_before_epoch_ends(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch('baseline.progress.time.monotonic',
                                                      side_effect=[100, 100, 101, 102, 103, 134, 135]):
            with BatchProgress('Epoch 1/5', 23) as progress:
                progress.update(1)
                progress.update(2)  # Too soon, avoid flooding saved logs.
                progress.update(10)  # Configured batch interval.
                progress.update(11)  # Time interval, even off the batch interval.
                self.assertTrue(output.getvalue().endswith('\n'))
                self.assertIn('Batch 11/23', output.getvalue())
                progress.update(23)  # Final batch even off the interval.
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertNotIn('\r', output.getvalue())
        self.assertNotIn('Batch 2/23', output.getvalue())
        self.assertIn('Batch 23/23 | Còn ~00:00:00', lines[-1])

    def test_captured_output_respects_custom_batch_interval(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch('baseline.progress.time.monotonic', return_value=100):
            with BatchProgress('Validation', 5, log_every=2) as progress:
                for batch in range(1, 6):
                    progress.update(batch)
        self.assertIn('Batch 2/5', output.getvalue())
        self.assertIn('Batch 4/5', output.getvalue())
        self.assertNotIn('Batch 3/5', output.getvalue())

    def test_secondary_worker_and_default_config_summary_are_silent(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with BatchProgress('Validation', 68, enabled=False) as progress:
                progress.update(10)
            print_training_config({}, [])
        self.assertEqual(output.getvalue(), '')

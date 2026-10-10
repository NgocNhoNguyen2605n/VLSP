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
        with contextlib.redirect_stdout(output), patch('baseline.progress.time.monotonic',
                                                      side_effect=[100, 100, 177, 632]):
            with BatchProgress('Epoch 1/5', 76, remaining_label='Còn epoch') as progress:
                progress.update(11)
                progress.update(76)
        text = output.getvalue()
        self.assertIn('Batch 0/76 | Còn epoch ~--:--:--', text)
        self.assertIn('Batch 11/76 | Còn epoch ~00:07:35', text)
        self.assertIn('Batch 76/76 | Còn epoch ~00:00:00', text)
        self.assertEqual(text.count('\n'), 1)

    def test_secondary_worker_and_default_config_summary_are_silent(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with BatchProgress('Validation', 68, enabled=False) as progress:
                progress.update(10)
            print_training_config({}, [])
        self.assertEqual(output.getvalue(), '')

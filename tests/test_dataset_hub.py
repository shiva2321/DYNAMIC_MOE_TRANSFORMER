import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_hub import MultiDomainDatasetHub, DOMAIN_LIST

class TestDatasetHub(unittest.TestCase):
    def setUp(self):
        self.hub_bpe = MultiDomainDatasetHub(seq_len=64, batch_size=4, use_bpe=True)
        self.hub_byte = MultiDomainDatasetHub(seq_len=64, batch_size=4, use_bpe=False)

    def test_domain_list_count(self):
        self.assertEqual(len(DOMAIN_LIST), 7)
        domains = self.hub_bpe.get_all_domains()
        self.assertEqual(len(domains), 7)

    def test_batch_shapes_bpe(self):
        for domain_idx in range(len(DOMAIN_LIST)):
            inputs, targets, name = self.hub_bpe.get_batch(domain_idx, batch_size=4, seq_len=64)
            self.assertEqual(inputs.shape, (4, 64))
            self.assertEqual(targets.shape, (4, 64))
            self.assertTrue(len(name) > 0)
            self.assertEqual(inputs.dtype, torch.long)

    def test_eval_batch_shapes(self):
        inputs, targets, name = self.hub_bpe.get_batch("code", batch_size=2, seq_len=32, is_eval=True)
        self.assertEqual(inputs.shape, (2, 32))
        self.assertEqual(targets.shape, (2, 32))

    def test_encode_decode_roundtrip(self):
        text = "def hello_world(): return 42"
        tokens = self.hub_bpe.encode(text)
        decoded = self.hub_bpe.decode(tokens)
        self.assertEqual(text, decoded)

    def test_byte_mode(self):
        inputs, targets, name = self.hub_byte.get_batch(0, batch_size=2, seq_len=16)
        self.assertEqual(inputs.shape, (2, 16))
        self.assertTrue((inputs < 256).all())

if __name__ == "__main__":
    unittest.main()

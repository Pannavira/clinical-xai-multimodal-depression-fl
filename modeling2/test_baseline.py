"""
test_baseline.py
================
Unit tests untuk memverifikasi komponen modeling2 (Loss, Model, Dataset, DataLoaders).
"""

import sys
import unittest
import torch
import numpy as np

from model import AudioEncoder, VisualEncoder, TextEncoder, MultimodalBaselineModel
from loss import ClassWeightedMSELoss
from dataset import EDAICMultimodalDataset, build_dataloaders


class TestModeling2(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.T = 512

    def test_loss_weighted_mse(self):
        """Uji ClassWeightedMSELoss terhadap batch normal dan edge cases."""
        criterion = ClassWeightedMSELoss(binarize_threshold=10.0)

        # Batch campuran (imbalanced: 3 normal, 1 depressed)
        pred = torch.tensor([5.0, 8.0, 12.0, 18.0], requires_grad=True)
        target = torch.tensor([4.0, 6.0, 8.0, 20.0])
        loss = criterion(pred, target)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

        # Backward test
        loss.backward()
        self.assertIsNotNone(pred.grad)

        # Edge case: semua class 0
        pred_0 = torch.tensor([2.0, 3.0, 4.0, 5.0])
        target_0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss_0 = criterion(pred_0, target_0)
        self.assertFalse(torch.isnan(loss_0))

        # Edge case: semua class 1
        pred_1 = torch.tensor([12.0, 15.0, 18.0, 22.0])
        target_1 = torch.tensor([11.0, 14.0, 17.0, 21.0])
        loss_1 = criterion(pred_1, target_1)
        self.assertFalse(torch.isnan(loss_1))

    def test_audio_encoder(self):
        """Uji AudioEncoder forward pass dan dimensi output (B, 256)."""
        encoder = AudioEncoder()
        # Input (B, 80, 512)
        x_c_first = torch.randn(self.batch_size, 80, self.T)
        out1 = encoder(x_c_first)
        self.assertEqual(out1.shape, (self.batch_size, 256))

        # Input (B, 512, 80)
        x_t_first = torch.randn(self.batch_size, self.T, 80)
        out2 = encoder(x_t_first)
        self.assertEqual(out2.shape, (self.batch_size, 256))

    def test_visual_encoder(self):
        """Uji VisualEncoder forward pass dan dimensi output (B, 256)."""
        encoder = VisualEncoder()
        # Input (B, 3, 512, 72)
        x_c_first = torch.randn(self.batch_size, 3, self.T, 72)
        out1 = encoder(x_c_first)
        self.assertEqual(out1.shape, (self.batch_size, 256))

        # Input (B, 512, 72, 3)
        x_t_first = torch.randn(self.batch_size, self.T, 72, 3)
        out2 = encoder(x_t_first)
        self.assertEqual(out2.shape, (self.batch_size, 256))

    def test_text_encoder(self):
        """Uji TextEncoder forward pass dan dimensi output (B, 256)."""
        encoder = TextEncoder()
        # Input (B, 768, 512)
        x_c_first = torch.randn(self.batch_size, 768, self.T)
        out1 = encoder(x_c_first)
        self.assertEqual(out1.shape, (self.batch_size, 256))

        # Input (B, 512, 768)
        x_t_first = torch.randn(self.batch_size, self.T, 768)
        out2 = encoder(x_t_first)
        self.assertEqual(out2.shape, (self.batch_size, 256))

    def test_multimodal_model(self):
        """Uji MultimodalBaselineModel end-to-end forward dan feature extraction."""
        model = MultimodalBaselineModel()
        xa = torch.randn(self.batch_size, self.T, 80)
        xv = torch.randn(self.batch_size, self.T, 72, 3)
        xt = torch.randn(self.batch_size, self.T, 768)

        pred = model(xa, xv, xt)
        self.assertEqual(pred.shape, (self.batch_size, 1))

        feats = model.extract_features(xa, xv, xt)
        self.assertEqual(feats["audio_latent"].shape, (self.batch_size, 256))
        self.assertEqual(feats["visual_latent"].shape, (self.batch_size, 256))
        self.assertEqual(feats["text_latent"].shape, (self.batch_size, 256))
        self.assertEqual(feats["fused_latent"].shape, (self.batch_size, 768))

    def test_dataset_and_dataloaders(self):
        """Uji dataset loading dan pembuatan DataLoader pada split E-DAIC."""
        train_loader, val_loader, test_loader = build_dataloaders(
            index_csv="modeling/01_Input_From_Partition/multimodal_feature_index.csv",
            labels_csv="data/detailed_lables.csv",
            feature_base_dir="modeling/01_Input_From_Partition",
            batch_size=8,
        )

        self.assertEqual(len(train_loader.dataset), 163)
        self.assertEqual(len(val_loader.dataset), 56)
        self.assertEqual(len(test_loader.dataset), 56)

        # Uji ambil 1 batch dari train loader
        batch = next(iter(train_loader))
        self.assertIn("audio", batch)
        self.assertIn("visual", batch)
        self.assertIn("text", batch)
        self.assertIn("phq8_score", batch)
        self.assertIn("label", batch)

        self.assertEqual(batch["audio"].shape, (8, 512, 80))
        self.assertEqual(batch["visual"].shape, (8, 512, 72, 3))
        self.assertEqual(batch["text"].shape, (8, 512, 768))
        self.assertEqual(batch["phq8_score"].shape, (8,))
        self.assertEqual(batch["label"].shape, (8,))


if __name__ == "__main__":
    unittest.main()

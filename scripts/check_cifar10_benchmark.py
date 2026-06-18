#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lcfm.datasets import CIFAR10Problem
from lcfm.cifar_metrics import build_cifar_resnet18, flat_to_uint8_images, make_eval_labels
from lcfm.models import build_model
from lcfm.pairing import apply_pairing, minibatch_ot_pair, pairing_features, pressure_aware_minibatch_ot_pair
from lcfm.utils import set_seed


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    set_seed(0)
    dataset_cfg = {"fake_data": True, "n_train": 16, "n_test": 8, "data_seed": 11, "class_conditional": True}
    problem = CIFAR10Problem(dataset_cfg)
    x0, x1, labels = problem.sample_train_batch(4, torch.device("cpu"))
    assert_true(problem.dim == 3072, "CIFAR dim should be 3072.")
    assert_true(problem.image_shape == (3, 32, 32), "CIFAR image shape should be 3x32x32.")
    assert_true(x0.shape == x1.shape == (4, 3072), "CIFAR batches should be flat image vectors.")
    assert_true(labels.shape == (4,), "Conditional CIFAR batches should include labels.")
    assert_true(labels.dtype == torch.long, "CIFAR labels should be int64.")
    assert_true(x1.dtype == torch.float32, "CIFAR targets should be float32.")
    assert_true(float(x1.min()) >= -1.0 and float(x1.max()) <= 1.0, "CIFAR targets should be in [-1, 1].")

    other = CIFAR10Problem(dataset_cfg)
    assert_true(
        torch.allclose(problem.target_eval(8, torch.device("cpu")), other.target_eval(8, torch.device("cpu"))),
        "Fake CIFAR eval split should be deterministic.",
    )

    model = build_model(
        "unet2d",
        problem.dim,
        {
            "model_kwargs": {
                "image_shape": [3, 32, 32],
                "base_channels": 8,
                "channel_mults": [1, 2],
                "num_res_blocks": 1,
                "time_dim": 32,
                "attention_resolutions": [],
                "num_classes": 10,
            }
        },
    )
    y = model(x0, torch.rand(4, 1), labels)
    assert_true(y.shape == x0.shape, "UNet2D output shape should match input shape.")

    pair_cfg = {
        "pairing_kwargs": {
            "cost_feature": "downsampled_pixels",
            "image_shape": [3, 32, 32],
            "downsample_size": 8,
        }
    }
    features = pairing_features(x1, pair_cfg)
    assert_true(features.shape == (4, 3 * 8 * 8), "Downsampled pairing features should be 8x8 RGB.")
    _, x1_ot = minibatch_ot_pair(x0, x1, pair_cfg)
    assert_true(x1_ot.shape == x1.shape, "Minibatch OT should preserve full image vectors.")
    _, x1_paired, labels_paired = apply_pairing(x0, x1, {"pairing": "minibatch_ot", **pair_cfg}, labels)
    for row, label in zip(x1_paired, labels_paired):
        match = torch.nonzero(torch.all(x1 == row, dim=1), as_tuple=False).flatten()[0]
        assert_true(torch.equal(label, labels[match]), "Paired labels should follow paired target images.")

    pressure_cfg = {
        "pairing_kwargs": {
            "pressure_beta": 0.0,
            "cost_feature": "downsampled_pixels",
            "image_shape": [3, 32, 32],
            "downsample_size": 8,
        }
    }
    _, x1_pressure = pressure_aware_minibatch_ot_pair(x0, x1, pressure_cfg)
    assert_true(torch.allclose(x1_pressure, x1_ot), "pressure_beta=0 should match minibatch OT.")

    eval_labels = make_eval_labels(23, torch.device("cpu"))
    assert_true(eval_labels.tolist()[:12] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1], "Eval labels should cycle through CIFAR classes.")
    uint8_images = flat_to_uint8_images(x1, problem.image_shape)
    assert_true(uint8_images.shape == (4, 3, 32, 32), "Metric image conversion should return image tensors.")
    assert_true(uint8_images.dtype == torch.uint8, "Metric images should be uint8.")
    classifier = build_cifar_resnet18()
    logits = classifier(uint8_images.float() / 255.0)
    assert_true(logits.shape == (4, 10), "CIFAR classifier should return 10 logits.")
    print("cifar10 benchmark checks passed")


if __name__ == "__main__":
    main()

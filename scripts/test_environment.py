# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys
import traceback


def check_packages(package_list):
    """Check basic package imports."""
    success = True
    for package in package_list:
        try:
            _ = importlib.import_module(package)
            print(f"\033[92m[SUCCESS]\033[0m {package} found")
        except Exception as e:
            print(f"\033[91m[ERROR]\033[0m Package not successfully imported: \033[93m{package}\033[0m")
            success = False
    return success


def test_pytorch_detailed():
    """Detailed PyTorch test with version and CUDA information."""
    try:
        import torch

        print("✅ PyTorch version preserved:", torch.__version__)
        print("✅ CUDA available:", torch.cuda.is_available())
        print("✅ CUDA version in PyTorch:", torch.version.cuda)
        return True
    except Exception as e:
        print("❌ PyTorch detailed test failed:", str(e))
        traceback.print_exc()
        return False


def test_transformer_engine_detailed():
    """Detailed Transformer Engine PyTorch import test."""
    try:
        import transformer_engine.pytorch as te  # noqa: F401

        print("✅ Transformer Engine successfully imported with pytorch submodule")
        return True
    except Exception as e:
        print("❌ Transformer Engine detailed test failed:", str(e))
        traceback.print_exc()
        return False


def test_apex_detailed():
    """Detailed APEX import test."""
    try:
        import apex  # noqa: F401

        print("✅ APEX successfully imported")
        return True
    except Exception as e:
        print("❌ APEX detailed test failed:", str(e))
        traceback.print_exc()
        return False


def test_transformers_image_processing_detailed():
    """Detailed transformers image processing auto import test."""
    try:
        from transformers.models.auto import image_processing_auto  # noqa: F401

        print("✅ transformers.models.auto.image_processing_auto imported successfully - no DictValue error")
        return True
    except Exception as e:
        print("❌ transformers image processing detailed test failed:", str(e))
        traceback.print_exc()
        return False


def main():
    """Run all environment verification tests."""
    print("=== COSMOS ENVIRONMENT VERIFICATION ===")

    # Check Python version
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        detected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"\033[91m[ERROR]\033[0m Python 3.10+ is required. You have: \033[93m{detected}\033[0m")
        sys.exit(1)

    print("\n--- Basic Package Import Tests ---")
    packages = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "megatron.core",
        "transformer_engine",
    ]
    packages_training = [
        "apex.multi_tensor_apply",
    ]

    basic_success = check_packages(packages + packages_training)

    print("\n--- Detailed Verification Tests ---")
    detailed_tests = [
        ("PyTorch Details", test_pytorch_detailed),
        ("Transformer Engine", test_transformer_engine_detailed),
        ("APEX", test_apex_detailed),
        ("Transformers Image Processing", test_transformers_image_processing_detailed),
    ]

    detailed_passed = 0
    detailed_failed = 0

    for test_name, test_func in detailed_tests:
        print(f"\n{test_name}:")
        if test_func():
            detailed_passed += 1
        else:
            detailed_failed += 1

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Basic imports: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Detailed tests: ✅ {detailed_passed} passed, ❌ {detailed_failed} failed")

    if basic_success and detailed_failed == 0:
        print("✅ All libraries verified successfully with preserved PyTorch version")
        sys.exit(0)
    else:
        print("❌ Environment verification failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base case for the CPU-backed memory-pool tests."""

import unittest

import torch


class CpuPoolMixin:
    """Pin torch's default device to CPU for the duration of each test.

    Mix into whatever base a test already uses -- `unittest.TestCase` via
    `CpuPoolTestCase` below, or sglang's `CustomTestCase`:

        class TestHostKVCache(CpuPoolMixin, CustomTestCase): ...

    See CpuPoolTestCase for why this is needed.

    Wrapping `run` rather than `setUp`/`tearDown` is deliberate. A subclass that
    defines `setUp` without chaining to `super().setUp()` -- which is common, and
    which several of these tests do -- would silently skip the pin while still
    inheriting the restore, so the restore blows up on state that was never set.
    Overriding `run` puts the pin around setUp, the test body, and tearDown alike,
    and cannot be defeated by a subclass forgetting to call super().
    """

    def run(self, result=None):
        prev = torch.get_default_device()
        torch.set_default_device("cpu")
        try:
            return super().run(result)
        finally:
            torch.set_default_device(prev)


class CpuPoolTestCase(CpuPoolMixin, unittest.TestCase):
    """A test that builds a CPU pool, and therefore needs CPU tensors.

    These tests construct pools on CPU and hand them index tensors. Both sides
    must agree on the device, and a bare `torch.tensor(...)` does NOT guarantee
    CPU: it follows torch's process-wide default device.

    That default is not ours to rely on. pytest imports every module during
    collection, so a module that calls `torch.set_default_device` at import scope
    leaves an accelerator default behind for everything that runs afterwards --
    and one in the tree does exactly that. On a GPU host our index tensors then
    come out on CUDA while the pool is on CPU:

        RuntimeError: indices should be either on cpu or on the same device
                      as the indexed tensor (cpu)

    which looks like a bug in the pool and is not one. It is invisible on a CPU
    laptop and depends on collection order, so it also does not reproduce
    reliably.

    Pin the default for the duration of each test and restore it afterwards. That
    makes these tests hermetic regardless of what else the suite imported, and it
    keeps the fix inside our own tests -- we do not get to reach into somebody
    else's test to make ours pass.
    """

    def setUp(self):
        super().setUp()
        self._prev_default_device = torch.get_default_device()
        torch.set_default_device("cpu")

    def tearDown(self):
        torch.set_default_device(self._prev_default_device)
        super().tearDown()

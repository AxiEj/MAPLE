import pytest
import torch

from maple.solvation.experimental.cuda_execution import (
    cuda_model_execution, prepare_cuda_execution,
)


def test_cuda_context_restores_caller_settings_after_failure(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled(),
              torch._C._get_graph_executor_optimize())
    with pytest.raises(RuntimeError, match='model failure'):
        with cuda_model_execution():
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert not torch._C._get_graph_executor_optimize()
            raise RuntimeError('model failure')
    assert before == (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch._C._get_graph_executor_optimize())


def test_cuda_workspace_is_set_only_before_context_initialization(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    prepare_cuda_execution()
    import os
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


def test_late_cuda_workspace_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(RuntimeError, match='before CUDA initialization'):
        prepare_cuda_execution()


def test_explicit_conflicting_workspace_is_not_overwritten(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    with pytest.raises(RuntimeError, match=':4096:8'):
        prepare_cuda_execution()


def test_runtime_context_does_not_accept_workspace_drift(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(RuntimeError, match='workspace'):
        with cuda_model_execution():
            pytest.fail('runtime drift was silently accepted')

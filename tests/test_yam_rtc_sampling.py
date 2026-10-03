import numpy as np
import pytest
from expo_ft.yam.rtc_sampling import RTCSampler


def fake_sampler():
    sampler = object.__new__(RTCSampler)
    sampler.state = None
    def sample(state, obs, prefix, noise):
        result = np.asarray(noise[0]).copy()
        result[:, :prefix.shape[1]] = prefix
        return result
    sampler._sample = sample
    return sampler


def test_prefix_and_executed_window():
    sampler = fake_sampler()
    noise = np.arange(2*30*32, dtype=np.float32).reshape(1,2,30,32)
    prefix = np.ones((1,5,32), np.float32)
    full, window = sampler.candidates(None, prefix, noise)
    np.testing.assert_array_equal(full[:, :5], np.ones((2,5,32)))
    np.testing.assert_array_equal(window, noise[0, :, 5:13, :14])


@pytest.mark.parametrize('prefix,noise,replan', [
    (np.zeros((1,9,32)),np.zeros((1,2,30,32)),8),
    (np.zeros((1,5,32)),np.zeros((1,2,30,32)),16),
    (np.zeros((1,5,14)),np.zeros((1,2,30,32)),8),
    (np.zeros((1,5,32)),np.zeros((1,2,30,14)),8),
    (np.full((1,5,32),np.nan),np.zeros((1,2,30,32)),8),
])
def test_reject_invalid_schedule_or_data(prefix,noise,replan):
    with pytest.raises(ValueError):
        fake_sampler().candidates(None,prefix,noise,replan)


def test_reject_changed_prefix():
    sampler = fake_sampler()
    sampler._sample = lambda *args: np.zeros((2,30,32))
    with pytest.raises(AssertionError, match='prefix changed'):
        sampler.candidates(None,np.ones((1,5,32)),np.zeros((1,2,30,32)))

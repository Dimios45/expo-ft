import importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

spec=importlib.util.spec_from_file_location('fit_base',Path(__file__).parents[1]/'scripts/yam/fit_base.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_gradient_accumulation_matches_mean_objective():
    p={'w':jnp.array([.3,-.2])}
    x=jnp.array([[1.,2.],[3.,-1.],[-2.,.5],[.1,.2]])
    keys=jax.random.split(jax.random.key(4),4)
    def loss(p,b,k):return ((p['w']*b).sum()-jax.random.normal(k,()))**2
    expected_loss,expected_grad=jax.value_and_grad(lambda p:jax.vmap(lambda b,k:loss(p,b,k))(x,keys).mean())(p)
    value,grad=jax.jit(lambda p:m.mean_gradients(loss,p,x,keys))(p)
    np.testing.assert_allclose(value,expected_loss,rtol=1e-6)
    np.testing.assert_allclose(grad['w'],expected_grad['w'],rtol=1e-6)

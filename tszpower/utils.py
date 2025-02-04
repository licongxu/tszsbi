import jax.numpy as jnp

def get_ell_range():
    # Predefined list of ell values
    ell_eval = jnp.array([10., 13.5, 18., 23.5, 30.5, 40., 52.5, 68.5, 89.5, 
                         117., 152.5, 198., 257.5, 335.5, 436.5, 567.5, 738., 959.5])

    return ell_eval

def simpson(y, *, x=None, dx=1.0, axis=-1):
    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    if x is not None:
        x = jnp.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd
        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)
            h = jnp.asarray([dx, dx], dtype=jnp.float64)
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))
                diffs = jnp.float64(jnp.diff(x, axis=axis))
                h = [jnp.squeeze(diffs[hm2], axis=axis),
                     jnp.squeeze(diffs[hm1], axis=axis)]
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = jnp.true_divide(
                num,
                den,
            )
            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = jnp.true_divide(
                num,
                den,
            )
            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = jnp.true_divide(
                num,
                den,
            )
            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]
        result += val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result
def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))
    if x is None:  # Even-spaced Simpson's rule.
        result = jnp.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = jnp.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float)
        h1 = h[sl1].astype(float)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = jnp.true_divide(h0, h1)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - jnp.true_divide(1.0, h0divh1)) +
                          y[slice1] * (hsum *
                                       jnp.true_divide(hsum, hprod)) +
                          y[slice2] * (2.0 - h0divh1))
        result = jnp.sum(tmp, axis=axis)
    return result
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)
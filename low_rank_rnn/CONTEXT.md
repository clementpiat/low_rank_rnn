## RNN

Neurons:

* internal state: $x(t) = (x_1(t),...,x_N (t))$ # (N, N)
* input: $u(t) = (u_1(t),...,u_P(t))$ # (N, P)
* scalar output: $z(t)$

Connectivity:

* recurrent connectivity: $J_{i,j}$ # (N, N)
* input connectivity: $I_i$ # (P, N)
* output connectivity: $w_i$ # (N, 1)

Dynamics:

$$\tau \frac{dx}{dt} = x(t) + J\phi(x(t)) + Iu(t)$$

With typically $\phi = \tanh$, and which gives immediately the following discrete formula (if $dt = \tau$):

$$x(t+1) = J\phi(x(t)) + Iu(t)$$

## Low-rank RNN

$$J = \sum_{i=1}^{R} m_i n_i^\top$$

Then J has only 2RN parameters


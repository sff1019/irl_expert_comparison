class GaussianMLPPolicy():
    def __init__(self,
                 env_spec,
                 hidden_size=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 name='GaussianMLPPolicy'):
        self.obs_dim = env_sepc.observation_space.flat_dim
        self.action_dim = env_sepc.action_space.flat_dim
        self.model = GaussianMLPModel(
            in_dim=self.obs_dim,
            out_dim=self.action_dim,
            hidden_dims=hidden_size,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization)
        )

    def forward(self, obs):
        dist = self.model(obs)
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))

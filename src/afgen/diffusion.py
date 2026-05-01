import torch
from math import sqrt

from boltz.model.modules.utils import (
    center_random_augmentation,
    default,
)
from boltz.model.loss.diffusion import weighted_rigid_align
from boltz.model.modules.diffusion import OutTokenFeatUpdate
from boltz.model.modules.diffusion import AtomDiffusion as BoltzAtomDiffusion


class AtomDiffusion(BoltzAtomDiffusion):
    def __init__(
        self,
        accumulate_token_repr_seq=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.accumulate_token_repr_seq = accumulate_token_repr_seq
        if self.accumulate_token_repr_seq:
            self.out_token_feat_update_seq = OutTokenFeatUpdate(
                sigma_data=kwargs['sigma_data'],
                token_s=kwargs['score_model_args']["token_s"],
                dim_fourier=kwargs['score_model_args']["dim_fourier"],
            )

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        train_accumulate_token_repr_seq=False,
        **network_condition_kwargs,
    ):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr_conf = None
        token_repr_seq = None
        token_a = None

        # gradually denoise
        for sigma_tm, sigma_t, gamma in sigmas_and_gammas:
            atom_coords, atom_coords_denoised = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
                return_second_coords=True,
                second_coords=atom_coords_denoised,
            )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised, token_a = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )

            if self.accumulate_token_repr:
                if token_repr_conf is None:
                    token_repr_conf = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr_conf = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr_conf, next_a=token_a
                    )
            
            if self.accumulate_token_repr_seq:
                if token_repr_seq is None:
                    token_repr_seq = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr_seq):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr_seq = self.out_token_feat_update_seq(
                        times=self.c_noise(sigma), acc_a=token_repr_seq, next_a=token_a
                    )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        return dict(
            sample_atom_coords=atom_coords,
            diff_token_repr=token_repr_conf,
            diff_token_repr_sequence=token_repr_seq)



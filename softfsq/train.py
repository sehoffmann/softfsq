import argparse

import dmlcloud as dml
import torch
import torch._dynamo
from omegaconf import OmegaConf
from torch.amp.grad_scaler import OptState
from torch.profiler import record_function

from .callbacks import SaveImageCallback
from .datasets import Imagenet
from .losses import CodebookLoss, DiscriminatorLoss, VQGANLoss
from .metrics import Perplexity


class VQStage(dml.Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(SaveImageCallback())

    def _build_datasets(self):
        self.train_ds = Imagenet.with_default_transforms('/scratch_local/datasets/ImageNet2012', split='train')
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_ds, shuffle=True, seed=self.config.seed
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        self.val_ds = Imagenet.with_default_transforms(
            '/scratch_local/datasets/ImageNet2012', split='val', random_crop=False, cache=True
        )
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_ds, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size // 2,
            sampler=self.val_sampler,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )

    def _build_model(self):
        model = dml.obj_from_cfg(
            self.config.model,
            input_dim=3,
            output_dim=3,
            codebook_dim=self.config.codebook_dim,
            codebook_size=self.config.codebook_size,
        )

        if self.config.model_checkpoint:
            dml.info(f'Initializing model from {self.config.model_checkpoint}')
            state_dict = torch.load(self.config.model_checkpoint, map_location='cpu')
            model.load_state_dict(state_dict)

        if self.config.compile:
            model.compile()
        self.model = dml.wrap_ddp(model, device=self.device, find_unused_parameters=False)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )  # taming uses betas=(0.5, 0.9)

        if self.config.rampup:
            self.rampup = torch.optim.lr_scheduler.LinearLR(
                self.optim, 1 / 1000, total_iters=self.config.rampup, verbose=True
            )
        else:

            self.rampup = None

        if self.config.cosine_annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, int(self.config.epochs * 1.2), 1e-6)
        else:
            self.scheduler = None

        # to prevent null pointer exceptions if not trainign with GAN
        self.disc_rampup = None
        self.disc_scheduler = None

        self.scaler = torch.amp.GradScaler(device=self.device)

    def _build_loss(self):
        pixel_loss = dml.obj_from_cfg(self.config.pixel_loss)

        if self.config.perceptual_loss:
            perceptual_loss = dml.obj_from_cfg(self.config.perceptual_loss)
        else:
            perceptual_loss = None

        if self.config.adv_loss:
            adv_loss = dml.obj_from_cfg(self.config.adv_loss)
        else:
            adv_loss = None

        codebook_loss = CodebookLoss(
            commitment_weight=self.config.loss_weights.commitment,
            codebook_weight=self.config.loss_weights.codebook,
            z_l2_weight=self.config.loss_weights.z_l2_reg,
        )

        self.loss_fn = VQGANLoss(
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            codebook_loss=codebook_loss,
            adv_loss=adv_loss,
            pixel_weight=self.config.loss_weights.pixel,
            perceptual_weight=self.config.loss_weights.perceptual,
            adv_weight=self.config.loss_weights.adv,
            adv_start_step=self.config.adv_start_step,
        )
        self.loss_fn.to(self.device)

        if self.discriminator:
            disc_loss = dml.obj_from_cfg(self.config.disc_loss)
            self.disc_loss = DiscriminatorLoss(disc_loss, self.config.adv_start_step)
            self.disc_loss.to(self.device)

    def _build_discriminator(self):
        if not self.config.discriminator:
            self.discriminator = None
        else:
            discriminator = dml.obj_from_cfg(self.config.discriminator)
            if self.config.compile:
                discriminator.compile()
            self.discriminator = dml.wrap_ddp(discriminator, device=self.device, find_unused_parameters=False)

            self.disc_optim = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )

            if self.config.rampup:
                self.disc_rampup = torch.optim.lr_scheduler.LinearLR(
                    self.disc_optim, 1 / 1000, total_iters=self.config.rampup
                )
            else:
                self.disc_rampup = None

            if self.config.cosine_annealing:
                self.disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.disc_optim, int(self.config.epochs * 1.2), 1e-6
                )
            else:
                self.disc_scheduler = None

    def _setup_metrics(self):
        self.perplexity_metric = Perplexity(compute_on_cpu=True)
        self.add_metric('train/codebook_perplexity', self.perplexity_metric)

        self.add_column('Total', 'train/total_loss', color='dim')
        self.add_column('Recon. (train)', 'train/recon_loss', color='green')
        self.add_column('Recon. (val)', 'val/recon_loss', color='green')
        self.add_column('MSE (val)', 'val/mse', color='green')
        self.add_column('Embedding', 'train/embedding_loss', color='cyan')

        if self.discriminator:
            self.add_column('Dyn. weight', 'train/dyn_adv_weight', color='magenta')
            self.add_column('G', 'train/adv_loss', color='blue')
            self.add_column('D', 'train/d_loss', color='blue')
            self.add_column('Real Acc.', 'train/real_acc', color='blue', formatter=lambda x: f'{100*x:.1f}%')
            self.add_column('Fake Acc.', 'train/fake_acc', color='blue', formatter=lambda x: f'{100*x:.1f}%')

        self.add_column(
            'Perplexity',
            'train/codebook_perplexity',
            color='red',
            formatter=lambda x: f'{x:.1f}',
        )

    def pre_stage(self):
        dml.seed(self.config.seed)

        self._build_datasets()
        self._build_model()
        self._build_discriminator()
        self._build_loss()

        self._setup_metrics()

        if self.config.profile:
            self.enable_profiler()

    def run_epoch(self):
        self.train()
        self.validate()

    @torch.no_grad()
    def post_epoch(self):
        if self.scheduler is not None:
            self.scheduler.step()

        if self.disc_scheduler is not None:
            self.disc_scheduler.step()

        if dml.is_root() and self.run_dir:
            torch.save(self.model.module.state_dict(), self.run_dir / 'latest.pth')

    @torch.no_grad()
    def _log_generator_metrics(self, x, pred, z, z_q, indices):
        if pred.dim() == 5:  # ensemble forecast: B, E, C, H, W
            eval_pred = pred.mean(dim=1)  # B, C, H, W
        else:
            eval_pred = pred

        self.log('mse', torch.nn.functional.mse_loss(eval_pred, x))
        self.log('mae', torch.nn.functional.l1_loss(eval_pred, x))

        if indices is not None:
            self.perplexity_metric.update(indices)

    @torch.no_grad()
    def _log_discriminator_metrics(self, loss, logits_real, logits_fake):
        self.log('d_loss', loss)
        self.log('real_acc', (logits_real > 0).float().mean())
        self.log('fake_acc', (logits_fake < 0).float().mean())

    def _generator_step(self, x):
        self.optim.zero_grad()

        with record_function('forward'), torch.amp.autocast(self.device.type):
            out, z, z_q, indices = self.model(x, swapout=self.config.get('swapout', 0))

            with record_function('discriminator_forward'):
                if self.discriminator and self.global_step >= self.config.adv_start_step:
                    logits_fake = self.discriminator(out)
                else:
                    logits_fake = None

            total_loss = self.loss_fn(
                model=self.model.module,
                pred=out,
                target=x,
                global_step=self.global_step,
                z=z,
                z_q=z_q,
                indices=indices,
                logits_fake=logits_fake,
            )

        with record_function('backward'):
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optim)
            if self.config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

        with record_function('optimization'):
            self.scaler.step(self.optim)
            was_stepped = self.scaler._per_optimizer_states[id(self.optim)]['stage'] is OptState.STEPPED
            if was_stepped and self.rampup is not None:
                self.rampup.step()

        self._log_generator_metrics(x, out, z, z_q, indices)

        return out, indices, total_loss

    def _discriminator_step(self, x, fakes):
        self.disc_optim.zero_grad()

        with torch.amp.autocast(self.device.type):
            with record_function('forward_real'):
                logits_real = self.discriminator(x)
            with record_function('forward_fake'):
                logits_fake = self.discriminator(fakes.detach())
            loss = self.disc_loss(
                logits_real=logits_real,
                logits_fake=logits_fake,
                global_step=self.global_step,
            )

        with record_function('backward'):
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.disc_optim)
            if self.config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.clip_grad_norm)

        with record_function('optimization'):
            self.scaler.step(self.disc_optim)
            was_stepped = self.scaler._per_optimizer_states[id(self.disc_optim)]['stage'] is OptState.STEPPED
            if was_stepped and self.disc_rampup is not None:
                self.disc_rampup.step()

        self._log_discriminator_metrics(loss, logits_real, logits_fake)

        return loss

    def train(self):
        self.metric_prefix = 'train'
        self.train_sampler.set_epoch(self.current_epoch)

        for i, (x, _) in enumerate(self.train_loader):
            if self.config.steps_per_epoch and i >= self.config.steps_per_epoch:
                break

            # callbacks etc. might change the model state, so really make sure we're in training mode
            self.model.train()
            if self.discriminator:
                self.discriminator.train()

            x = x.to(self.device, non_blocking=True)

            with record_function('generator_step'):
                out, indices, total_loss = self._generator_step(x)

            with record_function('discriminator_step'):
                if self.discriminator and self.global_step >= self.config.adv_start_step:
                    disc_loss = self._discriminator_step(x, out)

            self.scaler.update()

            self.log('misc/samples', len(x), reduction='sum', prefixed=False)
            if self.has_profiler:
                self.profiler.step()

            self.finish_step()

        self.log('misc/lr', self.optim.param_groups[0]['lr'], prefixed=False)

    @torch.no_grad()
    def validate(self):
        self.metric_prefix = 'val'
        for i, (x, _) in enumerate(self.val_loader):
            if self.config.steps_per_epoch and i >= self.config.steps_per_epoch:
                break

            # callbacks etc. might change the model state, so really make sure we're in eval mode
            self.model.eval()
            if self.discriminator:
                self.discriminator.eval()

            x = x.to(self.device)
            out, z, z_q, indices = self.model(x)

            self.loss_fn(
                model=self.model.module,
                pred=out,
                target=x,
                global_step=self.global_step,
                z=z,
                z_q=z_q,
                logits_fake=None,
            )  # logs internally
            self._log_generator_metrics(x, out, z, z_q, indices)


def train_vqgan():
    dml.init()

    torch._dynamo.config.optimize_ddp = 'ddp_optimizer'
    torch.set_float32_matmul_precision('high')  # TensorFloat32 format

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='+', help='Path to the configuration file')
    args = parser.parse_args()

    configs = [OmegaConf.load(c) for c in args.config]
    config = OmegaConf.merge(*configs)
    config.lr = dml.scale_lr(config.base_lr * config.batch_size)

    pipe = dml.Pipeline(config, name=config.get('name'))
    pipe.append(VQStage(epochs=config.epochs))

    if config.wandb:
        pipe.enable_wandb(project='vqgan')

    if config.checkpointing:
        pipe.enable_checkpointing('checkpoints')

    # Run the training
    pipe.run()


if __name__ == '__main__':
    train_vqgan()

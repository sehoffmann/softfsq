import dmlcloud as dml
import torch
import torch.nn as nn


class CodebookLoss(nn.Module):
    def __init__(self, commitment_weight, codebook_weight, z_l2_weight):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.z_l2_weight = z_l2_weight

    def forward(self, z, z_q):
        commitment = nn.functional.mse_loss(z, z_q.detach())
        embedding = nn.functional.mse_loss(z.detach(), z_q)
        l2_reg = torch.mean(z.pow(2))

        dml.log_metric('embedding_loss', embedding)
        dml.log_metric('z_l2_reg', l2_reg)

        loss = self.commitment_weight * commitment + self.codebook_weight * embedding + self.z_l2_weight * l2_reg
        return loss


def calc_dynamic_adv_weight(recon_loss, adv_loss, last_layer_weights):
    # has shape of last_layer_weights
    recon_grads = torch.autograd.grad(recon_loss, last_layer_weights, retain_graph=True)[0]
    adv_grads = torch.autograd.grad(adv_loss, last_layer_weights, retain_graph=True)[0]

    with torch.no_grad():
        weight = torch.norm(recon_grads) / (torch.norm(adv_grads) + 1e-4)
        weight = torch.clamp(weight, 1e-2, 10)

    return weight


def gan_schedule(global_step, start_step, rampup=500):
    return 1.0


class VQGANLoss(nn.Module):
    def __init__(
        self,
        pixel_loss,
        codebook_loss,
        perceptual_loss=None,
        adv_loss=None,
        pixel_weight=1.0,
        perceptual_weight=1.0,
        adv_weight=1.0,
        adv_start_step=0,
    ):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.codebook_loss = codebook_loss
        self.perceptual_loss = perceptual_loss
        self.adv_loss = adv_loss
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight
        self.adv_start_step = adv_start_step

    def forward(self, model, pred, target, global_step, z=None, z_q=None, logits_fake=None, indices=None):
        total_loss = 0.0

        # Pixel loss
        pixel = self.pixel_loss(pred, target)
        dml.log_metric('pixel_loss', pixel)

        # Perceptual loss
        if self.perceptual_loss:
            perceptual = self.perceptual_loss(pred, target)
            dml.log_metric('perceptual_loss', perceptual)
            reconstruction = self.pixel_weight * pixel + self.perceptual_weight * perceptual
        else:
            reconstruction = self.pixel_weight * pixel

        total_loss += reconstruction
        dml.log_metric('recon_loss', reconstruction)

        # Codebook loss
        if z is not None and z_q is not None:
            codebook = self.codebook_loss(z, z_q)  # CodebookLoss does more fine-grained logging
            total_loss += codebook

        # Adversarial loss
        if self.adv_loss and logits_fake is not None:
            adv = self.adv_loss(logits_fake)
            schedule_weight = gan_schedule(global_step, self.adv_start_step)
            if schedule_weight > 0:  # small optimization
                dyn_weight = calc_dynamic_adv_weight(reconstruction, adv, model.last_layer_weights)
                total_loss += schedule_weight * dyn_weight * self.adv_weight * adv

            dml.log_metric('adv_loss', adv)
            dml.log_metric('gan_schedule', schedule_weight)
            dml.log_metric('dyn_adv_weight', dyn_weight)
            dml.log_metric('dyn_adv_weight_max', dyn_weight, reduction='max')

        dml.log_metric('total_loss', total_loss)
        return total_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, adv_loss, adv_start_step=0):
        super().__init__()
        self.adv_loss = adv_loss
        self.adv_start_step = adv_start_step

    def forward(self, logits_real, logits_fake, global_step):
        schedule_weight = gan_schedule(global_step, self.adv_start_step)
        loss = self.adv_loss(logits_real, logits_fake)
        return schedule_weight * loss

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import get_contrastive_loss

logger = logging.getLogger(__name__)


class ViMixin:
    def vi_get_posteriori(self, inputs_hidden, inputs_bow):
        if self.config['vi_asin_bow']:
            inputs_bow += 1e-5 
            inputs_bow /= inputs_bow.sum(dim=-1, keepdim=True)  # Normalize
            hidden = F.gelu(self.post_bow(inputs_bow))
        else:
            hidden = inputs_hidden
        hidden = self.dropout(hidden)
        post_mu, post_logvar = self.post_mu(hidden), self.post_logvar(hidden)  # [bsz, num_topics]
        kl_post_prior = 0.5 * torch.sum(post_mu.pow(2) + post_logvar.exp() - post_logvar - 1, dim=-1)
        return post_mu, post_logvar, kl_post_prior

    @staticmethod
    def vi_sample_posteriori(mu, logvar, sampling=1):
        """ Reparameterization. """
        bsz, num_topics = mu.size()
        samples = torch.randn(bsz, sampling, num_topics, device=mu.device)

        std = torch.exp(logvar / 2)
        return samples.mul_(std.unsqueeze(1)).add_(mu.unsqueeze(1))

    def vi_get_likelihood(self, els, post_mu, post_logvar, sampling=1):
        topic_embs = self.dropout(F.normalize(self.emb_topics.weight, p=2, dim=-1, eps=1e-8))
        distr_topic2el = F.softmax(torch.matmul(topic_embs, els.t()) * 10, dim=-1)

        distr_input2topic = F.softmax(self.vi_sample_posteriori(post_mu, post_logvar, sampling=sampling), dim=-1)
        el_likelihood = torch.matmul(distr_input2topic, distr_topic2el).mean(dim=1)
        el_likelihood += torch.full_like(el_likelihood, fill_value=1e-5)
        return el_likelihood, (distr_input2topic, distr_topic2el)

    def vi_forward(self, inputs_hidden, inputs_bow, els, el_labels, sampling=1):
        post_mu, post_logvar, kl = self.vi_get_posteriori(inputs_hidden, inputs_bow)
        el_likelihood, _ = self.vi_get_likelihood(els, post_mu, post_logvar, sampling=sampling)
        nll = -el_likelihood.log() * el_labels
        loss = nll.sum(dim=-1) + (0 if self.config['vi_no_kl'] else kl)  # [bsz]
        return loss.mean(dim=-1)

    def diverse_topic_loss(self):
        topic_embs = self.dropout(F.normalize(self.emb_topics.weight, p=2, dim=-1, eps=1e-8))  # [num_topics, hidden]
        avg_sim = torch.matmul(topic_embs, topic_embs.t()).mean()
        loss = nn.MSELoss(reduction='mean')(avg_sim, torch.ones_like(avg_sim) * -1)
        return loss

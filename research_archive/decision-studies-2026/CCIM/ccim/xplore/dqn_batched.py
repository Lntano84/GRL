"""Batched encoder + batched update: one forward per branch instead of 600 per sample.

Reuses the authors' parameters and modules; only the *batching* is new.  ``_official/`` is untouched.

What had to change, and why
---------------------------
1. **Normalisation.**  ``GcnEncoderGraph.apply_bn`` (dqn.py:84) builds ``nn.BatchNorm1d(x.size()[1])``
   on an ``x`` of shape ``(batch, num_nodes, features)``.  In the published loop ``batch == 1``, so the
   statistics are taken over the **feature axis of each node individually**, and because the module is
   constructed inside the call no running statistics ever accumulate.  Padding to a common ``M`` and
   calling the same thing would take statistics over ``(batch, features)`` and let graphs contaminate each
   other, so :meth:`BatchedSoftPoolingGcnEncoder.apply_bn` computes the same per-(graph, node) quantity
   directly.  For a padded row (all zeros) this yields exactly zero, not NaN.
2. **The max at dqn.py:245** reduces over **original nodes**, so padded nodes must be excluded.  The
   corrected body fills them with ``-inf`` before the reduction.
3. **The max at dqn.py:278** reduces over the **clusters produced by DiffPool**, whose count is
   ``assign_dim``, not ``M``.  The original-node mask must NOT be applied there -- doing so would be a
   shape error at best and would silently drop real cluster nodes at worst.  It is left unmasked, which is
   also what the authors' code does because it sets ``embedding_mask = None`` for every pooling round
   after the first (dqn.py:252-255).
4. **Softmax rows for padding.**  A row of all ``-inf`` logits produces NaN, so padding rows are *not*
   forced to ``-inf`` before the softmax.  The softmax runs normally and the padding rows of the
   assignment matrix are zeroed afterwards, before any matmul -- which is exactly the order the authors
   already use (softmax at dqn.py:257, masking at dqn.py:262).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from rl_alg import utils as utls
from rl_alg.dqn import SoftPoolingGcnEncoder


class BatchedSoftPoolingGcnEncoder(SoftPoolingGcnEncoder):
    """The authors' encoder with a batch dimension that does not leak between graphs."""

    def apply_bn(self, x):
        """Per-(graph, node) normalisation over the feature axis, matching ``BatchNorm1d(m)`` with N=1."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)

    def forward(self, x, adj, batch_num_nodes=None, compute_loss=False, **kwargs):
        x_a = kwargs.get("assign_x", x)
        self.link_loss = torch.zeros(1).to(self.device)
        self.entropy_loss = torch.zeros(1).to(self.device)
        max_num_nodes = adj.size()[1]
        node_mask = (None if batch_num_nodes is None
                     else self.construct_mask(max_num_nodes, batch_num_nodes))   # (B, M, 1)

        out_all = []
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block,
                                            self.conv_last, node_mask)
        self.node_embeddings = embedding_tensor.clone()

        # over ORIGINAL nodes -> exclude padding (change 2)
        masked = embedding_tensor if node_mask is None else embedding_tensor.masked_fill(
            node_mask == 0, float("-inf"))
        out, _ = torch.max(masked, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out_all.append(torch.sum(embedding_tensor, dim=1))

        for i in range(self.num_pooling):
            embedding_mask = (self.construct_mask(max_num_nodes, batch_num_nodes)
                              if (batch_num_nodes is not None and i == 0) else None)
            self.assign_tensor = F.softmax(self.gcn_forward(
                x_a, adj, self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                self.assign_conv_last_modules[i], embedding_mask), -1)
            # softmax first, THEN zero the padding rows (change 4)
            if node_mask is not None:
                self.assign_tensor = self.assign_tensor * node_mask
            if compute_loss:
                self.link_loss += self.loss(adj)
                self.entropy_loss -= ((1 / adj.size()[-2])
                                      * torch.sum(self.assign_tensor
                                                  * torch.log(self.assign_tensor + 1e-12)))

            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj, self.conv_first_after_pool[i],
                                               self.conv_block_after_pool[i],
                                               self.conv_last_after_pool[i])
            # over CLUSTERS -> no original-node mask (change 3)
            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out_all.append(torch.sum(embedding_tensor, dim=1))

        return torch.cat(out_all, dim=1) if self.concat else out


def pad_batch(entries, device="cpu"):
    """``entries``: list of ``(x, adj)`` with x ``(1, m, F)`` and adj ``(1, m, m)`` -> padded batch."""
    sizes = [int(e[1].shape[1]) for e in entries]
    m_max = max(sizes)
    n_feat = entries[0][0].shape[2]
    x = torch.zeros(len(entries), m_max, n_feat)
    adj = torch.zeros(len(entries), m_max, m_max)
    for b, (xb, ab) in enumerate(entries):
        m = sizes[b]
        x[b, :m, :] = xb[0]
        adj[b, :m, :m] = ab[0]
    return x.to(device), adj.to(device), sizes


def swap_in_batched_encoder(acmodel):
    """Give a model the batched encoder while keeping the authors' trained parameters.

    The encoder is **deep-copied and its class swapped** rather than re-constructed: the parent class does
    not retain its constructor arguments as attributes, and a deep copy guarantees that every parameter,
    buffer and submodule is byte-identical to the one it replaces, which is what makes the comparison
    below a test of the batching alone.
    """
    import copy
    for holder in (acmodel.actor_critic, acmodel.target_actor_critic):
        old = holder.graph_embedder
        new = copy.deepcopy(old)
        new.__class__ = BatchedSoftPoolingGcnEncoder
        new.to(acmodel.device)
        holder.graph_embedder = new
    return acmodel


def batched_embeddings(holder, xs, adjs, sizes, node_indices=None, node_dim=None):
    """One encoder forward for the whole batch; returns ``(state_embed, per-sample action_embed)``."""
    out = holder.graph_embedder.forward(xs, adjs, batch_num_nodes=sizes, compute_loss=False)
    state_embed = out[:, -holder.graph_embedding_dim:]
    action_embed = None
    if node_indices is not None:
        ne = holder.graph_embedder.node_embeddings          # (B, M, F)
        action_embed = torch.cat([ne[b, int(node_indices[b]), :node_dim].reshape(1, -1)
                                  for b in range(len(sizes))], dim=0)
    return state_embed, action_embed


def gradient_update_sarsa_batched(acmodel, batch_size: int = 100):
    """The published update with one batched encoder forward per branch instead of one per sample."""
    s, a, r, s1, a1, _ano = acmodel.replay.sample_(batch_size)
    sa = [np.array(x[0], dtype=np.float32) for x in s]
    sb = [np.array(x[1], dtype=np.float32) for x in s]
    s1a = [np.array(x[0], dtype=np.float32) for x in s1]
    s1b = [np.array(x[1], dtype=np.float32) for x in s1]
    a1t = [np.array(x, dtype=np.float32) for x in a1]
    at = torch.from_numpy(np.array(a, dtype=np.float32)).to(acmodel.device)
    rt = torch.from_numpy(np.array(r, dtype=np.float32)).to(acmodel.device)

    xs, adjs, sizes = pad_batch([(torch.from_numpy(v).reshape((1,) + v.shape),
                                  torch.from_numpy(w).reshape((1,) + w.shape))
                                 for v, w in zip(sa, sb)], acmodel.device)
    xs1, adjs1, sizes1 = pad_batch([(torch.from_numpy(v).reshape((1,) + v.shape),
                                     torch.from_numpy(w).reshape((1,) + w.shape))
                                    for v, w in zip(s1a, s1b)], acmodel.device)
    del sizes1
    a1_stack = torch.from_numpy(np.stack(a1t)).to(acmodel.device)   # (B, D), matching cat(state, action)

    with torch.no_grad():
        st1, _ = batched_embeddings(acmodel.target_actor_critic, xs1, adjs1, sizes)
        q_next1, q_next2 = acmodel.target_actor_critic.critic(st1, a1_stack)
    q_next = torch.min(q_next1, q_next2)
    q_expected = rt.unsqueeze(1) + acmodel.gamma * q_next

    st, _ = batched_embeddings(acmodel.actor_critic, xs, adjs, sizes)
    q_predicted1, q_predicted2 = acmodel.actor_critic.critic(st, at)

    acmodel.q_next1, acmodel.q_next2, acmodel.q_next = q_next1, q_next2, q_next
    acmodel.q_expected = q_expected
    acmodel.r = rt
    acmodel.q_predicted1, acmodel.q_predicted2 = q_predicted1, q_predicted2
    acmodel.loss_critic = (F.mse_loss(q_predicted1, q_expected)
                           + F.mse_loss(q_predicted2, q_expected))
    acmodel.critic_opt.zero_grad()
    acmodel.loss_critic.backward()
    acmodel.critic_opt.step()
    utls.copy_parameters(acmodel.target_actor_critic, acmodel.actor_critic, acmodel.eta)

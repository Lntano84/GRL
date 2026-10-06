"""Optimised update: remove duplicated graph-representation forwards and unused auxiliary losses.

Separate from the authors' snapshot
-----------------------------------
``_official/rl_alg/dqn.py`` is left byte-for-byte as published.  This module provides an alternative
update, named separately, so the original stays available for comparison.

What is removed, and why it should be equivalent
------------------------------------------------
1. **Duplicated graph-representation forwards.**  ``GraphTD3.critic_forward1`` and ``critic_forward2``
   (dqn.py:496-520) each call ``get_embeddings``, which runs the whole DiffPool embedder.  The published
   update calls both for the target branch, and calls ``GraphTD3.forward`` **twice** for the online branch
   -- and ``forward`` (dqn.py:545) itself runs the embedder twice (once in ``actor_forward``, whose output
   is discarded, and once in ``critic_forward``, which already returns *both* Q values).  Per sample that
   is 2 target + 4 online = **6 embedder forwards**, i.e. **600 per batch of 100**.  Sharing one embedding
   per sample and reading both Q heads from it gives **2 per sample, 200 per batch**.
2. **Unused auxiliary losses.**  ``SoftPoolingGcnEncoder.forward`` with ``compute_loss=True``
   (dqn.py:264-266) accumulates ``link_loss`` and ``entropy_loss``.  The training loss (dqn.py:655) is only
   ``mse(q1) + mse(q2)``, and nothing ever reads those two attributes or backwards through them, so they
   cannot affect the Q values or the gradients along the Q path.

Both claims are checked empirically by ``scripts/equiv_check.py`` against the published update on a fixed
replay batch: Q values, loss, parameter gradients, parameters after stepping, and action selection after
consecutive updates.  If any of those differ, this is **not** an engineering-equivalent speed-up and must be
reported as a new implementation version.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Variable

from rl_alg import utils as utls
from rl_alg.dqn import GraphTD3


def embeddings(acmodel, nodes_attr, adj, nodes=None, compute_loss: bool = False):
    """``GraphTD3.get_embeddings`` (dqn.py:522-541), with ``compute_loss`` exposed.

    The published method always calls the embedder with its default ``compute_loss=True``; the only change
    here is passing the flag through.  Everything else -- the slicing of ``state_embed``, the per-sample
    gather of ``node_embeddings`` -- is identical.
    """
    graph_embedder = acmodel.graph_embedder
    state_embed = graph_embedder.forward(nodes_attr, adj, compute_loss=compute_loss)[
        :, -acmodel.graph_embedding_dim:]
    action_embed = None
    if nodes is not None:
        action_embeds = [graph_embedder.node_embeddings[i:i + 1, nodes[i], :acmodel.node_embedding_dim]
                         for i in range(len(nodes))]
        action_embed = torch.cat(action_embeds, dim=0)
    return state_embed, action_embed


def gradient_update_sarsa_fast(acmodel, batch_size: int = 100, compute_loss: bool = False,
                               share_embeddings: bool = True):
    """The published ``gradient_update_sarsa`` (dqn.py:631-663) with duplicated work removed.

    ``share_embeddings=False`` restores the published call pattern while still allowing
    ``compute_loss`` to be toggled, so the two changes can be attributed separately.
    """
    s, a, r, s1, a1, _ano = acmodel.replay.sample_(batch_size)
    sa = [np.array(x[0], dtype=np.float32) for x in s]
    sb = [np.array(x[1], dtype=np.float32) for x in s]
    at = Variable(torch.from_numpy(np.array(a, dtype=np.float32)).to(acmodel.device))
    rt = Variable(torch.from_numpy(np.array(r, dtype=np.float32)).to(acmodel.device))
    s1a = [np.array(x[0], dtype=np.float32) for x in s1]
    s1b = [np.array(x[1], dtype=np.float32) for x in s1]
    a1t = [np.array(x, dtype=np.float32) for x in a1]

    if share_embeddings:
        ta = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in sa]
        tb = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in sb]
        n1a = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in s1a]
        n1b = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in s1b]
        a1v = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in a1t]

        # ---- target branch: one embedding per sample, both Q heads from it
        nxt1, nxt2 = [], []
        for i in range(len(n1a)):
            st, act = embeddings(acmodel.target_actor_critic, n1a[i], n1b[i], nodes=None,
                                 compute_loss=compute_loss)
            q1, q2 = acmodel.target_actor_critic.critic(st, a1v[i])
            nxt1.append(q1.detach())
            nxt2.append(q2.detach())
        q_next1 = torch.squeeze(torch.cat(nxt1, dim=0))
        q_next2 = torch.squeeze(torch.cat(nxt2, dim=0))
        q_next = torch.min(q_next1, q_next2).detach()
        q_expected = rt + acmodel.gamma * q_next
        acmodel.q_next1, acmodel.q_next2, acmodel.q_next = q_next1, q_next2, q_next
        acmodel.q_expected, acmodel.r = q_expected, rt

        # ---- online branch: one critic pass per sample, actor never invoked
        cur1, cur2 = [], []
        for i in range(len(ta)):
            st, _ = embeddings(acmodel.actor_critic, ta[i], tb[i], nodes=None,
                               compute_loss=compute_loss)
            q1, q2 = acmodel.actor_critic.critic(st, at[i:i + 1])
            cur1.append(q1)
            cur2.append(q2)
        q_predicted1 = torch.squeeze(torch.cat(cur1, dim=0))
        q_predicted2 = torch.squeeze(torch.cat(cur2, dim=0))
    else:
        sa = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in sa]
        sb = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in sb]
        a1v = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in a1t]
        s1a = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in s1a]
        s1b = [Variable(torch.from_numpy(x.reshape((1,) + x.shape)).to(acmodel.device)) for x in s1b]
        q_next1 = torch.squeeze(torch.cat([
            GraphTD3.critic_forward1(
                acmodel.target_actor_critic, x[0], x[1], x[2], adjust_dim=False,
                convert_torch=False, node_labels=False).detach() for x in zip(s1a, s1b, a1v)], dim=0))
        q_next2 = torch.squeeze(torch.cat([
            GraphTD3.critic_forward2(
                acmodel.target_actor_critic, x[0], x[1], x[2], adjust_dim=False,
                convert_torch=False, node_labels=False).detach() for x in zip(s1a, s1b, a1v)], dim=0))
        q_next = torch.min(q_next1, q_next2).detach()
        q_expected = rt + acmodel.gamma * q_next
        acmodel.q_next1, acmodel.q_next2, acmodel.q_next = q_next1, q_next2, q_next
        acmodel.q_expected, acmodel.r = q_expected, rt
        q_predicted1 = torch.squeeze(torch.cat([
            GraphTD3.forward(
                acmodel.actor_critic, x[0], x[1], x[2].view((1,) + x[2].shape), adjust_dim=False,
                convert_torch=False, node_labels=False)[1] for x in zip(sa, sb, at)], dim=0))
        q_predicted2 = torch.squeeze(torch.cat([
            GraphTD3.forward(
                acmodel.actor_critic, x[0], x[1], x[2].view((1,) + x[2].shape), adjust_dim=False,
                convert_torch=False, node_labels=False)[2] for x in zip(sa, sb, at)], dim=0))

    acmodel.q_predicted1, acmodel.q_predicted2 = q_predicted1, q_predicted2
    acmodel.loss_critic = (torch.nn.functional.mse_loss(q_predicted1, q_expected)
                           + torch.nn.functional.mse_loss(q_predicted2, q_expected))
    acmodel.critic_opt.zero_grad()
    acmodel.loss_critic.backward()
    acmodel.critic_opt.step()
    utls.copy_parameters(acmodel.target_actor_critic, acmodel.actor_critic, acmodel.eta)

"""Passive operational metrics; never feed these evaluation fields back into search."""
import numpy as np
from grid2op.Reward import AlertReward, L2RPNSandBoxScore, _AlertTrustScore, _NewRenewableSourcesUsageScore
from grid2op.utils.underlying_statistics import EpisodeStatistics

PREFIX=EpisodeStatistics.KEY_SCORE


def operational_other_rewards():
    # Preserve the native alert auxiliary reward, add exactly the score-helper component names.
    return {"alert":AlertReward,
            f"{PREFIX}_grid_operational_cost":L2RPNSandBoxScore,
            f"{PREFIX}_assistant_confidence":_AlertTrustScore,
            f"{PREFIX}_new_renewable_sources_usage":_NewRenewableSourcesUsageScore}


def public_cost_ledger(obs,env,previous_curtailed_mw):
    """Reconcile the installed reward using current/previous public observations.

    Grid2Op 1.12.5 L2RPNSandBoxScore uses -env._sum_curtailment_mw,
    which baseEnv computes as the change in total curtailment, not its level.
    Retain the physical curtailment level separately; do not confuse this reward
    convention with total curtailed energy or feed the ledger back into actions.
    """
    dt=float(env.delta_time_seconds)/3600
    marginal=float(np.max(env.gen_cost_per_MW[np.asarray(obs.gen_p)>0]))
    losses=(float(np.sum(obs.gen_p,dtype=np.float64))-float(np.sum(obs.load_p,dtype=np.float64)))*dt
    redisp=float(np.abs(obs.actual_dispatch).sum(dtype=np.float64))*dt
    current_curtailed_mw=float(obs.curtailment_mw.sum(dtype=np.float64))
    curtailed=current_curtailed_mw*dt
    delta_curtailed=(current_curtailed_mw-float(previous_curtailed_mw))*dt
    storage=float(np.abs(obs.storage_power).sum(dtype=np.float64))*dt
    return {"marginal_cost":marginal,"losses_mwh":losses,"redispatch_mwh":redisp,
            "curtailed_mwh":curtailed,"previous_curtailed_mw":float(previous_curtailed_mw),
            "current_curtailed_mw":current_curtailed_mw,"curtailment_delta_mwh":delta_curtailed,
            "storage_throughput_mwh":storage,
            "recomputed_raw_cost":marginal*(losses+redisp+delta_curtailed+storage)}


def cost_rounding_tolerance(obs,env):
    # The official reward sums backend values in float32; the independent ledger uses
    # public float32 observations, then float64 accumulation. Bound this conversion/
    # pairwise-summation scale before any formal smoke result is available.
    magnitude=sum(float(np.abs(getattr(obs,k)).sum(dtype=np.float64)) for k in
                  ["gen_p","load_p","actual_dispatch","curtailment_mw","storage_power"])
    price=float(np.max(env.gen_cost_per_MW[np.asarray(obs.gen_p)>0]))
    return max(0.002,16*np.finfo(np.float32).eps*magnitude*float(env.delta_time_seconds)/3600*price)

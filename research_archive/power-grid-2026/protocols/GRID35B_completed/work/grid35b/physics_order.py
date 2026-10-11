"""Nonlearned geometric tail from GRID21 public_prior, without the PPO prefix."""
import numpy as np
def order_from_public_edges(edges,edge_rho,n_sub,eligible,subids,aliases):
    assert edges.shape[0]==2 and edges.shape[1]==2*len(edge_rho)
    live=edges[:,:len(edge_rho)]%n_sub
    top=sorted(range(len(edge_rho)),key=lambda l:(-float(edge_rho[l]),l))[:3]
    roots=sorted({int(s) for l in top for s in live[:,l]})
    distance=np.full(n_sub,n_sub+1,dtype=int);pressure=np.zeros(n_sub);neighbor=[[] for _ in range(n_sub)]
    queue=roots.copy()
    for s in roots:distance[s]=0
    for l in range(len(edge_rho)):
        u,v=(int(x) for x in live[:,l]);neighbor[u].append(v);neighbor[v].append(u)
        pressure[u]=max(pressure[u],float(edge_rho[l]));pressure[v]=max(pressure[v],float(edge_rho[l]))
    for s in queue:
        for t in neighbor[s]:
            if distance[t]>distance[s]+1:distance[t]=distance[s]+1;queue.append(t)
    tail=sorted(eligible,key=lambda i:(int(distance[subids[i]]),-float(pressure[subids[i]]),i))
    order=[];seen=set()
    for i in tail:
        rep=aliases[i]
        if rep not in seen:seen.add(rep);order.append(rep)
    assert set(order)==set(aliases.values())
    return order

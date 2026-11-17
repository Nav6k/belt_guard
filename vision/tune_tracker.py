"""Monte Carlo check of the loop tracker's confirm / resolve rules.    python -m vision.tune_tracker

Per-frame rates measured on held-out images: a planted defect is detected ~90 % of the time, an intact
frame is (wrongly) flagged ~31 % of the time. Each run plants 4 defects on the loop (one by the splice,
one repaired at lap 20, one appearing at lap 10) plus 15 random intact frames per lap, for 40 laps.
Reported per config (tpr, fpr, CONFIRM_N, CONFIRM_M, RESOLVE_AFTER, rel_tol):
  fake = false defects confirmed per run, missing = share of real defects not active at the end,
  repair = share of runs where the repaired defect wasn't resolved.
"""
import random
import vision.tracker as T
from vision.tracker import LoopTracker, Observation
L=10000
def run(tpr,fpr,N,M,R,rel,seed,laps=40):
    T.CONFIRM_N,T.CONFIRM_M,T.RESOLVE_AFTER=N,M,R
    rng=random.Random(seed); tr=LoopTracker(L,rel_tol=rel)
    planted=[9996,5809,2594,4703]; first_conf={}; false_res=0
    for lap in range(laps):
        obs=[];obsc=[]
        for k,p in enumerate(planted):
            if k==0 and lap>=20: continue  # repaired
            if k==3 and lap<10: continue
            m=(p+rng.gauss(0,.5+.0003*p))%L
            if rng.random()<.08: obsc.append(m); continue
            if rng.random()<tpr: obs.append(Observation(lap,0,m,'damage',.9,.3,''))
        for _ in range(15):
            m=rng.uniform(0,L)
            if rng.random()<.08: obsc.append(m); continue
            if rng.random()<fpr: obs.append(Observation(lap,0,m,'damage',.7,.1,''))
        tr.update_lap(lap,obs,obsc)
    real=[t for t in tr.tracks if t.status!='candidate' and min(tr.circ_dist(t.position_m,p) for p in planted)<30]
    fake=[t for t in tr.tracks if t.status in('confirmed','growing','resolved') and t not in real]
    act={round(t.position_m) for t in real if t.status!='resolved'}
    # expect: 5809,2594,4703 active; 9996 resolved
    missing=sum(not any(tr.circ_dist(a,p)<30 for a in act) for p in planted[1:])
    wrong0=any(tr.circ_dist(t.position_m,9996)<30 and t.status!='resolved' for t in real)
    return len(fake),missing,wrong0
for cfg in [(.90,.31,2,3,3,.002),(.71,.21,2,3,3,.002),(.71,.21,2,3,5,.001),(.71,.21,3,4,5,.001),(.71,.21,3,5,5,.001),(.90,.31,3,4,5,.001)]:
    r=[run(*cfg,s) for s in range(200)]
    print(cfg,'fake confirmed/run %.2f'%(sum(x[0] for x in r)/200),' real defect not active %.3f'%(sum(x[1] for x in r)/600),' repair not resolved %.2f'%(sum(x[2] for x in r)/200))

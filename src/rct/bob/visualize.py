from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def _line(xs,ys,title,path):
    fig=plt.figure(); ax=fig.add_subplot(111); ax.plot(xs,ys); ax.set_title(title); fig.savefig(path); plt.close(fig)

def render_trajectory(records,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); xs=[r.get('step',i) for i,r in enumerate(records)]; paths=[]
    specs=[('loss','Task / viability proxy','viability.png'),('retained','Retained route fraction','corrective_access_retained.png'),('reachable','Reachable route fraction','corrective_access_reachable.png'),('active_edges','Active edge count','active_edge_count.png'),('probe_spend','Probe spend','probe_spend.png'),('abstain_rate','ABSTAIN rate','abstain_rate.png'),('prediction_residual','Predicted minus realized return','valuation_residual.png')]
    for key,title,name in specs:
        _line(xs,[r.get(key,float('nan')) for r in records],title,out/name); paths.append(out/name)
    fig=plt.figure(); ax=fig.add_subplot(111)
    for j in range(38): ax.plot(xs,[((r.get('eta') or [float('nan')]*38)[j]) for r in records])
    ax.set_title('Eta coefficient trajectories'); p=out/'eta_coefficients.png'; fig.savefig(p); plt.close(fig); paths.append(p)
    kinds=sorted({k for r in records for k in (r.get('action_exposure') or {}).keys()})
    vals=[(records[-1].get('action_exposure') or {}).get(k,0) for k in kinds] if records else []
    fig=plt.figure(); ax=fig.add_subplot(111); ax.bar(kinds,vals); ax.set_title('Action-class exposure'); p=out/'action_class_exposure.png'; fig.savefig(p); plt.close(fig); paths.append(p)
    return tuple(paths)

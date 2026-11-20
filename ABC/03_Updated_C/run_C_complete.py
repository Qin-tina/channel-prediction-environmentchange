#!/usr/bin/env python3
"""Member C ONLY. B's original model and predictions remain untouched.
Python 3.10+, numpy, pandas, matplotlib, scipy, joblib, scikit-learn.
Run from any directory: python run_C_only.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
B = ROOT/'02_Original_B'/'Member_B_Deliverables'
A = ROOT/'01_Original_A'/'Member_A_Deliverables'/'outputs'
OUT = HERE/'results'
OUT.mkdir(exist_ok=True)
WINDOW, HOLD, QUANTILE = 16, 12, 0.99
N_VALIDATION_TRACES, N_STRESS_SEEDS = 8, 10
CAL_SEED = 71342

# Fallback decisions at t only depend on residuals from t-1 and earlier.
def sequential_fallback(actual, linear, last, threshold, window=WINDOW, hold=HOLD):
    actual, linear, last = map(lambda x: np.asarray(x,dtype=float),(actual,linear,last))
    if not (len(actual)==len(linear)==len(last)):
        raise ValueError('Input arrays must have identical lengths')
    n=len(actual); prediction=np.empty(n); chosen=np.zeros(n,dtype=bool)
    events=np.zeros(n,dtype=bool); decision_stat=np.full(n,np.nan)
    history=[]; remaining=0
    for i in range(n):
        # BEFORE seeing actual[i] decide based on past prediction errors.
        if remaining==0 and len(history)>=window:
            score=float(np.mean(np.square(history[-window:])))
            decision_stat[i]=score
            if score > threshold:
                remaining=hold; events[i]=True
        if remaining>0:
            prediction[i]=last[i]; chosen[i]=True; remaining-=1
        else:
            prediction[i]=linear[i]
        # Only after choosing, the already committed fixed-model residual is observed.
        history.append(float(actual[i]-linear[i]))
    return prediction,chosen,events,decision_stat

def snr_trace(rho, rng, n=1000, mean=10., innovation_sd=2.):
    # Explicitly C-generated *supplemental* traces; B's files are not regenerated.
    x=np.empty(n);x[0]=mean
    for i in range(1,n):
        x[i]=mean+rho*(x[i-1]-mean)+np.sqrt(1-rho*rho)*rng.normal(0,innovation_sd)
    return x

def predict_trace(model, trace):
    return trace[1:], trace[:-1],model.predict(trace[:-1].reshape(-1,1))

def rmse(y,p):return float(np.sqrt(np.mean(np.square(y-p))))

def metrics(name, y, last, fixed, hybrid, mode, events):
    return dict(scenario=name, method='last_value', rmse_db=rmse(y,last), fallback_share=np.nan, trigger_count=np.nan) if mode=='last' else None

model=joblib.load(B/'fixed_linear_model.pkl')
raw=pd.read_csv(B/'environment_B_predictions.csv')
needed=['actual_snr','last_value_prediction','linear_prediction']
if not set(needed).issubset(raw.columns):raise ValueError('Missing B prediction columns')
y=raw.actual_snr.to_numpy();last=raw.last_value_prediction.to_numpy();fixed=raw.linear_prediction.to_numpy()
# Verify B's fixed saved predictor agrees with B's *provided* predictions exactly.
verified=model.predict(last.reshape(-1,1))
max_delta=float(np.max(np.abs(verified-fixed)))
if max_delta>1e-7:raise AssertionError(f'Provided B model/predictions mismatch {max_delta}')

# Separate calibration data: generated ONLY for C. Frozen B model; no fitting anywhere.
# No Environment B observation is used to tune the detection threshold.
cal_rng=np.random.default_rng(CAL_SEED)
scores=[]
for k in range(N_VALIDATION_TRACES):
    yy,ll,ff=predict_trace(model,snr_trace(.95,cal_rng))
    resid=(yy-ff)**2
    scores.extend([np.mean(resid[j-WINDOW:j]) for j in range(WINDOW,len(resid))])
threshold=float(np.quantile(scores,QUANTILE))

hybrid,active,events,score=sequential_fallback(y,fixed,last,threshold)
res=raw.copy();res['member_C_hybrid_prediction']=hybrid
res['member_C_fallback_active']=active.astype(int)
res['member_C_trigger_event']=events.astype(int)
res['member_C_past_residual_mse']=score
res.to_csv(OUT/'C_predictions_on_B_original.csv',index=False)

rows=[]
def addrow(scenario, method, y,p, fallback_share=None, trigger_count=None, seed=None):
    rows.append(dict(scenario=scenario,seed=seed,method=method,rmse_db=rmse(y,p),
                     fallback_share=fallback_share,trigger_count=trigger_count))
for name,p in [('Last-value',last),('B fixed linear (unaltered)',fixed),('C residual fallback',hybrid)]:
    addrow('B_original_environment_B',name,y,p,float(active.mean()) if 'fallback' in name else None,int(events.sum()) if 'fallback' in name else None)

# A dataset diagnostics: frozen B model, NO retraining, no apples-to-apples claim.
for split in ['test_A','test_B']:
    frame=pd.read_csv(A/f'{split}.csv')
    yy=frame.target_true_snr_next_db.to_numpy()
    ll=frame.snr_lag0_db.to_numpy()
    ff=model.predict(ll.reshape(-1,1))
    hh=np.empty(len(frame)); aa=np.zeros(len(frame),bool);ee=np.zeros(len(frame),bool)
    # Reset on EACH independent sequence; no cross-sequence state leakage.
    for _,indices in frame.groupby('sequence_id',sort=False).indices.items():
        pp,active0,events0,_=sequential_fallback(yy[indices],ff[indices],ll[indices],threshold)
        hh[indices],aa[indices],ee[indices]=pp,active0,events0
    df=frame[['environment','sequence_id','time_index']].copy()
    df['actual_true_snr_next_db']=yy
    df['last_value_from_measured_snr_db']=ll
    df['B_frozen_linear_on_A_data_db']=ff
    df['C_hybrid_on_A_data_db']=hh
    df['fallback_active']=aa.astype(int)
    df.to_csv(OUT/f'C_diagnostic_using_A_{split}.csv',index=False)
    for name,p in [('Last-value',ll),('B fixed linear (unaltered)',ff),('C residual fallback',hh)]:
        addrow(f'A_original_{split}_diagnostic_NOT_same_model',name,yy,p,float(aa.mean()) if 'fallback' in name else None,int(ee.sum()) if 'fallback' in name else None)

# Supplemental stress test using newly generated traces, same frozen B model.
for rho in [.95,.80,.60,.40]:
    for seed in range(N_STRESS_SEEDS):
        rng=np.random.default_rng(81000+seed)
        yy,ll,ff=predict_trace(model,snr_trace(rho,rng))
        hh,aa,ee,_=sequential_fallback(yy,ff,ll,threshold)
        for name,p in [('Last-value',ll),('B fixed linear (unaltered)',ff),('C residual fallback',hh)]:
            addrow(f'C_supplementary_rho_{rho:.2f}',name,yy,p,float(aa.mean()) if 'fallback' in name else None,int(ee.sum()) if 'fallback' in name else None,seed)

details=pd.DataFrame(rows);details.to_csv(OUT/'C_metrics_all.csv',index=False)
stress=details[details.seed.notna()]
grp=stress.groupby(['scenario','method'],sort=False).rmse_db
summary=grp.agg(['mean','std','count']).reset_index()
summary['ci95_halfwidth']=tdist.ppf(.975,summary['count']-1)*summary['std']/np.sqrt(summary['count'])
summary.to_csv(OUT/'C_supplementary_95ci.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(12,4.3))
labels=['Last-value','B fixed linear (unaltered)','C residual fallback']
sub=details[details.scenario=='B_original_environment_B'].set_index('method')
axes[0].bar(range(3),[sub.loc[v,'rmse_db'] for v in labels],color=['#a6a6a6','#2673b8','#e2a33a'])
axes[0].set_xticks(range(3),['Last-value','B linear','C fallback'])
axes[0].set_ylabel('RMSE (dB)');axes[0].set_title('B original Environment B file')
start=50;stop=min(180,len(raw))
axes[1].plot(range(start,stop),y[start:stop],label='Actual SNR',color='#252525',lw=1.3)
axes[1].plot(range(start,stop),fixed[start:stop],label='B linear',lw=1,alpha=.8)
axes[1].plot(range(start,stop),hybrid[start:stop],label='C fallback',lw=1.0,alpha=.8)
axes[1].set_xlabel('Time index');axes[1].set_ylabel('SNR (dB)');axes[1].set_title('Predictions on B original file');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'C_figure1_B_original.png',dpi=200);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.4))
for name in labels:
    chunk=summary[summary.method==name].sort_values('scenario',ascending=False)
    rhos=[float(s.split('_')[-1]) for s in chunk.scenario]
    ax.errorbar(rhos,chunk['mean'],yerr=chunk['ci95_halfwidth'],marker='o',capsize=3,label=name)
ax.set_xlabel('Test correlation rho');ax.set_ylabel('RMSE (dB)');ax.set_title('C-generated supplemental traces (10 independent seeds)')
ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'C_figure2_supplemental.png',dpi=200);plt.close(fig)

with (OUT/'C_result_summary.txt').open('w') as f:
    f.write('UNCHANGED B MODEL / B CSV — C results only\n')
    f.write(f'B model max prediction discrepancy: {max_delta:.3g}\n')
    f.write(f'C calibration: rho=0.95, 8 independently generated traces; window={WINDOW}, hold={HOLD}, q={QUANTILE}\n')
    f.write(f'Threshold MSE: {threshold:.8f} dB^2\n')
    f.write(f'Fallback active time fraction: {active.mean():.6f}; trigger events: {events.sum()}\n')
    f.write(sub[['rmse_db']].to_string()+'\n')
    f.write('Do not claim packet goodput: D has not supplied shared rate table.\n')
print((OUT/'C_result_summary.txt').read_text())

# --- Member C extension (NEW): performance-aware residual-triggered gate ---
# No changes to any A/B source file or fitted model. Original basic method above is retained.
IMPROVEMENT_MARGIN=0.05  # pre-declared 5% squared-error margin; sensitivity is evaluated separately

def performance_aware_fallback(actual, fixed, last, threshold, window=16, margin=.05):
    """Causal gate: decisions at index i use only outcomes indexed < i.

    Both predictors are independently available each slot. Comparing their prior
    errors is a prequential evaluation, NOT future-target leakage.
    """
    actual,fixed,last=map(lambda z:np.asarray(z,dtype=float),(actual,fixed,last))
    if not(len(actual)==len(fixed)==len(last)):raise ValueError('Length mismatch')
    n=len(actual);out=fixed.copy();active=np.zeros(n,bool);event=np.zeros(n,bool)
    score=np.full(n,np.nan); advantage=np.full(n,np.nan)
    for i in range(window,n):
        sl=slice(i-window,i)
        e_fixed=float(np.mean((actual[sl]-fixed[sl])**2))
        e_last=float(np.mean((actual[sl]-last[sl])**2))
        score[i]=e_fixed
        advantage[i]=(e_fixed-e_last)/max(e_fixed,1e-12)
        active[i]=e_fixed>threshold and e_last<(1-margin)*e_fixed
        if active[i]:out[i]=last[i]
        if active[i] and not active[i-1]:event[i]=True
    return out,active,event,score,advantage

# 1) Exact B-provided file: never regenerate, refit, or replace.
aware,aware_active,aware_events,aware_scores,aware_advantage=performance_aware_fallback(
    y,fixed,last,threshold,WINDOW,IMPROVEMENT_MARGIN)
# Keep prior complete outputs, save an additional four-way comparison for D.
expanded=raw.copy()
expanded['C_basic_fallback_prediction']=hybrid
expanded['C_basic_active']=active.astype(int)
expanded['C_basic_trigger_event']=events.astype(int)
expanded['C_performance_aware_prediction']=aware
expanded['C_performance_aware_active']=aware_active.astype(int)
expanded['C_performance_aware_trigger_event']=aware_events.astype(int)
expanded['C_past_linear_mse']=aware_scores
expanded['C_past_relative_last_value_advantage']=aware_advantage
expanded.to_csv(OUT/'C_B_original_four_method_predictions.csv',index=False)

newrows=[]
def extended_row(scenario,method,truth,pred,act=None,event=None,seed=None):
    newrows.append(dict(scenario=scenario,seed=seed,method=method,
        rmse_db=rmse(truth,pred),fallback_share=float(act.mean()) if act is not None else np.nan,
        trigger_events=int(event.sum()) if event is not None else np.nan,
        samples=len(truth)))
for name,pred,act,ev in [('Last-value',last,None,None),('B fixed linear',fixed,None,None),
                         ('C basic fallback',hybrid,active,events),
                         ('C performance-aware fallback',aware,aware_active,aware_events)]:
    extended_row('B_original_environment_B',name,y,pred,act,ev)

# 2) A original environment A/B traces, used ONLY as out-of-source diagnostic.
# Reset sequential states for every independent sequence, never connect traces.
for split in ['test_A','test_B']:
    frame=pd.read_csv(A/f'{split}.csv')
    yy=frame.target_true_snr_next_db.to_numpy()
    ll=frame.snr_lag0_db.to_numpy()
    ff=model.predict(ll.reshape(-1,1))
    p_basic=np.empty(len(frame));a_basic=np.zeros(len(frame),bool);ev_basic=np.zeros(len(frame),bool)
    p_aware=np.empty(len(frame));a_aware=np.zeros(len(frame),bool);ev_aware=np.zeros(len(frame),bool)
    for _,idx in frame.groupby('sequence_id',sort=False).indices.items():
        p_basic[idx],a_basic[idx],ev_basic[idx],_=sequential_fallback(yy[idx],ff[idx],ll[idx],threshold)
        p_aware[idx],a_aware[idx],ev_aware[idx],_,_=performance_aware_fallback(
            yy[idx],ff[idx],ll[idx],threshold,WINDOW,IMPROVEMENT_MARGIN)
        for name,pred,act,ev in [('Last-value',ll[idx],None,None),('B fixed linear',ff[idx],None,None),
                            ('C basic fallback',p_basic[idx],a_basic[idx],ev_basic[idx]),
                            ('C performance-aware fallback',p_aware[idx],a_aware[idx],ev_aware[idx])]:
            extended_row(f'A_original_{split}_diagnostic',name,yy[idx],pred,act,ev,str(frame.sequence_id.iloc[idx[0]]))
    export=frame[['environment','sequence_id','time_index']].copy()
    export['actual_true_snr_next_db']=yy
    export['last_value_prediction_db']=ll
    export['B_frozen_linear_prediction_db']=ff
    export['C_basic_prediction_db']=p_basic
    export['C_performance_aware_prediction_db']=p_aware
    export['basic_active']=a_basic.astype(int)
    export['aware_active']=a_aware.astype(int)
    export.to_csv(OUT/f'C_A_{split}_four_method_diagnostic.csv',index=False)

# 3) C-generated sensitivity traces, 10 independent seeds * 4 correlations
# + 2 additional stress innovations, all predictions from frozen B model.
for rho in [.95,.80,.60,.40]:
    for seed in range(10):
        for sigma in [2.,3.5]:
            rng=np.random.default_rng(81000+seed+int(sigma*10000))
            yy,ll,ff=predict_trace(model,snr_trace(rho,rng,innovation_sd=sigma))
            pb,ab,eb,_=sequential_fallback(yy,ff,ll,threshold)
            pa,aa,ea,_,_=performance_aware_fallback(yy,ff,ll,threshold,WINDOW,IMPROVEMENT_MARGIN)
            for name,pred,act,ev in [('Last-value',ll,None,None),('B fixed linear',ff,None,None),
                         ('C basic fallback',pb,ab,eb),('C performance-aware fallback',pa,aa,ea)]:
                extended_row(f'C_supplemental_rho_{rho:.2f}_sigma_{sigma:.1f}',name,yy,pred,act,ev,seed)

result_df=pd.DataFrame(newrows)
result_df.to_csv(OUT/'C_four_method_run_metrics.csv',index=False)
agg=result_df[result_df.seed.notna()].groupby(['scenario','method']).agg(
    mean_rmse=('rmse_db','mean'),std_rmse=('rmse_db','std'),n=('rmse_db','size'),
    mean_fallback_share=('fallback_share','mean'),mean_trigger_events=('trigger_events','mean')).reset_index()
agg['ci95_halfwidth']=tdist.ppf(.975,agg.n-1)*agg.std_rmse/np.sqrt(agg.n)
agg.to_csv(OUT/'C_four_method_95ci.csv',index=False)

# Sensitivity: window and threshold quantile are fixed on C-only calibration.
# The sensitivity file reports what-if experiments; do NOT select best using B test.
sensitivity=[]
for w in [8,16,32]:
    for margin in [0.,.05,.10,.20]:
        pr,mask,ev,_,_=performance_aware_fallback(y,fixed,last,threshold,w,margin)
        sensitivity.append(dict(window=w,margin=margin,rmse_db=rmse(y,pr),fallback_share=mask.mean(),events=ev.sum()))
pd.DataFrame(sensitivity).to_csv(OUT/'C_sensitivity_DESCRIPTIVE_NOT_TUNING.csv',index=False)

# Visualizations
fig,ax=plt.subplots(figsize=(9,4.5))
methods=['Last-value','B fixed linear','C basic fallback','C performance-aware fallback']
subset=result_df[result_df.scenario=='B_original_environment_B'].set_index('method')
values=[subset.loc[m,'rmse_db'] for m in methods]
ax.bar(range(4),values,color=['#777777','#337aaf','#cf863b','#267b5c'])
ax.set_xticks(range(4),['Last-value','Fixed linear','Basic fallback','Aware fallback'])
ax.set_ylabel('RMSE (dB)');ax.set_ylim(0,max(values)*1.19)
for i,v in enumerate(values):ax.text(i,v+.015,f'{v:.4f}',ha='center',fontsize=10)
ax.set_title('B original Environment B — same targets and original predictors')
fig.tight_layout();fig.savefig(OUT/'C_figure3_four_method_B_original.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for j,sigma in enumerate([2.,3.5]):
    ax=axes[j]
    for method in methods:
        sub=agg[(agg.method==method)&(agg.scenario.str.endswith(f'sigma_{sigma:.1f}'))]
        xs=[float(v.split('_rho_')[1].split('_')[0]) for v in sub.scenario]
        order=np.argsort(xs)
        ax.errorbar(np.asarray(xs)[order],sub.mean_rmse.to_numpy()[order],
          yerr=sub.ci95_halfwidth.to_numpy()[order],marker='o',capsize=2,label=method)
    ax.set_title(f'Innovation std = {sigma:.1f} dB')
    ax.set_xlabel('AR(1) correlation ρ');ax.set_ylabel('RMSE (dB)');ax.grid(alpha=.22)
axes[0].legend(fontsize=7)
fig.suptitle('C-only stress tests, 10 paired independent seeds per scenario')
fig.tight_layout();fig.savefig(OUT/'C_figure4_four_method_stress_95ci.png',dpi=180);plt.close(fig)

with (OUT/'C_four_method_summary.txt').open('w',encoding='utf-8') as f:
    f.write('B ORIGINAL FILE / FROZEN MODEL: FOUR-WAY COMPARISON\n')
    f.write(f'Original model prediction max absolute mismatch = {max_delta:.12g}\n')
    f.write(f'C-only calibration threshold = {threshold:.8f} dB^2, quantile {QUANTILE}\n')
    f.write(f'Window = {WINDOW}, basic hold = {HOLD}, aware margin = {IMPROVEMENT_MARGIN:.0%}\n')
    f.write(subset[['rmse_db','fallback_share','trigger_events']].to_string()+'\n')
    f.write('B original file is one trace: do not claim independent-run confidence intervals for this comparison.\n')
    f.write('A tests are diagnostics ONLY; A/B have inconsistent parameters and should not be pooled.\n')
    f.write('No packet goodput (Member D rate-selection table absent).\n')
print('\n', (OUT/'C_four_method_summary.txt').read_text())

"""EEEN3008J Part D: reproduce communication evaluation from the supplied ABC outputs.
No fitting, no tuning on the test set. All rates/thresholds are illustrative.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RATES=np.array([1.,2.,4.,6.,8.])
THRESH=np.array([0.,5.,10.,15.,20.])
SLOT_S=0.010
PAYLOAD_BITS=10000  # 1250-byte packet, exact integral number per slot for each rate
METHODS={'Last-value':'last','B fixed linear':'linear','C basic fallback':'basic','C performance-aware fallback':'aware'}

def evaluate(df, actual, cols, basic_active, aware_active, scenario, run_id):
    out=[]
    truth=df[actual].to_numpy(dtype=float)
    for name,col in cols.items():
        pred=df[col].to_numpy(dtype=float)
        valid=np.isfinite(truth)&np.isfinite(pred)
        y=truth[valid]; p=pred[valid]
        ix=np.clip(np.searchsorted(THRESH,p,side='right')-1,0,len(RATES)-1)
        rate=RATES[ix]; required=THRESH[ix]
        success=y>=required
        # Each 10 ms slot sends floor(rate*1e6*slot / payload_bits) whole payload packets.
        packets=(rate*1e6*SLOT_S/PAYLOAD_BITS).astype(int)
        sent=int(packets.sum()); delivered=int(packets[success].sum())
        duration=len(y)*SLOT_S
        goodput=delivered*PAYLOAD_BITS/duration/1e6
        active_col=basic_active if name=='C basic fallback' else aware_active if name=='C performance-aware fallback' else None
        freq=float(df.loc[valid,active_col].mean()*100) if active_col else np.nan
        out.append(dict(scenario=scenario,run_id=run_id,method=name,n_slots=len(y),rmse_db=np.sqrt(np.mean((y-p)**2)),goodput_mbps=goodput,success_slot_pct=success.mean()*100,packet_delivery_pct=delivered/sent*100 if sent else np.nan,selected_rate_mbps=rate.mean(),fallback_active_pct=freq,sent_packets=sent,delivered_packets=delivered))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--abc-root',default='ABC')
    ap.add_argument('--out',default='.')
    args=ap.parse_args()
    root=Path(args.abc_root)
    out=Path(args.out); (out/'results').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(parents=True,exist_ok=True)
    c=root/'03_Updated_C'/'results'
    original=pd.read_csv(c/'C_B_original_four_method_predictions.csv')
    records=evaluate(original,'actual_snr',{'Last-value':'last_value_prediction','B fixed linear':'linear_prediction','C basic fallback':'C_basic_fallback_prediction','C performance-aware fallback':'C_performance_aware_prediction'},'C_basic_active','C_performance_aware_active','B_original_single_trace','B_original')
    # Supplemental A generator's independent sequences are diagnostics only, NOT same distribution as B original.
    for tag,filename in [('A_generator_test_A_DIAGNOSTIC','C_A_test_A_four_method_diagnostic.csv'),('A_generator_test_B_DIAGNOSTIC','C_A_test_B_four_method_diagnostic.csv')]:
        df=pd.read_csv(c/filename)
        for seq,part in df.groupby('sequence_id',sort=True):
            records.extend(evaluate(part,'actual_true_snr_next_db',{'Last-value':'last_value_prediction_db','B fixed linear':'B_frozen_linear_prediction_db','C basic fallback':'C_basic_prediction_db','C performance-aware fallback':'C_performance_aware_prediction_db'},'basic_active','aware_active',tag,seq))
    metrics=pd.DataFrame(records)
    metrics.to_csv(out/'results'/'part_d_run_metrics.csv',index=False)
    summary=metrics.groupby(['scenario','method'],sort=False).agg(n_runs=('run_id','size'),mean_goodput_mbps=('goodput_mbps','mean'),std_goodput_mbps=('goodput_mbps','std'),mean_rmse_db=('rmse_db','mean'),mean_delivery_pct=('packet_delivery_pct','mean'),mean_fallback_pct=('fallback_active_pct','mean')).reset_index()
    # t 95% intervals for independent sequences only; B original single trace gets NaN.
    from scipy.stats import t
    summary['ci95_goodput_halfwidth']=summary.apply(lambda r: t.ppf(.975,r.n_runs-1)*r.std_goodput_mbps/np.sqrt(r.n_runs) if r.n_runs>1 else np.nan,axis=1)
    summary.to_csv(out/'results'/'part_d_summary.csv',index=False)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    names=list(METHODS)
    colors=['#4978b9','#d99837','#ad6c9f','#399a83']
    base=metrics[metrics.scenario=='B_original_single_trace'].set_index('method').loc[names]
    fig,ax=plt.subplots(figsize=(9,4.6))
    bars=ax.bar(range(4),base.goodput_mbps,color=colors)
    ax.set_xticks(range(4),['Last-value','Fixed linear','Basic fallback','Performance-aware'],rotation=12)
    ax.set_ylabel('Packet goodput (Mbps)');ax.set_title('B original test trace: packet goodput (n=1 trace)')
    for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.03,f'{b.get_height():.3f}',ha='center',fontsize=9)
    ax.set_ylim(0,max(base.goodput_mbps)*1.2);fig.tight_layout();fig.savefig(out/'figures'/'D_fig1_B_original_goodput.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.bar(range(4),base.rmse_db,color=colors);ax.set_xticks(range(4),['Last-value','Fixed linear','Basic fallback','Performance-aware'],rotation=12)
    ax.set_ylabel('RMSE (dB)');ax.set_title('B original test trace: prediction RMSE (n=1 trace)')
    ax.set_ylim(0,max(base.rmse_db)*1.23);fig.tight_layout();fig.savefig(out/'figures'/'D_fig2_B_original_rmse.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4))
    vals=base.loc[['C basic fallback','C performance-aware fallback'],'fallback_active_pct']
    ax.bar(['Basic fallback','Performance-aware'],vals,color=[colors[2],colors[3]])
    ax.set_ylim(0,110);ax.set_ylabel('Fallback-active slots (%)');ax.set_title('B original test trace: fallback use')
    for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.2f}%',ha='center')
    fig.tight_layout();fig.savefig(out/'figures'/'D_fig3_B_original_fallback.png',dpi=180);plt.close(fig)
    diag=summary[summary.scenario.str.contains('DIAGNOSTIC')]
    fig,ax=plt.subplots(figsize=(9,4.8))
    labels=['A generator: test A','A generator: test B']
    x=np.arange(2);width=.19
    for j,name in enumerate(names):
        ss=diag[diag.method==name].set_index('scenario').loc[['A_generator_test_A_DIAGNOSTIC','A_generator_test_B_DIAGNOSTIC']]
        ax.bar(x+(j-1.5)*width,ss.mean_goodput_mbps,width,yerr=ss.ci95_goodput_halfwidth,capsize=3,label=name,color=colors[j])
    ax.set_xticks(x,labels);ax.set_ylabel('Packet goodput (Mbps)');ax.set_title('A-generator diagnostics: mean of 12 independent traces (95% CI)')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(out/'figures'/'D_fig4_A_diagnostics_goodput.png',dpi=180);plt.close(fig)
    with open(out/'results'/'method_notes.txt','w') as f:
        f.write('MODEL: 10 ms saturated slots; 1250-byte payload; rates [1,2,4,6,8] Mbps; required SNR [0,5,10,15,20] dB; deterministic hard-threshold packet success; no PHY/MAC overhead or retries.\n')
        f.write('All rates and thresholds are illustrative, not standards-derived. B original test has one trace: NO 95% CI. A generator test A/B are separate diagnostic domain, NOT comparable as matched before/after of B original.\n')
        f.write('Packet goodput counts payload bits successfully delivered divided by total slot duration.\n')
        f.write('C performance-aware fallback is identical to B fixed linear on B original, because it never activates.\n')
if __name__=='__main__':main()

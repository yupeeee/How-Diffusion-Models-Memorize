from pathlib import Path
import json, warnings, argparse
import numpy as np
import pandas as pd
import torch
from scipy.integrate import quad
from scipy.special import logsumexp
from utils.experiments.cache import GenerationPaths
from utils.experiments.theory.support import FiniteSupport
from utils.experiments.theory.scheduler_adapter import SchedulerAdapter
from utils.experiments.theory.metrics import clean_estimates
from utils.experiments.theory.feedback import _logits, _partition, DEFAULT_INTEGRATION

root=Path('/nas/home/juyeop/workspace/How-Diffusion-Models-Memorize')
parser=argparse.ArgumentParser();parser.add_argument('--audit',required=True);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
audit=json.loads(Path(args.audit).read_text())['configurations'][args.config]
bundle=Path(audit['bundle'])
issue_ids={row['original_index'] for row in audit['all_raw_findings'].get('saved_integral_QA_inconsistency',{}).get('all_rows',audit['all_raw_findings'].get('saved_integral_QA_inconsistency',{}).get('examples',[]))}
meta=json.loads((bundle/'support_metadata.json').read_text())
bank=FiniteSupport(torch.load(bundle/'support.pt',map_location='cpu',weights_only=True),meta['atom_ids'],meta['aliases'],weights=meta['weights'])
paths=GenerationPaths(root/'logs'/args.config/'experiment_S0_N20')
schedule=torch.load(paths.schedule,map_location='cpu',weights_only=True)
run=json.loads(paths.run_config.read_text())
adapter=SchedulerAdapter(schedule,recorded_diffusers_version=run['scientific_config']['package_versions']['diffusers'])
frames=[]; m_records=[]
for index in sorted(issue_ids):
    paths_matching=list((bundle/'trajectory_metrics').glob(index+'.parquet'))
    if not paths_matching:
        paths_matching=[p for p in (bundle/'trajectory_metrics').glob('*.parquet') if index in p.stem]
    if len(paths_matching)!=1:raise RuntimeError((index,paths_matching))
    frame=pd.read_parquet(paths_matching[0])
    eligible=frame[frame.feedback_eligible]
    mask=eligible.candidate_integral_qa_status.eq('numerical_inconsistency')
    if mask.any(): frames.append(eligible[mask])
    m=eligible.branch_gap_l2-eligible.candidate_conditional_reference_error_l2-eligible.candidate_unconditional_reference_error_l2-eligible.candidate_variation_l2
    residual=(eligible.candidate_condition_margin_l2-m).abs()
    worst=residual.idxmax()
    row=eligible.loc[worst]
    m_records.append({'original_index':str(row.original_index),'seed':int(row.seed),'step':int(row.step_index),'M_reconstruction_abs_residual':float(residual.loc[worst]),'recorded_source_sensitivity':float(row.candidate_source_sensitivity_l2),'recorded_total_condition_uncertainty':float(row.candidate_condition_numerical_uncertainty_l2),'ratio_to_recorded_source_sensitivity':float(residual.loc[worst]/row.candidate_source_sensitivity_l2)})
bad=pd.concat(frames,ignore_index=True)
bad=bad[bad.original_index.astype(str).eq("1683590374") & bad.seed.eq(0) & bad.step_index.eq(48)]
results=[]
for index,record in bad.groupby('original_index'):
    z=torch.load(paths.latent_path(index),map_location='cpu',weights_only=True)
    u,c=torch.load(paths.noise_prediction_path(index),map_location='cpu',weights_only=True)
    for _,row in record.iterrows():
        seed,k=int(row.seed),int(row.step_index);coefficient=adapter.coefficients(k)
        start=(seed//16)*16;end=min(start+16,20);offset=seed-start
        state=z[start:end,k];next_state=z[start:end,k+1]
        mu,mc,delta,mg=clean_estimates(state,u[start:end,k],c[start:end,k],coefficient.alpha,coefficient.sigma,7.5)
        _,matched,_=adapter.matched_update(state,next_state,u[start:end,k],c[start:end,k],7.5,k)
        target=bank.aliases[row.support_target_id]
        a,s=coefficient.destination_alpha,coefficient.destination_sigma
        intercept=_logits(bank,matched.flatten(1),target,a,s)[offset].numpy()
        ideal_shift=7.5*coefficient.B*(mc-mu).flatten(1)
        slopes=((a/s**2)*(ideal_shift@bank.target_geometry(target)[0].T))[offset].numpy()
        direct_endpoint=_logits(bank,next_state.double().flatten(1),target,a,s)[offset].numpy()
        affine_H=-logsumexp(intercept+slopes)+logsumexp(intercept)
        direct_H=-logsumexp(direct_endpoint)+logsumexp(intercept)
        def derivative(position):
            logits=intercept+position*slopes
            weights=np.exp(logits-logsumexp(logits))
            return -float(weights@slopes)
        intervals,truncated=_partition(intercept,slopes,DEFAULT_INTEGRATION)
        points=sorted({left for left,right in intervals}|{right for left,right in intervals})[1:-1]
        with warnings.catch_warnings(record=True) as messages:
            integration=quad(derivative,0.,1.,points=points,epsabs=1e-9,epsrel=1e-12,limit=1000,full_output=1)
            value,error,info=integration[:3]
            quadpack_message=integration[3] if len(integration)>3 else None
        order=sorted(range(len(slopes)),key=lambda i:(slopes[i],intercept[i]))
        hull=[]; starts=[]
        for atom in order:
            if hull and slopes[atom]==slopes[hull[-1]]:hull.pop();starts.pop()
            crossing=-np.inf
            while hull:
                old=hull[-1];crossing=(intercept[old]-intercept[atom])/(slopes[atom]-slopes[old])
                if crossing>starts[-1]:break
                hull.pop();starts.pop()
            hull.append(atom);starts.append(crossing if len(hull)>1 else -np.inf)
        tail_checks=[]
        for extent in (8,16,32,64):
            widened={0.,.5,1.}
            for h in range(1,len(hull)):
                crossing=starts[h]
                if not 0<crossing<1:continue
                steepness=abs(slopes[hull[h]]-slopes[hull[h-1]])
                for width in (0.,-2.,2.,-8.,8.,-16.,16.,-32.,32.,-64.,64.):
                    position=crossing+width/steepness
                    if abs(width)<=extent and 0<position<1:widened.add(float(position))
            wide=quad(derivative,0.,1.,points=sorted(widened)[1:-1],epsabs=1e-9,epsrel=1e-12,limit=1000,full_output=1)
            tail_checks.append({'width_extent':extent,'points':len(widened),'integrated_H':float(wide[0]),'reported_error':float(wide[1]),'neval':int(wide[2]['neval']),'integral_minus_affine_endpoint':float(wide[0]-affine_H),'integral_minus_saved_endpoint':float(wide[0]-row.candidate_log_probability_gain),'QUADPACK_message':wide[3] if len(wide)>3 else None})
        out_tail={'tail_checks':tail_checks,'log_one_plus_exp_minus_8':float(np.log1p(np.exp(-8.)))}
        saved_allowance=float(row.candidate_integral_identity_error_estimate+row.candidate_gain_numerical_tolerance)
        out={'record':str(index),'seed':seed,'step':k,'target_atom':target,
             'saved_H':float(row.candidate_log_probability_gain),'saved_integrated_H':float(row.candidate_integrated_log_probability_gain),
             'saved_identity_residual':float(row.candidate_integral_identity_residual),'saved_quadrature_estimate':float(row.candidate_integral_identity_error_estimate),'saved_gain_tolerance':float(row.candidate_gain_numerical_tolerance),'saved_combined_allowance':saved_allowance,
             'saved_margin':float(row.candidate_condition_margin_l2),'saved_proof_slack':float(row.candidate_proof_lower_bound_slack),'saved_condition_status':row.candidate_condition_status,
             'recomputed_direct_H':float(direct_H),'recomputed_direct_minus_saved_H':float(direct_H-row.candidate_log_probability_gain),'affine_endpoint_H':float(affine_H),'direct_minus_affine_H':float(direct_H-affine_H),
             'tight_independent_QUADPACK_H':float(value),'tight_QUADPACK_error_estimate':float(error),'tight_QUADPACK_evaluations':int(info['neval']),
             'tight_integral_minus_affine_H':float(value-affine_H),'tight_integral_minus_saved_H':float(value-row.candidate_log_probability_gain),
             'maximum_relative_logit_magnitude':float(max(abs(intercept).max(),abs(intercept+slopes).max())),
             'maximum_endpoint_logit_reconstruction_difference':float(np.max(np.abs(direct_endpoint-(intercept+slopes)))),
             'quadrature_warnings':[str(message.message) for message in messages], 'QUADPACK_message':quadpack_message,'tight_requested_tolerance_achieved':quadpack_message is None,
             'source_timestep':coefficient.timestep,'destination_timestep':coefficient.destination_timestep,'kappa':coefficient.B,'sigma_next':s,
             'support_size':bank.size,'weights_positive_normalized':bool((bank.weights>0).all() and torch.isclose(bank.weights.sum(),torch.tensor(1.,dtype=torch.float64))),
        }
        out.update(out_tail)
        results.append(out)
        print(json.dumps(out),flush=True)
report={'scope':'read-only CPU independent investigation of all producer-unresolved integral-QA rows in this configuration; original scalar outputs and thresholds untouched','configuration':args.config,'analysis_hash':audit['analysis_hash'],'reference_backend':'CPU; GPU-produced bundles can differ in reduction order','rows':results,'M_component_reconstruction_maxima_per_record':m_records,'largest_M_residual_record':max(m_records,key=lambda x:x['M_reconstruction_abs_residual']),'maximum_M_residual_to_source_sensitivity_ratio':max(x['ratio_to_recorded_source_sensitivity'] for x in m_records)}
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

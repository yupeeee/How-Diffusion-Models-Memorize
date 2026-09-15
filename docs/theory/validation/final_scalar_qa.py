"""Read-only complete-bundle QA, using saved scalar artifacts exclusively.

This script does not import the theory reducer, torch, a model library, or a
scheduler. It never reads .pt files and never recalculates candidate posteriors
or quadrature. All comparisons use saved scalar values and recorded budgets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path('/nas/home/juyeop/workspace/How-Diffusion-Models-Memorize')
FORMULA_VERSION = 'cache-theory-3.0-proposition5'
BUNDLES = {
    'sdv1_ddim_g7.5_T50_N20': '1598f207670f784edadc621f97ccfd31bc8aab880eb507f43489ba4a3b73b10a',
    'sdv1_ddpm_g7.5_T50_N20': 'ba4b71becddb8797155d09c08c3dc31b3667573ab4df8046614b07d60ff15d38',
    'sdv2_ddim_g7.5_T50_N20': '80966a1f0c9d5cff3b833695a13555a39b2dca41462225038309c8438048e50e',
    'realvis_ddim_g7.5_T50_N20': '8289be884b96c62300cdb2d3cc10342a01ba259eb863a6c68123b3a50a9c9f12',
}
OVERRIDES = {}
KEYS = ['run_id', 'original_index', 'record_id', 'target_id', 'seed']
COLUMNS = KEYS + [
    'support_target_id', 'candidate_target_atom', 'step_index', 'destination_step_index',
    'destination_update_index', 'timestep', 'destination_timestep', 'is_initial',
    'is_final_update', 'seed_role', 'included', 'include_prompt', 'alpha', 'sigma',
    'snr', 'destination_alpha', 'destination_sigma', 'destination_snr', 'kappa',
    'affine_applicable', 'latent_dimension', 'terminal_sscd', 'branch_status',
    'branch_gap_l2', 'branch_gap_rmse', 'conditional_target_error_rmse',
    'unconditional_target_error_rmse', 'joint_target_error_rmse',
    'feedback_eligible', 'feedback_status', 'candidate_conditional_reference_error_l2',
    'candidate_unconditional_reference_error_l2', 'candidate_variation_l2',
    'candidate_variation_error_l2', 'candidate_condition_margin_l2',
    'candidate_condition_margin_rmse', 'candidate_condition_numerical_uncertainty_l2',
    'candidate_condition_status', 'candidate_gain_status', 'candidate_log_probability_gain',
    'candidate_log_odds_gain', 'candidate_matched_log_probability',
    'candidate_guided_log_probability', 'candidate_matched_log_complement',
    'candidate_guided_log_complement', 'candidate_matched_log_odds',
    'candidate_guided_log_odds', 'candidate_gain_numerical_tolerance',
    'candidate_gain_saturated', 'candidate_gain_underflow',
    'candidate_integrated_log_probability_gain', 'candidate_integral_identity_residual',
    'candidate_integral_identity_error_estimate', 'candidate_integral_status',
    'candidate_integral_qa_status', 'candidate_proof_lower_bound',
    'candidate_proof_lower_bound_slack', 'candidate_gram_roundoff_allowance_l2',
    'candidate_source_sensitivity_l2', 'candidate_quadrature_evaluations',
    'candidate_quadrature_refinements', 'candidate_quadrature_budget_exhausted',
    'candidate_integration_absolute_tolerance', 'candidate_integration_relative_tolerance',
    'candidate_integration_max_evaluations', 'candidate_segment_displacement_residual_l2',
]
EPS = np.finfo(np.float64).eps


def scalar_path(bundle, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix not in {'.json', '.csv', '.parquet'}:
        raise ValueError(f'Non-scalar/unsafe audit dependency: {relative}')
    path = bundle / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Missing/unsafe scalar audit dependency: {path}')
    return path


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def verified_path(bundle, manifest, relative):
    path = scalar_path(bundle, relative)
    expected = manifest['numerical_files'].get(relative)
    if expected is None or digest(path) != expected:
        raise ValueError(f'Unverified or changed scalar artifact: {path}')
    return path


def counts(values):
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).items()}


def quantiles(values):
    values = pd.to_numeric(values, errors='coerce')
    values = values[np.isfinite(values)]
    return {str(q): float(values.quantile(q)) for q in (0, .01, .25, .5, .75, .99, 1)} if len(values) else {}


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_complete_manifest(config):
    parent = ROOT / 'outputs' / config / 'theory_v2'
    selected_hash = OVERRIDES.get(config, BUNDLES[config])
    index_path = parent / 'index.json'
    if config not in OVERRIDES and index_path.is_file():
        index = json.loads(index_path.read_text())
        matches = [entry for entry in index.get('analyses', []) if
                   (entry.get('config', {}).get('num_seeds'), entry.get('config', {}).get('num_inference_steps'),
                    entry.get('config', {}).get('guidance_scale'), entry.get('config', {}).get('center')) == (20, 50, 7.5, 'reference-initial')
                   and entry.get('config', {}).get('target_error_tolerance') is None]
        if len(matches) != 1:
            raise RuntimeError(f'Ambiguous/missing published default analysis: {index_path}')
        selected_hash = matches[0]['analysis_hash']
    bundle = parent / selected_hash
    path = bundle / 'manifest.json'
    if not path.is_file():
        raise RuntimeError(f'Not yet published: {config}: {path}')
    manifest = json.loads(path.read_text())
    if (manifest.get('complete') is not True or manifest.get('schema_version') != 3
            or manifest.get('formula_version') != FORMULA_VERSION
            or manifest.get('analysis_hash') != selected_hash
            or manifest.get('analysis_scope') != 'complete_frozen_selection'):
        raise RuntimeError(f'Not a complete full-selection V3 bundle: {path}')
    run_config = manifest['config']
    if (run_config['num_seeds'], run_config['num_inference_steps'], run_config['guidance_scale']) != (20, 50, 7.5):
        raise RuntimeError(f'Unexpected scientific configuration: {path}')
    return bundle, manifest


def audit(config, bundle, manifest):
    n, steps, g = (manifest['config'][key] for key in ('num_seeds', 'num_inference_steps', 'guidance_scale'))
    expected = manifest['expected_counts']
    shard_names = sorted(name for name in manifest['numerical_files'] if name.startswith('trajectory_metrics/') and name.endswith('.parquet'))
    if len(shard_names) != expected['prompts']:
        raise RuntimeError(f'Incomplete trajectory shard inventory: {config}')
    all_shards = sorted(path.relative_to(bundle).as_posix() for path in (bundle / 'trajectory_metrics').glob('*.parquet'))
    if all_shards != shard_names:
        raise RuntimeError(f'Unmanaged/missing trajectory scalar shard: {config}')
    frames = [pd.read_parquet(verified_path(bundle, manifest, name), columns=COLUMNS) for name in shard_names]
    frame = pd.concat(frames, ignore_index=True)
    initial = pd.read_parquet(verified_path(bundle, manifest, 'initial_metrics.parquet'), columns=KEYS + ['step_index', 'terminal_sscd'])
    endpoint = pd.read_parquet(verified_path(bundle, manifest, 'endpoint_metrics.parquet'), columns=KEYS + ['step_index', 'terminal_sscd', 'terminal_clean_applicable', 'terminal_rmse'])
    source = pd.read_parquet(verified_path(bundle, manifest, 'source_records.parquet'))
    baseline = pd.read_parquet(verified_path(bundle, manifest, 'plotdata/unconditional_center.parquet'), columns=['run_id', 'seed', 'step_index', 'unconditional_center_rmse'])
    schedule = pd.read_csv(verified_path(bundle, manifest, 'schedule.csv'))
    support = json.loads(verified_path(bundle, manifest, 'support_metadata.json').read_text())
    violations = {}

    def check(name, bad, rows=frame):
        mask = np.asarray(bad, dtype=bool)
        if mask.ndim == 0:
            violations[name] = {'count': int(mask), 'examples': []}
        else:
            selected = rows.loc[mask]
            columns = [k for k in KEYS + [
                'step_index', 'destination_step_index', 'timestep', 'destination_timestep',
                'feedback_status', 'candidate_condition_status', 'candidate_gain_status',
                'branch_gap_l2', 'candidate_conditional_reference_error_l2',
                'candidate_unconditional_reference_error_l2', 'candidate_variation_l2',
                'candidate_condition_margin_l2', 'candidate_source_sensitivity_l2',
                'candidate_condition_numerical_uncertainty_l2', 'candidate_log_probability_gain',
                'candidate_log_odds_gain', 'candidate_gain_numerical_tolerance',
                'candidate_integrated_log_probability_gain', 'candidate_integral_identity_residual',
                'candidate_integral_identity_error_estimate', 'candidate_integral_qa_status',
                'candidate_integral_status', 'candidate_proof_lower_bound_slack',
                'candidate_quadrature_evaluations', 'candidate_quadrature_budget_exhausted',
                'audit_actual', 'audit_expected', 'audit_residual', 'audit_roundoff_allowance',
            ] if k in selected]
            raw_rows = selected[columns].to_dict('records')
            violations[name] = {'count': int(mask.sum()), 'examples': raw_rows[:10], 'all_rows': raw_rows}

    def scalar_identity(name, actual, wanted, scale=None, rows=frame):
        actual, wanted = np.asarray(actual, dtype=float), np.asarray(wanted, dtype=float)
        magnitude = np.maximum(1., np.abs(wanted)) if scale is None else np.maximum(1., np.asarray(scale, dtype=float))
        allowance = 256 * EPS * magnitude
        bad = ~np.isfinite(actual) | ~np.isfinite(wanted) | (np.abs(actual - wanted) > allowance)
        check(name, bad, rows.assign(audit_actual=actual, audit_expected=wanted,
                                     audit_residual=actual-wanted, audit_roundoff_allowance=allowance))
        return float(np.max(np.abs(actual - wanted))) if len(actual) else None

    check('no_current_failed_computational_records', manifest.get('current_failed_record_count') != 0)
    check('manifest_trajectory_count', len(frame) != expected['trajectory_rows'])
    check('manifest_initial_count', len(initial) != expected['initial_rows'])
    check('manifest_endpoint_count', len(endpoint) != expected['endpoint_rows'])
    check('full_seed_step_grid_size', len(frame) != expected['prompts'] * n * steps)
    check('stable_identity_missing', frame[KEYS].isna().any(axis=1))
    check('duplicate_sample_steps', frame.duplicated(KEYS + ['step_index']))
    check('source_run_identity', frame.run_id.ne(manifest['scientific_generation_hash']))
    check('only_evaluation_seeds', ~frame.seed.isin(range(n)))
    check('only_stored_prediction_indices', ~frame.step_index.isin(range(steps)))
    check('evaluation_seed_role', frame.seed_role.ne('experiment'))
    check('retained_prompt_flags', ~(frame.included & frame.include_prompt))
    sample_counts = frame.groupby(KEYS, dropna=False).size()
    check('all_steps_per_sample', bool(sample_counts.ne(steps).any()))
    prompt_counts = frame.groupby(KEYS[:-1], dropna=False).seed.nunique()
    check('all_seeds_per_prompt', bool(prompt_counts.ne(n).any()))
    check('prompt_identity_count', len(prompt_counts) != expected['prompts'])
    retained_ids = set(source.loc[source.included, 'original_index'].astype(str))
    check('frozen_retained_prompt_identity', set(frame.original_index) != retained_ids)
    check('target_membership_alias', frame.support_target_id.map(support['aliases']).ne(frame.candidate_target_atom))
    weights = np.asarray(support['weights'], dtype=float)
    check('declared_weights_normalized_positive', not (np.isfinite(weights).all() and (weights > 0).all() and np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)))
    check('current_step_initial_flag', frame.is_initial.ne(frame.step_index.eq(0)))
    check('current_step_final_flag', frame.is_final_update.ne(frame.step_index.eq(steps - 1)))
    check('destination_update_index', frame.destination_update_index.ne(frame.step_index + 1))
    check('destination_step_index', frame.destination_step_index.ne(frame.step_index + 1))
    schedule = schedule.set_index('step_index')
    check('saved_source_training_index', frame.timestep.ne(frame.step_index.map(schedule.timestep)))
    check('saved_destination_training_index', frame.destination_timestep.ne(frame.step_index.map(schedule.destination_timestep)))
    for column in ('alpha', 'sigma', 'snr', 'destination_alpha', 'destination_sigma'):
        scalar_identity('schedule_' + column, frame[column], frame.step_index.map(schedule[column]))
    scalar_identity('schedule_kappa', frame.kappa, frame.step_index.map(schedule.B))
    nonfinal = frame.loc[~frame.is_final_update]
    for source_column, destination_column in (('alpha', 'destination_alpha'), ('sigma', 'destination_sigma'), ('timestep', 'destination_timestep')):
        scalar_identity('next_cached_level_' + source_column, nonfinal[destination_column], (nonfinal.step_index + 1).map(schedule[source_column]), rows=nonfinal)
    same_scores = frame.groupby(KEYS).terminal_sscd.nunique(dropna=False)
    check('same_seed_score_constant_across_steps', bool(same_scores.ne(1).any()))
    check('initial_key_and_score_identity', not initial[KEYS + ['terminal_sscd']].sort_values(KEYS).reset_index(drop=True).equals(frame.loc[frame.is_initial, KEYS + ['terminal_sscd']].sort_values(KEYS).reset_index(drop=True)))
    check('endpoint_key_and_score_identity', not endpoint[KEYS + ['terminal_sscd']].sort_values(KEYS).reset_index(drop=True).equals(frame.loc[frame.is_final_update, KEYS + ['terminal_sscd']].sort_values(KEYS).reset_index(drop=True)))
    check('baseline_true_unique_seed_count', len(baseline) != n or set(baseline.seed) != set(range(n)) or bool(baseline.duplicated(['run_id', 'seed']).any()) or bool(baseline.step_index.ne(0).any()))
    scalar_identity('joint_errors_preserve_both_branches', frame.joint_target_error_rmse, np.maximum(frame.conditional_target_error_rmse, frame.unconditional_target_error_rmse))
    expected_eligible = (~frame.is_final_update & frame.affine_applicable & frame.kappa.gt(0) & frame.alpha.gt(0) & frame.sigma.gt(0) & frame.destination_alpha.gt(0) & frame.destination_sigma.gt(0) & frame.candidate_target_atom.notna() & (support['support_size'] > 1) & (g > 0))
    check('feedback_eligibility', frame.feedback_eligible.ne(expected_eligible))
    e = frame.loc[frame.feedback_eligible].copy()
    square_dimension = np.sqrt(e.latent_dimension)
    scale = e.branch_gap_l2 + e.candidate_conditional_reference_error_l2 + e.candidate_unconditional_reference_error_l2 + e.candidate_variation_l2
    margin = e.branch_gap_l2 - e.candidate_conditional_reference_error_l2 - e.candidate_unconditional_reference_error_l2 - e.candidate_variation_l2
    maximum_margin_identity_residual = scalar_identity('condition_component_M_identity', e.candidate_condition_margin_l2, margin, scale=scale, rows=e)
    margin_reconstruction_residual = np.abs(e.candidate_condition_margin_l2 - margin)
    check('condition_M_identity_outside_recorded_source_sensitivity',
          margin_reconstruction_residual > e.candidate_source_sensitivity_l2 + 256 * EPS * np.maximum(1., scale), e)
    scalar_identity('condition_RMSE_normalization', e.candidate_condition_margin_rmse * square_dimension, e.candidate_condition_margin_l2, scale=scale, rows=e)
    scalar_identity('conditional_reference_under_single_target_idealization', e.candidate_conditional_reference_error_l2, e.conditional_target_error_rmse * square_dimension, rows=e)
    scalar_identity('log_probability_H_identity', e.candidate_log_probability_gain, e.candidate_guided_log_probability - e.candidate_matched_log_probability, rows=e)
    scalar_identity('log_odds_gain_identity', e.candidate_log_odds_gain, e.candidate_guided_log_odds - e.candidate_matched_log_odds, rows=e)
    for endpoint_name in ('matched', 'guided'):
        odds = e[f'candidate_{endpoint_name}_log_odds']
        scalar_identity(endpoint_name + '_stable_log_probability', e[f'candidate_{endpoint_name}_log_probability'], -np.logaddexp(0., -odds), rows=e)
        scalar_identity(endpoint_name + '_stable_log_complement', e[f'candidate_{endpoint_name}_log_complement'], -np.logaddexp(0., odds), rows=e)
    gain = e.candidate_log_probability_gain
    odds_gain = e.candidate_log_odds_gain
    gain_tolerance = e.candidate_gain_numerical_tolerance
    expected_underflow = gain.eq(0) & odds_gain.abs().gt(gain_tolerance)
    check('underflow_flag_matches_resolved_sign', e.candidate_gain_underflow.ne(expected_underflow), e)
    check('positive_gain_sign', e.candidate_gain_status.eq('positive') & ~odds_gain.gt(gain_tolerance), e)
    check('negative_gain_sign', e.candidate_gain_status.eq('negative') & ~odds_gain.lt(-gain_tolerance), e)
    check('resolved_odds_gain_preserves_strict_sign', odds_gain.abs().gt(gain_tolerance) & ~e.candidate_gain_status.isin(['positive', 'negative']), e)
    check('zero_gain_sign', e.candidate_gain_status.eq('zero') & odds_gain.abs().gt(gain_tolerance), e)
    check('saturation_flag_has_rounded_one_endpoint', e.candidate_gain_saturated & (np.minimum(np.abs(e.candidate_matched_log_probability), np.abs(e.candidate_guided_log_probability)) > 4 * EPS), e)
    check('nonnegative_variation_and_error', e.candidate_variation_l2.lt(0) | e.candidate_variation_error_l2.lt(0), e)
    check('negative_source_or_roundoff_allowance', e.candidate_source_sensitivity_l2.lt(0) | e.candidate_gram_roundoff_allowance_l2.lt(0), e)
    scalar_identity('condition_uncertainty_components', e.candidate_condition_numerical_uncertainty_l2, e.candidate_variation_error_l2 + e.candidate_gram_roundoff_allowance_l2 + e.candidate_source_sensitivity_l2, rows=e)
    prefactor = e.destination_alpha * g * e.kappa / e.destination_sigma.pow(2)
    scalar_identity('proof_lower_bound_formula', e.candidate_proof_lower_bound, prefactor * e.branch_gap_l2 * e.candidate_condition_margin_l2, scale=np.abs(prefactor * e.branch_gap_l2 * e.candidate_condition_margin_l2), rows=e)
    scalar_identity('proof_lower_bound_slack_formula', e.candidate_proof_lower_bound_slack, gain - e.candidate_proof_lower_bound, rows=e)
    scalar_identity('integral_residual_formula', e.candidate_integral_identity_residual, gain - e.candidate_integrated_log_probability_gain, rows=e)
    identity_allowance = e.candidate_integral_identity_error_estimate + gain_tolerance
    slack_allowance = prefactor * e.branch_gap_l2 * e.candidate_condition_numerical_uncertainty_l2 + gain_tolerance
    identity_outside = e.candidate_integral_identity_residual.abs().gt(identity_allowance)
    check('integral_identity_outside_recorded_uncertainty', identity_outside, e)
    check('integral_identity_failure_not_recorded_unresolved', identity_outside &
          (~e.candidate_condition_status.eq('numerically_unresolved') |
           e.candidate_integral_qa_status.eq('consistent_with_numerical_estimates')), e)
    check('proof_lower_bound_slack_outside_recorded_uncertainty', e.candidate_proof_lower_bound_slack.lt(-slack_allowance), e)
    check('saved_integral_QA_inconsistency', e.candidate_integral_qa_status.ne('consistent_with_numerical_estimates'), e)
    check('quadrature_budget_exceeded', e.candidate_quadrature_evaluations.gt(e.candidate_integration_max_evaluations), e)
    check('estimated_met_outside_positive_margin', e.candidate_condition_status.eq('estimated_met') & ~e.candidate_condition_margin_l2.gt(e.candidate_condition_numerical_uncertainty_l2), e)
    check('estimated_not_met_outside_negative_margin', e.candidate_condition_status.eq('estimated_not_met') & ~e.candidate_condition_margin_l2.lt(-e.candidate_condition_numerical_uncertainty_l2), e)
    check('unresolved_integral_cannot_classify_condition', e.candidate_integral_status.ne('estimated_converged') & ~e.candidate_condition_status.eq('numerically_unresolved'), e)
    check('budget_exhausted_cannot_classify_condition', e.candidate_quadrature_budget_exhausted & ~e.candidate_condition_status.eq('numerically_unresolved'), e)
    check('estimated_condition_positive_negative_gain_inconsistency', e.candidate_condition_status.eq('estimated_met') & e.candidate_gain_status.eq('negative'), e)
    e['high_sscd'] = e.terminal_sscd > .75
    met = e.candidate_condition_status.eq('estimated_met')
    negative_high = e.high_sscd & e.candidate_gain_status.eq('negative')
    examples = KEYS + ['step_index', 'snr', 'destination_snr', 'terminal_sscd', 'candidate_condition_margin_rmse', 'candidate_log_probability_gain', 'candidate_log_odds_gain', 'candidate_gain_status', 'candidate_gain_underflow']
    per_step = []
    for step, group in e.groupby('step_index'):
        covered = group.candidate_condition_status.eq('estimated_met')
        per_step.append({
            'step_index': int(step), 'source_timestep': int(group.timestep.iloc[0]), 'destination_timestep': int(group.destination_timestep.iloc[0]),
            'source_snr': float(group.snr.iloc[0]), 'destination_snr': float(group.destination_snr.iloc[0]),
            'eligible_count': len(group), 'condition_met_count': int(covered.sum()), 'condition_status_counts': counts(group.candidate_condition_status),
            'gain_status_counts': counts(group.candidate_gain_status), 'conditional_gain_status_counts': counts(group.loc[covered, 'candidate_gain_status']),
            'saturated_count': int(group.candidate_gain_saturated.sum()), 'underflow_count': int(group.candidate_gain_underflow.sum()),
            'high_sscd_negative_gain_count': int((group.high_sscd & group.candidate_gain_status.eq('negative')).sum()),
            'log_probability_gain_quantiles': quantiles(group.candidate_log_probability_gain),
        })
    all_findings = {name: value for name, value in violations.items() if value['count']}
    numerical_findings = {}
    # Preserve the strict reconstruction finding; the producer already logs a
    # separate source-sensitivity allowance. This does not change its thresholds.
    if (all_findings.get('condition_component_M_identity') and
            not violations['condition_M_identity_outside_recorded_source_sensitivity']['count']):
        numerical_findings['condition_component_M_identity'] = {
            **all_findings['condition_component_M_identity'],
            'classification': 'independent_reduction_roundoff_within_preexisting_recorded_source_sensitivity',
            'maximum_raw_residual': maximum_margin_identity_residual,
            'maximum_ratio_to_recorded_source_sensitivity': float((margin_reconstruction_residual / e.candidate_source_sensitivity_l2.replace(0, np.nan)).max()),
        }
    inconsistent = e.candidate_integral_qa_status.ne('consistent_with_numerical_estimates')
    all_already_unresolved = e.loc[inconsistent, 'candidate_condition_status'].eq('numerically_unresolved').all()
    if all_already_unresolved:
        for name in ('integral_identity_outside_recorded_uncertainty', 'saved_integral_QA_inconsistency'):
            if name in all_findings:
                numerical_findings[name] = {
                    **all_findings[name], 'classification': 'producer_recorded_numerically_unresolved; flags_and_thresholds_unchanged',
                    'investigation': 'docs/theory/numerical_qa_investigation.md' if config in BUNDLES else 'requires_configuration_specific_investigation',
                }
    failures = {name: value for name, value in all_findings.items() if name not in numerical_findings}
    return {
        'configuration': config, 'bundle': str(bundle), 'analysis_hash': manifest['analysis_hash'],
        'schema_version': manifest['schema_version'], 'formula_version': manifest['formula_version'],
        'manifest_complete': True, 'current_failed_computational_records': manifest.get('current_failed_record_count'),
        'resolved_devices': manifest.get('execution', {}).get('resolved_devices', []),
        'reducer_source_hashes': manifest.get('reducer_source_hashes', {}),
        'scope': 'all retained prompts, all evaluation seeds, all stored prediction steps',
        'validation_status': ('hard_contract_checks_failed' if failures else 'hard_contract_checks_passed_with_recorded_numerical_issues' if numerical_findings else 'hard_contract_checks_passed'),
        'hard_contract_status': 'passed' if not failures else 'failed',
        'formal_numerical_certification': False,
        'checks_count': len(violations), 'violations': failures,
        'all_raw_findings': all_findings, 'numerical_findings': numerical_findings,
        'producer_recorded_numerical_issue_rows': int(inconsistent.sum()),
        'prompt_count': len(prompt_counts), 'seeds_per_prompt': n, 'steps_per_seed': steps, 'trajectory_row_count': len(frame),
        'initial_row_count': len(initial), 'endpoint_row_count': len(endpoint), 'unique_initial_baseline_count': len(baseline),
        'eligible_feedback_count': len(e), 'ineligible_feedback_status_counts': counts(frame.loc[~frame.feedback_eligible, 'feedback_status']),
        'condition_status_counts': counts(e.candidate_condition_status), 'condition_met_count': int(met.sum()),
        'condition_coverage_denominator': len(e), 'condition_coverage': float(met.mean()) if len(e) else None,
        'condition_interpretation': 'numerically estimated, not a formal certificate',
        'gain_status_counts': counts(e.candidate_gain_status), 'condition_met_gain_status_counts': counts(e.loc[met, 'candidate_gain_status']),
        'saturated_count': int(e.candidate_gain_saturated.sum()), 'underflow_count': int(e.candidate_gain_underflow.sum()),
        'underflow_resolved_sign_counts': counts(e.loc[e.candidate_gain_underflow, 'candidate_gain_status']),
        'high_sscd_eligible_count': int(e.high_sscd.sum()), 'negative_high_sscd_count': int(negative_high.sum()),
        'negative_high_sscd_most_negative_H_examples': e.loc[negative_high].nsmallest(10, 'candidate_log_probability_gain')[examples].to_dict('records'),
        'integral_status_counts': counts(e.candidate_integral_status), 'integral_QA_counts': counts(e.candidate_integral_qa_status),
        'quadrature_budget_exhausted_count': int(e.candidate_quadrature_budget_exhausted.sum()),
        'maximum_integral_identity_error_allowance_ratio': float((e.candidate_integral_identity_residual.abs() / identity_allowance.replace(0, np.nan)).max()),
        'minimum_proof_lower_bound_slack': float(e.candidate_proof_lower_bound_slack.min()),
        'maximum_margin_component_identity_residual': maximum_margin_identity_residual,
        'terminal_clean_applicable_count': int(endpoint.terminal_clean_applicable.sum()),
        'terminal_clean_inapplicable_count': int((~endpoint.terminal_clean_applicable).sum()),
        'metric_quantiles': {column: quantiles(e[column]) for column in ('candidate_condition_margin_rmse', 'candidate_variation_l2', 'candidate_variation_error_l2', 'candidate_condition_numerical_uncertainty_l2', 'candidate_log_probability_gain', 'candidate_log_odds_gain', 'candidate_quadrature_evaluations', 'candidate_quadrature_refinements', 'candidate_integral_identity_residual')},
        'timestep_summaries': per_step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', choices=tuple(BUNDLES), help='Audit only this completed configuration; repeatable. Default requires all four complete.')
    parser.add_argument('--analysis', action='append', default=[], metavar='CONFIG=SHA256', help='Pin a complete bundle instead of resolving its published index. Repeatable.')
    parser.add_argument('--output', default='/tmp/theory-v3-final-scalar-qa.json')
    args = parser.parse_args()
    for requested in args.analysis:
        parts = requested.split('=', 1)
        if len(parts) != 2 or parts[0] not in BUNDLES or re.fullmatch(r'[0-9a-f]{64}', parts[1]) is None:
            parser.error('--analysis requires a known CONFIG and a lowercase SHA256')
        OVERRIDES[parts[0]] = parts[1]
    configurations = args.config or list(BUNDLES)
    # Preflight every requested completion before auditing or emitting a result.
    try:
        ready = {config: load_complete_manifest(config) for config in configurations}
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'requested_configurations': configurations,
              'all_four_configurations_requested': set(configurations) == set(BUNDLES),
              'scope': 'complete requested scalar bundles only; no raw tensor access, posterior recomputation, or model inference',
              'configurations': {}}
    for config, (bundle, manifest) in ready.items():
        result = audit(config, bundle, manifest)
        report['configurations'][config] = result
        print(json.dumps({key: result[key] for key in ('configuration', 'validation_status', 'prompt_count', 'trajectory_row_count', 'eligible_feedback_count', 'condition_status_counts', 'gain_status_counts', 'saturated_count', 'underflow_count', 'negative_high_sscd_count', 'quadrature_budget_exhausted_count', 'violations')}), flush=True)
    additive = ('prompt_count', 'trajectory_row_count', 'initial_row_count', 'endpoint_row_count',
                'eligible_feedback_count', 'condition_met_count', 'saturated_count', 'underflow_count',
                'negative_high_sscd_count', 'quadrature_budget_exhausted_count',
                'producer_recorded_numerical_issue_rows', 'current_failed_computational_records')
    report['aggregate_counts'] = {key: sum(value[key] for value in report['configurations'].values()) for key in additive}
    for key in ('condition_status_counts', 'gain_status_counts', 'underflow_resolved_sign_counts'):
        combined = {}
        for value in report['configurations'].values():
            for label, count in value[key].items(): combined[label] = combined.get(label, 0) + count
        report['aggregate_counts'][key] = combined
    report['aggregate_condition_coverage'] = (report['aggregate_counts']['condition_met_count'] /
                                            report['aggregate_counts']['eligible_feedback_count'])
    report['hard_contract_status'] = 'passed' if all(value['hard_contract_status'] == 'passed' for value in report['configurations'].values()) else 'failed'
    report['validation_status'] = 'hard_contract_checks_failed' if report['hard_contract_status'] == 'failed' else 'hard_contract_checks_completed; inspect_preserved_numerical_findings'
    report['formal_numerical_certification'] = False
    destination = Path(args.output)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(safe_json(report), indent=2, allow_nan=False) + '\n')
    os.replace(temporary, destination)
    print(destination, flush=True)
    if report['hard_contract_status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

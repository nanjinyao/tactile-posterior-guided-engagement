function result = run_m12_proposed_vs_direct_trial()
%RUN_M12_PROPOSED_VS_DIRECT_TRIAL Two-method M12 comparison at one hard offset.
%
% Place the upright, unloaded M12 tip a little above the approximate hole
% centre and run this file.  There is deliberately no prompt, menu, token or
% method argument.  Edit only selectedMethodIndex below, then run this file.
% After each trial, recover the robot manually to the same above-centre
% teaching condition and run this same file again.
%
% This launcher is a focused two-method companion to
% run_m12_paired_full_loop_trial.  It shares that file's engine, truth
% firewall, checkpoint semantics and private-record handling, but keeps its
% own campaign file so neither launcher can disturb the other's history.
%
% HONEST STATEMENT OF WHAT THIS OFFSET IS AND IS NOT
%
% The offset below was chosen deliberately to discriminate between the two
% methods.  It is NOT a random draw and must never be presented as one.  A
% single discriminating case study is legitimate evidence about *where* the
% two methods differ; it is not evidence about average performance, and any
% claim of general superiority needs the unbiased offset campaign instead.
%
% HOW THIS POINT WAS DERIVED
%
% Measured direction error (= |true - inferred|, true = offset angle + 180):
%
%   5.25 deg  ->   1.50 deg   inferred 183.75, four-method campaign, r 4.600
%  20.000 deg ->  16.25 deg   inferred 183.75, Direct captured on attempt 1
%  30.000 deg -> unstable     three cones gave 33.8 / 48.8 / 273.8 deg
%  35.000 deg -> 166.25 deg   inferred  48.75, confidently antipodal
%
% Inside its trained lobe the model reports 183.75 deg regardless of the true
% bearing, so the error is just the offset minus 3.75 deg -- the 5.25 and
% 20 deg points match that to the digit.  By 30 deg the lobe is abandoned and
% the direction becomes unusable, so the angle must stay near 20-25 deg.
%
% Whether Direct misses is decided by the PHYSICAL capture radius, and three
% successive estimates of it were all beaten by measurement:
%
%   0.750 mm  the planner constant.  Not a physical opening at all.
%   1.88 mm   mouth apothem 6.50 minus tip circumradius 4.619, worst clocking.
%   2.50 mm   half the AF 13.00 - AF 8.00 clearance, hexes aligned.
%
% What all three ignore is the CHAMFER.  The mouth carries a 1.20 mm deep
% 45 deg chamfer, so the opening at the top face is a further 1.20 mm wider in
% radius, and a tip landing anywhere on that cone is guided in rather than
% stopped.  Corrected, the capture radius is 3.08 mm at worst clocking and
% 3.70 mm aligned -- which finally agrees with every measurement:
%
%   20 deg / r 4.55  miss 1.27 mm  -> captured on attempt 1
%   24 deg / r 6.00  miss 2.08 mm  -> captured on attempt 1
%   29 deg / r 6.00  miss 2.56 mm  -> captured on attempt 1
%   39 deg / r 6.00  miss 3.46 mm  -> MISSED on all three attempts
%
% So the threshold is now bracketed: capture at 2.56 mm, no capture at 3.46 mm,
% which intersects the chamfer-corrected 3.08..3.70 mm band at 3.08..3.46 mm.
% observedCapturingMissMm below records the lower bound and is asserted against
% the assumed radius; raise it whenever a larger miss captures.
%
% 39 deg is therefore the established demo point: Direct exhausts all three
% attempts there (IMPROVED_DIRECT_STRICT_THREE_ATTEMPTS_COMPLETE_NO_CAPTURE),
% and the run confirmed the route stays on 183.75 deg for every attempt even
% though the third cone's own top1 drifted to 243.75 deg -- the route is held
% inside the first attempt's sector by FRESH_POSTERIOR_FIRST_DIRECTION_
% MICROADJUST_BOUNDED, so the predicted 35.25 deg error held for all three.
%
% Four constraints bracket the design, all asserted below rather than left as
% prose:
%
%   1  angle inside the model's trained lobe, so the error stays predictable
%   2  attempt-1 miss above the capture radius, so Direct brushes past
%   3  peg tip stays on the D 21.275 head at touchdown, or it tips off the rim
%   4  post-shortcut distance + touchdown lock <= 4.80 mm grid radius, so the
%      proposed method can still cover the hole after its own shortcut
%
% Constraint 3 caps the radius near 6.0 mm, so the angle has to carry the
% design.  At 39 deg:
%
%   attempt 1  miss 3.46 mm    attempt 2  3.21 mm    attempt 3  3.71 mm
%   proposed   3.95 mm from the hole after its shortcut, 0.60 mm inside the
%              grid radius even after the touchdown lock error
%   head       peg tip reaches 10.02 mm against a 10.637 mm head radius
%
% Each angle increase is a change of PHYSICAL POSE, not a bias added to the
% model's estimate.  Injecting an offset into the inferred direction would
% manufacture the baseline's failure: the honest answer to "why did Direct
% miss?" would become "because we degraded its estimate", which is worth
% nothing as evidence.  Moving the peg produces the same error for free and
% stays a real measurement.
%
% The proposed method is expected to succeed at this same 39 deg point without
% any change: its shortcut is capped at 3.00 mm, so after the same wrong-by-
% 35.25 deg direction it sits 3.95 mm from the hole, well inside the 4.80 mm
% residual grid.  That gap between the two methods is the demonstration, not a
% confound to remove.  Matching the proposed shortcut to Direct's 5.70 mm would
% also land inside the grid (3.55 mm), but it would delete the very mechanism
% on show -- the proposed route deliberately does not over-commit to a learned
% direction, and keeps the residual inside what its grid can cover.
%
% WHAT IS STILL NOT GUARANTEED.  "Direct fails once or twice, then succeeds" was
% never achieved and is not what this point produces: the retry only shifts the
% start by 0.25 mm perpendicular, so it can only rescue a miss that sits inside
% a 0.25 mm window around the threshold.  At 39 deg all three attempts miss
% cleanly instead.  If a partial recovery is wanted, the window is near 36 deg
% (3.19 / 2.94 / 3.44 mm against a 3.08..3.46 mm threshold), but hitting it
% depends on where in that band the true threshold sits.

% THE ONLY DEMO METHOD SELECTOR -- edit this literal to 1 or 2.
%   1: Proposed+ learned posterior + residual entropy search
%   2: Improved Direct x3 learned
selectedMethodIndex = 2;
assert(isnumeric(selectedMethodIndex) && ...
        isscalar(selectedMethodIndex) && ...
        isfinite(selectedMethodIndex) && ...
        selectedMethodIndex == fix(selectedMethodIndex) && ...
        selectedMethodIndex >= 1 && selectedMethodIndex <= 2, ...
    'M12Demo:ProposedVsDirectMethodSelector', ...
    'selectedMethodIndex must be one integer literal, 1 or 2.');

realDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(realDir, 'data');
if ~isfolder(dataDir)
    [made, message] = mkdir(dataDir);
    assert(made, 'M12Demo:ProposedVsDirectDataDirectory', ...
        'Cannot create %s: %s', dataDir, message);
end
% Fixture-frame polar design.  Both knobs are set together; the four asserts
% below are the actual design constraints and will reject a pair that breaks
% any of them.  See the header for how this point was derived.
demoRadiusMm = 6.000;
demoAngleFixtureDeg = 31.000;
offsetsFixtureMm = demoRadiusMm * ...
    [cosd(demoAngleFixtureDeg), sind(demoAngleFixtureDeg)];
offsetIndex = 1;
offsetFixtureMm = offsetsFixtureMm(offsetIndex, :);
assert(all(isfinite(offsetFixtureMm)), ...
    'M12Demo:ProposedVsDirectOffsetFinite', ...
    'The fixed two-method demo offset is not finite.');
sourceSampleOffsetFixtureMm = [ ...
    5.657069292, 0.370783911];
% Derive the design identity from the geometry actually in force.  A literal
% would go stale the first time the angle knob moved, and a stale identity is
% worse than none because the checkpoint would still look authoritative.
demoDesignId = sprintf( ...
    'M12_DISCRIMINATING_OFFSET_PROPOSED_VS_DIRECT_R%s_A%s_V1', ...
    strrep(sprintf('%.3f', demoRadiusMm), '.', 'P'), ...
    strrep(sprintf('%.3f', demoAngleFixtureDeg), '.', 'P'));
demoSeedToken = sprintf('M12_PROPOSED_VS_DIRECT_20260729_A%s', ...
    strrep(sprintf('%.3f', demoAngleFixtureDeg), '.', 'P'));
% One campaign file per angle.  Sweeping the knob is meant to be a one-line
% edit, so a shared checkpoint would either reject the new angle as design
% drift or quietly blend two different experiments into one history.
campaignPath = fullfile(dataDir, sprintf( ...
    'proposed_vs_direct_demo_campaign_v1_A%s.json', ...
    strrep(sprintf('%.3f', demoAngleFixtureDeg), '.', 'P')));
sourceSampleId = ...
    'M12-CLEAN_CANDIDATE-SUPPORTED-A003p75';
offsetSelectionRationale = ...
    ['DELIBERATELY_DISCRIMINATING_NOT_RANDOM_', ...
     'RADIUS_AT_PROPOSED_GRID_COVERAGE_LIMIT_', ...
     'ANGLE_ON_TRAINED_DIRECTION_LOBE_SHOULDER_', ...
     'DIRECT_RETRY_RECOVERY_IS_EMPIRICAL_NOT_CONSTRUCTED'];
proposedGridReachMm = 4.800;
proposedShortcutCapMm = 3.000;
directRayLimitMm = 5.700;
directAdjustmentMm = 0.250;
conservativeCaptureRadiusMm = 0.750;
touchdownXyLockBoundMm = 0.250;
% Physical capture radius: the eccentricity at which the peg tip still drops in
% during a swept ray.  Three predictions were wrong before this was right, each
% time because the model of the opening was too small:
%
%   0.750 mm  the planner constant.  Not a physical opening at all.
%   1.88 mm   mouth apothem 6.50 minus tip circumradius 4.619, worst clocking.
%   2.50 mm   half the AF 13.00 - AF 8.00 clearance, hexes aligned.
%
% All three were beaten by measurement.  What they miss is the CHAMFER: the
% mouth carries a 1.20 mm deep 45 deg chamfer, so the opening at the top face
% is a further 1.20 mm wider in radius than AF 13.00, and a tip landing on that
% cone is guided in rather than stopped.  Correcting for it:
physicalCaptureRadiusLowMm = (13.00 / 2.0) + 1.20 - (8.00 / sqrt(3.0));
physicalCaptureRadiusHighMm = (13.00 + 2.0 * 1.20 - 8.00) / 2.0;
% Every run so far has captured, so the threshold is only bounded from below.
% Update this literal whenever a larger miss captures; it is the one number the
% whole design depends on and it must never be guessed downward again.
observedCapturingMissMm = 2.56;
assert(physicalCaptureRadiusLowMm >= observedCapturingMissMm, ...
    'M12Demo:ProposedVsDirectCaptureRadius', ...
    ['The assumed capture radius must not fall below the largest miss ', ...
     'already observed to capture.']);
physicalCaptureRadiusMm = physicalCaptureRadiusLowMm;
% Inside its trained lobe the model reports 183.75 deg regardless of the true
% bearing, so the direction error is the offset minus 3.75 deg.  Measured at
% 5.25 deg (1.50 deg error) and 20 deg (16.25 deg error), both exact; by
% 30 deg the lobe is abandoned and three cones gave 33.8/48.8/273.8 deg.
predictedDirectionErrorDeg = abs(demoAngleFixtureDeg - 3.750);
alongRayMm = demoRadiusMm * cosd(predictedDirectionErrorDeg);
perpendicularMm = demoRadiusMm * sind(predictedDirectionErrorDeg);
if alongRayMm >= directRayLimitMm
    firstAttemptMissMm = hypot(perpendicularMm, ...
        alongRayMm - directRayLimitMm);
else
    firstAttemptMissMm = perpendicularMm;
end
% Retry shifts are +/-0.25 mm perpendicular with signs [0, +1, -1], and +1
% moves toward the hole for a positive angular error.
predictedAttemptMissMm = firstAttemptMissMm + [0.0, -1.0, 1.0] * ...
    directAdjustmentMm;
% After its 3.00 mm shortcut along the same wrong direction, the proposed
% method still has to reach the hole with its residual grid.  Checking this at
% the touchdown anchor instead, as an earlier version did, ignores that the
% shortcut runs first and can carry the hole further away.
proposedPostShortcutMm = sqrt(demoRadiusMm^2 + proposedShortcutCapMm^2 - ...
    2.0 * demoRadiusMm * proposedShortcutCapMm * ...
    cosd(predictedDirectionErrorDeg));
assert(proposedPostShortcutMm + touchdownXyLockBoundMm <= ...
        proposedGridReachMm, ...
    'M12Demo:ProposedVsDirectProposedReachability', ...
    ['After the learned shortcut the hole must still lie inside the ', ...
     'proposed residual grid, or the proposed method cannot win either.']);
% Fairness: with a perfect direction the Direct ray must still capture, so any
% miss it suffers is attributable to aim and not to a short ray.
assert(abs(demoRadiusMm - directRayLimitMm) < physicalCaptureRadiusMm, ...
    'M12Demo:ProposedVsDirectBaselineFairness', ...
    ['A perfectly aimed Direct ray must still capture at this radius so ', ...
     'the comparison measures aim, not reach.']);
% The peg tip must stay on the socket head at touchdown or it tips off the
% rim.  Hex support radius toward the offset bearing, flat normals at
% 45 + k*60 deg with apothem 4.00 mm, against the 21.275 mm head diameter.
tipBearingDeviationDeg = min(abs(demoAngleFixtureDeg - (45.0 + (-2:3) * 60.0)));
tipSupportRadiusMm = 4.00 / cosd(tipBearingDeviationDeg);
assert(demoRadiusMm + tipSupportRadiusMm <= 21.275 / 2.0, ...
    'M12Demo:ProposedVsDirectPegOffHead', ...
    'The peg tip would overhang the socket head rim at touchdown.');
% The peg is displaced from the hole by +offset, so the hole lies back along
% the opposite bearing.  Printing this at startup turns the post-cone
% screening read into a subtraction the operator can do at a glance.
trueDirectionToHoleDeg = mod(demoAngleFixtureDeg + 180.0, 360.0);
methodModes = { ...
    'FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY', ...
    'FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED'};
trialCount = numel(methodModes);
assert(trialCount == 2, 'M12Demo:ProposedVsDirectMethodTable', ...
    'This launcher must retain exactly two methods.');

if isfile(campaignPath)
    campaign = jsondecode(fileread(campaignPath));
    validateProposedVsDirectCampaign(campaign, offsetsFixtureMm, ...
        methodModes, demoDesignId, demoSeedToken, sourceSampleId, ...
        sourceSampleOffsetFixtureMm, demoRadiusMm, ...
        demoAngleFixtureDeg, proposedGridReachMm, directRayLimitMm, ...
        conservativeCaptureRadiusMm, touchdownXyLockBoundMm, ...
        proposedPostShortcutMm, offsetSelectionRationale);
    campaign = recoverCampaignCenterFromPrivateRecords( ...
        campaign, dataDir);
else
    campaign = struct();
    campaign.schema = 'M12_PROPOSED_VS_DIRECT_DEMO_CAMPAIGN_V1';
    campaign.campaign_id = ['M12_PVD_DEMO_', ...
        char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'))];
    campaign.status = 'ACTIVE';
    campaign.created_local = pairedIsoTimestamp();
    campaign.updated_local = campaign.created_local;
    campaign.offset_distribution = ...
        'ONE_FIXED_DELIBERATELY_DISCRIMINATING_POINT_NOT_A_RANDOM_DRAW';
    campaign.offset_selection_rationale = offsetSelectionRationale;
    campaign.demo_design_id = demoDesignId;
    campaign.demo_seed_token = demoSeedToken;
    campaign.source_sample_id = sourceSampleId;
    campaign.source_sample_offset_fixture_mm = ...
        sourceSampleOffsetFixtureMm;
    campaign.demo_radius_mm = demoRadiusMm;
    campaign.demo_angle_fixture_deg = demoAngleFixtureDeg;
    campaign.proposed_grid_reach_mm = proposedGridReachMm;
    campaign.direct_ray_limit_mm = directRayLimitMm;
    campaign.conservative_capture_radius_mm = ...
        conservativeCaptureRadiusMm;
    campaign.touchdown_xy_lock_bound_mm = touchdownXyLockBoundMm;
    campaign.proposed_post_shortcut_distance_mm = ...
        proposedPostShortcutMm;
    campaign.true_direction_to_hole_deg = trueDirectionToHoleDeg;
    campaign.physical_capture_radius_mm = physicalCaptureRadiusMm;
    campaign.predicted_direction_error_deg = predictedDirectionErrorDeg;
    campaign.predicted_attempt_lateral_miss_mm = predictedAttemptMissMm;
    campaign.offsets_fixture_mm = offsetsFixtureMm;
    campaign.method_modes = methodModes;
    campaign.method_order = 1:trialCount;
    campaign.campaign_center_world_xy_mm = [];
    campaign.campaign_center_source_public_trial_token = '';
    campaign.next_trial_index = 1;
    campaign.completed_trial_count = 0;
    campaign.trials = repmat(struct( ...
        'trial_index', 0, ...
        'offset_index', 0, ...
        'order_position', 0, ...
        'method_index', 0, ...
        'method_mode', '', ...
        'status', 'PENDING', ...
        'attempt_count', 0, ...
        'current_public_trial_token', '', ...
        'last_started_local', '', ...
        'last_finished_local', '', ...
        'run_directory', '', ...
        'outcome_kind', 'PENDING', ...
        'last_error_identifier', '', ...
        'last_error_message', ''), trialCount, 1);
    for methodIndex = 1:trialCount
        executionMode = methodModes{methodIndex};
        trialIndex = methodIndex;
        campaign.trials(trialIndex).trial_index = trialIndex;
        campaign.trials(trialIndex).offset_index = offsetIndex;
        campaign.trials(trialIndex).order_position = methodIndex;
        campaign.trials(trialIndex).method_index = methodIndex;
        campaign.trials(trialIndex).method_mode = executionMode;
    end
    writeProposedVsDirectCheckpoint(campaignPath, campaign);
end

% Direct selector semantics: status and next_trial_index are historical
% checkpoint data only.  They never choose the method, and a COMPLETE
% selected slot is intentionally repeatable for rapid real-robot iteration.
trialIndex = selectedMethodIndex;
methodIndex = selectedMethodIndex;
trial = campaign.trials(trialIndex);
executionMode = methodModes{methodIndex};
assert(double(trial.offset_index) == offsetIndex && ...
        double(trial.method_index) == methodIndex && ...
        strcmp(char(trial.method_mode), executionMode), ...
    'M12Demo:ProposedVsDirectMethodDrift', ...
    'Campaign trial %d method no longer matches the fixed table.', ...
    trialIndex);

publicTrialToken = char(java.util.UUID.randomUUID());
privateTrialSpec = struct();
privateTrialSpec.schema = 'M12_PROPOSED_VS_DIRECT_PRIVATE_TRIAL_V1';
privateTrialSpec.campaign_id = char(campaign.campaign_id);
privateTrialSpec.trial_index = trialIndex;
privateTrialSpec.offset_index = offsetIndex;
privateTrialSpec.offset_fixture_mm = offsetFixtureMm;
privateTrialSpec.method_index = methodIndex;
privateTrialSpec.method_mode = executionMode;
privateTrialSpec.order_position = double(trial.order_position);
privateTrialSpec.public_trial_token = publicTrialToken;
privateTrialSpec.campaign_center_world_xy_mm = ...
    double(campaign.campaign_center_world_xy_mm);
privateTrialSpec.public_entrypoint = ...
    'run_m12_proposed_vs_direct_trial';
privateTrialSpec.truth_firewall = ...
    'OFFSET_USED_ONLY_FOR_PRE_TOUCHDOWN_AIR_TRANSPORT_NOT_POLICY_INPUT';
privateTrialSpec.demo_design_id = demoDesignId;
privateTrialSpec.demo_seed_token = demoSeedToken;
privateTrialSpec.source_sample_id = sourceSampleId;
privateTrialSpec.offset_selection_rationale = offsetSelectionRationale;

campaign.trials(trialIndex).status = 'RUNNING';
campaign.trials(trialIndex).attempt_count = ...
    double(campaign.trials(trialIndex).attempt_count) + 1;
campaign.trials(trialIndex).current_public_trial_token = publicTrialToken;
campaign.trials(trialIndex).last_started_local = pairedIsoTimestamp();
campaign.trials(trialIndex).last_finished_local = '';
campaign.trials(trialIndex).run_directory = '';
campaign.trials(trialIndex).outcome_kind = 'RUNNING';
campaign.trials(trialIndex).last_error_identifier = '';
campaign.trials(trialIndex).last_error_message = '';
campaign.status = 'ACTIVE';
campaign.completed_trial_count = sum(strcmp( ...
    {campaign.trials.status}, 'COMPLETE'));
campaign.next_trial_index = trialIndex;
campaign.updated_local = pairedIsoTimestamp();
writeProposedVsDirectCheckpoint(campaignPath, campaign);

fprintf(['\nM12 proposed-vs-direct selected method %d/2: %s; ', ...
    'repeat attempt %d.\n', ...
    'Deliberately discriminating offset r %.3f mm at %.3f deg (fixture).\n', ...
    '>>> TRUE DIRECTION TO HOLE = %.2f deg <<<\n', ...
    'Model is mode-locked on 183.75 deg inside its trained lobe, so the ', ...
    'predicted direction error is %.2f deg.\n', ...
    'Chamfer-corrected capture radius %.2f..%.2f mm; every run so far ', ...
    'captured, so it is only bounded below at %.2f mm.\n', ...
    'Predicted lateral miss:  attempt 1  %.2f mm    attempt 2  %.2f mm', ...
    '    attempt 3  %.2f mm\n', ...
    'Check the prediction against LEARNED HOLE DIRECTION at ', ...
    'ACTIVE_MODEL_INFERENCE_OK just after the cone.\n', ...
    'No confirmation is required; live preflight and motion begin now.\n'], ...
    methodIndex, executionMode, ...
    double(campaign.trials(trialIndex).attempt_count), ...
    demoRadiusMm, demoAngleFixtureDeg, trueDirectionToHoleDeg, ...
    predictedDirectionErrorDeg, physicalCaptureRadiusLowMm, ...
    physicalCaptureRadiusHighMm, observedCapturingMissMm, ...
    predictedAttemptMissMm(1), predictedAttemptMissMm(2), ...
    predictedAttemptMissMm(3));
% Consecutive real-robot iterations edit the shared engine in one MATLAB
% session.  Reload that function automatically so F5 never runs a stale
% cached controller and the operator needs no extra command.
clear execute_m12_contact_engagement_policy
try
    result = execute_m12_contact_engagement_policy( ...
        executionMode, privateTrialSpec);
catch trialError
    campaign = readProposedVsDirectCheckpoint(campaignPath);
    campaign = adoptCampaignCenterFromPrivateRecord( ...
        campaign, dataDir, publicTrialToken);
    % Direct x3 finite exhaustion returns normally from the controller and
    % never enters this catch.  Every exception here still represents a hard
    % state, stale/unknown feedback, or implementation failure and remains
    % available for immediate same-method debugging/retry.
    campaign.trials(trialIndex).status = 'RETRY_PENDING';
    campaign.trials(trialIndex).outcome_kind = ...
        'HARD_STATE_OR_IMPLEMENTATION_RETRY_REQUIRED';
    campaign.trials(trialIndex).last_finished_local = pairedIsoTimestamp();
    campaign.next_trial_index = trialIndex;
    campaign.trials(trialIndex).last_error_identifier = ...
        char(trialError.identifier);
    campaign.trials(trialIndex).last_error_message = ...
        pairedOneLine(trialError.message);
    campaign.completed_trial_count = sum(strcmp( ...
        {campaign.trials.status}, 'COMPLETE'));
    if campaign.completed_trial_count == numel(campaign.trials)
        campaign.status = 'COMPLETE';
    else
        campaign.status = 'ACTIVE';
    end
    campaign.updated_local = pairedIsoTimestamp();
    writeProposedVsDirectCheckpoint(campaignPath, campaign);
    rethrow(trialError);
end

campaign = readProposedVsDirectCheckpoint(campaignPath);
campaign = adoptCampaignCenterFromPrivateRecord( ...
    campaign, dataDir, publicTrialToken);
campaign.trials(trialIndex).status = 'COMPLETE';
campaign.trials(trialIndex).last_finished_local = pairedIsoTimestamp();
campaign.trials(trialIndex).run_directory = char(result.run_directory);
finiteDirectExhaustion = strcmp(executionMode, ...
    'FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED') && ...
    isfield(result, 'method_execution_completed') && ...
    logical(result.method_execution_completed) && ...
    isfield(result, 'capture_found') && ~logical(result.capture_found);
if finiteDirectExhaustion
    campaign.trials(trialIndex).outcome_kind = ...
        'METHOD_EXHAUSTED_NO_CAPTURE';
else
    campaign.trials(trialIndex).outcome_kind = 'TASK_SUCCESS';
end
campaign.trials(trialIndex).last_error_identifier = '';
campaign.trials(trialIndex).last_error_message = '';
campaign.completed_trial_count = sum(strcmp( ...
    {campaign.trials.status}, 'COMPLETE'));
campaign.next_trial_index = trialIndex;
if campaign.completed_trial_count == numel(campaign.trials)
    campaign.status = 'COMPLETE';
else
    campaign.status = 'ACTIVE';
end
campaign.updated_local = pairedIsoTimestamp();
writeProposedVsDirectCheckpoint(campaignPath, campaign);
result.pvd_campaign_id = char(campaign.campaign_id);
result.pvd_trial_index = trialIndex;
result.pvd_selected_method_index = selectedMethodIndex;
result.pvd_selected_method_mode = executionMode;
result.pvd_outcome_kind = ...
    char(campaign.trials(trialIndex).outcome_kind);
result.pvd_selected_method_attempt_count = ...
    double(campaign.trials(trialIndex).attempt_count);
result.pvd_completed_trial_count = campaign.completed_trial_count;
result.pvd_offset_selection_rationale = offsetSelectionRationale;
result.pvd_true_direction_to_hole_deg = trueDirectionToHoleDeg;
result.pvd_physical_capture_radius_mm = physicalCaptureRadiusMm;
result.pvd_predicted_direction_error_deg = predictedDirectionErrorDeg;
result.pvd_predicted_attempt_lateral_miss_mm = predictedAttemptMissMm;
end

function campaign = recoverCampaignCenterFromPrivateRecords( ...
        campaign, dataDir)
if ~isempty(campaign.campaign_center_world_xy_mm)
    return;
end
for trialIndex = numel(campaign.trials):-1:1
    token = char(campaign.trials( ...
        trialIndex).current_public_trial_token);
    if isempty(token)
        continue;
    end
    campaign = adoptCampaignCenterFromPrivateRecord( ...
        campaign, dataDir, token);
    if ~isempty(campaign.campaign_center_world_xy_mm)
        return;
    end
end
end

function campaign = adoptCampaignCenterFromPrivateRecord( ...
        campaign, dataDir, publicTrialToken)
% The engine writes its private record under the paired directory regardless
% of which launcher started the trial, so the campaign centre is recovered
% from the same place both launchers use.  The centre is a physical property
% of the fixture setup, not of either campaign.
privatePath = fullfile(dataDir, 'paired_full_loop_private', ...
    [publicTrialToken, '.json']);
if ~isfile(privatePath)
    return;
end
record = jsondecode(fileread(privatePath));
candidate = double(reshape( ...
    record.campaign_center_world_xy_mm, 1, []));
assert(numel(candidate) == 2 && all(isfinite(candidate)), ...
    'M12Demo:ProposedVsDirectCampaignCenterRecord', ...
    'The private record contains an invalid campaign centre.');
if isempty(campaign.campaign_center_world_xy_mm)
    campaign.campaign_center_world_xy_mm = candidate;
    campaign.campaign_center_source_public_trial_token = ...
        publicTrialToken;
else
    existing = double(reshape( ...
        campaign.campaign_center_world_xy_mm, 1, []));
    assert(norm(existing - candidate) <= 1e-9, ...
        'M12Demo:ProposedVsDirectCampaignCenterDrift', ...
        'A private trial attempted to change the fixed campaign centre.');
end
end

function validateProposedVsDirectCampaign( ...
        campaign, offsetsFixtureMm, methodModes, demoDesignId, ...
        demoSeedToken, sourceSampleId, sourceSampleOffsetFixtureMm, ...
        demoRadiusMm, demoAngleFixtureDeg, proposedGridReachMm, ...
        directRayLimitMm, conservativeCaptureRadiusMm, ...
        touchdownXyLockBoundMm, proposedPostShortcutMm, ...
        offsetSelectionRationale)
assert(isstruct(campaign) && ...
    isfield(campaign, 'schema') && ...
    strcmp(char(campaign.schema), ...
        'M12_PROPOSED_VS_DIRECT_DEMO_CAMPAIGN_V1'), ...
    'M12Demo:ProposedVsDirectCampaignSchema', ...
    'The checkpoint is not a proposed-vs-direct campaign.');
assert(isfield(campaign, 'trials') && ...
    numel(campaign.trials) == numel(methodModes), ...
    'M12Demo:ProposedVsDirectCampaignTrialCount', ...
    'The checkpoint does not hold exactly %d trials.', ...
    numel(methodModes));
storedOffsets = double(campaign.offsets_fixture_mm);
storedOffsets = reshape(storedOffsets, size(offsetsFixtureMm));
assert(isequal(size(storedOffsets), size(offsetsFixtureMm)) && ...
    max(abs(storedOffsets(:) - offsetsFixtureMm(:))) <= 1e-9, ...
    'M12Demo:ProposedVsDirectCampaignOffsetDrift', ...
    'The checkpoint offset no longer matches this file.');
assert(strcmp(char(campaign.demo_design_id), demoDesignId) && ...
    strcmp(char(campaign.demo_seed_token), demoSeedToken) && ...
    strcmp(char(campaign.source_sample_id), sourceSampleId) && ...
    strcmp(char(campaign.offset_selection_rationale), ...
        offsetSelectionRationale), ...
    'M12Demo:ProposedVsDirectCampaignDesignDrift', ...
    'The checkpoint design identity no longer matches this file.');
storedSourceOffset = double(reshape( ...
    campaign.source_sample_offset_fixture_mm, 1, []));
assert(numel(storedSourceOffset) == 2 && ...
    max(abs(storedSourceOffset - sourceSampleOffsetFixtureMm)) <= 1e-9, ...
    'M12Demo:ProposedVsDirectCampaignSourceOffsetDrift', ...
    'The checkpoint source sample offset no longer matches this file.');
scalarFields = { ...
    'demo_radius_mm', demoRadiusMm; ...
    'demo_angle_fixture_deg', demoAngleFixtureDeg; ...
    'proposed_grid_reach_mm', proposedGridReachMm; ...
    'direct_ray_limit_mm', directRayLimitMm; ...
    'conservative_capture_radius_mm', conservativeCaptureRadiusMm; ...
    'touchdown_xy_lock_bound_mm', touchdownXyLockBoundMm; ...
    'proposed_post_shortcut_distance_mm', proposedPostShortcutMm};
for fieldIndex = 1:size(scalarFields, 1)
    name = scalarFields{fieldIndex, 1};
    expected = scalarFields{fieldIndex, 2};
    assert(isfield(campaign, name) && ...
        abs(double(campaign.(name)) - expected) <= 1e-9, ...
        'M12Demo:ProposedVsDirectCampaignGeometryDrift', ...
        'The checkpoint field %s no longer matches this file.', name);
end
for trialIndex = 1:numel(campaign.trials)
    assert(strcmp(char(campaign.trials(trialIndex).method_mode), ...
            methodModes{trialIndex}), ...
        'M12Demo:ProposedVsDirectCampaignMethodTable', ...
        'Checkpoint trial %d method no longer matches this file.', ...
        trialIndex);
end
end

function campaign = readProposedVsDirectCheckpoint(path)
assert(isfile(path), 'M12Demo:ProposedVsDirectCheckpointMissing', ...
    'The campaign checkpoint %s is missing.', path);
campaign = jsondecode(fileread(path));
end

function writeProposedVsDirectCheckpoint(path, campaign)
text = jsonencode(campaign, 'PrettyPrint', true);
temporaryPath = [path, '.tmp'];
fileId = fopen(temporaryPath, 'w');
assert(fileId > 0, 'M12Demo:ProposedVsDirectCheckpointOpen', ...
    'Cannot open %s for writing.', temporaryPath);
cleanup = onCleanup(@() fclose(fileId));
fwrite(fileId, text, 'char');
clear cleanup;
[moved, message] = movefile(temporaryPath, path, 'f');
assert(moved, 'M12Demo:ProposedVsDirectCheckpointWrite', ...
    'Cannot write %s: %s', path, message);
end

function value = pairedIsoTimestamp()
value = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS'));
end

function value = pairedOneLine(value)
value = char(value);
value = regexprep(value, '\s+', ' ');
value = strtrim(value);
end

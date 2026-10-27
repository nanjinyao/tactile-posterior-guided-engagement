function result = run_m12_sampling_conical_fingerprints_26pt()
%RUN_M12_SAMPLING_CONICAL_FINGERPRINTS_26PT Acquire the M12 sampling plan.
%
% This is a data-only lane.  Calling this dedicated entrypoint authorizes the
% run; no second command-window phrase is requested.  The first fresh
% stationary TCP XY is this run's M12 centre; 26 preregistered local offsets are
% visited in the printed-CAD-generated, sentinel-constrained short route.
% Each point gets a fresh air
% baseline, anchored touchdown, stable 3.5 N contact and one full cone.
% No simulation/model inference, full-loop insertion or online update runs.

% Real-robot iteration commonly edits the large executor between consecutive
% MATLAB calls.  Refresh only this dependency so the next invocation cannot
% run a stale in-memory executor while leaving the rest of the session intact.
rehash path;
clear execute_m12_contact_engagement_policy
result = execute_m12_contact_engagement_policy( ...
    'DATASET_M12_CONICAL_FINGERPRINTS_26PT');
end

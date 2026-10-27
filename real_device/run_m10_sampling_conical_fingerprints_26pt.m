function result = run_m10_sampling_conical_fingerprints_26pt()
%RUN_M10_SAMPLING_CONICAL_FINGERPRINTS_26PT Acquire the M10 26-point plan.
%
% Calling this dedicated entrypoint authorizes the run without a second
% command-window phrase.  The fresh stationary TCP XY becomes the run-local
% M10 centre.  Current installed-tool gravity is absorbed by the fresh start
% baseline and a new 30-sample air baseline at every point; raw Bota data is
% still preserved.  No simulation/model inference or online model update runs.

rehash path;
clear execute_m12_contact_engagement_policy
result = execute_m12_contact_engagement_policy( ...
    'DATASET_M10_CONICAL_FINGERPRINTS_26PT');
end

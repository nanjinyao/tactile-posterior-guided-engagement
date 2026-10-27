function result = run_m12_full_loop_baseline_improved_direct_x3()
%RUN_M12_FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3 Run Improved Direct x3.
%
% Each of at most three attempts acquires a fresh conical fingerprint, runs
% the promoted real-data model, and executes one bounded ray in that fresh
% learned direction.  Perpendicular reset anchors are 0/+0.25/-0.25 mm;
% there is no hand-written angle fan or direction fallback.  Surface XY
% transport uses the shared 4/2/1 mm/s adaptive, 0.30 mm endpoint-observed
% controller.  Capture ends early; otherwise three clean misses complete this
% finite baseline at the measured current pose with no outer x3 restart.

rehash path;
clear execute_m12_contact_engagement_policy
result = execute_m12_contact_engagement_policy( ...
    'FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED');
end

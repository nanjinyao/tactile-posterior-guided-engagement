function result = run_m12_full_loop_baseline_pure_outward_spiral()
%RUN_M12_FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL Run the geometric baseline.
%
% After guarded touchdown, search directly with one time- and radius-bounded
% outward Archimedean spiral.  This method uses neither a conical fingerprint
% nor a learned model.  It shares the 4/2/1 mm/s adaptive, 0.30 mm
% endpoint-observed surface controller and 4.0 +/- 1.0 N preferred
% normal-force band.

rehash path;
clear execute_m12_contact_engagement_policy
result = execute_m12_contact_engagement_policy( ...
    'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL');
end

function result = run_m12_full_loop_proposed_local_spiral()
%RUN_M12_FULL_LOOP_PROPOSED_LOCAL_SPIRAL Run the Proposed full loop.
%
% One conical fingerprint produces the same learned CAD-posterior bounded
% shortcut as Proposed+.  A posterior-sized local outward Archimedean spiral
% resolves residual uncertainty under the shared 4/2/1 mm/s adaptive,
% 0.30 mm endpoint-observed surface controller.  The promoted model runtime
% must complete its preflight and post-cone inference; no hidden classical
% fallback is permitted.

rehash path;
clear execute_m12_contact_engagement_policy
result = execute_m12_contact_engagement_policy( ...
    'FULL_LOOP_PROPOSED_LEARNED_LOCAL_SPIRAL');
end

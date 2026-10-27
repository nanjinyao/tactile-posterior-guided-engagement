function result = run_m12_full_loop_proposed_plus_entropy_grid()
%RUN_M12_FULL_LOOP_PROPOSED_PLUS_ENTROPY_GRID Run the new shared task core.
%
% F5 immediately executes the new shared controller:
% touchdown -> continuous 15 degree cone -> learned CAD-posterior bounded
% shortcut -> learned-direction CAD corridor -> 95%-credible residual
% entropy grid -> physical chamfer capture -> segmented insertion ->
% persistent bottom event -> KST end.  The promoted real-data model owns only
% high-level direction/state-posterior routing; MATLAB retains every physical
% guard, capture, insertion and terminal decision.
%
% The full-loop core uses an explicitly logged predecessor-compatibility
% commissioning lane until final pivot/payload/fixture measurements exist.
% Missing commissioning IDs no longer prevent motion.  Fresh Bota/diagnostic
% data, one KST command owner, 25 N / 0.75 Nm hard guards, joint/workspace
% envelopes, and KST end on completion/exception remain active.

rehash path;
clear execute_m12_contact_engagement_policy
result = execute_m12_contact_engagement_policy( ...
    'FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY');
end

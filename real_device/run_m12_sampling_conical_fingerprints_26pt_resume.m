function result = run_m12_sampling_conical_fingerprints_26pt_resume(previousRunDir)
%RUN_M12_SAMPLING_CONICAL_FINGERPRINTS_26PT_RESUME Continue an incomplete run.
%
% With no input, select the most recently updated resumable M12 campaign.
% An explicit parent directory is still accepted.  The parent remains
% immutable: a new child validates its plan/CAD/evidence lineage, establishes
% fresh stationary robot/Bota state, verifies clear load, and skips every
% execution index whose cone evidence is already complete.

assert(nargin <= 1, ...
    'M12Demo:SamplingResumePathInput', ...
    'Pass no input for automatic resume, or one previous run directory.');
if nargin == 0 || isempty(previousRunDir)
    previousRunDir = selectLatestResumableM12Campaign();
    fprintf('[M12 AUTO RESUME] Selected parent run:\n  %s\n', ...
        previousRunDir);
end
assert(ischar(previousRunDir) || ...
    (isstring(previousRunDir) && isscalar(previousRunDir)), ...
    'M12Demo:SamplingResumePathInput', ...
    'The previous M12 sampling run directory must be text.');
rehash path;
clear execute_m12_contact_engagement_policy
result = execute_m12_contact_engagement_policy( ...
    'DATASET_M12_CONICAL_FINGERPRINTS_26PT', previousRunDir);
end

function runDir = selectLatestResumableM12Campaign()
entryDir = fileparts(mfilename('fullpath'));
dataRoot = fullfile(entryDir, 'data');
candidates = dir(fullfile( ...
    dataRoot, 'M12_conical_fingerprints_26pt_*'));
candidates = candidates([candidates.isdir]);
if isempty(candidates)
    error('M12Demo:SamplingResumeAutoSelect', ...
        'No M12 26-point sampling campaign exists under %s.', dataRoot);
end
[checkpointTimes, checkpointExists] = deal( ...
    -inf(1, numel(candidates)), false(1, numel(candidates)));
for candidateIndex = 1:numel(candidates)
    checkpointPath = fullfile( ...
        candidates(candidateIndex).folder, ...
        candidates(candidateIndex).name, ...
        'sampling_campaign.json');
    checkpointInfo = dir(checkpointPath);
    if ~isempty(checkpointInfo)
        checkpointTimes(candidateIndex) = checkpointInfo(1).datenum;
        checkpointExists(candidateIndex) = true;
    end
end
candidates = candidates(checkpointExists);
checkpointTimes = checkpointTimes(checkpointExists);
[~, order] = sort(checkpointTimes, 'descend');
candidates = candidates(order);

runDir = '';
for candidateIndex = 1:numel(candidates)
    candidateDir = fullfile( ...
        candidates(candidateIndex).folder, ...
        candidates(candidateIndex).name);
    checkpointPath = fullfile(candidateDir, 'sampling_campaign.json');
    if ~isfile(checkpointPath)
        continue
    end
    try
        campaign = jsondecode(fileread(checkpointPath));
    catch
        continue
    end
    [~, candidateName] = fileparts(candidateDir);
    if ~isfield(campaign, 'campaign') || ...
            ~strcmp(char(campaign.campaign), ...
                'm12_run_local_conical_fingerprints_26pt_v2') || ...
            ~isfield(campaign, 'run_session_id') || ...
            ~strcmp(char(campaign.run_session_id), candidateName) || ...
            ~isfield(campaign, 'attempt_count_planned') || ...
            double(campaign.attempt_count_planned) ~= 26 || ...
            ~isfield(campaign, 'status') || ...
            ~isResumableCampaignStatus(char(campaign.status)) || ...
            ~isfield(campaign, 'attempts') || ...
            ~hasCompletedCone(campaign.attempts)
        continue
    end
    runDir = candidateDir;
    break
end

if isempty(runDir)
    error('M12Demo:SamplingResumeAutoSelect', ...
        ['No resumable M12 campaign with at least one completed cone was ', ...
         'found under %s. Start a fresh sampling run instead.'], dataRoot);
end
end

function eligible = isResumableCampaignStatus(status)
eligible = contains(status, 'ABORTED') || ...
    strcmp(status, 'RUNNING') || ...
    strcmp(status, ...
        'RETURNED_TO_INITIAL_POSE_BUT_CAMPAIGN_FINALIZATION_FAILED');
end

function completed = hasCompletedCone(attempts)
completed = false;
if ~isstruct(attempts) || isempty(attempts)
    return
end
for attemptIndex = 1:numel(attempts)
    if isfield(attempts(attemptIndex), 'cone_completed') && ...
            isscalar(attempts(attemptIndex).cone_completed) && ...
            logical(attempts(attemptIndex).cone_completed)
        completed = true;
        return
    end
end
end

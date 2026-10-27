function dashboard = m12_sampling_topview_dashboard(action, dashboard, varargin)
%M12_SAMPLING_TOPVIEW_DASHBOARD Lightweight M12 sampling top-view state machine.
%
%   dashboard = m12_sampling_topview_dashboard( ...
%       action, dashboard, payload1, payload2, ...)
%
% Payload arguments must be scalar structs.  The first payload is
% authoritative; later payloads (for example the shared controller cfg)
% contribute only fields absent from earlier payloads.  This permits both
% compact action payloads and the controller's initialize call:
%
%   dashboard = m12_sampling_topview_dashboard( ...
%       'initialize', [], samplingPlan, cfg);
%
% Actions and payload fields:
%   initialize
%       plan or points              Existing 26-point M12 sampling plan.
%       socket_center_fixture_xy_mm Optional [x_F, y_F] centre override.
%       route_fixture_xy_mm         Optional N-by-2 route; point order is used
%                                   when this field is absent.
%       route_indices               Optional point-order route indices.
%       visible                     Optional logical or 'on'/'off'.
%   target
%       index                       Attempt index in 1:26.
%   endpoint
%       measured_fixture_xy_mm      Optional N-by-2 measured endpoint batch.
%       command_fixture_xy_mm       Optional N-by-2 command endpoint batch.
%       break_before                Optional logical path separator.
%   phase
%       name                        Phase/status text.
%   complete
%       index                       Secured motion-lifecycle attempt index.
%       partition                   Optional effective partition override.
%   cone_complete
%       index                       Attempt whose cone trace is complete.
%   cone_batch
%       measured_fixture_xy_mm      Optional N-by-2 measured cone points.
%       command_fixture_xy_mm       Optional N-by-2 command-propagated points.
%       break_before                Optional logical path separator.
%   finalize
%       name                        Optional final status text.
%
% Every action may carry allow_drawnow=true.  Without that explicit field no
% draw is requested; endpoint paths remain buffered in the dashboard state.
% Even when allowed, rendering is limited to 2 Hz and uses
% "drawnow limitrate nocallbacks".  This helper owns no communication or
% scheduling resource and never reads robot/sensor data.
%
% Input coordinates are fixture coordinates.  The displayed mathematical
% top view is (u_right, v_up) = (-x_F, -y_F), so +X_F points left, +Y_F
% points down, and +Z_F points out of the fixture toward the observer.

if nargin < 1
    action = '';
end
if nargin < 2 || isempty(dashboard)
    dashboard = disabledDashboard('NOT_INITIALIZED');
end

try
    payload = mergePayloadArguments(varargin);
    assert(isstruct(payload) && isscalar(payload), ...
        'M12TopView:PayloadType', 'Payload must be a scalar struct.');
    action = lower(strtrim(char(action)));
    if strcmp(action, 'initialize')
        dashboard = initializeDashboard(dashboard, payload);
    else
        assertDashboardReady(dashboard);
        if ~dashboard.enabled
            return;
        end
        switch action
            case 'target'
                dashboard = applyTarget(dashboard, payload);
            case 'endpoint'
                dashboard = appendEndpoint(dashboard, payload);
            case 'phase'
                dashboard = applyPhase(dashboard, payload);
            case 'complete'
                dashboard = applyComplete(dashboard, payload);
            case 'cone_complete'
                dashboard = applyConeComplete(dashboard, payload);
            case 'cone_batch'
                dashboard = appendEndpoint(dashboard, payload);
            case 'finalize'
                dashboard = applyFinalize(dashboard, payload);
            otherwise
                error('M12TopView:UnknownAction', ...
                    'Unknown dashboard action "%s".', action);
        end
    end
    dashboard = renderIfExplicitlyAllowed(dashboard, payload, action);
catch dashboardError
    dashboard = disableAfterError(dashboard, dashboardError);
end
end

function dashboard = initializeDashboard(previousDashboard, payload)
if isstruct(previousDashboard) && isscalar(previousDashboard) && ...
        isfield(previousDashboard, 'figure') && ...
        isscalar(previousDashboard.figure) && ...
        isgraphics(previousDashboard.figure)
    delete(previousDashboard.figure);
end

sizeLabel = samplingSizeLabel(payload);
[points, pointFixtureXY, partitions, contactStates, ...
    pointNumbers, socketCenter] = ...
    unpackSamplingPlan(payload);
printedCad = unpackPrintedCadGeometry(payload, sizeLabel);
routeFixtureXY = unpackRoute(payload, pointFixtureXY);
visible = parseFigureVisibility(payload);

figureHandle = figure( ...
    'Name', sprintf('%s 26-point sampling top view', sizeLabel), ...
    'NumberTitle', 'off', ...
    'Color', 'w', ...
    'Visible', visible);
axesHandle = axes('Parent', figureHandle);
hold(axesHandle, 'on');
grid(axesHandle, 'on');
axis(axesHandle, 'equal');
box(axesHandle, 'on');
xlabel(axesHandle, 'display u [mm] = -x_F');
ylabel(axesHandle, 'display v_{up} [mm] = -y_F');
title(axesHandle, sprintf([ ...
    '%s sampling from +Z_F: +X_F left, +Y_F down, +Z_F toward viewer'], ...
    sizeLabel));

outerDiameterMm = printedCad.head_outer_diameter;
mouthAfMm = printedCad.socket_mouth_af;
wallAfMm = printedCad.socket_deep_wall_af;
throughHoleDiameterMm = printedCad.center_through_hole_diameter;
pegTipAfMm = printedCad.peg_tip_face_af;
pegTipFlatNormalYawDeg = ...
    printedCad.peg_tip_face_flat_normal_fixture_deg;

theta = linspace(0.0, 2.0 * pi, 121);
outerFixtureXY = socketCenter + ...
    0.5 * outerDiameterMm * [cos(theta(:)), sin(theta(:))];
outerUv = fixtureToDisplay(outerFixtureXY);
outerHandle = plot(axesHandle, outerUv(:, 1), outerUv(:, 2), ...
    '-', 'Color', [0.30, 0.34, 0.39], 'LineWidth', 1.4);

mouthFixtureXY = regularHexagon(socketCenter, mouthAfMm, 0.0);
mouthUv = fixtureToDisplay(mouthFixtureXY);
mouthHandle = patch(axesHandle, ...
    'XData', mouthUv(:, 1), 'YData', mouthUv(:, 2), ...
    'FaceColor', [0.98, 0.88, 0.90], 'FaceAlpha', 0.22, ...
    'EdgeColor', [0.65, 0.18, 0.28], 'LineWidth', 1.6);

wallFixtureXY = regularHexagon(socketCenter, wallAfMm, 0.0);
wallUv = fixtureToDisplay(wallFixtureXY);
wallHandle = patch(axesHandle, ...
    'XData', wallUv(:, 1), 'YData', wallUv(:, 2), ...
    'FaceColor', [1.00, 1.00, 1.00], 'FaceAlpha', 1.0, ...
    'EdgeColor', [0.48, 0.10, 0.18], ...
    'LineStyle', '--', 'LineWidth', 1.2);

throughHoleFixtureXY = socketCenter + ...
    0.5 * throughHoleDiameterMm * [cos(theta(:)), sin(theta(:))];
throughHoleUv = fixtureToDisplay(throughHoleFixtureXY);
throughHoleHandle = patch(axesHandle, ...
    'XData', throughHoleUv(:, 1), 'YData', throughHoleUv(:, 2), ...
    'FaceColor', [0.10, 0.10, 0.10], ...
    'EdgeColor', [0.10, 0.10, 0.10], 'LineWidth', 0.8);

routeUv = fixtureToDisplay(routeFixtureXY);
routeHandle = plot(axesHandle, routeUv(:, 1), routeUv(:, 2), ...
    ':', 'Color', [0.58, 0.61, 0.66], 'LineWidth', 1.0);

pointUv = fixtureToDisplay(pointFixtureXY);
pointHandles = gobjects(26, 1);
numberHandles = gobjects(26, 1);
for index = 1:26
    pointHandles(index) = plot(axesHandle, ...
        pointUv(index, 1), pointUv(index, 2), 'o', ...
        'LineStyle', 'none', ...
        'MarkerSize', 7.0, ...
        'MarkerEdgeColor', [0.42, 0.45, 0.49], ...
        'MarkerFaceColor', [0.78, 0.80, 0.83]);
    numberHandles(index) = text(axesHandle, ...
        pointUv(index, 1), pointUv(index, 2), ...
        sprintf('%d', pointNumbers(index)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 7.0, ...
        'FontWeight', 'bold', ...
        'Color', [0.05, 0.05, 0.05], ...
        'Visible', 'off');
end

measuredPathHandle = plot(axesHandle, NaN, NaN, ...
    '-', 'Color', [0.00, 0.35, 0.90], 'LineWidth', 1.8, ...
    'Marker', '.', 'MarkerSize', 7.0);
commandPathHandle = plot(axesHandle, NaN, NaN, ...
    '--', 'Color', [0.95, 0.45, 0.05], 'LineWidth', 1.2);
currentPegHandle = plot(axesHandle, NaN, NaN, ...
    '-', 'Color', [0.02, 0.02, 0.02], 'LineWidth', 2.0);

allFixtureXY = [pointFixtureXY; routeFixtureXY; outerFixtureXY];
allUv = fixtureToDisplay(allFixtureXY);
minimumUv = min(allUv, [], 1);
maximumUv = max(allUv, [], 1);
marginMm = 2.5;
xlim(axesHandle, [minimumUv(1) - marginMm, maximumUv(1) + marginMm]);
ylim(axesHandle, [minimumUv(2) - marginMm, maximumUv(2) + marginMm]);

axisSpan = max(maximumUv - minimumUv);
arrowLength = min(5.0, max(2.5, 0.14 * axisSpan));
arrowOrigin = [maximumUv(1) + 0.55 * marginMm, ...
    maximumUv(2) + 0.55 * marginMm];
quiver(axesHandle, arrowOrigin(1), arrowOrigin(2), ...
    -arrowLength, 0.0, 0.0, ...
    'Color', [0.08, 0.08, 0.08], ...
    'LineWidth', 1.4, 'MaxHeadSize', 0.55);
text(axesHandle, arrowOrigin(1) - arrowLength, arrowOrigin(2), ...
    '+X_F', 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
quiver(axesHandle, arrowOrigin(1), arrowOrigin(2), ...
    0.0, -arrowLength, 0.0, ...
    'Color', [0.08, 0.08, 0.08], ...
    'LineWidth', 1.4, 'MaxHeadSize', 0.55);
text(axesHandle, arrowOrigin(1), arrowOrigin(2) - arrowLength, ...
    '+Y_F', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'FontWeight', 'bold');
text(axesHandle, arrowOrigin(1) - 0.10 * arrowLength, ...
    arrowOrigin(2) - 0.10 * arrowLength, '\odot +Z_F', ...
    'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', 'FontWeight', 'bold');

statusHandle = text(axesHandle, 0.01, 0.99, ...
    'phase: INITIALIZED | cones: 0/26 | secured: 0/26 | target: none', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [1.0, 1.0, 1.0], ...
    'Margin', 3.0);
legend(axesHandle, ...
    [outerHandle, mouthHandle, wallHandle, throughHoleHandle, ...
     routeHandle, measuredPathHandle, commandPathHandle, currentPegHandle], ...
    {sprintf('%s outer D%.3f', sizeLabel, outerDiameterMm), ...
     sprintf('entry chamfer mouth AF%.2f', mouthAfMm), ...
     sprintf('deep wall AF%.2f', wallAfMm), ...
     sprintf('through-hole D%.1f', throughHoleDiameterMm), ...
     '26-point route', 'measured endpoint', 'command path', ...
     sprintf('current peg tip AF%.1f', pegTipAfMm)}, ...
    'Location', 'eastoutside');

dashboard = struct( ...
    'enabled', true, ...
    'failure', '', ...
    'sampling_size_label', sizeLabel, ...
    'figure', figureHandle, ...
    'axes', axesHandle, ...
    'points', points, ...
    'point_fixture_xy_mm', pointFixtureXY, ...
    'point_partitions', {partitions}, ...
    'point_contact_states', {contactStates}, ...
    'point_numbers', pointNumbers, ...
    'point_handles', pointHandles, ...
    'number_handles', numberHandles, ...
    'cone_completed_mask', false(26, 1), ...
    'committed_mask', false(26, 1), ...
    'completed_mask', false(26, 1), ...
    'target_index', 0, ...
    'phase', 'INITIALIZED', ...
    'socket_center_fixture_xy_mm', socketCenter, ...
    'route_fixture_xy_mm', routeFixtureXY, ...
    'outer_handle', outerHandle, ...
    'mouth_handle', mouthHandle, ...
    'wall_handle', wallHandle, ...
    'through_hole_handle', throughHoleHandle, ...
    'route_handle', routeHandle, ...
    'measured_path_handle', measuredPathHandle, ...
    'command_path_handle', commandPathHandle, ...
    'current_peg_handle', currentPegHandle, ...
    'status_handle', statusHandle, ...
    'measured_path_fixture_xy_mm', zeros(0, 2), ...
    'command_path_fixture_xy_mm', zeros(0, 2), ...
    'current_fixture_xy_mm', [NaN, NaN], ...
    'peg_tip_af_mm', pegTipAfMm, ...
    'peg_tip_flat_normal_yaw_deg', pegTipFlatNormalYawDeg, ...
    'printed_cad_geometry_mm', printedCad, ...
    'maximum_path_points', 20000, ...
    'maximum_refresh_hz', 2.0, ...
    'minimum_refresh_interval_s', 0.5, ...
    'refresh_clock', tic, ...
    'last_refresh_elapsed_s', -Inf, ...
    'finalized', false);
end

function dashboard = applyTarget(dashboard, payload)
index = payloadIndex(payload);
if dashboard.target_index >= 1 && dashboard.target_index <= 26
    dashboard = restorePointStyle(dashboard, dashboard.target_index);
end
dashboard.target_index = index;
set(dashboard.point_handles(index), ...
    'MarkerEdgeColor', [0.72, 0.48, 0.00], ...
    'MarkerFaceColor', [1.00, 0.78, 0.08], ...
    'MarkerSize', 9.0);
dashboard = maybeApplyPhaseFromPayload(dashboard, payload);
end

function dashboard = appendEndpoint(dashboard, payload)
measured = payloadMatrix(payload, 'measured_fixture_xy_mm');
command = payloadMatrix(payload, 'command_fixture_xy_mm');
breakBefore = payloadLogical(payload, 'break_before', false);

if breakBefore
    if ~isempty(dashboard.measured_path_fixture_xy_mm)
        dashboard.measured_path_fixture_xy_mm(end + 1, :) = [NaN, NaN];
    end
    if ~isempty(dashboard.command_path_fixture_xy_mm)
        dashboard.command_path_fixture_xy_mm(end + 1, :) = [NaN, NaN];
    end
end
if ~isempty(measured)
    dashboard.measured_path_fixture_xy_mm = ...
        appendBoundedPath(dashboard.measured_path_fixture_xy_mm, ...
        measured, dashboard.maximum_path_points);
end
if ~isempty(command)
    dashboard.command_path_fixture_xy_mm = ...
        appendBoundedPath(dashboard.command_path_fixture_xy_mm, ...
        command, dashboard.maximum_path_points);
end

if ~isempty(measured)
    dashboard.current_fixture_xy_mm = lastFiniteRow(measured);
elseif ~isempty(command)
    dashboard.current_fixture_xy_mm = lastFiniteRow(command);
end
% Keep endpoint samples in lightweight MATLAB state.  Graphics handles are
% synchronized only by the common 2 Hz render path, so a later phase/finalize
% action also flushes any endpoint that arrived inside the rate-limit window.
dashboard = maybeApplyPhaseFromPayload(dashboard, payload);
end

function dashboard = applyPhase(dashboard, payload)
dashboard.phase = payloadText(payload, {'name', 'phase', 'status'}, ...
    'PHASE');
end

function dashboard = applyComplete(dashboard, payload)
index = payloadIndex(payload);
partition = dashboard.point_partitions{index};
if isfield(payload, 'partition') && ~isempty(payload.partition)
    partition = char(payload.partition);
elseif isfield(payload, 'effective_partition') && ...
        ~isempty(payload.effective_partition)
    partition = char(payload.effective_partition);
end
partition = canonicalPartition(partition);
dashboard.point_partitions{index} = partition;
dashboard.committed_mask(index) = true;
dashboard.completed_mask(index) = true;
if dashboard.target_index == index
    dashboard.target_index = 0;
end
dashboard = restorePointStyle(dashboard, index);
set(dashboard.number_handles(index), 'Visible', 'on');
dashboard = maybeApplyPhaseFromPayload(dashboard, payload);
end

function dashboard = applyConeComplete(dashboard, payload)
index = payloadIndex(payload);
dashboard.cone_completed_mask(index) = true;
if dashboard.target_index ~= index
    dashboard = restorePointStyle(dashboard, index);
end
set(dashboard.number_handles(index), 'Visible', 'on');
dashboard = maybeApplyPhaseFromPayload(dashboard, payload);
end

function dashboard = applyFinalize(dashboard, payload)
if dashboard.target_index >= 1 && dashboard.target_index <= 26
    dashboard = restorePointStyle(dashboard, dashboard.target_index);
end
dashboard.target_index = 0;
dashboard.phase = payloadText(payload, {'name', 'phase', 'status'}, ...
    'FINALIZED');
dashboard.finalized = true;
end

function dashboard = restorePointStyle(dashboard, index)
if dashboard.committed_mask(index)
    partition = dashboard.point_partitions{index};
    color = partitionColor(partition);
    marker = 'o';
    markerSize = 8.0;
    if strcmp(partition, 'sentinel')
        marker = 'p';
        markerSize = 11.0;
    end
    set(dashboard.point_handles(index), ...
        'Marker', marker, ...
        'MarkerEdgeColor', color, ...
        'MarkerFaceColor', color, ...
        'MarkerSize', markerSize);
elseif dashboard.cone_completed_mask(index)
    set(dashboard.point_handles(index), ...
        'Marker', 'o', ...
        'MarkerEdgeColor', [0.00, 0.35, 0.90], ...
        'MarkerFaceColor', [0.72, 0.85, 1.00], ...
        'MarkerSize', 8.0);
else
    set(dashboard.point_handles(index), ...
        'Marker', 'o', ...
        'MarkerEdgeColor', [0.42, 0.45, 0.49], ...
        'MarkerFaceColor', [0.78, 0.80, 0.83], ...
        'MarkerSize', 7.0);
end
end

function dashboard = maybeApplyPhaseFromPayload(dashboard, payload)
if isfield(payload, 'phase') && ~isempty(payload.phase)
    dashboard.phase = char(payload.phase);
elseif isfield(payload, 'status') && ~isempty(payload.status)
    dashboard.phase = char(payload.status);
elseif isfield(payload, 'name') && ~isempty(payload.name)
    dashboard.phase = char(payload.name);
end
end

function dashboard = updateStatusText(dashboard)
if ~dashboard.enabled || ~isgraphics(dashboard.status_handle)
    return;
end
if dashboard.target_index >= 1
    targetText = sprintf('%d', ...
        dashboard.point_numbers(dashboard.target_index));
    targetText = sprintf('%s %s', targetText, strrep( ...
        dashboard.point_contact_states{dashboard.target_index}, '_', '-'));
else
    targetText = 'none';
end
set(dashboard.status_handle, 'String', sprintf( ...
    ['phase: %s | cones: %d/26 | secured: %d/26 | ', ...
     'target: %s'], ...
    strrep(dashboard.phase, '_', '\_'), ...
    sum(dashboard.cone_completed_mask), ...
    sum(dashboard.committed_mask), targetText));
end

function dashboard = renderIfExplicitlyAllowed(dashboard, payload, action)
forceRender = payloadLogical(payload, 'force_render', false) || ...
    strcmp(action, 'finalize');
if ~dashboard.enabled || ...
        ~payloadLogical(payload, 'allow_drawnow', false) || ...
        (~forceRender && ~refreshIsDue(dashboard))
    return;
end
dashboard = synchronizeDynamicGraphics(dashboard);
dashboard = updateStatusText(dashboard);
dashboard.last_refresh_elapsed_s = toc(dashboard.refresh_clock);
drawnow limitrate nocallbacks;
end

function dashboard = synchronizeDynamicGraphics(dashboard)
setPathHandle(dashboard.measured_path_handle, ...
    dashboard.measured_path_fixture_xy_mm);
setPathHandle(dashboard.command_path_handle, ...
    dashboard.command_path_fixture_xy_mm);
if all(isfinite(dashboard.current_fixture_xy_mm))
    pegFixtureXY = regularHexagon( ...
        dashboard.current_fixture_xy_mm, dashboard.peg_tip_af_mm, ...
        dashboard.peg_tip_flat_normal_yaw_deg);
    pegUv = fixtureToDisplay(pegFixtureXY);
    set(dashboard.current_peg_handle, ...
        'XData', pegUv(:, 1), 'YData', pegUv(:, 2));
end
end

function due = refreshIsDue(dashboard)
elapsed = toc(dashboard.refresh_clock);
due = elapsed - dashboard.last_refresh_elapsed_s >= ...
    dashboard.minimum_refresh_interval_s;
end

function assertDashboardReady(dashboard)
assert(isstruct(dashboard) && isscalar(dashboard) && ...
    isfield(dashboard, 'enabled'), ...
    'M12TopView:DashboardType', ...
    'Dashboard must be the scalar state returned by initialize.');
if ~dashboard.enabled
    return;
end
assert(isfield(dashboard, 'figure') && ...
    isgraphics(dashboard.figure) && ...
    isfield(dashboard, 'axes') && isgraphics(dashboard.axes), ...
    'M12TopView:FigureClosed', ...
    'Top-view figure or axes was closed.');
end

function dashboard = disableAfterError(dashboard, dashboardError)
if ~isstruct(dashboard) || ~isscalar(dashboard)
    dashboard = disabledDashboard('INVALID_DASHBOARD_STATE');
end
dashboard.enabled = false;
dashboard.failure = sprintf('%s: %s', ...
    dashboardError.identifier, oneLine(dashboardError.message));
if isfield(dashboard, 'status_handle') && ...
        isgraphics(dashboard.status_handle)
    try
        set(dashboard.status_handle, ...
            'String', ['dashboard disabled: ', dashboard.failure], ...
            'Color', [0.80, 0.05, 0.05]);
    catch
        % Plotting failures never escape into robot or sensor code.
    end
end
end

function dashboard = disabledDashboard(reason)
dashboard = struct( ...
    'enabled', false, ...
    'failure', char(reason), ...
    'figure', [], ...
    'axes', [], ...
    'last_refresh_elapsed_s', -Inf);
end

function payload = mergePayloadArguments(arguments)
payload = struct();
for argumentIndex = 1:numel(arguments)
    candidate = arguments{argumentIndex};
    if isempty(candidate)
        continue;
    end
    assert(isstruct(candidate) && isscalar(candidate), ...
        'M12TopView:PayloadType', ...
        'Every dashboard payload argument must be a scalar struct.');
    fields = fieldnames(candidate);
    for fieldIndex = 1:numel(fields)
        fieldName = fields{fieldIndex};
        if ~isfield(payload, fieldName)
            payload.(fieldName) = candidate.(fieldName);
        end
    end
end
end

function [points, pointFixtureXY, partitions, contactStates, ...
        pointNumbers, socketCenter] = ...
        unpackSamplingPlan(payload)
plan = payload;
if isfield(payload, 'plan') && ~isempty(payload.plan)
    plan = payload.plan;
end
if isfield(payload, 'points')
    points = payload.points;
elseif isstruct(plan) && isfield(plan, 'points')
    points = plan.points;
else
    error('M12TopView:PointsMissing', ...
        'Initialize payload must contain points or plan.points.');
end
if iscell(points)
    points = [points{:}];
end
if istable(points)
    points = table2struct(points);
end
assert(isstruct(points) && numel(points) == 26, ...
    'M12TopView:PointCount', ...
    'The M12 top view requires exactly 26 sampling points.');

socketCenter = [0.0, 0.0];
if isfield(payload, 'socket_center_fixture_xy_mm')
    socketCenter = rowVector2( ...
        payload.socket_center_fixture_xy_mm, 'socket centre');
elseif isstruct(plan) && isfield(plan, 'fixture_center_xy_mm')
    socketCenter = rowVector2(plan.fixture_center_xy_mm, 'socket centre');
elseif isstruct(plan) && isfield(plan, 'fixture_centre_xy_mm')
    socketCenter = rowVector2(plan.fixture_centre_xy_mm, 'socket centre');
elseif isstruct(plan) && isfield(plan, 'nominal_fixture_center_xy_mm')
    socketCenter = rowVector2( ...
        plan.nominal_fixture_center_xy_mm, 'socket centre');
end

pointFixtureXY = zeros(26, 2);
partitions = cell(26, 1);
contactStates = cell(26, 1);
pointNumbers = zeros(26, 1);
for index = 1:26
    point = points(index);
    if isfield(point, 'fixture_xy_mm')
        pointFixtureXY(index, :) = ...
            rowVector2(point.fixture_xy_mm, 'point fixture XY');
    elseif isfield(point, 'local_offset_fixture_mm')
        pointFixtureXY(index, :) = socketCenter + ...
            rowVector2(point.local_offset_fixture_mm, ...
            'point local fixture offset');
    else
        error('M12TopView:PointCoordinate', ...
            'Sampling point %d has no fixture coordinate.', index);
    end
    assert(isfield(point, 'partition'), ...
        'M12TopView:PointPartition', ...
        'Sampling point %d has no partition.', index);
    partitions{index} = canonicalPartition(point.partition);
    assert(isfield(point, 'contact_state') && ...
        ~isempty(point.contact_state), ...
        'M12TopView:PointContactState', ...
        'Sampling point %d has no printed-CAD contact state.', index);
    contactStates{index} = char(point.contact_state);
    if isfield(point, 'execution_index')
        pointNumbers(index) = double(point.execution_index);
    elseif isfield(point, 'attempt_index')
        pointNumbers(index) = double(point.attempt_index);
    else
        pointNumbers(index) = index;
    end
end
assert(numel(unique(pointNumbers)) == 26 && ...
    all(isfinite(pointNumbers)) && ...
    all(pointNumbers == round(pointNumbers)), ...
    'M12TopView:PointNumbers', ...
    'The 26 displayed attempt numbers must be unique finite integers.');
end

function geometry = unpackPrintedCadGeometry(payload, sizeLabel)
plan = payload;
if isfield(payload, 'plan') && ~isempty(payload.plan)
    plan = payload.plan;
end
assert(isstruct(plan) && isfield(plan, 'geometry_mm') && ...
    isstruct(plan.geometry_mm), ...
    'M12TopView:PrintedCadGeometryMissing', ...
    'Initialize payload must contain plan.geometry_mm from printed CAD.');
geometry = plan.geometry_mm;
required = {'head_outer_diameter', 'socket_mouth_af', ...
    'socket_deep_wall_af', 'center_through_hole_diameter', ...
    'peg_tip_face_af', 'peg_shaft_af', ...
    'peg_tip_face_flat_normal_fixture_deg'};
for index = 1:numel(required)
    fieldName = required{index};
    assert(isfield(geometry, fieldName) && ...
        isscalar(geometry.(fieldName)) && ...
        isfinite(double(geometry.(fieldName))), ...
        'M12TopView:PrintedCadGeometryField', ...
        'Printed-CAD geometry field %s is missing or invalid.', fieldName);
end
if strcmp(sizeLabel, 'M8')
    expected = [14.95, 8.1, 6.5, 0.5, 4.8, 6.0];
elseif strcmp(sizeLabel, 'M10')
    expected = [18.975, 10.54, 8.54, 0.5, 6.4, 8.0];
else
    expected = [21.275, 13.0, 10.6, 0.5, 8.0, 10.0];
end
assert(abs(double(geometry.head_outer_diameter) - expected(1)) < 1e-9 && ...
    abs(double(geometry.socket_mouth_af) - expected(2)) < 1e-9 && ...
    abs(double(geometry.socket_deep_wall_af) - expected(3)) < 1e-9 && ...
    abs(double(geometry.center_through_hole_diameter) - expected(4)) < 1e-9 && ...
    abs(double(geometry.peg_tip_face_af) - expected(5)) < 1e-9 && ...
    abs(double(geometry.peg_shaft_af) - expected(6)) < 1e-9, ...
    'M12TopView:PrintedCadGeometryChanged', ...
    'The active printed %s CAD dimensions changed.', sizeLabel);
end

function sizeLabel = samplingSizeLabel(payload)
sizeLabel = 'M12';
if isfield(payload, 'sampling_size_label') && ...
        ~isempty(payload.sampling_size_label)
    sizeLabel = upper(char(payload.sampling_size_label));
elseif isfield(payload, 'sampling_size') && ...
        ~isempty(payload.sampling_size)
    sizeLabel = upper(char(payload.sampling_size));
end
assert(strcmp(sizeLabel, 'M12') || strcmp(sizeLabel, 'M10') || ...
    strcmp(sizeLabel, 'M8'), ...
    'M12TopView:SamplingSize', ...
    'Sampling dashboard size must be M12, M10, or M8.');
end

function routeFixtureXY = unpackRoute(payload, pointFixtureXY)
if isfield(payload, 'route_fixture_xy_mm') && ...
        ~isempty(payload.route_fixture_xy_mm)
    routeFixtureXY = finiteMatrix2( ...
        payload.route_fixture_xy_mm, 'route fixture XY');
elseif isfield(payload, 'route_indices') && ...
        ~isempty(payload.route_indices)
    indices = double(payload.route_indices(:));
    assert(all(isfinite(indices)) && all(indices == round(indices)) && ...
        all(indices >= 1) && all(indices <= 26), ...
        'M12TopView:RouteIndices', ...
        'Route indices must be integers in 1:26.');
    routeFixtureXY = pointFixtureXY(indices, :);
else
    routeFixtureXY = pointFixtureXY;
end
assert(~isempty(routeFixtureXY), 'M12TopView:RouteEmpty', ...
    'The 26-point route cannot be empty.');
end

function visible = parseFigureVisibility(payload)
visible = 'on';
if ~isfield(payload, 'visible')
    return;
end
value = payload.visible;
if islogical(value) || isnumeric(value)
    assert(isscalar(value) && isfinite(double(value)), ...
        'M12TopView:VisibleValue', ...
        'visible must be a scalar logical or on/off text.');
    if logical(value)
        visible = 'on';
    else
        visible = 'off';
    end
else
    value = lower(char(value));
    assert(strcmp(value, 'on') || strcmp(value, 'off'), ...
        'M12TopView:VisibleValue', ...
        'visible text must be on or off.');
    visible = value;
end
end

function index = payloadIndex(payload)
if isfield(payload, 'index')
    index = double(payload.index);
elseif isfield(payload, 'attempt_index')
    index = double(payload.attempt_index);
elseif isfield(payload, 'target_index')
    index = double(payload.target_index);
else
    error('M12TopView:IndexMissing', ...
        'This dashboard action requires an attempt index.');
end
assert(isscalar(index) && isfinite(index) && ...
    index == round(index) && index >= 1 && index <= 26, ...
    'M12TopView:IndexRange', ...
    'Attempt index must be an integer in 1:26.');
end

function value = payloadMatrix(payload, fieldName)
if ~isfield(payload, fieldName) || isempty(payload.(fieldName))
    value = zeros(0, 2);
    return;
end
value = double(payload.(fieldName));
assert(ismatrix(value) && size(value, 2) == 2 && ...
    all(all(isfinite(value) | isnan(value))) && ...
    all(all(isfinite(value), 2) | all(isnan(value), 2)), ...
    'M12TopView:PathCoordinate', ...
    '%s must be N-by-2 finite rows with optional [NaN NaN] breaks.', ...
    fieldName);
end

function value = payloadLogical(payload, fieldName, defaultValue)
value = defaultValue;
if ~isfield(payload, fieldName)
    return;
end
candidate = payload.(fieldName);
assert((islogical(candidate) || isnumeric(candidate)) && ...
    isscalar(candidate) && isfinite(double(candidate)), ...
    'M12TopView:LogicalField', ...
    '%s must be a finite scalar logical value.', fieldName);
value = logical(candidate);
end

function textValue = payloadText(payload, candidates, defaultValue)
textValue = defaultValue;
for index = 1:numel(candidates)
    fieldName = candidates{index};
    if isfield(payload, fieldName) && ~isempty(payload.(fieldName))
        textValue = char(payload.(fieldName));
        return;
    end
end
end

function path = appendBoundedPath(path, additions, maximumPoints)
path = [path; additions];
if size(path, 1) > maximumPoints
    path = path(end - maximumPoints + 1:end, :);
end
end

function setPathHandle(handle, fixtureXY)
if isempty(fixtureXY)
    set(handle, 'XData', NaN, 'YData', NaN);
    return;
end
uv = fixtureToDisplay(fixtureXY);
set(handle, 'XData', uv(:, 1), 'YData', uv(:, 2));
end

function row = lastFiniteRow(values)
index = find(all(isfinite(values), 2), 1, 'last');
assert(~isempty(index), 'M12TopView:CurrentPointMissing', ...
    'Path batch contains no finite point for the current peg marker.');
row = values(index, :);
end

function value = rowVector2(value, description)
value = double(value(:).');
assert(numel(value) == 2 && all(isfinite(value)), ...
    'M12TopView:CoordinateShape', ...
    '%s must contain two finite coordinates.', description);
end

function value = finiteMatrix2(value, description)
value = double(value);
assert(ismatrix(value) && size(value, 2) == 2 && ...
    all(isfinite(value(:))), ...
    'M12TopView:CoordinateMatrix', ...
    '%s must be a finite N-by-2 matrix.', description);
end

function uv = fixtureToDisplay(fixtureXY)
uv = -double(fixtureXY);
end

function vertices = regularHexagon( ...
        centerFixtureXY, acrossFlatsMm, flatNormalYawDeg)
if nargin < 3
    flatNormalYawDeg = 0.0;
end
centerFixtureXY = rowVector2(centerFixtureXY, 'hexagon centre');
assert(isscalar(acrossFlatsMm) && isfinite(acrossFlatsMm) && ...
    acrossFlatsMm > 0.0 && isscalar(flatNormalYawDeg) && ...
    isfinite(flatNormalYawDeg), ...
    'M12TopView:HexagonSize', ...
    'Hexagon across-flats size must be positive and finite.');
radiusMm = acrossFlatsMm / sqrt(3.0);
angles = deg2rad(flatNormalYawDeg + 30.0 + 60.0 * (0:6));
vertices = centerFixtureXY + ...
    radiusMm * [cos(angles(:)), sin(angles(:))];
end

function partition = canonicalPartition(partition)
partition = lower(strtrim(char(partition)));
valid = {'clean_candidate', 'holdout', 'quarantine', 'sentinel'};
assert(any(strcmp(partition, valid)), ...
    'M12TopView:Partition', ...
    'Unknown sampling partition "%s".', partition);
end

function color = partitionColor(partition)
switch canonicalPartition(partition)
    case 'clean_candidate'
        color = [0.10, 0.40, 0.90];
    case 'holdout'
        color = [0.95, 0.55, 0.10];
    case 'quarantine'
        color = [0.85, 0.18, 0.14];
    case 'sentinel'
        color = [0.55, 0.25, 0.75];
end
end

function text = oneLine(text)
text = regexprep(char(text), '[\r\n]+', ' ');
text = strtrim(text);
end

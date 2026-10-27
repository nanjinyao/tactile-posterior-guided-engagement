function dashboard = m8_sampling_topview_dashboard(action, dashboard, varargin)
%M8_SAMPLING_TOPVIEW_DASHBOARD M8 facade over the shared top-view renderer.
%
% The M8 facade pins the printed-CAD size identity while preserving the
% field-proven M12/M10 action interface and the rule that UI failures never
% stop acquisition.  The shared renderer validates the active printed M8
% geometry before it creates the M8-labelled figure.

sizePayload = struct('sampling_size_label', 'M8');
dashboard = m12_sampling_topview_dashboard( ...
    action, dashboard, sizePayload, varargin{:});
end

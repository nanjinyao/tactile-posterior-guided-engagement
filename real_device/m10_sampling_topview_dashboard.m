function dashboard = m10_sampling_topview_dashboard(action, dashboard, varargin)
%M10_SAMPLING_TOPVIEW_DASHBOARD M10 facade over the shared top-view renderer.
%
% The M10 facade pins the printed-CAD size identity while preserving the
% field-proven M12 action interface and the rule that UI failures never stop
% acquisition.

sizePayload = struct('sampling_size_label', 'M10');
dashboard = m12_sampling_topview_dashboard( ...
    action, dashboard, sizePayload, varargin{:});
end

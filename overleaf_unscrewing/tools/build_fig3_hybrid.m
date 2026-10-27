function build_fig3_hybrid()
%BUILD_FIG3_HYBRID Build the action--inference--execution workflow in Fig. 3.
% The real-video keyframe strip anchors the physical motion.  The tactile
% traces, circular summary, KUKA photograph, routing geometry, arrows, and all
% typography are retained from data or drawn deterministically here.

root = fileparts(fileparts(mfilename('fullpath')));
fig_dir = fullfile(root, 'figures');

probe_frames = imread(fullfile(fig_dir, 'fig3_real_probe_keyframes.png'));
[ph, ~, ~] = size(probe_frames);
probe_frames = probe_frames(round(0.12*ph):round(0.90*ph), :, :);

source = imread(fullfile(fig_dir, 'tactile_fingerprint_and_direction.png'));
[sh, sw, ~] = size(source);
fingerprint = source(1:min(sh,900), 1:min(sw,1435), :);
posterior = source(1:min(sh,858), min(sw,1450):sw, :);

hardware = imread(fullfile(fig_dir, 'fig1_hardware_coordinate_frames.png'));
[hh, hw, ~] = size(hardware);
hardware = hardware(round(0.035*hh):round(0.90*hh), ...
    round(0.220*hw):round(0.685*hw), :);

ink = [0.09 0.09 0.09];
blue = [0.12 0.42 0.72];
green = [0.08 0.58 0.32];
red = [0.82 0.18 0.17];
grey = [0.42 0.44 0.47];

fig = figure('Visible','off','Color','w','Position',[80 80 1900 760]);
ax = axes(fig,'Position',[0 0 1 1]);
hold(ax,'on'); axis(ax,[0 1900 0 760]); axis(ax,'off');
set(ax,'YDir','reverse');

% Top row: acquire one tactile observation and infer its direction.
r1 = [55 62 650 260];
r3 = [760 62 550 260];
r4 = [1360 62 360 260];
place_image_panel(ax, probe_frames, r1, blue);
draw_conical_overlay(ax, r1);
place_image_panel(ax, fingerprint, r3, blue);
place_image_panel_fit(ax, posterior, r4, blue);

panel_title(ax, r1, 'Real $15^\circ$ conical probing sequence', ink);
panel_title(ax, r3, 'Four-channel tactile fingerprint', ink);
panel_title(ax, r4, 'Calibrated circular direction summary', ink);

flow_arrow(ax, r1(1)+r1(3), 192, r3(1)-r1(1)-r1(3), 0, blue);
flow_arrow(ax, r3(1)+r3(3), 192, r4(1)-r3(1)-r3(3), 0, blue);
text(ax,1335,174,'CNN ensemble','HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontSize',12,'FontWeight','bold', ...
    'Color',grey);

% Bottom row: posterior-supported route, guarded motion, and verification.
rr = [1375 474 335 185];
rc = [1020 492 275 145];
rh = [665 435 275 225];
rv = [390 510 195 92];

rounded_box(ax, rr, green, 1.45);
draw_route_proposal(ax, rr, ink, blue, green, [0.90 0.62 0.00], grey);
text(ax,rr(1)+rr(3)/2,451,'Bounded route proposal', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',16,'FontWeight','bold','Color',ink);

rounded_box(ax, rc, red, 1.35);
text(ax,rc(1)+rc(3)/2,rc(2)+47,'Admittance / RCC', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',15,'Color',ink);
text(ax,rc(1)+rc(3)/2,rc(2)+82,'wrench and tilt guards', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',12.5,'FontWeight','bold','Color',red);
text(ax,rc(1)+rc(3)/2,rc(2)+113,'unload / retreat / recover', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',11,'Color',grey);

place_image_panel(ax, hardware, rh, [0.22 0.22 0.22]);
text(ax,rh(1)+rh(3)/2,414,'KUKA LBR iiwa + hex driver', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',16,'FontWeight','bold','Color',ink);

rounded_box(ax, rv, blue, 1.35);
text(ax,rv(1)+rv(3)/2,rv(2)+36,'Terminal verifier', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',14,'Color',ink);
text(ax,rv(1)+rv(3)/2,rv(2)+67,'success / abort', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',11.5,'Color',grey);

% Clockwise execution path, matching the compact loop used in the reference.
flow_arrow(ax, r4(1)+r4(3)/2, r4(2)+r4(4), 0, ...
    rr(2)-r4(2)-r4(4), green);
flow_arrow(ax, rr(1), rr(2)+rr(4)/2, ...
    rc(1)+rc(3)-rr(1), 0, green);
flow_arrow(ax, rc(1), rc(2)+rc(4)/2, ...
    rh(1)+rh(3)-rc(1), 0, blue);
flow_arrow(ax, rh(1), rh(2)+rh(4)/2, ...
    rv(1)+rv(3)-rh(1), 0, blue);
flow_arrow(ax, rv(1), rv(2)+rv(4)/2, -150, 0, blue);
text(ax,205,545,'verified','HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontSize',11.5,'FontWeight','bold', ...
    'Color',blue);

% A failed terminal check returns physical negative evidence before re-probing.
plot(ax,[rv(1)+rv(3)/2 rv(1)+rv(3)/2 r1(1)+r1(3)/2], ...
    [rv(2) 370 370],'--','Color',green,'LineWidth',1.65);
flow_arrow(ax,r1(1)+r1(3)/2,370,0,r1(2)+r1(4)-370,green);
text(ax,365,356,'miss: update belief and re-probe', ...
    'HorizontalAlignment','left','FontName','Times New Roman', ...
    'FontSize',12,'FontAngle','italic','FontWeight','bold','Color',green);

out = fullfile(fig_dir,'fig3_action_observation_routing.png');
exportgraphics(ax,out,'Resolution',220,'BackgroundColor','white');
close(fig);
end


function place_image_panel(ax, img, r, edge)
rectangle(ax,'Position',r,'Curvature',0.08,'FaceColor','white', ...
    'EdgeColor','none');
pad = 7;
image(ax,'XData',[r(1)+pad r(1)+r(3)-pad], ...
    'YData',[r(2)+pad r(2)+r(4)-pad],'CData',img);
rectangle(ax,'Position',r,'Curvature',0.08,'FaceColor','none', ...
    'EdgeColor',edge,'LineWidth',1.7);
end


function place_image_panel_fit(ax, img, r, edge)
rectangle(ax,'Position',r,'Curvature',0.08,'FaceColor','white', ...
    'EdgeColor','none');
pad = 7;
avail_w = r(3)-2*pad;
avail_h = r(4)-2*pad;
[ih, iw, ~] = size(img);
img_ratio = iw/ih;
if avail_w/avail_h > img_ratio
    draw_h = avail_h;
    draw_w = img_ratio*draw_h;
else
    draw_w = avail_w;
    draw_h = draw_w/img_ratio;
end
x0 = r(1)+(r(3)-draw_w)/2;
y0 = r(2)+(r(4)-draw_h)/2;
image(ax,'XData',[x0 x0+draw_w],'YData',[y0 y0+draw_h],'CData',img);
rectangle(ax,'Position',r,'Curvature',0.08,'FaceColor','none', ...
    'EdgeColor',edge,'LineWidth',1.7);
end


function draw_conical_overlay(ax, r)
% Explicit geometric cue over the real sequence.  The cone is an annotation
% only: the three hardware poses remain the unaltered video-derived images.
red = [0.88 0.18 0.14];
ink = [0.10 0.10 0.10];
centers = r(1) + r(3)*[1/6 1/2 5/6];
labels = {'(i) Initial contact','(ii) Probing phase','(iii) Conical scan'};
for k = 1:3
    text(ax,centers(k),r(2)+18,labels{k},'HorizontalAlignment','center', ...
        'FontName','Times New Roman','FontSize',10.5,'FontWeight','bold', ...
        'Color',ink,'BackgroundColor','white','Margin',1.2);
end

% The middle frame receives the compact cone: its apex is the observed
% contact region, its dashed centerline is the nominal vertical, and the
% ellipse denotes one revolution of the 15-degree tilted tool axis.
apex = [centers(2)+20, r(2)+0.75*r(4)];
ellipse_c = [centers(2)+20, r(2)+0.40*r(4)];
rx = 38;
ry = 9;
t = linspace(0.12*pi,1.88*pi,160);
plot(ax,ellipse_c(1)+rx*cos(t),ellipse_c(2)+ry*sin(t),'--', ...
    'Color',red,'LineWidth',1.7);
plot(ax,[apex(1) ellipse_c(1)],[apex(2) ellipse_c(2)],'--', ...
    'Color',red,'LineWidth',1.35);
plot(ax,[apex(1) ellipse_c(1)-rx],[apex(2) ellipse_c(2)],':', ...
    'Color',red,'LineWidth',1.25);
plot(ax,[apex(1) ellipse_c(1)+rx],[apex(2) ellipse_c(2)],':', ...
    'Color',red,'LineWidth',1.25);
plot(ax,apex(1),apex(2),'o','MarkerSize',6.5,'MarkerFaceColor',red, ...
    'MarkerEdgeColor','w','LineWidth',0.9);

q = round(0.69*numel(t));
quiver(ax,ellipse_c(1)+rx*cos(t(q)),ellipse_c(2)+ry*sin(t(q)), ...
    -7*sin(t(q)),1.8*cos(t(q)),0,'Color',red,'LineWidth',1.5, ...
    'MaxHeadSize',0.9,'AutoScale','off');
text(ax,ellipse_c(1)+44,ellipse_c(2)-14,'$\phi:0\!\rightarrow\!2\pi$', ...
    'Interpreter','latex','FontName','Times New Roman','FontSize',10.5, ...
    'FontWeight','bold','Color',red,'BackgroundColor','white','Margin',1);
text(ax,apex(1)+19,apex(2)-39,'$\alpha=15^\circ$', ...
    'Interpreter','latex','FontName','Times New Roman','FontSize',10.5, ...
    'FontWeight','bold','Color',red,'BackgroundColor','white','Margin',1);
end


function rounded_box(ax, r, edge, width)
rectangle(ax,'Position',r,'Curvature',0.10,'FaceColor','white', ...
    'EdgeColor',edge,'LineWidth',width);
end


function panel_title(ax, r, label, color)
interpreter = 'none';
if contains(label,'$')
    interpreter = 'latex';
end
text(ax,r(1)+r(3)/2,r(2)-18,label,'HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontSize',16,'FontWeight','bold', ...
    'Interpreter',interpreter,'Color',color);
end


function flow_arrow(ax, x, y, dx, dy, color)
quiver(ax,x,y,dx,dy,0,'Color',color,'LineWidth',1.9, ...
    'MaxHeadSize',min(0.8,18/max(18,hypot(dx,dy))),'AutoScale','off');
end


function draw_route_proposal(ax, r, ink, blue, green, orange, grey)
mapx = @(u) r(1) + u*r(3);
mapy = @(v) r(2) + v*r(4);

theta = linspace(0,2*pi,7)+pi/6;
plot(ax,mapx(0.78+0.10*cos(theta)),mapy(0.49+0.20*sin(theta)), ...
    '-','Color',ink,'LineWidth',1.0);
t = linspace(0,2*pi,220);
plot(ax,mapx(0.76+0.20*cos(t)),mapy(0.49+0.34*sin(t)), ...
    '--','Color',green,'LineWidth',1.0);

x0 = [mapx(0.10) mapy(0.78)];
xs = [mapx(0.52) mapy(0.57)];
plot(ax,x0(1),x0(2),'o','MarkerSize',4.8,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',ink,'LineWidth',0.9);
quiver(ax,x0(1),x0(2),xs(1)-x0(1),xs(2)-x0(2),0, ...
    'Color',blue,'LineWidth',1.8,'MaxHeadSize',0.28,'AutoScale','off');
plot(ax,xs(1),xs(2),'o','MarkerSize',5,'MarkerFaceColor',blue, ...
    'MarkerEdgeColor','w','LineWidth',0.7);

s = linspace(0,3.1*pi,160);
rad = linspace(0.012,0.18,numel(s));
plot(ax,mapx(0.53+rad.*cos(s)),mapy(0.58+0.62*rad.*sin(s)), ...
    '-','Color',orange,'LineWidth',1.15);

text(ax,mapx(0.08),mapy(0.91),'$x_0$','Interpreter','latex', ...
    'FontName','Times New Roman','FontSize',10.5,'Color',ink);
text(ax,mapx(0.31),mapy(0.65),'$L^*\hat d$','Interpreter','latex', ...
    'FontName','Times New Roman','FontSize',10.5,'Color',blue);
text(ax,mapx(0.50),mapy(0.48),'$x_s$','Interpreter','latex', ...
    'FontName','Times New Roman','FontSize',10.5,'Color',blue);
text(ax,mapx(0.72),mapy(0.09),'residual support', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',10.5,'FontWeight','bold','Color',green);
text(ax,mapx(0.50),mapy(0.93),'capped shortcut + residual search', ...
    'HorizontalAlignment','center','FontName','Times New Roman', ...
    'FontSize',10.5,'Color',grey);
end

function pointCloudIndex=ClickA3DPoint(pointCloud,figure_used,node_selected,no_new_point)
% CALLBACKCLICK3DPOINT mouse click callback function for CLICKA3DPOINT
%
%   The transformation between the viewing frame and the point cloud frame
%   is calculated using the camera viewing direction and the 'up' vector.
%   Then, the point cloud is transformed into the viewing frame. Finally,
%   the z coordinate in this frame is ignored and the x and y coordinates
%   of all the points are compared with the mouse click location and the
%   closest point is selected.
%
%   Babak Taati - May 4, 2005
%   revised Oct 31, 2007
%   revised Jun 3, 2008
%   revised May 19, 2009

if nargin==2
    dcm_obj = datacursormode(figure_used);
    datacursormode on;
    set(figure_used,'Visible','on');
    w = waitforbuttonpress;
    if ~w
        info_struct = getCursorInfo(dcm_obj);
        while isempty(info_struct)
            pause(0.01)
            info_struct = getCursorInfo(dcm_obj);
        end
        pos_coordinates = info_struct.Position;
        pointCloudIndex = dsearchn(pointCloud, ...
            pos_coordinates);     
    else    
       pointCloudIndex = [];
       return;
    end
elseif nargin==3
    pointCloudIndex=node_selected;
    selectedPoint = pointCloud(pointCloudIndex,:);
    hold on,
    % highlight the selected point
    h_1 = scatter3(selectedPoint(1:end-1,1), selectedPoint(1:end-1,2), ...
        selectedPoint(1:end-1,3), 'filled',...
        'MarkerFaceColor','k');
    alpha(h_1,.4)
    h_2 = plot3(selectedPoint(end,1), selectedPoint(end,2), ...
        selectedPoint(end,3), 'r.', 'MarkerSize', 20);
    set(h_1,'Tag','pt'); % set its Tag property for later use
    set(h_2,'Tag','pt'); % set its Tag property for later use
    hold off,
else
    pointCloudIndex=node_selected;
    selectedPoint = pointCloud(pointCloudIndex,:);
    hold on,
    % highlight the selected point
    h_1 = plot3(selectedPoint(1), selectedPoint(2), ...
        selectedPoint(3),'r.','MarkerSize', 10);
    %alpha(h_1,.4)
    set(h_1,'Tag','pt'); % set its Tag property for later use
    hold off,
end
function atrialrepresentation (Model, MAG, ming, maxg)
% It displays an instant time.

% Input variables:
% Model: structure´s atria.
% MAG: potentials.
% ming: minimal value of the scale.
% maxg: maximum value of the scale.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

faces    = Model.face; % in this structure´s model they are triangles.
vertices = Model.node;  % nodes.

if nargin >2
    caxis([ming maxg]);
else
    maxCAXIS=max(max(abs( MAG(:,2:end) )));
    caxis(maxCAXIS.*[-1 1]);
end

patch('Faces', faces,'Vertices', vertices,'FaceVertexCData', MAG,'FaceColor','interp','EdgeAlpha',0);  
xlabel('X'); ylabel('Y'); zlabel('Z');
axis equal;
axis off
caxis([ming maxg]);
shading interp
light
lighting gouraud

view(90,30);
%zoom(1.13);
%set(gca,'Position',[0.09 0.05 0.42 1]);
camlight('headlight'),
material([0.3 0.6 0.3]),

end
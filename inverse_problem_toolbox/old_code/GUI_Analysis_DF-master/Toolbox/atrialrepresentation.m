function atrialrepresentation (Model, MAG, ming, maxg)
% It displays an instant time.

% Input variables:
% Model: structure´s atria.
% MAG: potentials.
% ming: minimal value of the scale.
% maxg: maximum value of the scale.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

faces    = Model.faces(1:3860,:); % in this structure´s model they are triangles.
vertices = Model.vertices(1:2039,:);  % nodes.

if nargin >2
    caxis([ming maxg]);
else
    maxCAXIS=max(max(abs( MAG(:,2:end) )));
    caxis(maxCAXIS.*[-1 1]);
end

window1=subplot(121);
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
set(window1,'Position',[0.09 0.05 0.42 1]);
camlight('headlight'),
material([0.3 0.6 0.3]),

window2=subplot(122);
patch('Faces', faces,'Vertices', vertices,'FaceVertexCData', MAG,'FaceColor','interp','EdgeAlpha',0);  
xlabel('X'); ylabel('Y'); zlabel('Z');
axis equal;
axis off
caxis([ming maxg]);
shading interp
light
lighting gouraud

view(-90,30);
camlight('headlight'),
camlight('headlight'),
material([0.3 0.6 0.3]),
%zoom(1.13);
set(window2,'Position',[0.48 0 0.42 1]);

h=colorbar('southoutside');
set(h, 'Position', [0.1 0.175 .815 .03]);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',25)

end
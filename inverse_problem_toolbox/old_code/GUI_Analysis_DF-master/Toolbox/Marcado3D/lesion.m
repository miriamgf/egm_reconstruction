function cel_fibr=lesion(nombre_proyecto,Mod_XL,indices);

umbral=0.04 * (max(Mod_XL.vertices(:,1))-min(Mod_XL.vertices(:,1)));

load indices_vecinos
lesions=indices_vecinos(indices);

ind=zeros(size(Mod_XL.vertices,1),1);

for i=1:length(lesions)
    R=ones(size(Mod_XL.vertices,1),1)*Mod_XL.vertices(lesions(i),:) - Mod_XL.vertices;
    R=sqrt(sum(R.^2,2));
    ind(R<=umbral)=1;
end

cel_fibr=find(ind==1);
% 
load 140709_AFMiguel
generaFib(FV2,cel_fibr,nombre_proyecto);

figure
hold on
axis equal
plot3(Mod_XL.vertices(ind==0,1),Mod_XL.vertices(ind==0,2),Mod_XL.vertices(ind==0,3),'b.')
plot3(Mod_XL.vertices(ind==1,1),Mod_XL.vertices(ind==1,2),Mod_XL.vertices(ind==1,3),'r.')

function cel_fibr=fibrosis(nombre_proyecto,Mod_XL);

load indices_vecinos
% pos=Mod_XL.vertices(indices_vecinos(5072),:);
% pos=Mod_XL.vertices(indices_vecinos(5196),:); %%LA
% pos=Mod_XL.vertices(indices_vecinos(1176),:);   %%RA
% pos=Mod_XL.vertices(indices_vecinos(1094),:);   %%RA
% pos=Mod_XL.vertices(indices_vecinos(813),:);   %%RAA
pos=Mod_XL.vertices(indices_vecinos(4881),:);   %%LSPV
% pos=Mod_XL.vertices(indices_vecinos(4415),:);   %%RSPV
% pos=Mod_XL.vertices(indices_vecinos(5271),:);   %%LIPV
% pos=Mod_XL.vertices(indices_vecinos(5245),:);   %%RIPV

% radio1=3e4; %ancho
radio1=1.5e4;  %estrecho
% radio1=2e4; %medio
% radio1=1e4;  %MUY estrecho
p_lineal=0.25;

R=ones(size(Mod_XL.vertices,1),1)*pos - Mod_XL.vertices;
R=sqrt(sum(R.^2,2));

ind_rot=find(R<=radio1);
ind_fuera=find(R>radio1);

N_fibr=floor(p_lineal*length(ind_fuera));
p=randperm(length(ind_fuera));
cel_fibr=ind_fuera(p(1:N_fibr));

load 140709_AFMiguel
generaFib(FV2,cel_fibr,nombre_proyecto);

load Mod_S
load indices_vecinos
vector=zeros(size(Mod_XL.vertices,1),1);
vector(cel_fibr)=1;
imagen_torso(Mod_S,vector(indices_vecinos))
colormap(jet)

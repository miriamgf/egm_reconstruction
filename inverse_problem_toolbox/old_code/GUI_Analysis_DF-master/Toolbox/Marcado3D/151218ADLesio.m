%Carreguem el model sinus
load('140707_AurAniso.mat')
load('E:\0Docs\MATLAB\20140702_Marcado3D\ModelsGraniMenut')

close all
[xo,vo]=definelinea2puntos(Mod_S);
    pas=[];
    for i=1:3
    pas(:,i)=Mod_XL.vertices(:,i)-xo(i);
    end
    dists=[];
    for i=1:length(pas)
    dists(i)=norm(cross(pas(i,:),vo))/norm(vo);
    end
    lesio=find(dists<2); %4
    
jep=find(Mod_XL.vertices(lesio,2)<1); lesio(jep)=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%     dibuixa 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
patch('Faces',Mod_S.faces,'Vertices',Mod_S.vertices,'facecolor','c','edgealpha',0.0) 
hold on;    axis equal;
% light('Style','infinite');
    punts=zonaLent;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'*k')
    punts=lesio;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'*r')
   axis equal
   axis off

   %Reduce Conductividad cerca de la lesion
    RemMax=1.0; Remmin=0.5; distmax=10; distmin=0.1;
    m=(Remmin-RemMax)/(distmin-distmax);
%     X1=RemMax+abs(m*distmax)
%     Rem= dists*m+X1;
    Rem= dists*m+Remmin;
    Rem(find(Rem>RemMax))=RemMax;
    Rem(find(Rem<Remmin))=Remmin;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%
%Generar model
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
nom='151218ADLesion';
load('140707_AurAniso.mat')
[FV2]=generaFib(FV2,lesio);
%Reduce Conductividad cerca de la lesion
for i=size(FV2.conductancias,1)-1
    FV2.conductancias(i,:)=FV2.conductancias(i,:)*Rem(i);
end
save('151218ADLesion.mat','FV2','lesio','dists','xo','vo');
guardaConductancias(FV2,nom);
guardaVecinos(FV2,nom);

%%%%%%%%%%%%%%%%%%%%%
% Para iniciar
%%%%%%%%%%%%%%%%%%%%%
load('E:\0Docs\MATLAB\20140702_Marcado3D\ModelsGraniMenut')
load('151218ADLesion.mat')
close all
[xo,vo]=definelinea2puntos(Mod_S,Mod_XL,lesio);
    pas=[];
    for i=1:3
    pas(:,i)=Mod_XL.vertices(:,i)-xo(i);
    end
    dists=[];
    for i=1:length(pas)
    dists(i)=norm(cross(pas(i,:),vo))/norm(vo);
    end
    lesioST=find(dists<4); %4
    
close all
[xo,vo]=definelinea2puntos(Mod_S,Mod_XL,lesioST);
    pas=[];
    for i=1:3
    pas(:,i)=Mod_XL.vertices(:,i)-xo(i);
    end
    dists=[];
    for i=1:length(pas)
    dists(i)=norm(cross(pas(i,:),vo))/norm(vo);
    end
    stim=find(dists<2); %4
    
    jep=find(Mod_XL.vertices(lesioST,1)<-24); lesioST2=lesioST; lesioST2(jep)=[];
    jep=find(Mod_XL.vertices(stim,1)<-24); stim2=stim; stim2(jep)=[];
    
    stim2 = setdiff(stim2,lesioST2);
    stim2 = setdiff(stim2,lesio);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%     dibuixa 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
patch('Faces',Mod_S.faces,'Vertices',Mod_S.vertices,'facecolor','c','edgealpha',0.0) 
hold on;    axis equal;
% light('Style','infinite');
    punts=lesio;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'*k')
    punts=lesioST2;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'*r')
       punts=stim2;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'*y')
   axis equal    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%guarda
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nom='151220ADLesionini';
[FV2]=generaFib(FV2,lesioST2);
save('151218ADLesionin.mat','FV2','lesio','lesioST2','stim2');
guardaConductancias(FV2,nom);
guardaVecinos(FV2,nom);

    stim=stim2(1:10:end);
    guardaStim(stim,stim,nom);

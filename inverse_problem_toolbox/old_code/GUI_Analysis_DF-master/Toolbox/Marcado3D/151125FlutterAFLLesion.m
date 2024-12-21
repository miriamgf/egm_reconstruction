% en Koivumaki
% load('140613Aur.mat')
load('E:\0Docs\MATLAB\20140702_Marcado3D\140613Aur')
load('E:\0Docs\MATLAB\20140702_Marcado3D\ModelsGraniMenut')
%directament mod

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Seleccion de nodos:
close all
% [xo,vo]=definelinea2puntos(Mod_S);
[xo,vo]=definelinea2puntos(Mod_S,Mod_XL, lesio6);
    pas=[];
    for i=1:3
    pas(:,i)=Mod_XL.vertices(:,i)-xo(i);
    end
    dists=[];
    for i=1:length(pas)
    dists(i)=norm(cross(pas(i,:),vo))/norm(vo);
    end
    zona=find(dists<12); %4
lesio6=[lesio6 zona];  lesio6=sort(unique(lesio6)); 
lesio7=zona;    lesio7=[lesio7 lesio5];     lesio7=sort(unique(lesio7));
lesio8=zona;    lesio8=[lesio8 lesio4];     lesio8=sort(unique(lesio8));
jepa=find(Mod_XL.vertices(lesio8,1)<0);     lesio8(jepa)=[];
% lesio6=zona;
    
% lesio5=zona;
lesio5= [lesio5 zona]; lesio5=sort(unique(lesio5));
lesiopersi=lesio5;
jepa=find(Mod_XL.vertices(lesio5,1)<4.5);  lesio5(jepa)=[];
jepa=find(Mod_XL.vertices(lesio5,3)>26);  lesio5(jepa)=[]; 
jepa=find(Mod_XL.vertices(lesio5,3)<-13);  lesio5(jepa)=[]; 
jepa=find(Mod_XL.vertices(lesio5,2)>35);  lesio5(jepa)=[];
% lesio4= zona;
lesio4= [lesio4 zona]; lesio4=sort(unique(lesio4));
jepa=find(Mod_XL.vertices(lesio4,1)<-3);  lesio4(jepa)=[];
jepa=find(Mod_XL.vertices(lesio4,2)<1);  lesio4(jepa)=[];
jepa=find(Mod_XL.vertices(lesio4,3)<44);  lesio4(jepa)=[];


lesio= [lesio zona]; lesio=sort(unique(lesio));
jepa=find(Mod_XL.vertices(lesio,1)<0.2);  lesio(jepa)=[];
jepa=find(Mod_XL.vertices(lesio,3)<45);  lesio(jepa)=[];

% busca pulmonares
PV1=find(FV.matLado(:,1)==15);
PV2=find(FV.matLado(:,1)==16);
% jepa=find(FV.matLado(:,1)==12);

%dibuja
plot3(Mod_XL.vertices(:,1),Mod_XL.vertices(:,2),Mod_XL.vertices(:,3),'ob');hold on
punts=lesionaca;
plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'or')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%
%Generar model
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
load('151125AFLLesion.mat')
lesionaca=[lesio4 lesio6 lesio7 PV1' PV2'];    lesionaca=sort(unique(lesionaca));
nom='151204flutterAFLLesion';
load('140709_AFL2.mat')
[FV2]=generaFib(FV2,lesionaca);
save('151204AFLLesion.mat','FV2','lesionaca','lesio4','lesio6','lesio7','PV1','PV2');
guardaConductancias(FV2,nom);
guardaVecinos(FV2,nom);

%%%%%%%%%%%%%%%%%%%%%
% Para iniciar
%%%%%%%%%%%%%%%%%%%%%
load('140717AFLNodos.mat')
lesioini=lesio;
nom='151204FlutterAFLLesionini';
load('151204AFLLesion.mat')
[FV2]=generaFib(FV2,lesioini);
save('151204AFLLesionini.mat','FV2','lesionaca','lesio4','lesio6','lesio7','lesioini','PV1','PV2');
guardaConductancias(FV2,nom);
guardaVecinos(FV2,nom);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%     dibuixa resultat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
patch('Faces',Mod_S.faces,'Vertices',Mod_S.vertices,'facecolor','c','edgealpha',0.0) 
hold on;    axis equal;
% light('Style','infinite');
    punts=lesio4;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'or')
    punts=lesio6;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'ob')
       punts=lesio7;
   plot3(Mod_XL.vertices(punts,1),Mod_XL.vertices(punts,2),Mod_XL.vertices(punts,3),'.y')
   axis equal
   axis off
   
   nom='1511ModLesionini';
   [FV2]=generaFib(FV2,lesio);
   save('1511ModLesionini','FV2','lesio','stim');
   guardaConductancias(FV2,nom);
    guardaVecinos(FV2,nom);
    stim=stim(1:10:end);
    guardaStim(stim,stim,nom);
    
%%%%%%%%%%%%
%Remod evita rotors
%%%%%%%%%%%%
close all; clear all;
load('E:\0Docs\MATLAB\20140702_Marcado3D\ModelsGraniMenut')
load('151118ModLesion.mat')
[xo,vo]=definelinea2puntos(Mod_S,Mod_XL, zona);
    pas=[];
    for i=1:3
    pas(:,i)=Mod_XL.vertices(:,i)-xo(i);
    end
    dists=[];
    for i=1:length(pas)
    dists(i)=norm(cross(pas(i,:),vo))/norm(vo);
    end
    RemMax=0.4; Remmin=0.01; distmax=20; distmin=50;
    m=(Remmin-RemMax)/(distmin-distmax);
    X1=RemMax+abs(m*distmax)
    Rem= dists*m+X1;
    Rem(find(Rem>RemMax))=RemMax;
    Rem(find(Rem<Remmin))=Remmin;
    
    figure
    aux= patch('Faces',Mod_XL.faces,'Vertices',Mod_XL.vertices,'FaceVertexCData', Rem');
    caxis([Remmin RemMax]);   shading interp; 

    guardaRem( Rem,'1511ModLesion')

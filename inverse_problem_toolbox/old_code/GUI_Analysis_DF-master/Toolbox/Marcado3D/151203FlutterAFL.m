% en Koivumaki
% load('140613Aur.mat')
load('E:\0Docs\MATLAB\20140702_Marcado3D\140613Aur')
load('E:\0Docs\MATLAB\20140702_Marcado3D\ModelsGraniMenut')
%Carregar Models

nom='151204FlutterAFL';
% load('140716AFL3.mat') %no va
load('140709_AFL2.mat')

guardaConductancias(FV2,nom);
guardaVecinos(FV2,nom);

%%%%%%%%%%%%%%%%%%%%%
% Para iniciar
%%%%%%%%%%%%%%%%%%%%%
load('140717AFLNodos.mat')
lesioini=lesio;
nom='151204FlutterAFLini';
[FV2]=generaFib(FV2,lesioini);
guardaConductancias(FV2,nom);
guardaVecinos(FV2,nom);


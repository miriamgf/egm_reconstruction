function [EE,row,col] = bemEEMatrix(model,surf1,surf2)
 
% Algoritmo que procesa la influencia de los potenciales de un nodo a otro
% node. Para ello de calcularan previamente los diferentes angulos solidos.

% Basado en: J.C de Munck, "A linear Discretization of the Volume Conductor 
% Boundary Integral Equation Using Analytically Integral Elements", 
% 1992, IEEE Transaction on Biomedical Engineering, 39(9):986-990. 

% También en: H.A. Schlitt, "Evaluation of Boundary Element Methods for 
% the EEG Forward Problem: Effect of Linear Interpolation", 1995, 
% IEEE Transaction on Biomedical Engineering, 42(1):52-58.

% -- Las filas "row" corresponderan a la surf1 --
% Calculamos el numero de nodos totales de surf1, alamcenado en "rowlen".
% Surf1 puede tener 1 o mas superficies. Ej: Pulmones, Torso, etc (Todo en surf1).
    rowlen = 0;
    for p = surf1, 
        rowlen = rowlen + size(model.surface{p}.node,2);
    end
    
    % Alamaceno en "rows" un vector de 1 hasta numero de nodos de cada 
    % superficie de surf1. 1:Nnodosdecadasuperficie.
    % 
    row = zeros(1,rowlen);
    rows = cell(1,length(surf1));
    rowstart = 0;
    for  p = 1:length(surf1),
        rows{p} = rowstart + [1:size(model.surface{surf1(p)}.node,2)];
        row(rowstart+[1:length(rows{p})]) = surf1(p);
        rowstart = rows{p}(end);
    end    
    
    % -- Las columnas "col" corresponderan a la surf2 --
    % Hago exactamente los mismo que para surf1 y row.
    collen = 0;
    for p = surf2,
        collen = collen + size(model.surface{p}.node,2);
    end
    
    col = zeros(1,collen);
    cols = cell(1,length(surf2));
    colstart = 0;
    for p = 1:length(surf2),
        cols{p} = colstart + [1:size(model.surface{surf2(p)}.node,2)];
        col(colstart+[1:length(cols{p})]) = surf2(p);
        colstart = cols{p}(end);
    end    

    EE = zeros(rowstart,colstart);

    for p = 1:length(surf1),
        for q = 1:length(surf2),
%             tic
            fprintf(1,'\nCalculando matriz de transferencia de potencial superficie %d y %d\n',surf1(p), surf2(q));
            EE(rows{p},cols{q}) = CalcMatrix(model,surf1(p),surf2(q));
%             toc
         end
    end        

    % Para poder ahorrar memoria RAM. Divide y vencerás. Más lento, si no
    % hace falta no usar.
% E22=CalcMatrix(model,surf1(2),surf2(2));
% E11=CalcMatrix(model,surf1(1),surf2(1));
% E12=CalcMatrix(model,surf1(1),surf2(2));
% E21=CalcMatrix(model,surf1(2),surf2(1));


return



function EE = CalcMatrix(model,surf1,surf2)

% A partir de las características geométricas (distancia nodo-nodo y faces)
% calcula los angulos solidos y los procesa para poder calcular la matriz
% influencia nodo-nodo. Utiliza ponderación lineal.

    sigmaIN=model.surface{surf2}.sigma(1);
    sigmaOUT=model.surface{surf2}.sigma(2);
    
    Pts = model.surface{surf1}.node;
    Pos = model.surface{surf2}.node;
    Tri = model.surface{surf2}.face;

    NumPts = size(Pts,2);
    NumPos = size(Pos,2);
    NumTri = size(Tri,2);
    
 if (sigmaIN-sigmaOUT) ~= 0 % Condicional solo para ahorrar tiempo y RAM. 
      
    
    In = ones(1,NumTri);  
      
    GeoData1 = zeros(NumPts,NumTri);
    GeoData2 = zeros(NumPts,NumTri);
    GeoData3 = zeros(NumPts,NumTri);
      
    for p = 1:NumPts,
                  
        % Todos los triangulos, sin considerar autoangulos.
        % Si un nodo forma parte de un triangulo, obviamente no calculo el
        % angulo solido desde ese nodo a su mismo face.
        if surf1 == surf2,
            Sel = find((Tri(1,:) ~= p)&(Tri(2,:)~=p)&(Tri(3,:)~=p));
        else
            Sel = 1:NumTri;
        end    
        
        % Define vectores para nodos p
        ym = Pts(:,p)*ones(1,NumTri);
        y1 = Pos(:,Tri(1,:))-ym;
        y2 = Pos(:,Tri(2,:))-ym;
        y3 = Pos(:,Tri(3,:))-ym;

        epsilon = 1e-15;		% para saber si un nodo se encuentra en el plano de un triángulo %%%%%%%%%%
        gamma = zeros(3,NumTri);

        % En este caso hay ponderación lineal node-node.
        y21 = y2 - y1;
        y32 = y3 - y2;
        y13 = y1 - y3;
        Ny1 = sqrt(sum(y1.^2));
        Ny2 = sqrt(sum(y2.^2));
        Ny3 = sqrt(sum(y3.^2));
        Ny21 = sqrt(sum((y21).^2));
        Ny32 = sqrt(sum((y32).^2));
        Ny13 = sqrt(sum((y13).^2));
        
        % Formule(14)(J.C de Munck, 1992) & Form(A11)(H.A Schlitt, 1995)
        NomGamma = Ny1.*Ny21 + sum(y1.*y21);
        DenomGamma = Ny2.*Ny21 + sum(y2.*y21);
        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(1,W) = -ones(1,size(W,2))./Ny21(W).*log(NomGamma(W)./DenomGamma(W));

        % Formule(14)(J.C de Munck, 1992) & Form(A11)(H.A Schlitt, 1995)
        NomGamma = Ny2.*Ny32 + sum(y2.*y32);
        DenomGamma = Ny3.*Ny32 + sum(y3.*y32);
        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(2,W) = -ones(1,size(W,2))./Ny32(W).*log(NomGamma(W)./DenomGamma(W));

        % Formule(14)(J.C de Munck, 1992) & Form(A11)(H.A Schlitt, 1995)
        NomGamma = Ny3.*Ny13 + sum(y3.*y13);
        DenomGamma = Ny1.*Ny13 + sum(y1.*y13);
        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(3,W) = -ones(1,size(W,2))./Ny13(W).*log(NomGamma(W)./DenomGamma(W));

        % Formule(13)(J.C de Munck, 1992)
        OmegaVec = [1 1 1]'*(gamma(3,:)-gamma(1,:)).*y1 + [1 1 1]'*(gamma(1,:)-gamma(2,:)).*y2 +[1 1 1]'*(gamma(2,:)-gamma(3,:)).*y3; 
        
        d = sum(y1.*cross(y2,y3));
        N = cross(y21,-y13);
        A2 = sum(N.*N);
        
        % Formule (9) (J.C de Munck, 1992).
        Nn = (Ny1.*Ny2.*Ny3+Ny1.*sum(y2.*y3)+Ny3.*sum(y1.*y2)+Ny2.*sum(y3.*y1));
        Omega = zeros(1,NumTri);
        Vz = find(Nn(Sel) == 0);
        Vp = find(Nn(Sel) > 0); 
        Vn = find(Nn(Sel) < 0);
        if size(Vp,1) > 0, Omega(Sel(Vp)) = 2*atan(d(Sel(Vp))./Nn(Sel(Vp))); end;
        if size(Vn,1) > 0, Omega(Sel(Vn)) = 2*atan(d(Sel(Vn))./Nn(Sel(Vn)))+2*pi; end;
        if size(Vz,1) > 0, Omega(Sel(Vz)) = pi*sign(d(Sel(Vz))); end;

        % Formule (11) (J.C de Munck, 1992).
        zn1 = sum(cross(y2,y3).*N); 
        zn2 = sum(cross(y3,y1).*N);
        zn3 = sum(cross(y1,y2).*N);

        % Angulos esféricos. Formule (19) (J.C de Munck, 1992) 
        % Usa tambien: Form(A10)(H.A Schlitt, 1995)
        GeoData1(p,Sel) = In(Sel)./A2(Sel).*((zn1(Sel).*Omega(Sel)) + d(Sel).*sum((y32(:,Sel)).*OmegaVec(:,Sel))); %  vertice1. Interp lineal.
        GeoData2(p,Sel) = In(Sel)./A2(Sel).*((zn2(Sel).*Omega(Sel)) + d(Sel).*sum((y13(:,Sel)).*OmegaVec(:,Sel))); %  vertice2. Interp lineal.
        GeoData3(p,Sel) = In(Sel)./A2(Sel).*((zn3(Sel).*Omega(Sel)) + d(Sel).*sum((y21(:,Sel)).*OmegaVec(:,Sel))); %  vertice3. Interp lineal.
    end 
                  
    EE = zeros(NumPts,NumPos);
    C = (1/(4*pi))*(sigmaIN - sigmaOUT);
%     display(num2str([model.surface{surf2}.sigma(1) model.surface{surf2}.sigma(2)]))
    for q=1:NumPos,
        V = find(Tri(1,:)==q);
        for r = 1:size(V,2), EE(:,q) = EE(:,q) - C*GeoData1(:,V(r)); end;
        V = find(Tri(2,:)==q);
        for r = 1:size(V,2), EE(:,q) = EE(:,q) - C*GeoData2(:,V(r)); end;
        V = find(Tri(3,:)==q);
        for r = 1:size(V,2), EE(:,q) = EE(:,q) - C*GeoData3(:,V(r)); end;
    end
  else
     EE = zeros(NumPts,NumPos);
  end
    
    
    % Formule (21) (J.C de Munck, 1992).
    % En la diagonal. Nodo(i)-Nodo(i)
    if surf1 == surf2,       
        for p = 1:NumPts,
            EE(p,p) = -sum(EE(p,:))+model.surface{surf2}.sigma(1);
%             EE(p,p) = -sum(EE(p,:))+(sigmaIN+sigmaOUT)/2;
        end           
    end        
    

return


function T = bemMatrixPP(model)

% En este programa se computa la matriz de transferencia a partir de la
% matriz de influencia de los potenciales y los gradientes de potencial.

% "hsurf" y "bsurf" contienen el numero de suèrficie. En este caso la
% ultima cada es el heart. Se resituan las capas en bemMatrixPP.
% "hsurf" es la superficie "source". Vamos donde está la densidad de
% corriente. The heart.
% "bsurf" contiene las diferentes superficies del cuerpo. The body.
% Ej: Torso, Lungs, etc.

%     hsurf = length(model.surface);
%     bsurf = 1:(hsurf-1);

    hsurf = 1;
    bsurf = 2:length(model.surface);
    
    Ghh = bemEJMatrix(model,hsurf,hsurf);
    Gbh = bemEJMatrix(model,bsurf,hsurf);  % Y la de los gradientes.   
    [EE,row] = bemEEMatrix(model,[hsurf bsurf],[hsurf bsurf]); % Obtengo la matriz de influencia de potenciales
    
    b = find(row ~= hsurf);   % Indices de Torso
    h = find(row == hsurf);   % Indices de Corazon
    
    Pbb = EE(b,b); % Torso a Torso
    Phh = EE(h,h); % Corazon a Corazon
    Pbh = EE(b,h); % Corazon a Torso
    Phb = EE(h,b); % Torso a Corazon
     
    iGhh=inv(Ghh); % Corazon a Corazon
    
%      % Formula obtenida en diferentes papers como Horacek o Stenroos.
       T = inv(Pbb - Gbh*iGhh*Phb)*(Gbh*iGhh*Phh-Pbh);  % Si el input es el potencial en el epicardio 

% % Otra forma
% % M*[dUh Utorso]=b*Uh;
% M= [Ghh Phb; Gbh Pbb];
% b= -[Phh; Pbh];
% T=M\b;


    
return     





